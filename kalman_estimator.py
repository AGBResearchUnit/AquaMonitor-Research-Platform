"""
kalman_estimator.py — Sediment Cr State Estimator (Kalman Filter)
==================================================================
Reconstructs the unmeasured sediment chromium stock S(t) from
routinely available tissue Cr concentration data.

This implements the M2-based Kalman filter described in:
- Supplementary Data SD-P.1 (derivation)
- Methods §2.5 of the companion paper
- AquaMonitor App P1 (field deployment)

Mathematical basis:
  Observation model (SD-P.1.1e):
    ẑ = CM × ke(kLe+kLM) / (kLM · ku^(s) · fing)

  Prediction step (SD-P.1.2b):
    Ŝ(t|t-1) = Ŝ(t-1|t-1) + [E − δs·Ŝ(t-1) − μ·I] × Δt

  Update step (SD-P.1.3e):
    Ŝ(t|t) = Ŝ(t|t-1) + K · (ẑ − Ŝ(t|t-1))

  Optimal Kalman gain (SD-P.1.4d):
    K* = P⁻ / (P⁻ + R)
    K* = 0.16 / (0.16 + 0.1225) ≈ 0.566 ≈ 0.60 (pre-tuned fixed gain)

Reference: Uno et al. (2026), Exposure and Health. Methods §2.5.
           Kalman (1960), J. Basic Eng. 82:35–45.

Usage:
    from kalman_estimator import KalmanEstimator
    est = KalmanEstimator()
    S_est = est.update(CM_obs=27.40, E=1.715, I=0.0, dt=1.0)
    print(f"Estimated sediment Cr: {S_est:.1f} mg/kg")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from model_chain import BASELINE_PARAMS


@dataclass
class KalmanState:
    """State of the Kalman estimator at a single time step."""
    t: float                    # Time (years since baseline)
    S_prior: float              # Prior estimate (before observation)
    S_posterior: float          # Posterior estimate (after update)
    P_prior: float              # Prior error covariance
    P_posterior: float          # Posterior error covariance
    innovation: float           # ẑ − Ŝ(t|t-1)
    K_gain: float               # Kalman gain applied
    CM_obs: Optional[float]     # Tissue Cr observation used
    S_direct: Optional[float]   # Direct sediment measurement (overrides filter)
    data_source: str            # "tissue_kalman", "direct", or "prediction"


class KalmanEstimator:
    """
    Kalman filter state estimator for sediment Cr reconstruction.

    The estimator fuses:
      (a) Forward ODE prediction from M1 (sediment accumulation model)
      (b) Tissue Cr observation inverted through M2 steady-state

    When a direct sediment measurement is available, it overrides the
    Kalman estimate entirely (equivalent to K=1, R=0).

    Parameters
    ----------
    params : dict, optional
        Override BASELINE_PARAMS values.
    K_fixed : float
        Fixed Kalman gain (pre-tuned; optimal K* ≈ 0.566 ≈ 0.60).
        Justification: model CV ≈ 40% (P⁻ = 0.16),
                       observation CV ≈ 35% (R = 0.1225).
        K* = 0.16 / (0.16 + 0.1225) = 0.566 ≈ 0.60.
    S_init : float, optional
        Initial sediment estimate. If None, uses BASELINE_PARAMS["S0"].
    P_init : float, optional
        Initial error covariance. Default: 0.16 (model variance).
    """

    def __init__(self,
                 params: dict = None,
                 K_fixed: float = 0.60,
                 S_init: float = None,
                 P_init: float = 0.16):

        self.p = {**BASELINE_PARAMS, **(params or {})}
        self.K_fixed = K_fixed      # Fixed gain (pre-tuned optimal)
        self.S_est = S_init if S_init is not None else self.p["S0"]
        self.P_est = P_init         # Error covariance
        self.Q = self._compute_Q()  # Process noise
        self.R = 0.1225             # Observation noise variance (CV=35%)
        self.history: List[KalmanState] = []
        self.t_current = 0.0

    def _compute_Q(self) -> float:
        """
        Process noise variance from CIR-SDE (M6).
        Q = σ₀² · E₀ / (2 · δs)  [variance of CIR stationary distribution]
        """
        p = self.p
        Q = p["sigma0"]**2 * p["E0"] / (2 * p["delta_s"])
        return Q

    def _invert_M2(self, CM_obs: float) -> float:
        """
        Invert M2 steady-state to get sediment estimate from tissue observation.

        Equation (SD-P.1.1e):
            ẑ = CM × ke(kLe + kLM) / (kLM · ku^(s) · fing)

        Valid only when sediment pathway dominates (Cw ≈ 0, i.e.,
        Bioaccumulation Paradox condition is met).
        """
        p = self.p
        ke = p["k_ML"] + p["k_Me"]
        ke_eff = p["k_Le"] + p["k_LM"]
        denom = p["k_LM"] * p["ku_s"] * p["f_ing"]
        S_obs = CM_obs * ke * ke_eff / denom
        return max(S_obs, 0.0)

    def _predict(self, E: float, I: float, dt: float) -> tuple[float, float]:
        """
        Kalman prediction step (SD-P.1.2b–c).

        S_prior(t) = S_post(t-1) + [E − δs·S_post(t-1) − μ·I] × Δt
        P_prior(t) = (1 − δs·Δt)² × P_post(t-1) + Q
        """
        p = self.p
        ds = p["delta_s"]
        mu = p["mu"]

        # Euler forward integration of M1
        dS = E - ds * self.S_est - mu * I
        S_prior = max(self.S_est + dS * dt, 0.0)

        # Covariance prediction (linearised)
        F = 1 - ds * dt                     # State transition coefficient
        P_prior = F**2 * self.P_est + self.Q * dt
        P_prior = max(P_prior, 1e-6)        # Numerical floor

        return S_prior, P_prior

    def _update(self, S_prior: float, P_prior: float,
                 S_obs: float) -> tuple[float, float, float, float]:
        """
        Kalman update step (SD-P.1.3d–f).

        Uses fixed gain K = 0.60 (pre-tuned optimal K* ≈ 0.566).

        innovation = S_obs − S_prior
        S_post = S_prior + K × innovation
        P_post = (1 − K) × P_prior
        """
        innovation = S_obs - S_prior
        K = self.K_fixed
        S_post = S_prior + K * innovation
        P_post = (1 - K) * P_prior
        return S_post, P_post, innovation, K

    def update(self,
               CM_obs: Optional[float] = None,
               S_direct: Optional[float] = None,
               E: float = None,
               I: float = 0.0,
               dt: float = 1.0) -> float:
        """
        Run one Kalman filter step.

        Priority order:
          1. Direct sediment measurement (S_direct) — overrides filter completely
          2. Tissue observation (CM_obs) — inverted via M2, then Kalman update
          3. ODE forward prediction only — when no observation available

        Parameters
        ----------
        CM_obs : float, optional
            Observed muscle Cr concentration (mg/kg ww).
        S_direct : float, optional
            Direct sediment Cr measurement (mg/kg). Overrides filter if provided.
        E : float, optional
            Current emission rate (mg/kg/yr). Defaults to BASELINE_PARAMS["E0"].
        I : float
            Remediation investment this period (M THB).
        dt : float
            Time step (years). Default: 1 year.

        Returns
        -------
        float : Updated sediment Cr estimate Ŝ (mg/kg).
        """
        if E is None:
            E = self.p["E0"]

        self.t_current += dt

        # STEP 1: Prediction
        S_prior, P_prior = self._predict(E, I, dt)

        # STEP 2: Update
        if S_direct is not None:
            # Direct measurement overrides completely
            S_post = max(S_direct, 0.0)
            P_post = 1e-4               # Near-zero uncertainty (measured)
            innovation = S_direct - S_prior
            K_used = 1.0
            source = "direct"
            obs = None

        elif CM_obs is not None and CM_obs > 0:
            # Invert M2 to get observation in S-space
            S_obs = self._invert_M2(CM_obs)
            S_post, P_post, innovation, K_used = self._update(S_prior, P_prior, S_obs)
            S_post = max(S_post, 0.0)
            source = "tissue_kalman"
            obs = CM_obs

        else:
            # No observation — use prediction only
            S_post = S_prior
            P_post = P_prior
            innovation = 0.0
            K_used = 0.0
            source = "prediction"
            obs = None

        # Update state
        self.history.append(KalmanState(
            t=self.t_current,
            S_prior=S_prior, S_posterior=S_post,
            P_prior=P_prior, P_posterior=P_post,
            innovation=innovation, K_gain=K_used,
            CM_obs=obs, S_direct=S_direct,
            data_source=source,
        ))
        self.S_est = S_post
        self.P_est = P_post

        return S_post

    def get_uncertainty(self) -> dict:
        """
        Return current posterior uncertainty (95% CI on S estimate).
        Assumes Gaussian posterior: Ŝ ± 1.96 × √P_posterior.
        """
        sigma_S = np.sqrt(self.P_est)
        return {
            "S_estimate": self.S_est,
            "sigma_S": sigma_S,
            "CI_lower": max(self.S_est - 1.96 * sigma_S, 0),
            "CI_upper": self.S_est + 1.96 * sigma_S,
            "P_posterior": self.P_est,
        }

    def reset(self, S_init: float = None, P_init: float = 0.16):
        """Reset estimator to initial state."""
        self.S_est = S_init if S_init is not None else self.p["S0"]
        self.P_est = P_init
        self.t_current = 0.0
        self.history.clear()

    @classmethod
    def compute_optimal_gain(cls, sigma_model_cv: float = 0.40,
                              sigma_obs_cv: float = 0.35) -> dict:
        """
        Compute the analytically optimal Kalman gain.

        Equation (SD-P.1.4d):
            K* = P⁻ / (P⁻ + R)

        where P⁻ = σ²_model (model error variance),
              R  = σ²_obs   (observation error variance).

        For normalised variances (CV expressed as fraction):
            P⁻ = CV_model² = 0.16
            R  = CV_obs²   = 0.1225
            K* = 0.16 / (0.16 + 0.1225) = 0.566

        Parameters
        ----------
        sigma_model_cv : float
            Model prediction CV (fraction). Default: 0.40 (40%).
        sigma_obs_cv : float
            Tissue observation CV (fraction). Default: 0.35 (35%).

        Returns
        -------
        dict with K_star and variance components.
        """
        P_minus = sigma_model_cv**2
        R_obs   = sigma_obs_cv**2
        K_star  = P_minus / (P_minus + R_obs)
        MSE_at_K_star = (1 - K_star) * P_minus

        return {
            "K_star": round(K_star, 3),
            "P_minus": P_minus,
            "R_obs": R_obs,
            "sigma_model_cv": sigma_model_cv,
            "sigma_obs_cv": sigma_obs_cv,
            "MSE_at_K_star": MSE_at_K_star,
            "MSE_unfiltered": min(P_minus, R_obs),
            "gain_description": (
                f"K* = {P_minus:.4f} / ({P_minus:.4f} + {R_obs:.4f}) "
                f"= {K_star:.3f} ≈ 0.60"
            ),
        }


# ─── DEMO ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AquaMonitor Kalman Estimator — Demo")
    print("=" * 60)

    # 1. Optimal gain derivation
    print("\n▸ Optimal Kalman gain derivation (SD-P.1.4d):")
    gain_result = KalmanEstimator.compute_optimal_gain()
    for k, v in gain_result.items():
        print(f"  {k:<26} = {v}")

    # 2. Single-step inversion (Pond 11 August 2025)
    print("\n▸ Single-step M2 inversion — Pond 11 (August 2025):")
    est = KalmanEstimator(S_init=0.0)   # No prior knowledge
    S_inverted = est._invert_M2(CM_obs=27.40)
    print(f"  Input:  CM_obs = 27.40 mg/kg (measured muscle Cr)")
    print(f"  Output: Ŝ_obs  = {S_inverted:.1f} mg/kg (sediment Cr, inferred)")
    print(f"  Target: S₀     = {BASELINE_PARAMS['S0']} mg/kg (baseline)")
    print(f"  Match:  {abs(S_inverted - BASELINE_PARAMS['S0']):.2f} mg/kg deviation")

    # 3. Full sequential estimation (5-year simulation with annual tissue samples)
    print("\n▸ Sequential Kalman estimation — 5-year field simulation:")
    est2 = KalmanEstimator()
    # Simulate annual tissue observations under Scenario A
    from model_chain import ModelChain, SCENARIOS
    mc = ModelChain()
    t_yr = np.arange(1, 6)
    S_true = mc.M1_sediment_ode(t_yr, E_frac=1.0)
    _, CM_true = zip(*[mc.M2_steady_state(S) for S in S_true])
    # Add measurement noise (CV=35%)
    np.random.seed(42)
    CM_obs_noisy = np.array(CM_true) * (1 + np.random.normal(0, 0.35, 5))
    CM_obs_noisy = np.maximum(CM_obs_noisy, 0.1)

    print(f"  {'Year':<6}  {'S_true':>10}  {'CM_obs':>10}  {'Ŝ_posterior':>13}  {'Error':>8}")
    print(f"  {'-'*52}")
    for i, yr in enumerate(t_yr):
        S_est = est2.update(CM_obs=CM_obs_noisy[i], E=BASELINE_PARAMS["E0"], dt=1.0)
        err = S_est - S_true[i]
        print(f"  {2026+yr:<6}  {S_true[i]:>10.1f}  {CM_obs_noisy[i]:>10.2f}  "
              f"{S_est:>13.1f}  {err:>+8.1f}")

    unc = est2.get_uncertainty()
    print(f"\n  Final estimate: Ŝ = {unc['S_estimate']:.1f} ± {unc['sigma_S']:.1f} mg/kg")
    print(f"  95% CI: [{unc['CI_lower']:.1f}, {unc['CI_upper']:.1f}] mg/kg")

    # 4. Direct measurement override
    print("\n▸ Direct measurement override (highest priority):")
    est3 = KalmanEstimator()
    S_kalman = est3.update(CM_obs=27.40)
    S_direct_override = est3.update(S_direct=38.5)   # Direct ICP-OES
    print(f"  Kalman estimate (tissue only): {S_kalman:.1f} mg/kg")
    print(f"  After direct measurement:      {S_direct_override:.1f} mg/kg (overrides filter)")
    print(f"  Data source: {est3.history[-1].data_source}")
