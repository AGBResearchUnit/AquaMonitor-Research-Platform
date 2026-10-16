# Model Parameters — AquaMonitor M0–M6 Framework

Complete parameter documentation with values, units, ranges, sources, and calibration methods.

Data source for all empirical values: Uno U et al. (2026) *Environments* 13(4), 231. https://doi.org/10.3390/environments13040231

---

## M0 — ADR-PDE Canal Transport

| Symbol | Name | Value | Unit | Range | Source |
|--------|------|-------|------|-------|--------|
| u | Canal flow velocity | 0.15 | m/s | 0.05–0.50 | Pathum Thani canal measurements |
| D | Longitudinal dispersion | 0.80 | m²/s | 0.1–5.0 | Elder (1959) empirical; canal width ~8m |
| k_d | First-order settling rate | 0.005 | d⁻¹ | 0.001–0.020 | Fine suspended sediment literature [1] |
| Q₀ | Industrial discharge load | 2.4 | mg/kg/s·m | 0.5–8.0 | Estimated from M0 fit; uncertainty = land-use variability |
| ℓ | Attenuation length (derived) | ~480 | m | — | Computed: 2D/[√(u²+4Dk_d)−u] |

**Note on E₀ derivation:** The baseline emission deposition rate E₀ = 1.715 mg/kg/yr is derived from M0 by integrating the settling flux k_d·Cw(s) over the canal reach to Pond 11 (480 m distance) and over the historical loading period. E₀ = k_d·[Q₀/√(u²+4Dk_d)]·exp(−480/ℓ)·T_years, where T_years ≈ 20 yr is the estimated industrial operating period.

---

## M1 — Sediment Accumulation ODE

| Symbol | Name | Value | Unit | Range (SA) | CV | Source |
|--------|------|-------|------|------------|-----|--------|
| E₀ | Baseline emission rate | 1.715 | mg/kg/yr | 0.50–4.00 | 35% | M0 derivation (see above) |
| δ_s | Natural sediment decay rate | 0.050 | yr⁻¹ | 0.010–0.150 | 40% | Freshwater sediment literature [2,3] |
| μ | Remediation efficiency | 0.050 | yr⁻¹/M THB | — | — | Expert elicitation; calibrated to BCR data |
| S₀ | Initial sediment Cr (Pond 11) | 34.3 | mg/kg | 10–80 | 25% | **Kalman inversion** from CM_obs = 27.40 mg/kg (see §2.5) |

**Critical note on S₀:** This value is NOT directly measured. It is estimated by inverting the M2 steady-state equation from tissue data. This creates a potential circularity that is addressed in three ways (see Methods §4.5 Limitation 2):
1. Sensitivity ranking (ku^(s) > δs) is invariant across S₀ ∈ [10, 80] mg/kg
2. Profile-likelihood confirms separate identifiability of ku^(s) and S₀ via liver:muscle ratio
3. The qualitative conclusion (sediment pathway dominates when Cw = BDL) is independent of S₀ value

**Sediment decay rate (δs) literature references:**
- [2] Wudtisin I & Boyd CE (2006) *Aquaculture Research* 37:1202 — catfish pond sediments in Thailand: decomposition rates 0.04–0.08 yr⁻¹
- [3] Lin CK & Yi Y (2003) *Aquaculture* 226:57 — freshwater aquaculture sediment burial: typical t½ = 8–15 years → δs = 0.046–0.087 yr⁻¹

---

## M2 — Two-Compartment Bioaccumulation

All kinetic rate constants calibrated from M2 steady-state fit to Pond 11 tissue data (liver and muscle) from Uno et al. (2026). The liver:muscle ratio provides a second constraint that partially separates ku^(s) from S₀.

| Symbol | Name | Value | Unit | Range (SA) | CV | Source |
|--------|------|-------|------|------------|-----|--------|
| ku^(s) | Sediment–fish uptake rate | 0.032 | d⁻¹ | 0.005–0.080 | 45% | **M2 calibration** from Pond 11 (see below) |
| ku^(w) | Aqueous uptake rate | 0.0015 | d⁻¹ | — | — | Zero at Pond 11 (Cw=BDL); from Ruangsomboon & Wongrat (2006) [4] |
| ku^(d) | Dietary uptake rate | 0.0008 | d⁻¹ | — | — | Assumed negligible; catfish fed commercial feed |
| k_LM | Liver→muscle transfer | 0.280 | d⁻¹ | 0.10–0.60 | 30% | M2 calibration |
| k_ML | Muscle→liver transfer | 0.180 | d⁻¹ | — | — | M2 calibration |
| k_Le | Liver elimination | 0.120 | d⁻¹ | — | — | M2 calibration |
| k_Me | Muscle elimination | 0.140 | d⁻¹ | 0.05–0.35 | 30% | M2 calibration |
| f_ing | Sediment ingestion fraction | 0.350 | — | 0.10–0.70 | 35% | Literature; benthic-feeding Clarias [5] |

**M2 calibration procedure:**
The five calibration parameters (ku^(s), k_LM, k_ML, k_Le, k_Me) were determined by minimising the sum of squared residuals between model-predicted and observed (liver Cr, muscle Cr) pairs at the five ponds with detectable tissue Cr (5A, 5B, 11, 14A; three Cr-positive ponds + 14A). The fitting criterion was:

  argmin_{θ} Σ_ponds [(CL_pred − CL_obs)² + (CM_pred − CM_obs)²]

subject to all rates > 0. The objective function was minimised using scipy.optimize.minimize (L-BFGS-B method). Goodness of fit: R² = 0.91 for muscle Cr, R² = 0.87 for liver Cr across calibration ponds.

**References:**
- [4] Ruangsomboon S & Wongrat L (2006) *Aquatic Toxicology* 78:15
- [5] Avenant-Oldewage A & Marx HM (2000) *Water SA* 26:569

---

## M3 — Oxidative-Stress Bistability (Hill-function ODE)

| Symbol | Name | Value | Unit | Interpretation |
|--------|------|-------|------|----------------|
| α | ROS production coefficient | 0.15 | — | Rate of ROS increase per unit tissue Cr |
| β | Quenching rate | 2.80 | — | Antioxidant quench effectiveness |
| γ | AOX induction rate | 0.60 | — | ROS-induced antioxidant upregulation |
| δ | AOX degradation rate | 0.25 | d⁻¹ | Antioxidant turnover |
| K | Hill saturation constant | 3.0 | mg/kg | Cr at half-maximal antioxidant response |
| n | Hill coefficient | 2.5 | — | >2 required for bistability |

**Bistability threshold: Ĉf ∈ [2.03, 5.19] mg/kg Cr**

The bounds 2.03 and 5.19 mg/kg correspond to the observed muscle Cr concentrations at the two ponds on opposite sides of the phenotypic transition:
- 2.03 mg/kg = mean Cr at Pond 5A (adaptive phenotype: high CAT, GPx; low MDA)
- 5.19 mg/kg = mean Cr at Pond 14A (injury phenotype: high MDA, NO; low GSH, CAT)

These serve as empirical bounds on the bistability window. The Hill-function parameters above are calibrated to produce a cubic nullcline with saddle-node bifurcation within this empirical range.

---

## M4 — Gene Regulatory Jacobian (6×6)

Genes: cat, cyp1a, gpx, met, tnf, star

Key off-diagonal element explaining the NYC star paradox:

| Element | Value | Interpretation |
|---------|-------|----------------|
| J[star, tnf] | −0.68 | TNF-α represses steroidogenesis (dominant at low Cf) |
| J[cat, Cf] | +0.12 | Cr exposure induces CAT at low Cf |
| J[cyp1a, Cf] | +0.28 | Cr(VI) activates AhR pathway |
| J[met, Cf] | −0.15 | Met repressed when saturation capacity exceeded |

All six eigenvalues of J satisfy Re(λ) < 0, confirming network stability (see SD1 §S4).

NYC star paradox: ΔCt(star) = −6.5 despite lowest CM(Cr) = 0.0 mg/kg. Explained by peak tnf (+2.8) at NYC activating J[star,tnf] < 0 coupling, suppressing steroidogenesis independent of metal burden.

---

## M5 — USEPA Health Risk

| Symbol | Name | Value | Unit | Source |
|--------|------|-------|------|--------|
| IR | Fish ingestion rate | 21.5 | g/day (wet wt) | Thai dietary survey, Fisheries Dept 2024 |
| EF | Exposure frequency | 365 | days/yr | USEPA standard assumption |
| ED | Exposure duration | 70 | years | USEPA lifetime assumption |
| BW_F | Female body weight | 50.0 | kg | Thai NHES 2023 |
| BW_M | Male body weight | 57.0 | kg | Thai NHES 2023 |
| AT | Averaging time (carcinogen) | 25,550 | days | 70 × 365 |
| RfD_Cr | Oral reference dose, Cr | 0.003 | mg/kg/day | USEPA IRIS |
| RfD_Pb | Oral reference dose, Pb | 0.0035 | mg/kg/day | USEPA IRIS |
| RfD_Ni | Oral reference dose, Ni | 0.020 | mg/kg/day | USEPA IRIS |
| RfD_Cd | Oral reference dose, Cd | 0.001 | mg/kg/day | USEPA IRIS |
| CSF_Cr | Cancer slope factor, Cr | 0.5 | (mg/kg/day)⁻¹ | USEPA IRIS |

**CRITICAL NOTE ON CHROMIUM CARCINOGENICITY:**
CSF_Cr = 0.5 applies specifically to hexavalent chromium [Cr(VI)], which is IARC Group 1 (carcinogenic to humans). Trivalent chromium [Cr(III)] is IARC Group 3 (not classifiable). ICP-OES measures **total chromium**. Therefore, all TR values in this study are **conservative upper bounds** assuming 100% Cr(VI).

If Cr(VI) constitutes fraction f of total Cr: TR_corrected = f × TR_total
- f = 1.0 (worst case): TR_Pond11_W = 7.0×10⁻³ (as reported)
- f = 0.30 (typical for industrial effluent): TR_Pond11_W ≈ 2.1×10⁻³ (still 21× USEPA limit)
- f = 0.10 (low Cr(VI)): TR_Pond11_W ≈ 7.0×10⁻⁴ (still 7× USEPA limit)

Even under the lowest plausible Cr(VI) fraction, TR exceeds the USEPA upper acceptable limit.

---

## M6 — CIR-SDE Stochastic Model

| Symbol | Name | Value | Unit | Derivation |
|--------|------|-------|------|-----------|
| κ | Mean-reversion rate | = δs | yr⁻¹ | κ ≡ δs (links SDE to M1 scenario params) |
| θ | Long-run equilibrium | = E₀/δs | mg/kg | θ ≡ E₀/δs (M1 steady state) |
| σS | State-dependent volatility | σ₀√S | — | CIR specification |
| σ₀ | Noise coefficient | 0.300 | — | Estimated from inter-pond S variability |
| S_crit | Ecological collapse threshold | 150 | mg/kg | Upper bound of bistability bifurcation |

**Note on 90% collapse probability (Scenario A, 2036):**
This is a STOCHASTIC quantity (cumulative first-passage probability P(τ_fp ≤ 10 yr)). The deterministic mean path reaches S(10yr) = 134 mg/kg, which is BELOW the 150 mg/kg threshold. The 90% figure represents the probability that the CIR stochastic trajectory (with diffusive tail) crosses the threshold by 2036. These are fundamentally different quantities and must not be conflated.

---

## Kalman Filter (SD-P.1)

| Symbol | Name | Value | Derivation |
|--------|------|-------|-----------|
| K | Fixed Kalman gain | 0.60 | Optimal K* = P⁻/(P⁻+R) = 0.16/(0.16+0.1225) = 0.566 ≈ 0.60 |
| P⁻ | Model error variance | 0.16 | CV_model = 40% → P⁻ = 0.40² = 0.16 |
| R | Observation error variance | 0.1225 | CV_obs = 35% → R = 0.35² = 0.1225 |
| Q | Process noise | 1.54 (mg/kg)² | σ₀²·E₀/(2δs) = 0.09×1.715/0.1 |

---

## Sensitivity Analysis Parameter Ranges

All ranges used in Sobol' quasi-Monte Carlo sampling (uniform distribution; N=4,096 base samples):

| Parameter | Min | Max | CV | Rank (ST_i) |
|-----------|-----|-----|-----|------------|
| ku^(s) | 0.005 | 0.080 | 45% | 1 (ST=0.56) |
| δs | 0.010 | 0.150 | 40% | 2 (ST=0.38) |
| IR | 0.005 | 0.050 | 25% | 3 (ST=0.16) |
| BW_F | 40.0 | 65.0 | 12% | 4 (ST=0.12) |
| E₀ | 0.50 | 4.00 | 35% | 5 (ST=0.14) |
| CSF_Cr | 0.20 | 0.80 | 20% | 6 (ST=0.08) |
| k_LM | 0.10 | 0.60 | 30% | 7 |
| f_ing | 0.10 | 0.70 | 35% | 8 |
| k_Me | 0.05 | 0.35 | 30% | 9 |
| S₀ | 10.0 | 80.0 | 25% | 10 |
| σ₀ | 0.05 | 0.60 | 40% | 11 (non-influential for TR) |
| k_d | 0.001 | 0.020 | 30% | 12 (non-influential for TR) |

Note: σ₀ and k_d are non-influential for TR (μ*<0.10, ST<0.06) but matter for P(collapse) under M6.
