# AquaMonitor Research Platform

**Heavy Metal Cancer Risk in Canal-Fed *Clarias* Catfish — Complete Source Code Repository**

> Companion to: Uno et al. (2026). "Decades of Sediment, Not Present-Day Water, Drive Heavy Metal Cancer Risk in Canal-Fed *Clarias* Catfish: A Mathematical Framework for Detection and Remediation." *Exposure and Health.* DOI: [to be inserted on acceptance]
>
> Primary empirical data: Uno U, Singchat W, Panthum T, Chaiyes A, Ekerette E, Edem U, Jeamsripong S, Uchuwittayakul A, Sirisaksoontorn W, Sinthuvanich C, Srikulnath K (2026). Spatial land use dynamics driving molecular stress and unacceptable human health risks in standardized catfish aquaculture systems. *Environments* 13(4), 231. https://doi.org/10.3390/environments13040231

---

## Repository Overview

This repository contains the complete source code for the AquaMonitor Research Platform — a suite of nine interoperable, zero-dependency web tools implementing the M0–M6 mathematical framework for heavy metal bioaccumulation and cancer risk assessment in canal-fed aquaculture systems.

**Zero dependencies.** Every file is a single self-contained HTML5/CSS3/JavaScript document. No server, no database, no installation required. Download and open in any modern browser.

**Fully traceable.** Every numerical output is computed from the framework equations documented in Supplementary Data SD1 and SD-P of the companion paper. No hardcoded results.

---

## Repository Structure

```
aquamonitor_repo/
│
├── README.md                          ← This file
├── LICENSE                            ← MIT License
├── CITATION.cff                       ← Machine-readable citation
├── INDEX.html                         ← Master index (open this first)
│
├── platform/                          ← All 16 platform HTML files
│   │
│   ├── CORE APPLICATION
│   │   ├── aquamonitor_app.html       ← P1: Field Data App (Kalman + M1–M5)
│   │   ├── aquamonitor_system.html    ← P1b: Full system dashboard
│   │   ├── website_home.html          ← P2: Public landing page
│   │   ├── website_portal.html        ← P3: Research portal / navigation hub
│   │
│   ├── ANALYTICAL TOOLS
│   │   ├── aquaculture_results_dashboard.html   ← P4: Interactive results (Chart.js)
│   │   ├── aquaculture_math_model.html          ← Math framework (M0–M5)
│   │   ├── aquaculture_math_forecast_explained.html  ← Forecast math explained
│   │   ├── aquaculture_economics_model.html     ← P5: Economics (E1–E8)
│   │   ├── aquaculture_forecast_2026_2036.html  ← Scenario forecasts
│   │   ├── aquaculture_interventions.html       ← P6: 25-intervention catalogue
│   │
│   ├── SUPPLEMENTARY CONTENT (A–E)
│   │   ├── A_complete_study_results.html        ← All Tables T1–T3, S4–S5
│   │   ├── B_complete_economics.html            ← Full economic derivations
│   │   ├── C_complete_forecast.html             ← Full forecast layers
│   │   ├── D_complete_math_derivations.html     ← All model derivations
│   │   └── E_intervention_mathematics.html      ← Intervention calculus
│
├── analysis/                          ← Python analysis and figure scripts
│   ├── requirements.txt               ← Python dependencies
│   ├── model_chain.py                 ← M0–M6 model chain implementation
│   ├── sensitivity_analysis.py        ← OAT, Morris, Sobol' analysis
│   ├── scenario_forecasts.py          ← 10-year scenario computations
│   ├── health_risk.py                 ← USEPA health risk calculations
│   ├── stochastic_model.py            ← CIR-SDE / Fokker-Planck (M6)
│   ├── make_figures.py                ← Figure 1–5 generation (300 dpi)
│   └── kalman_estimator.py            ← Kalman filter state estimator
│
└── docs/
    ├── PARAMETERS.md                  ← All model parameters with sources
    ├── EQUATIONS.md                   ← All equations in readable form
    └── DATA_DICTIONARY.md             ← Variable names and units
```

---

## Quick Start

### Open the platform (no installation)

1. Download or clone this repository
2. Open `INDEX.html` in any modern browser
3. Navigate to any component from the index

**Recommended reading order:**
1. `A_complete_study_results.html` — understand the data
2. `aquaculture_results_dashboard.html` — interactive visualisation
3. `aquaculture_math_model.html` — the model chain
4. `D_complete_math_derivations.html` — step-by-step proofs
5. `B_complete_economics.html` — economic frameworks
6. `C_complete_forecast.html` — scenario forecasts
7. `aquamonitor_app.html` — field deployment application

### Run the Python analysis

```bash
cd analysis/
pip install -r requirements.txt
python make_figures.py          # Reproduces Figures 1–5 at 300 dpi
python sensitivity_analysis.py  # Reproduces Tables 4–5 (Morris + Sobol')
python scenario_forecasts.py    # Reproduces Table 6
```

---

## Platform Components

| ID | File | Description | Models |
|----|------|-------------|--------|
| P1 | `aquamonitor_app.html` | Field data collection and forecast app with Kalman estimator | M0–M5, Kalman |
| P1b | `aquamonitor_system.html` | Full system dashboard | All |
| P2 | `website_home.html` | Public landing page | Communication |
| P3 | `website_portal.html` | Research portal and navigation hub | All |
| P4 | `aquaculture_results_dashboard.html` | 8-panel interactive results | M5 outputs |
| P5 | `aquaculture_economics_model.html` | 8 economic frameworks (E1–E8) | E1–E8 |
| P6 | `aquaculture_interventions.html` | 25-intervention catalogue with BCR | M1–M5 modified |
| — | `aquaculture_math_model.html` | Mathematical framework (M0–M5) | M0–M5 |
| — | `aquaculture_forecast_2026_2036.html` | 3-scenario × 6-domain forecast | M1, M5, M6 |
| — | `aquaculture_math_forecast_explained.html` | How math generates forecasts | M1–M5 |
| A | `A_complete_study_results.html` | All data tables (T1–T3, S4–S5) | Data |
| B | `B_complete_economics.html` | Game theory, Coase, tax, trade | E1–E8 |
| C | `C_complete_forecast.html` | Uncertainty, seasonal, climate, tipping | M1, M5, M6 |
| D | `D_complete_math_derivations.html` | All model derivations step-by-step | M0–M6 |
| E | `E_intervention_mathematics.html` | Calculus for all 25 interventions | M1–M5 |

---

## Mathematical Framework

The platform implements the following model chain:

```
Industrial discharge
        ↓
M0: 2D ADR-PDE (canal transport)         Cw(s) = Q₀/√(u²+4Dk_d) · exp(−s/ℓ)
        ↓
M1: Sediment ODE                          dS/dt = E(t) − δₛS − μI(t)
        ↓
M2: Two-compartment bioaccumulation       CM* = kLM·ku(s)·fing·S / ke(kLe+kLM)
        ↓
M3: Oxidative-stress bistability          Cubic nullcline system; Ĉf ∈ [2.03, 5.19]
        ↓
M4: Gene regulatory Jacobian              6×6 J; all Re(λ) < 0
        ↓
M5: USEPA health risk                     TR = CDI_Cr × CSF_Cr
        ↓
M6: CIR-SDE / Fokker-Planck              P(collapse) = 1 − γ(a, β·Scrit)/Γ(a)
```

**Key finding:** Pond 11, Pathum Thani: TR(Women) = 7.0×10⁻³ (70× USEPA limit) despite all waterborne metals below detection. The Bioaccumulation Paradox is formally proven: Δᵢ → ∞ as Cw → 0 with finite sediment stock S.

**Dominant parameters (Sobol'):** ku^(s) (STi = 0.56), δs (STi = 0.38) jointly explain 67% of TR variance.

---

## Model Parameters

All calibrated parameter values are in `docs/PARAMETERS.md`. Key values:

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Sediment–fish uptake rate | ku^(s) | 0.032 d⁻¹ | M2 fit to tissue data |
| Sediment decay rate | δs | 0.05 yr⁻¹ | Freshwater sediment literature |
| Baseline emission | E₀ | 1.715 mg/kg/yr | M0 ADR-PDE calibration |
| Initial sediment Cr (Pond 11) | S₀ | 34.3 mg/kg | Kalman inversion from tissue |
| Ingestion rate | IR | 21.5 g/day | Thai dietary survey (Fisheries Dept 2024) |
| Body weight (female) | BW_F | 50 kg | Thai NHES 2023 |
| Cancer slope factor | CSF_Cr | 0.5 (mg/kg/d)⁻¹ | USEPA IRIS (total Cr, upper bound) |
| Kalman gain | K | 0.60 | Analytical optimum K* = 0.566 |

> **Note on chromium carcinogenicity:** CSF_Cr = 0.5 applies to hexavalent chromium (Cr(VI)). ICP-OES measures total Cr. All TR values are therefore conservative upper bounds. If only 30% of tissue Cr is hexavalent, TR values would decrease by a factor of ~3 while remaining well above the USEPA limit.

---

## Intervention Portfolio

The optimal four-intervention portfolio achieves a 1,000-fold TR reduction:

| Intervention | Mechanism | TR multiplier | BCR |
|---|---|---|---|
| I12: Pigouvian tax | E → 0.05E₀ | ×0.05 | 9.0+ |
| I1: Sediment dredging | S → (1−φ)S, φ=0.80 | ×0.20 | 8.2 |
| I18: Biochar amendment | fing → 0.15 fing | ×0.15 | 4.2 |
| I21: Dietary advisory | IR → 0.50 IR | ×0.50 | 75 |
| **Portfolio combined** | **All separable terms** | **×0.00075** | **NPV +1,400M THB** |

> **Proof of multiplicativity:** Because each intervention modifies a different separable factor in the master TR equation, their fractional reductions multiply: 0.05 × 0.20 × 0.15 × 0.50 = 0.00075 ≈ 10⁻³. See `E_intervention_mathematics.html` and SD-P.4 for full derivation.

> **Important note on canal filtration (I4):** This intervention has **zero effect at Pond 11** (where Cw = BDL). Proven algebraically: ku^(w) × 0 × ηfilt = 0 regardless of filtration efficiency. See `E_intervention_mathematics.html` SD-P.3.

---

## Scenario Forecasts (2026–2036, Pond 11)

| Variable | 2026 | A (2036, no action) | B (2036, partial) | C (2036, full) |
|---|---|---|---|---|
| S(t) Cr mg/kg | 34.3 | **134** (163 w/climate) | 47 | 5.0 |
| CM(Cr) mg/kg | 27.4 | **93** | 24 | 2.1 |
| TR (Women) | 7.0×10⁻³ | **2.4×10⁻²** | 5.8×10⁻³ | 8×10⁻⁵ |
| P(collapse) | 2% | **90%** (>99% w/climate) | 11% | 0.2% |
| NPV M THB | — | **−8,200** | −2,800 | +1,400 |

95% CI on TR: multiply by [0.32, 3.14] (lognormal, σ_lnTR = 0.58).

---

## Technical Details

### Platform technology stack

| Component | Technology |
|---|---|
| Structure | HTML5 |
| Styling | CSS3 (custom properties, flexbox, grid) |
| Interactivity | Vanilla JavaScript (ES2020) |
| Charts | Chart.js v4.4.1 (CDN) |
| Math typesetting | MathJax v3.2.2 (CDN) |
| Data persistence | Browser localStorage (AquaMonitor App) |
| Fonts | Google Fonts (Syne, JetBrains Mono, Spectral) |
| Server required | None |

### Python analysis stack

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥1.24 | Array operations |
| scipy | ≥1.11 | ODE solving, statistics |
| matplotlib | ≥3.7 | Figure generation |
| SALib | ≥1.4 | Sensitivity analysis (Morris, Sobol') |
| pandas | ≥2.0 | Data handling |

### Browser compatibility

Tested on: Chrome 120+, Firefox 120+, Safari 17+, Edge 120+.
Mobile: iOS Safari 16+, Android Chrome 120+.

---

## Data Availability

All empirical data underlying this platform are available from the companion paper:
- Uno et al. (2026), *Environments* 13(4), 231. https://doi.org/10.3390/environments13040231

Raw measurement data (tissue metals, biomarkers, gene expression) are reproduced in `A_complete_study_results.html` (Tables T1–T3, S4–S5) and in `analysis/` as structured data.

---

## Citation

If you use this platform or its code in your research, please cite:

```bibtex
@software{aquamonitor2026,
  title   = {AquaMonitor Research Platform},
  author  = {Uno, Ukam and Singchat, Worapong and Panthum, Thitipong
             and Srikulnath, Kornsorn and {AquaMonitor Development Team}},
  year    = {2026},
  url     = {https://zenodo.org/[DOI to be inserted on acceptance]},
  version = {1.0.0}
}
```

And the companion paper:
```bibtex
@article{uno2026sediment,
  title   = {Decades of Sediment, Not Present-Day Water, Drive Heavy Metal
             Cancer Risk in Canal-Fed \textit{Clarias} Catfish:
             A Mathematical Framework for Detection and Remediation},
  author  = {Uno, Ukam and Singchat, Worapong and Panthum, Thitipong
             and Chaiyes, Aingorn and Ekerette, Ekerette and Edem, Uduak
             and Jeamsripong, Saharuetai and Uchuwittayakul, Anurak
             and Sirisaksoontorn, Weekit and Sinthuvanich, Chomdao
             and Srikulnath, Kornsorn},
  journal = {Exposure and Health},
  year    = {2026},
  doi     = {[to be inserted on acceptance]}
}
```

---

## License

MIT License. See `LICENSE` for full text.

Copyright (c) 2026 Uno et al. / AquaMonitor Development Team

Permission is hereby granted, free of charge, to any person obtaining a copy of this software to use, copy, modify, merge, publish, and distribute it, subject to the conditions in LICENSE.

---

## Contact

Corresponding author: Kornsorn Srikulnath (kornsorn.s@ku.ac.th)
Laboratory of Animal Cytogenetics and Comparative Genomics, Kasetsart University, Bangkok, Thailand

---

*AquaMonitor v1.0.0 · Pathum Thani Catfish Heavy Metal Study · June 2026*
