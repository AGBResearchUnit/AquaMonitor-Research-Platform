"""
make_figures.py — Generate Figures 1–5 at 300 dpi
====================================================
Reproduces all five figures from the companion paper.
Output: fig1_framework.png, fig2_contamination.png,
        fig3_biomarkers.png, fig4_sensitivity.png, fig5_scenarios.png

Reference: Uno et al. (2026), Exposure and Health.
Data:      Uno et al. (2026), Environments 13(4), 231.

Usage:
    python make_figures.py [--dpi 300] [--outdir ./figures]
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from model_chain import ModelChain, BASELINE_PARAMS, SCENARIOS, STUDY_DATA

# ─── STYLE ────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":    "#0d1520",
    "panel": "#111c2a",
    "rim":   "#1e3048",
    "text":  "#c8d8e8",
    "muted": "#4a6880",
    "gold":  "#e8a820",
    "red":   "#e83840",
    "amber": "#f09030",
    "teal":  "#18b898",
    "blue":  "#4898e0",
    "violet":"#9870d0",
}

def set_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["panel"],
        "axes.edgecolor":    PALETTE["rim"],
        "axes.labelcolor":   PALETTE["text"],
        "axes.titlecolor":   PALETTE["text"],
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "grid.color":        PALETTE["rim"],
        "grid.linewidth":    0.5,
        "text.color":        PALETTE["text"],
        "font.family":       "DejaVu Sans",
        "font.size":         8,
        "legend.facecolor":  PALETTE["panel"],
        "legend.edgecolor":  PALETTE["rim"],
        "legend.fontsize":   7,
        "lines.linewidth":   1.5,
        "figure.dpi":        100,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.facecolor": PALETTE["bg"],
    })

PONDS = STUDY_DATA["ponds"]
METALS = ["Cr", "Pb", "Ni", "Cd", "Fe", "Mn", "Zn"]

# ─── FIGURE 2: CONTAMINATION & HEALTH RISK ───────────────────────────────────
def make_fig2(outdir: str):
    """
    Figure 2: Heavy metal contamination and human health risk.
    (a) Waterborne metals heatmap  (b) Muscle tissue heatmap
    (c) Hazard Index bar chart      (d) Target Cancer Risk bar chart
    """
    # Study data from Uno et al. (2026), Environments 13(4), 231
    water_data = np.array([
        [0,    0,    0,    0,    0,    0,    0   ],  # Cr — all BDL
        [0,    0,    0, 0.07, 15.76, 0.11, 0.13 ],  # Zn
        [0,    0,    0, 0.02, 83.99, 0.02, 0.06 ],  # Fe
        [0,    0,    0, 0.014,56.64, 0.003,0.010],  # Mn
        [0,    0,    0, 0.016, 0,    0.019, 0.023], # Pb
        [0,    0,    0,    0,    0,    0, 0.001 ],   # Ni
        [0,    0,    0,    0,    0,    0,    0   ],  # Cd
    ])
    muscle_data = np.array([
        [2.03, 3.03, 27.40, 0,     5.19,  0,     0    ],  # Cr
        [0,    0,    0,     0.018, 2.08,  0.022, 0.012],  # Pb
        [0,    0,    4.40,  0,     3.15,  0,     0.001],  # Ni
        [0,    0,    0,     0,     0.32,  0,     0    ],  # Cd
        [142,  58.3, 308.7, 0.037, 483.6, 0.082, 0.082],  # Fe
        [9.78, 14.21,56.14, 0.001, 55.69, 0,     0    ],  # Mn
        [87.5, 87.3, 122.2, 0.035, 130.3, 0.051, 0.046],  # Zn
    ])
    HI_W = [0.271, 0.404, 3.741, 0.002, 1.121, 0.002, 0.001]
    HI_M = [0.237, 0.354, 3.282, 0.0018, 0.983, 0.0022, 0.0012]
    TR_W = [4.1e-4, 6.1e-4, 7.0e-3, 6.1e-8, 2.9e-3, 7.5e-8, 4.1e-7]
    TR_M = [3.6e-4, 5.3e-4, 6.1e-3, 5.4e-8, 2.6e-3, 6.6e-8, 3.3e-7]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                            left=0.08, right=0.95, top=0.92, bottom=0.08)

    # (a) Water heatmap
    ax_a = fig.add_subplot(gs[0, 0])
    wlog = np.log10(np.where(water_data > 0, water_data, np.nan))
    im_a = ax_a.imshow(wlog, aspect="auto", cmap="RdYlBu_r",
                        vmin=-3, vmax=2, interpolation="nearest")
    ax_a.set_xticks(range(7)); ax_a.set_xticklabels(PONDS, fontsize=7)
    ax_a.set_yticks(range(7)); ax_a.set_yticklabels(METALS, fontsize=7)
    ax_a.set_title("(a) Waterborne metals (log₁₀ ppm)", pad=8)
    ax_a.axvline(2.5, color=PALETTE["red"], lw=2, ls="--", alpha=0.7)
    ax_a.text(2, -0.7, "Pond 11\n(all BDL)", color=PALETTE["red"],
              fontsize=6, ha="center", va="center")
    plt.colorbar(im_a, ax=ax_a, label="log₁₀ ppm", fraction=0.04)

    # (b) Muscle heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    mlog = np.log10(np.where(muscle_data > 0, muscle_data, np.nan))
    im_b = ax_b.imshow(mlog, aspect="auto", cmap="RdYlBu_r",
                        vmin=-2, vmax=3, interpolation="nearest")
    ax_b.set_xticks(range(7)); ax_b.set_xticklabels(PONDS, fontsize=7)
    ax_b.set_yticks(range(7)); ax_b.set_yticklabels(METALS, fontsize=7)
    ax_b.set_title("(b) Muscle tissue (log₁₀ mg kg⁻¹ ww)", pad=8)
    ax_b.add_patch(plt.Rectangle((1.5, -0.5), 1, 1,
                                   fill=False, edgecolor=PALETTE["red"], lw=2))
    ax_b.text(2, 0, "★", color=PALETTE["gold"], fontsize=14, ha="center", va="center")
    plt.colorbar(im_b, ax=ax_b, label="log₁₀ mg/kg", fraction=0.04)

    # (c) Hazard Index
    ax_c = fig.add_subplot(gs[1, 0])
    x = np.arange(7)
    ax_c.bar(x - 0.2, HI_M, 0.38, label="Men", color=PALETTE["blue"], alpha=0.85)
    ax_c.bar(x + 0.2, HI_W, 0.38, label="Women", color=PALETTE["violet"], alpha=0.85)
    ax_c.axhline(1.0, color=PALETTE["red"], ls="--", lw=1.5, label="HI threshold")
    ax_c.set_xticks(x); ax_c.set_xticklabels(PONDS, fontsize=7)
    ax_c.set_ylabel("Hazard Index (HI)"); ax_c.set_title("(c) Hazard Index", pad=8)
    ax_c.legend(fontsize=7); ax_c.grid(True, axis="y", alpha=0.3)
    ax_c.set_ylim(0, max(HI_W) * 1.15)

    # (d) Cancer Risk
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.bar(x - 0.2, TR_M, 0.38, label="Men", color=PALETTE["blue"], alpha=0.85)
    ax_d.bar(x + 0.2, TR_W, 0.38, label="Women", color=PALETTE["violet"], alpha=0.85)
    ax_d.axhline(1e-4, color=PALETTE["red"], ls="--", lw=1.5, label="USEPA upper")
    ax_d.axhline(1e-6, color=PALETTE["amber"], ls=":", lw=1.2, label="USEPA lower")
    ax_d.set_yscale("log"); ax_d.set_ylim(1e-8, 1e-1)
    ax_d.set_xticks(x); ax_d.set_xticklabels(PONDS, fontsize=7)
    ax_d.set_ylabel("Target Cancer Risk (TR)"); ax_d.set_title("(d) Cancer Risk (TR)", pad=8)
    ax_d.legend(fontsize=7); ax_d.grid(True, axis="y", alpha=0.3, which="both")

    fig.suptitle("Figure 2 — Heavy Metal Contamination and Human Health Risk",
                  fontsize=10, fontweight="bold", y=0.98)
    path = os.path.join(outdir, "fig2_contamination.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── FIGURE 3: BIOMARKERS & GENE EXPRESSION ──────────────────────────────────
def make_fig3(outdir: str):
    """
    Figure 3: Oxidative stress biomarkers and gene expression profiles.
    (a) Biomarker bar chart   (b) Gene expression heatmap
    """
    # Biomarker data (mean; n=6/pond) from Uno et al. (2026), Table S4
    # Values normalised to within-study max for comparison on single axis
    raw = {
        "CAT":  [194.56, 142.80, 126.40, 98.20, 71.30, None, 112.40],
        "SOD":  [2980.40, 2150.80, 2340.60, 1840.20, 1640.80, None, 6663.60],
        "GSH":  [9.24, 8.14, 11.04, 7.42, 4.32, None, 8.92],
        "GR":   [3.42, 2.85, 2.68, 2.14, 1.84, None, 2.98],
        "GPx":  [0.130, 0.088, 0.074, 0.082, 0.048, None, 0.112],
        "MDA":  [1.82, 2.40, 3.50, 1.64, 7.80, None, 1.44],
        "NO":   [18.40, 24.20, 32.10, 14.80, 42.60, None, 12.40],
    }
    gene_data = {
        "cat":   [2.4, 1.8, -2.1, 0.8, -4.2, None, 0.4],
        "cyp1a": [1.2, 1.4, 3.2, 0.4, 2.8, None, -0.2],
        "gpx":   [1.8, 1.2, 2.1, 0.6, -1.8, None, 0.8],
        "met":   [0.8, 0.6, -3.5, 1.2, -2.8, None, 0.4],
        "tnf":   [-1.2, -0.8, -2.8, -0.4, 1.4, None, 2.8],
        "star":  [0.6, 0.8, 0.2, 1.4, -2.4, None, -6.5],
    }

    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 2, wspace=0.4, left=0.07, right=0.96, top=0.90, bottom=0.12)

    # (a) Biomarkers — normalised bar chart (6 ponds; 14B excluded as ref)
    ax_a = fig.add_subplot(gs[0, 0])
    ponds_6 = ["5A", "5B", "11", "13", "14A", "NYC"]
    colors_bm = [PALETTE["teal"], PALETTE["blue"], PALETTE["violet"],
                  PALETTE["amber"], PALETTE["red"], PALETTE["gold"]]
    markers = list(raw.keys())

    for j, pond in enumerate(ponds_6):
        values = []
        for m in markers:
            vals_all = raw[m][:6] + [raw[m][6]]   # remap: skip 14B
            remap = [raw[m][0], raw[m][1], raw[m][2], raw[m][3], raw[m][4], raw[m][6]]
            v = remap[j]
            if v is not None:
                vmax = max(x for x in raw[m] if x is not None)
                values.append(v / vmax)
            else:
                values.append(0)

        x = np.arange(len(markers))
        ax_a.bar(x + j * 0.12, values, 0.11, color=colors_bm[j], label=pond, alpha=0.85)

    ax_a.set_xticks(np.arange(len(markers)) + 0.3)
    ax_a.set_xticklabels(markers, fontsize=8)
    ax_a.set_ylabel("Normalised activity (fraction of max)")
    ax_a.set_title("(a) Oxidative stress biomarkers (normalised)", pad=8)
    ax_a.legend(title="Pond", fontsize=6, title_fontsize=7, ncol=2)
    ax_a.grid(True, axis="y", alpha=0.3)
    ax_a.set_ylim(0, 1.15)

    # Bistability annotation
    ax_a.axhline(0.37, color=PALETTE["amber"], ls=":", lw=1, alpha=0.6)
    ax_a.text(0, 0.39, "Bistability window", color=PALETTE["amber"],
              fontsize=6, style="italic")

    # (b) Gene expression heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    genes = list(gene_data.keys())
    ponds_hm = ["5A", "5B", "11", "13", "14A", "NYC"]
    Z = []
    for g in genes:
        row_vals = gene_data[g][:6] + [gene_data[g][6]]
        row = [row_vals[0], row_vals[1], row_vals[2], row_vals[3], row_vals[4], row_vals[6]]
        Z.append([v if v is not None else 0.0 for v in row])
    Z = np.array(Z, dtype=float)

    im = ax_b.imshow(Z, cmap="RdBu_r", vmin=-7, vmax=7, aspect="auto")
    ax_b.set_xticks(range(6)); ax_b.set_xticklabels(ponds_hm, fontsize=8)
    ax_b.set_yticks(range(6)); ax_b.set_yticklabels(genes, fontsize=8, style="italic")
    ax_b.set_title("(b) Gene expression ΔCt (positive = upregulated)", pad=8)

    # Annotate values
    for i in range(6):
        for j in range(6):
            val = Z[i, j]
            color = "white" if abs(val) > 3 else PALETTE["text"]
            ax_b.text(j, i, f"{val:+.1f}", ha="center", va="center",
                       fontsize=6, color=color)

    # Highlight NYC star
    ax_b.add_patch(plt.Rectangle((5 - 0.5, 5 - 0.5), 1, 1,
                                   fill=False, edgecolor=PALETTE["red"], lw=2))
    ax_b.text(5.6, 5, "★", color=PALETTE["gold"], fontsize=10)

    plt.colorbar(im, ax=ax_b, label="ΔCt", fraction=0.04)

    fig.suptitle("Figure 3 — Oxidative Stress Biomarkers and Gene Expression Profiles",
                  fontsize=10, fontweight="bold", y=0.98)
    path = os.path.join(outdir, "fig3_biomarkers.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── FIGURE 4: SENSITIVITY ANALYSIS ─────────────────────────────────────────
def make_fig4(outdir: str):
    """
    Figure 4: Sensitivity analysis results.
    (a) Morris μ*/σ scatter   (b) Sobol' index bar chart   (c) OAT tornado
    Uses pre-computed values matching Tables 4–5 of the paper.
    Run sensitivity_analysis.py to recompute with full N=4096 Sobol' scheme.
    """
    # Pre-computed values matching Tables 4 & 5 (from N=4096 Sobol' run)
    morris_data = {
        "ku_s":    (0.48, 0.31, "Highly influential; nonlinear"),
        "delta_s": (0.39, 0.22, "Highly influential; interacts"),
        "IR":      (0.31, 0.09, "Influential; near-linear"),
        "BW_F":    (0.27, 0.08, "Influential; near-linear"),
        "E0":      (0.24, 0.19, "Influential; nonlinear"),
        "CSF_Cr":  (0.21, 0.07, "Influential; near-linear"),
        "k_LM":    (0.18, 0.12, "Moderately influential"),
        "f_ing":   (0.16, 0.10, "Moderately influential"),
        "k_Me":    (0.14, 0.09, "Moderately influential"),
        "S0":      (0.12, 0.06, "Marginally influential"),
        "sigma0":  (0.06, 0.08, "Non-influential for TR"),
        "k_d":     (0.04, 0.03, "Non-influential for TR"),
    }
    sobol_S1  = {"ku_s": 0.41, "delta_s": 0.26, "IR": 0.12, "BW_F": 0.09,
                  "E0": 0.07, "CSF_Cr": 0.06}
    sobol_STi = {"ku_s": 0.56, "delta_s": 0.38, "IR": 0.16, "BW_F": 0.12,
                  "E0": 0.14, "CSF_Cr": 0.08}
    sobol_CI_S1  = {"ku_s": (0.34,0.48), "delta_s": (0.21,0.31), "IR": (0.09,0.15),
                    "BW_F": (0.07,0.12), "E0": (0.05,0.10), "CSF_Cr": (0.04,0.09)}
    sobol_CI_STi = {"ku_s": (0.49,0.63), "delta_s": (0.33,0.43), "IR": (0.13,0.20),
                    "BW_F": (0.09,0.15), "E0": (0.11,0.18), "CSF_Cr": (0.06,0.11)}
    oat_Si = {"ku_s": 0.94, "delta_s": -0.82, "IR": 0.75, "BW_F": -0.68,
               "E0": 0.65, "CSF_Cr": 0.60}
    labels = {
        "ku_s": "ku^(s)", "delta_s": "δs", "IR": "IR", "BW_F": "BW_F",
        "E0": "E₀", "CSF_Cr": "CSF_Cr",
    }

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.4, left=0.06, right=0.97, top=0.87, bottom=0.14)

    # (a) Morris μ*/σ scatter
    ax_a = fig.add_subplot(gs[0, 0])
    for name, (mu_star, sigma, cls) in morris_data.items():
        color = (PALETTE["red"] if mu_star > 0.30 else
                 PALETTE["amber"] if mu_star > 0.10 else PALETTE["muted"])
        size = 100 if mu_star > 0.20 else 60 if mu_star > 0.10 else 30
        ax_a.scatter(mu_star, sigma, color=color, s=size, alpha=0.85, zorder=3)
        short = name.replace("_s", "ₛ").replace("ku_s", "ku^s").replace("delta", "δ")
        if mu_star > 0.10:
            ax_a.annotate(short, (mu_star, sigma), xytext=(4, 2),
                          textcoords="offset points", fontsize=6, color=PALETTE["text"])

    ax_a.axvline(0.10, color=PALETTE["amber"], ls="--", lw=1, alpha=0.7)
    ax_a.axvline(0.20, color=PALETTE["red"], ls=":", lw=1, alpha=0.5)
    ax_a.set_xlabel("μ* (mean |elementary effect|)"); ax_a.set_ylabel("σ (std dev)")
    ax_a.set_title("(a) Morris Global Screening", pad=8)
    ax_a.grid(True, alpha=0.2)
    patches = [mpatches.Patch(color=PALETTE["red"], label="μ*>0.20 (highly influential)"),
               mpatches.Patch(color=PALETTE["amber"], label="μ*>0.10 (influential)"),
               mpatches.Patch(color=PALETTE["muted"], label="Non-influential")]
    ax_a.legend(handles=patches, fontsize=6)

    # (b) Sobol' bar chart
    ax_b = fig.add_subplot(gs[0, 1])
    params_sob = list(sobol_S1.keys())
    x = np.arange(len(params_sob))
    w = 0.38
    s1_vals = [sobol_S1[p] for p in params_sob]
    sti_vals = [sobol_STi[p] for p in params_sob]
    s1_errs = [sobol_CI_S1[p][1] - sobol_S1[p] for p in params_sob]
    sti_errs = [sobol_CI_STi[p][1] - sobol_STi[p] for p in params_sob]

    ax_b.bar(x - w/2, sti_vals, w, color=PALETTE["blue"], alpha=0.85,
              label="STi (total-order)", yerr=sti_errs, capsize=3,
              error_kw={"color": PALETTE["muted"], "lw": 0.8})
    ax_b.bar(x + w/2, s1_vals, w, color=PALETTE["teal"], alpha=0.70,
              label="Si (first-order)", yerr=s1_errs, capsize=3,
              error_kw={"color": PALETTE["muted"], "lw": 0.8})
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([labels[p] for p in params_sob], fontsize=7)
    ax_b.set_ylabel("Sobol' index")
    ax_b.set_title("(b) Sobol' Variance Decomposition (primary)", pad=8)
    ax_b.legend(fontsize=7); ax_b.grid(True, axis="y", alpha=0.3)
    ax_b.set_ylim(0, 0.72)
    ax_b.text(0.02, 0.98, f"ΣSi = 0.62  (38% interaction variance)",
               transform=ax_b.transAxes, fontsize=6.5,
               color=PALETTE["muted"], va="top")

    # (c) OAT tornado
    ax_c = fig.add_subplot(gs[0, 2])
    params_oat = list(oat_Si.keys())[:6]
    si_vals = [oat_Si[p] for p in params_oat]
    colors_t = [PALETTE["red"] if v > 0 else PALETTE["blue"] for v in si_vals]
    y = np.arange(len(params_oat))
    ax_c.barh(y, si_vals, color=colors_t, alpha=0.80)
    ax_c.axvline(0, color=PALETTE["text"], lw=0.8)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([labels[p] for p in params_oat], fontsize=7)
    ax_c.set_xlabel("Normalised sensitivity index Si (±25% perturbation)")
    ax_c.set_title("(c) OAT Tornado (±25% perturbation)", pad=8)
    ax_c.grid(True, axis="x", alpha=0.3)
    ax_c.text(0.98, 0.02, "Note: OAT is a local method (preliminary screen only).\nSobol' (panel b) is the primary result.",
               transform=ax_c.transAxes, fontsize=5.5, color=PALETTE["muted"],
               ha="right", va="bottom", style="italic")

    fig.suptitle("Figure 4 — Sensitivity Analysis Results for Target Cancer Risk (TR), T=10yr, Scenario A",
                  fontsize=9, fontweight="bold", y=0.99)
    path = os.path.join(outdir, "fig4_sensitivity.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── FIGURE 5: SCENARIO FORECASTS ────────────────────────────────────────────
def make_fig5(outdir: str):
    """
    Figure 5: Ten-year scenario forecasts (2026–2036) at Pond 11.
    (a) Sediment Cr stock   (b) Muscle Cr burden
    (c) Target cancer risk  (d) Collapse probability
    """
    mc = ModelChain()
    t = np.linspace(0, 10, 300)
    years = 2026 + t

    sc_colors = {"A": PALETTE["red"], "B": PALETTE["amber"], "C": PALETTE["teal"]}
    sc_ls = {"A": "-", "B": "--", "C": "-."}
    sc_names = {
        "A": "Scenario A: Collapse (no intervention)",
        "B": "Scenario B: Partial intervention",
        "C": "Scenario C: Full recovery",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(hspace=0.42, wspace=0.35, left=0.09, right=0.97,
                         top=0.90, bottom=0.09)

    for sc_name in ["A", "B", "C"]:
        res = mc.run_full_chain(t_years=10, scenario=sc_name, n_points=300)
        col = sc_colors[sc_name]
        ls  = sc_ls[sc_name]

        # (a) Sediment
        axes[0, 0].plot(years, res.S_t, color=col, ls=ls, lw=2,
                         label=sc_names[sc_name])
        # (b) Muscle
        axes[0, 1].plot(years, res.CM_t, color=col, ls=ls, lw=2)
        # (c) TR
        axes[1, 0].semilogy(years, res.TR_women, color=col, ls=ls, lw=2)
        if sc_name == "A" and res.TR_CI_lower is not None:
            axes[1, 0].fill_between(years, res.TR_CI_lower, res.TR_CI_upper,
                                     color=col, alpha=0.15, label="95% CI (Sc. A)")
        # (d) P(collapse)
        axes[1, 1].plot(years, res.P_collapse * 100, color=col, ls=ls, lw=2)

    # Climate-adjusted Scenario A
    mc_clim = ModelChain({**BASELINE_PARAMS, "E0": BASELINE_PARAMS["E0"] * 1.22})
    res_clim = mc_clim.run_full_chain(t_years=10, scenario="A", n_points=300)
    axes[0, 0].plot(years, res_clim.S_t, color=PALETTE["red"], ls=":", lw=1.5,
                     alpha=0.7, label="Sc. A + AR6 climate (+22%)")

    # Reference lines
    axes[0, 0].axhline(150, color=PALETTE["violet"], ls=":", lw=1.5, alpha=0.8,
                        label="Collapse threshold (150 mg/kg)")
    axes[0, 1].axhline(1.0, color=PALETTE["muted"], ls=":", lw=1.2, alpha=0.8,
                        label="Background (~1 mg/kg)")
    axes[1, 0].axhline(1e-4, color=PALETTE["red"], ls=":", lw=1.5, alpha=0.8,
                        label="USEPA upper (10⁻⁴)")
    axes[1, 1].axhline(50, color=PALETTE["violet"], ls=":", lw=1.2, alpha=0.7,
                        label="50% threshold")

    # Labels
    axes[0, 0].set_ylabel("Sediment Cr S(t) (mg kg⁻¹)")
    axes[0, 0].set_title("(a) Sediment Chromium Stock", pad=8)
    axes[0, 0].legend(fontsize=6, loc="upper left")

    axes[0, 1].set_ylabel("Muscle Cr C_M(t) (mg kg⁻¹ ww)")
    axes[0, 1].set_title("(b) Muscle Chromium Burden", pad=8)
    axes[0, 1].legend(fontsize=6)

    axes[1, 0].set_ylabel("Target Cancer Risk TR (log scale)")
    axes[1, 0].set_title("(c) Cancer Risk — Women (conservative upper bound)", pad=8)
    axes[1, 0].set_ylim(1e-6, 1e-1)
    axes[1, 0].legend(fontsize=6)
    axes[1, 0].text(2027, 1.5e-4, "USEPA limit", color=PALETTE["red"],
                     fontsize=6, style="italic")
    axes[1, 0].text(2027.2, 5e-3, "Total Cr (conservative UB)", color=PALETTE["muted"],
                     fontsize=5.5, style="italic", alpha=0.8)

    axes[1, 1].set_ylabel("P(collapse) (%)")
    axes[1, 1].set_title("(d) Stochastic Collapse Probability — CIR-SDE / Fokker-Planck", pad=8)
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].legend(fontsize=6)
    axes[1, 1].text(0.60, 0.45,
                     "Note: 90% is stochastic (first-passage)\nDeterministic mean: S(10yr)=134<150",
                     transform=axes[1, 1].transAxes, fontsize=5.5,
                     color=PALETTE["muted"], style="italic", va="center")

    for ax in axes.flat:
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(2026, 2036)

    fig.suptitle("Figure 5 — Ten-Year Scenario Forecasts (2026–2036), Pond 11",
                  fontsize=10, fontweight="bold", y=0.97)
    path = os.path.join(outdir, "fig5_scenarios.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AquaMonitor paper figures")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
    parser.add_argument("--outdir", default="./figures", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_style()

    print(f"Generating figures at {args.dpi} dpi → {args.outdir}/")
    print("Note: Figure 1 (causal framework diagram) is provided as a")
    print("      vector graphic in fig1_framework.svg (see docs/).")
    print()

    make_fig2(args.outdir)
    make_fig3(args.outdir)
    make_fig4(args.outdir)
    make_fig5(args.outdir)

    print(f"\nDone. 4 figures saved to {args.outdir}/")
    print("For Figure 1, open platform/aquaculture_math_model.html")
    print("and export the causal chain diagram at required resolution.")
