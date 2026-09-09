"""
Publication-quality figures for the Biomedical Engineering Letters submission.

BMEL requires: EPS for vector graphics or TIFF for halftones; 600 dpi for
combination art; Helvetica or Arial lettering; figures named Fig1, Fig2, ...
with captions kept in the manuscript text rather than in the image files.

This script regenerates all four figures from the cached results, so nothing
has to be retrained. It writes both EPS (vector, preferred) and TIFF (600 dpi)
into figures/.

    Fig1  four-model comparison: ranking inversion + generalization gap
    Fig2  ROC curves, both domains
    Fig3  multi-seed replication
    FigS1 confusion matrices (Electronic Supplementary Material)

USAGE
    python make_figures.py

INPUTS (all produced by the earlier scripts; nothing is recomputed)
    processed/analysis_scores.npz     <- statistical_analysis.py
    processed/ae_threshold.npy        <- train_autoencoder.py
    results/multiseed_runs.json       <- multiseed_experiment.py
    results/multiseed_summary.json    <- multiseed_experiment.py
    results/classical_baseline.json   <- classical_baseline.py
"""
import os
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# BMEL: Helvetica or Arial. Fall back gracefully if neither is installed.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans",
                                          "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["axes.labelsize"] = 9
matplotlib.rcParams["pdf.fonttype"] = 42      # embed as TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["legend.framealpha"] = 1.0

CNN, AE, RF, LR = "#4C72B0", "#DD8452", "#55A868", "#8172B2"
# EPS does not support transparency, so every fill uses an explicit light
# tint rather than alpha; this keeps the EPS identical to the PNG.
CNN_L, AE_L, RF_L, LR_L = "#A9BEDB", "#EFC0A2", "#A9D3B5", "#C0B8D9"
GRID = "0.88"


def need(path):
    if not os.path.exists(path):
        sys.exit(f"\nERROR: {path} not found.\n"
                 "Run the earlier scripts first (see the header of this file).\n")
    return path


def _ghostscript():
    import shutil as _sh
    for c in ("gs", "gswin64c", "gswin32c"):
        p = _sh.which(c)
        if p:
            return p
    return None


def _outline_eps(path):
    """Convert every glyph in an EPS from embedded-font text into vector paths.

    matplotlib embeds the font as Type 42 (TrueType inside PostScript). Several
    publisher submission systems -- Editorial Manager's PDF builder among them
    -- cannot resolve Type 42 in EPS and silently drop ALL text, leaving axes
    and curves with no labels. Ghostscript's -dNoOutputFonts re-emits the
    glyphs as filled paths, so there is no font left to fail on. Still vector,
    still editable in production, and impossible to drop.
    """
    import subprocess
    gs = _ghostscript()
    if gs is None:
        print("  WARNING: Ghostscript not found; EPS text left as embedded fonts.")
        print("           Some submission systems will drop it. Run fix_eps_fonts.py")
        print("           later, or upload the TIFF versions instead.")
        return
    tmp = path + ".tmp.eps"
    subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-dNoOutputFonts",
                    "-dEPSCrop", "-sDEVICE=eps2write", "-o", tmp, path], check=True)
    os.replace(tmp, path)


def save(fig, name):
    """Write EPS (vector, text outlined) and TIFF (600 dpi); PNG as a copy."""
    eps = os.path.join(OUT, f"{name}.eps")
    fig.savefig(eps, format="eps", bbox_inches="tight")
    _outline_eps(eps)
    try:
        fig.savefig(os.path.join(OUT, f"{name}.tif"), format="tiff", dpi=600,
                    bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    except Exception as e:
        print(f"  (TIFF for {name} skipped: {e}; EPS and PNG still written)")
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.eps (text outlined), .tif, .png")


# ---------------------------------------------------------------------------
# Load cached results
# ---------------------------------------------------------------------------
print("Loading cached results...")
d = np.load(need("processed/analysis_scores.npz"))
cnn_ptb, ae_ptb, y_ptb = d["cnn_ptb"], d["ae_ptb"], d["y_ptb"].astype(int)
cnn_mit, ae_mit, y_mit = d["cnn_mit"], d["ae_mit"], d["y_mit"].astype(int)
ae_thr = float(d["ae_threshold"])

with open(need("results/multiseed_runs.json"), encoding="utf-8") as f:
    runs = sorted(json.load(f), key=lambda r: r["seed"])
with open(need("results/multiseed_summary.json"), encoding="utf-8") as f:
    ms = json.load(f)
with open(need("results/classical_baseline.json"), encoding="utf-8") as f:
    cb = json.load(f)

cp = np.array([r["cnn_ptbxl"]["AUC"] for r in runs])
cm = np.array([r["cnn_mitbih"]["AUC"] for r in runs])
ap = np.array([r["ae_ptbxl"]["AUC"] for r in runs])
am = np.array([r["ae_mitbih"]["AUC"] for r in runs])
gc, ga = cp - cm, ap - am

# ---------------------------------------------------------------------------
# Fig 1 — four-model comparison
# ---------------------------------------------------------------------------
print("Fig1: four-model comparison")
M = [("Supervised CNN", cp.mean(), cm.mean(), CNN, gc.std(ddof=1)),
     ("Random forest", cb["Random forest"]["auc_in"], cb["Random forest"]["auc_out"], RF, None),
     ("Logistic regression", cb["Logistic regression"]["auc_in"],
      cb["Logistic regression"]["auc_out"], LR, None),
     ("Autoencoder", ap.mean(), am.mean(), AE, ga.std(ddof=1))]

fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})

ax = axes[0]
for name, a, b, c, _ in M:
    ax.plot([0, 1], [a, b], "-o", color=c, lw=1.8, ms=5, label=name)
    ax.text(-0.05, a, f"{a:.3f}", ha="right", va="center", fontsize=7.5, color=c)
    ax.text(1.05, b, f"{b:.3f}", ha="left", va="center", fontsize=7.5, color=c)
ax.set_xticks([0, 1])
ax.set_xticklabels(["PTB-XL\n(in-domain)", "MIT-BIH\n(cross-dataset)"])
ax.set_xlim(-0.34, 1.34)
ax.set_ylim(0.66, 0.92)
ax.set_ylabel("AUC")
ax.set_title("(a) Ranking inversion under domain shift")
ax.legend(loc="lower left", fontsize=6.8, framealpha=1.0)
ax.grid(axis="y", color=GRID, lw=0.5)

ax = axes[1]
names = [m[0] for m in M]
gaps = [m[1] - m[2] for m in M]
cols = [m[3] for m in M]
errs = [m[4] for m in M]
y = np.arange(len(M))[::-1]
ax.barh(y, gaps, color=[{"#4C72B0": CNN_L, "#DD8452": AE_L, "#55A868": RF_L,
                         "#8172B2": LR_L}[c] for c in cols],
        edgecolor=cols, lw=0.8, height=0.6)
for i, (g, e) in enumerate(zip(gaps, errs)):
    if e is not None:
        ax.errorbar(g, y[i], xerr=e, fmt="none", ecolor="0.2", capsize=3, lw=1.0)
        ax.text(g + e + 0.008, y[i], f"{g:.3f} ± {e:.3f}", va="center", fontsize=7.5)
    else:
        ax.text(g + 0.008, y[i], f"{g:.3f}", va="center", fontsize=7.5)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Generalization gap (AUC$_{source}$ − AUC$_{target}$)")
ax.set_title("(b) Loss on transfer")
ax.set_xlim(0, 0.27)
ax.grid(axis="x", color=GRID, lw=0.5)
fig.tight_layout()
save(fig, "Fig1")

# ---------------------------------------------------------------------------
# Fig 2 — ROC curves
# ---------------------------------------------------------------------------
print("Fig2: ROC curves")
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
for ax, (sc, sa, yt, title) in zip(axes, [
        (cnn_ptb, ae_ptb, y_ptb, "(a) PTB-XL (in-domain)"),
        (cnn_mit, ae_mit, y_mit, "(b) MIT-BIH (cross-dataset)")]):
    f1, t1, _ = roc_curve(yt, sc)
    f2, t2, _ = roc_curve(yt, sa)
    ax.plot(f1, t1, color=CNN, lw=1.6, label=f"Supervised CNN (AUC = {roc_auc_score(yt, sc):.3f})")
    ax.plot(f2, t2, color=AE, lw=1.6, label=f"Autoencoder (AUC = {roc_auc_score(yt, sa):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.65", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(color=GRID, lw=0.5)
fig.tight_layout()
save(fig, "Fig2")

# ---------------------------------------------------------------------------
# Fig 3 — multi-seed replication
# ---------------------------------------------------------------------------
print("Fig3: multi-seed replication")
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1))
ax = axes[0]
for i in range(len(runs)):
    ax.plot([0, 1], [cp[i], cm[i]], "-o", color=CNN_L, ms=3, lw=0.8)
    ax.plot([0, 1], [ap[i], am[i]], "-o", color=AE_L, ms=3, lw=0.8)
ax.plot([0, 1], [cp.mean(), cm.mean()], "-s", color=CNN, lw=2.2, ms=6, label="Supervised CNN (mean)")
ax.plot([0, 1], [ap.mean(), am.mean()], "-s", color=AE, lw=2.2, ms=6, label="Autoencoder (mean)")
ax.set_xticks([0, 1])
ax.set_xticklabels(["PTB-XL\n(in-domain)", "MIT-BIH\n(cross-dataset)"])
ax.set_xlim(-0.25, 1.25)
ax.set_ylabel("AUC")
ax.set_title(f"(a) Per-seed AUC, n = {len(runs)} trainings")
ax.legend(loc="lower left", fontsize=7)
ax.grid(axis="y", color=GRID, lw=0.5)

ax = axes[1]
xp = np.arange(2)
ax.bar(xp[0], gc.mean(), 0.55, color=CNN_L, edgecolor=CNN, lw=0.8, label="Supervised CNN")
ax.bar(xp[1], ga.mean(), 0.55, color=AE_L, edgecolor=AE, lw=0.8, label="Autoencoder")
jit = np.linspace(-0.13, 0.13, len(runs))
ax.scatter(xp[0] + jit, gc, color="0.15", zorder=3, s=14)
ax.scatter(xp[1] + jit, ga, color="0.15", zorder=3, s=14, label="individual seeds")
ax.errorbar(xp, [gc.mean(), ga.mean()], yerr=[gc.std(ddof=1), ga.std(ddof=1)],
            fmt="none", ecolor="0.15", capsize=4, lw=1.2, zorder=4)
ax.axhline(0, color="0.5", lw=0.7)
ax.set_xticks(xp)
ax.set_xticklabels(["Supervised CNN", "Autoencoder"], fontsize=8)
ax.set_ylabel("Generalization gap")
ax.set_title("(b) Generalization gap per seed")
ax.set_ylim(-0.01, 0.30)
ax.legend(loc="upper right", fontsize=6.5)
ax.grid(axis="y", color=GRID, lw=0.5)
gd = gc - ga
_t, _p = stats.ttest_rel(gc, ga)
_ptxt = "p < 0.001" if _p < 0.001 else f"p = {_p:.3f}"
ax.text(0.0, 0.262,
        f"difference {gd.mean():.3f} ± {gd.std(ddof=1):.3f}\n"
        f"{int((gd > 0).sum())} / {len(runs)} seeds, {_ptxt}",
        ha="center", va="center", fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.6))
fig.tight_layout()
save(fig, "Fig3")

# ---------------------------------------------------------------------------
# Fig S1 — confusion matrices
# ---------------------------------------------------------------------------
print("FigS1: confusion matrices")
cfgs = [(y_ptb, (cnn_ptb >= 0.5).astype(int), "Supervised CNN – PTB-XL"),
        (y_ptb, (ae_ptb >= ae_thr).astype(int), "Autoencoder – PTB-XL"),
        (y_mit, (cnn_mit >= 0.5).astype(int), "Supervised CNN – MIT-BIH"),
        (y_mit, (ae_mit >= ae_thr).astype(int), "Autoencoder – MIT-BIH")]
fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.0))
for (yt, yp, title), ax in zip(cfgs, axes.ravel()):
    m = confusion_matrix(yt, yp)
    ax.imshow(m, cmap="Blues")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Abnormal"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Abnormal"], fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8.5); ax.set_ylabel("True", fontsize=8.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=10,
                    color="white" if m[i, j] > m.max() / 2 else "black")
fig.tight_layout()
save(fig, "FigS1")

print(f"\nAll figures written to {OUT}/")
print("Submit the .eps files (BMEL prefers vector); .tif at 600 dpi are included")
print("as an alternative, and .png copies for your own use.")
