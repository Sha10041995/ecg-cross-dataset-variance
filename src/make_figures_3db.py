"""
Figures 1 and 3 for the three-database version of the manuscript.
Fig 2 (ROC curves, MIT-BIH) and Fig S1 are unchanged and are NOT regenerated here.

    python make_figures_3db.py          (run from the project root)

Inputs: results/multiseed3_runs.json, results/classical_baseline.json,
        results/classical_chapman.json
Outputs: figures_3db/Fig1.eps/.tif/.png, Fig3.eps/.tif/.png
"""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

BASE = os.environ.get("ECG_BASE", ".")
OUT = "figures_3db"; os.makedirs(OUT, exist_ok=True)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.titlesize"] = 9.5
matplotlib.rcParams["axes.labelsize"] = 9
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42
CNN, AE, RF, LR = "#4C72B0", "#DD8452", "#55A868", "#8172B2"
CNN_L, AE_L, RF_L, LR_L = "#A9BEDB", "#EFC0A2", "#A9D3B5", "#C0B8D9"
GRID = "0.88"

runs = sorted(json.load(open(f"{BASE}/results/multiseed3_runs.json")), key=lambda r: r["seed"])
cb = json.load(open(f"{BASE}/results/classical_baseline.json"))
cc = json.load(open(f"{BASE}/results/classical_chapman.json"))
cp = np.array([r["cnn_ptbxl"]["AUC"] for r in runs]); cm = np.array([r["cnn_mitbih"]["AUC"] for r in runs])
cx = np.array([r["cnn_chapman"]["AUC"] for r in runs])
ap = np.array([r["ae_ptbxl"]["AUC"] for r in runs]); am = np.array([r["ae_mitbih"]["AUC"] for r in runs])
ax_ = np.array([r["ae_chapman"]["AUC"] for r in runs])

def outline_eps(path):
    """Matplotlib EPS text can render blank in Ghostscript/Distiller/submission
    systems (see README). Convert text to outlines with Ghostscript, and keep the
    %%Orientation comment that some systems use to decide whether to rotate."""
    import shutil, subprocess
    gs = shutil.which("gs") or shutil.which("gswin64c") or shutil.which("gswin32c")
    if gs is None:
        print("  warning: Ghostscript not found; EPS left unconverted. Use the TIFF instead.")
        return
    tmp = path + ".tmp"
    subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=eps2write",
                    "-dNoOutputFonts", "-dEPSCrop", f"-sOutputFile={tmp}", path], check=True)
    data = open(tmp, "rb").read()
    if b"%%Orientation" not in data:
        data = data.replace(b"%%BoundingBox", b"%%Orientation: portrait\n%%BoundingBox", 1)
    open(path, "wb").write(data)
    os.remove(tmp)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.eps", format="eps", bbox_inches="tight")
    outline_eps(f"{OUT}/{name}.eps")
    try:
        fig.savefig(f"{OUT}/{name}.tif", format="tiff", dpi=600, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"})
    except Exception as e:
        print("  (TIFF skipped:", e, ")")
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  wrote", name)

def spread(vals, gap):
    """push labels apart so they do not overprint: returns adjusted y positions"""
    order = np.argsort(vals); y = np.array(vals, float)
    for _ in range(50):
        for a, b in zip(order[:-1], order[1:]):
            if y[b] - y[a] < gap:
                mid = (y[a] + y[b]) / 2; y[a] = mid - gap / 2; y[b] = mid + gap / 2
    return y

# ------------------------------------------------------------------ Fig 1
M = [("Supervised CNN", [cp.mean(), cm.mean(), cx.mean()], CNN, CNN_L,
      [cp.std(ddof=1), cm.std(ddof=1), cx.std(ddof=1)], (cp-cm).std(ddof=1), (cp-cx).std(ddof=1)),
     ("Random forest", [cb["Random forest"]["auc_in"], cb["Random forest"]["auc_out"],
                        cc["Random forest"]["auc_cha"]], RF, RF_L, None, None, None),
     ("Logistic regression", [cb["Logistic regression"]["auc_in"], cb["Logistic regression"]["auc_out"],
                              cc["Logistic regression"]["auc_cha"]], LR, LR_L, None, None, None),
     ("Autoencoder", [ap.mean(), am.mean(), ax_.mean()], AE, AE_L,
      [ap.std(ddof=1), am.std(ddof=1), ax_.std(ddof=1)], (ap-am).std(ddof=1), (ap-ax_).std(ddof=1))]

fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), gridspec_kw={"width_ratios": [1.3, 1]})
a = axes[0]
for name, v, c, _, _, _, _ in M:
    a.plot([0, 1, 2], v, "-o", color=c, lw=1.8, ms=4.5, label=name)
for j, side in ((0, "l"), (2, "r")):
    vals = [m[1][j] for m in M]; ys = spread(vals, 0.021)
    for m, y0, v in zip(M, ys, vals):
        if side == "l":
            a.text(-0.07, y0, f"{v:.3f}", ha="right", va="center", fontsize=7, color=m[2])
        else:
            a.text(2.07, y0, f"{v:.3f}", ha="left", va="center", fontsize=7, color=m[2])
a.set_xticks([0, 1, 2])
a.set_xticklabels(["PTB-XL\n(in-domain)", "MIT-BIH\n(severe shift)", "Chapman-Shaoxing\n(second target)"], fontsize=7.8)
a.set_xlim(-0.45, 2.45); a.set_ylim(0.54, 0.93); a.set_ylabel("AUC")
a.set_title("(a) Model ranking differs by target")
a.legend(loc="lower left", fontsize=6.5, framealpha=1.0)
a.grid(axis="y", color=GRID, lw=0.5)

b = axes[1]
n = len(M); h = 0.36; y = np.arange(n)[::-1]
for k, (name, v, c, cl, sds, sd_m, sd_c) in enumerate(M):
    g_m, g_c = v[0] - v[1], v[0] - v[2]
    b.barh(y[k] + h/2, g_m, h, color=cl, edgecolor=c, lw=0.8)
    b.barh(y[k] - h/2, g_c, h, color=c, edgecolor=c, lw=0.8)
    for gg, ee, yy in ((g_m, sd_m, y[k] + h/2), (g_c, sd_c, y[k] - h/2)):
        if ee is not None:
            b.errorbar(gg, yy, xerr=ee, fmt="none", ecolor="0.2", capsize=2.5, lw=0.9)
            b.text(gg + ee + 0.006, yy, f"{gg:.3f}", va="center", fontsize=6.8)
        else:
            b.text(gg + 0.006, yy, f"{gg:.3f}", va="center", fontsize=6.8)
b.set_yticks(y); b.set_yticklabels([m[0] for m in M], fontsize=7.6)
b.set_xlabel("Generalization gap (AUC$_{source}$ − AUC$_{target}$)")
b.set_title("(b) Loss on transfer")
b.set_xlim(0, 0.36); b.grid(axis="x", color=GRID, lw=0.5)
b.legend(handles=[Patch(fc="0.85", ec="0.3", label="MIT-BIH"), Patch(fc="0.35", ec="0.3", label="Chapman (strict label)")],
         loc="center right", fontsize=6.5, framealpha=1.0)
fig.tight_layout(); save(fig, "Fig1")

# ------------------------------------------------------------------ Fig 3
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), gridspec_kw={"width_ratios": [1.15, 1]})
a = axes[0]
for i in range(len(runs)):
    a.plot([0, 1, 2], [cp[i], cm[i], cx[i]], "-o", color=CNN_L, ms=3, lw=0.8)
    a.plot([0, 1, 2], [ap[i], am[i], ax_[i]], "-o", color=AE_L, ms=3, lw=0.8)
a.plot([0, 1, 2], [cp.mean(), cm.mean(), cx.mean()], "-s", color=CNN, lw=2.2, ms=5.5, label="Supervised CNN (mean)")
a.plot([0, 1, 2], [ap.mean(), am.mean(), ax_.mean()], "-s", color=AE, lw=2.2, ms=5.5, label="Autoencoder (mean)")
a.set_xticks([0, 1, 2]); a.set_xticklabels(["PTB-XL\n(in-domain)", "MIT-BIH", "Chapman-\nShaoxing"], fontsize=7.8)
a.set_xlim(-0.25, 2.25); a.set_ylabel("AUC")
a.set_title(f"(a) Per-seed AUC, n = {len(runs)} trainings")
a.legend(loc="lower left", fontsize=6.8); a.grid(axis="y", color=GRID, lw=0.5)

b = axes[1]
gc_m, ga_m = cp - cm, ap - am; gc_c, ga_c = cp - cx, ap - ax_
groups = [(gc_m, CNN, CNN_L, 0.0), (ga_m, AE, AE_L, 1.0), (gc_c, CNN, CNN_L, 2.4), (ga_c, AE, AE_L, 3.4)]
jit = np.linspace(-0.13, 0.13, len(runs))
for g, c, cl, x in groups:
    b.bar(x, g.mean(), 0.6, color=cl, edgecolor=c, lw=0.8)
    b.scatter(x + jit, g, color="0.15", zorder=3, s=12)
    b.errorbar(x, g.mean(), yerr=g.std(ddof=1), fmt="none", ecolor="0.15", capsize=3.5, lw=1.1, zorder=4)
b.axhline(0, color="0.5", lw=0.7)
b.set_xticks([0, 1, 2.4, 3.4]); b.set_xticklabels(["CNN", "AE", "CNN", "AE"], fontsize=8)
b.text(0.5, -0.115, "MIT-BIH", ha="center", fontsize=8, transform=b.get_xaxis_transform())
b.text(2.9, -0.115, "Chapman-Shaoxing", ha="center", fontsize=8, transform=b.get_xaxis_transform())
b.set_ylabel("Generalization gap"); b.set_ylim(-0.01, 0.30)
b.set_title("(b) Gap per seed, by target")
b.grid(axis="y", color=GRID, lw=0.5)
for (g1, g2, x) in ((gc_m, ga_m, 0.5), (gc_c, ga_c, 2.9)):
    d = g1 - g2; t, p = stats.ttest_rel(g1, g2)
    s = f"CNN − AE {d.mean():+.3f} ± {d.std(ddof=1):.3f}\nCNN larger in {int((d>0).sum())}/{len(runs)} seeds".replace("-0.","−0.")
    b.text(x, 0.262, s, ha="center", va="center", fontsize=6.4,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
fig.tight_layout(); save(fig, "Fig3")
