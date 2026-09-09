"""
Statistical analysis for the ECG cross-dataset anomaly detection manuscript.

Adds the uncertainty quantification that reviewers of a Q1/Q2 journal will ask for,
WITHOUT retraining anything:

  1. Stratified bootstrap 95% confidence intervals for every reported metric
     (AUC, accuracy, precision, recall, F1, specificity) on both datasets.
  2. DeLong test comparing the two models' AUCs on the SAME samples
     (paired, within PTB-XL and within MIT-BIH).
  3. Bootstrap CI for each model's generalization gap  (AUC_PTB-XL - AUC_MIT-BIH).
  4. Bootstrap CI + p-value for the DIFFERENCE between the two models'
     generalization gaps -- this is the paper's central claim.

Run from the project root (the folder containing baseline_cnn.pt):

    python statistical_analysis.py

Runtime: roughly 1-3 minutes on CPU. Nothing is retrained; the saved models are
only used for a forward pass. Output goes to results/statistical_analysis.txt
and the raw score arrays are cached in processed/ so the analysis can be rerun
instantly.
"""
import os
import json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
# torch is imported lazily further below: it is only needed the first time this
# script runs, when the anomaly scores still have to be computed.

SEED = 42
N_BOOT = 2000
os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# Load or compute the anomaly scores
# ----------------------------------------------------------------------------
CACHE = "processed/analysis_scores.npz"

if os.path.exists(CACHE):
    print("Loading cached scores from", CACHE)
    d = np.load(CACHE)
    cnn_ptb, ae_ptb, y_ptb = d["cnn_ptb"], d["ae_ptb"], d["y_ptb"]
    cnn_mit, ae_mit, y_mit = d["cnn_mit"], d["ae_mit"], d["y_mit"]
    ae_threshold = float(d["ae_threshold"])
else:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    # Model architectures -- identical to train_baseline_cnn.py / train_autoencoder.py
    class ECG_CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.AdaptiveAvgPool1d(1)
            )
            self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

        def forward(self, x):
            return self.fc(self.net(x)).squeeze(-1)

    class ECG_Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            )
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2), nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Upsample(scale_factor=2), nn.Conv1d(32, 16, kernel_size=5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
                nn.Upsample(scale_factor=2), nn.Conv1d(16, 1, kernel_size=7, padding=3),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class ECGDataset(Dataset):
        def __init__(self, X):
            self.X = torch.from_numpy(X).unsqueeze(1).float()

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i]

    print("Loading data and models...")
    X = np.load("processed/ptbxl_X.npy")
    y_is_normal = np.load("processed/ptbxl_y_is_normal.npy")
    folds = np.load("processed/ptbxl_folds.npy")
    X_mit = np.load("processed/mitbih_X.npy")
    y_mit_is_normal = np.load("processed/mitbih_y_is_normal.npy")

    LEAD_II_IDX = 1
    test_mask = folds == 10
    X_ptb = X[:, :, LEAD_II_IDX][test_mask]
    y_ptb = (~y_is_normal[test_mask]).astype(int)
    y_mit = (~y_mit_is_normal).astype(int)

    baseline = ECG_CNN()
    baseline.load_state_dict(torch.load("baseline_cnn.pt", map_location="cpu"))
    baseline.eval()
    autoencoder = ECG_Autoencoder()
    autoencoder.load_state_dict(torch.load("autoencoder.pt", map_location="cpu"))
    autoencoder.eval()
    ae_threshold = float(np.load("processed/ae_threshold.npy")[0])

    def cnn_scores(Xd):
        out = []
        with torch.no_grad():
            for xb in DataLoader(ECGDataset(Xd), batch_size=128):
                out.extend(torch.sigmoid(baseline(xb)).numpy())
        return np.array(out)

    def ae_scores(Xd):
        out = []
        with torch.no_grad():
            for xb in DataLoader(ECGDataset(Xd), batch_size=128):
                rec = autoencoder(xb)
                out.extend(((rec - xb) ** 2).mean(dim=[1, 2]).numpy())
        return np.array(out)

    print("Scoring PTB-XL test set...")
    cnn_ptb, ae_ptb = cnn_scores(X_ptb), ae_scores(X_ptb)
    print("Scoring MIT-BIH...")
    cnn_mit, ae_mit = cnn_scores(X_mit), ae_scores(X_mit)

    np.savez(CACHE, cnn_ptb=cnn_ptb, ae_ptb=ae_ptb, y_ptb=y_ptb,
             cnn_mit=cnn_mit, ae_mit=ae_mit, y_mit=y_mit,
             ae_threshold=np.array(ae_threshold))
    print("Scores cached to", CACHE)

CNN_THRESHOLD = 0.5

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def point_metrics(y, score, thr):
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "AUC": roc_auc_score(y, score),
        "Accuracy": (tp + tn) / len(y),
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Specificity": spec,
        "BalancedAcc": 0.5 * (rec + spec),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def stratified_boot_indices(y, n_boot, rng):
    """Resample within each class so class prevalence is preserved."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, pos.size, replace=True),
                              rng.choice(neg, neg.size, replace=True)])


def ci(vals, alpha=0.05):
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ----------------------------------------------------------------------------
# DeLong test (Sun & Xu fast algorithm)
# ----------------------------------------------------------------------------
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    k, N = preds_sorted.shape
    n = N - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, N])
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_test(y, s1, s2):
    """Paired comparison of two AUCs on the same samples."""
    order = (-y).argsort(kind="mergesort")   # positives first
    m = int(y.sum())
    aucs, cov = _fast_delong(np.vstack((s1, s2))[:, order], m)
    l = np.array([[1.0, -1.0]])
    var = float((l @ cov @ l.T)[0, 0])
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs, float(z), float(2 * stats.norm.sf(abs(z))), np.sqrt(np.diag(cov))


# ----------------------------------------------------------------------------
# Run the analysis
# ----------------------------------------------------------------------------
report = []
def out(line=""):
    print(line)
    report.append(line)

out("=" * 78)
out("STATISTICAL ANALYSIS -- ECG cross-dataset anomaly detection")
out(f"seed={SEED}   bootstrap replicates={N_BOOT}   stratified by class")
out("=" * 78)

datasets = {
    "PTB-XL (in-domain)": (y_ptb, {"Baseline CNN": (cnn_ptb, CNN_THRESHOLD),
                                   "Autoencoder": (ae_ptb, ae_threshold)}),
    "MIT-BIH (cross-dataset)": (y_mit, {"Baseline CNN": (cnn_mit, CNN_THRESHOLD),
                                        "Autoencoder": (ae_mit, ae_threshold)}),
}

results = {}
METRICS = ["AUC", "Accuracy", "Precision", "Recall", "F1", "Specificity", "BalancedAcc"]

for ds_name, (y, models) in datasets.items():
    out()
    out("-" * 78)
    out(f"{ds_name}   n={len(y)}  (normal={int((y == 0).sum())}, abnormal={int((y == 1).sum())})")
    out("-" * 78)
    results[ds_name] = {}

    for mdl_name, (score, thr) in models.items():
        pt = point_metrics(y, score, thr)
        boot = {m: [] for m in METRICS}
        r = np.random.default_rng(SEED)
        for idx in stratified_boot_indices(y, N_BOOT, r):
            bm = point_metrics(y[idx], score[idx], thr)
            for m in METRICS:
                boot[m].append(bm[m])

        out()
        out(f"  {mdl_name}  (threshold = {thr:.6g})")
        out(f"    confusion matrix: TN={pt['TN']}  FP={pt['FP']}  FN={pt['FN']}  TP={pt['TP']}")
        results[ds_name][mdl_name] = {}
        for m in METRICS:
            lo, hi = ci(np.array(boot[m]))
            out(f"    {m:<12} {pt[m]:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
            results[ds_name][mdl_name][m] = {"point": pt[m], "ci_low": lo, "ci_high": hi}
        results[ds_name][mdl_name]["confusion"] = {k: pt[k] for k in ("TN", "FP", "FN", "TP")}

    # DeLong: paired AUC comparison on identical samples
    s_cnn = models["Baseline CNN"][0]
    s_ae = models["Autoencoder"][0]
    aucs, z, p, se = delong_test(y, s_cnn, s_ae)
    diff = aucs[0] - aucs[1]
    out()
    out(f"  DeLong test (Baseline CNN vs Autoencoder, paired on the same {len(y)} samples):")
    out(f"    AUC difference = {diff:+.4f}   z = {z:+.3f}   p = {p:.3g}")
    out(f"    DeLong SE: CNN={se[0]:.4f}, AE={se[1]:.4f}")
    out(f"    -> {'STATISTICALLY SIGNIFICANT' if p < 0.05 else 'not significant'} at alpha=0.05")
    results[ds_name]["delong_cnn_vs_ae"] = {"auc_diff": float(diff), "z": z, "p": p}

# ----------------------------------------------------------------------------
# Generalization gap: the paper's central claim
# ----------------------------------------------------------------------------
out()
out("=" * 78)
out("GENERALIZATION GAP  (AUC on PTB-XL  minus  AUC on MIT-BIH)")
out("=" * 78)

r = np.random.default_rng(SEED + 1)
ptb_idx = list(stratified_boot_indices(y_ptb, N_BOOT, r))
r = np.random.default_rng(SEED + 2)
mit_idx = list(stratified_boot_indices(y_mit, N_BOOT, r))

gap_cnn, gap_ae, gap_diff = [], [], []
for ip, im in zip(ptb_idx, mit_idx):
    g_c = roc_auc_score(y_ptb[ip], cnn_ptb[ip]) - roc_auc_score(y_mit[im], cnn_mit[im])
    g_a = roc_auc_score(y_ptb[ip], ae_ptb[ip]) - roc_auc_score(y_mit[im], ae_mit[im])
    gap_cnn.append(g_c); gap_ae.append(g_a); gap_diff.append(g_c - g_a)

gap_cnn = np.array(gap_cnn); gap_ae = np.array(gap_ae); gap_diff = np.array(gap_diff)

pt_gap_cnn = roc_auc_score(y_ptb, cnn_ptb) - roc_auc_score(y_mit, cnn_mit)
pt_gap_ae = roc_auc_score(y_ptb, ae_ptb) - roc_auc_score(y_mit, ae_mit)

lo, hi = ci(gap_cnn)
out(f"  Baseline CNN gap:  {pt_gap_cnn:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
res_gap = {"cnn": {"point": float(pt_gap_cnn), "ci_low": lo, "ci_high": hi}}
lo, hi = ci(gap_ae)
out(f"  Autoencoder  gap:  {pt_gap_ae:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
res_gap["ae"] = {"point": float(pt_gap_ae), "ci_low": lo, "ci_high": hi}

lo, hi = ci(gap_diff)
# two-sided bootstrap p-value for H0: the two gaps are equal
p_boot = 2 * min((gap_diff <= 0).mean(), (gap_diff >= 0).mean())
p_boot = max(p_boot, 1.0 / N_BOOT)
out()
out(f"  DIFFERENCE (CNN gap - AE gap): {pt_gap_cnn - pt_gap_ae:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
if p_boot <= 1.0 / N_BOOT:
    out(f"  Bootstrap two-sided p < {1.0 / N_BOOT:.4g}   "
        f"(no bootstrap replicate crossed zero at {N_BOOT} replicates)")
else:
    out(f"  Bootstrap two-sided p = {p_boot:.4g}")
out(f"  -> The supervised model's gap is {'significantly' if lo > 0 else 'NOT significantly'} "
    f"larger than the autoencoder's.")
res_gap["difference"] = {"point": float(pt_gap_cnn - pt_gap_ae), "ci_low": lo, "ci_high": hi, "p": float(p_boot)}
results["generalization_gap"] = res_gap

# ----------------------------------------------------------------------------
# Ready-to-paste sentences
# ----------------------------------------------------------------------------
out()
out("=" * 78)
out("READY-TO-PASTE TEXT FOR THE MANUSCRIPT")
out("=" * 78)
g = results["generalization_gap"]
out(f"  The supervised baseline lost {g['cnn']['point']:.3f} AUC "
    f"(95% CI [{g['cnn']['ci_low']:.3f}, {g['cnn']['ci_high']:.3f}]) under domain shift,")
out(f"  whereas the autoencoder lost {g['ae']['point']:.3f} AUC "
    f"(95% CI [{g['ae']['ci_low']:.3f}, {g['ae']['ci_high']:.3f}]).")
out(f"  The difference between the two generalization gaps was {g['difference']['point']:.3f} "
    f"(95% CI [{g['difference']['ci_low']:.3f}, {g['difference']['ci_high']:.3f}],")
p_txt = (f"p < {1.0 / N_BOOT:.4g}" if g['difference']['p'] <= 1.0 / N_BOOT
         else f"p = {g['difference']['p']:.4g}")
out(f"  bootstrap {p_txt}).")

with open("results/statistical_analysis.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
with open("results/statistical_analysis.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print()
print("Saved: results/statistical_analysis.txt")
print("Saved: results/statistical_analysis.json")
print()
print(">>> Send me results/statistical_analysis.txt and I will insert the")
print(">>> confidence intervals and p-values into the manuscript tables.")
