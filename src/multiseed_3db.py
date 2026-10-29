"""
THREE-DATABASE VERSION of multiseed_experiment.py.

Identical training code, seeds, splits and thresholds. The only additions:
  * Chapman-Shaoxing/Ningbo (processed/chapman_*.npy) is scored as a third,
    "mild shift" target, with the same frozen models and the same AE threshold.
  * Results go to NEW files (results/multiseed3_*), so the existing
    multiseed_runs.json is never overwritten.
  * After each seed, the PTB-XL and MIT-BIH numbers are compared with the
    existing results/multiseed_runs.json. They should match exactly; if not,
    the environment has drifted and nothing from this run should be used.

Usage (from the project root):
    python multiseed_3db.py --quick       # plumbing check
    python multiseed_3db.py               # full run, seeds 0-4, ~45-60 min

Original docstring follows.

Multi-seed replication for the ECG cross-dataset anomaly detection manuscript.

This closes the last remaining statistical gap: the bootstrap intervals produced by
statistical_analysis.py quantify EVALUATION-SET sampling variability, but not
TRAINING variability (weight initialisation, batch ordering, dropout sampling).
This script retrains both models from scratch under several random seeds and
reports the across-seed distribution of every headline number, including the
generalization gap and the difference between the two models' gaps.

Everything -- architectures, optimiser, learning rate, batch size, epoch counts,
data splits, threshold-selection rule -- is identical to train_baseline_cnn.py and
train_autoencoder.py. The ONLY thing that varies between runs is the random seed.

Run from the project root (the folder containing baseline_cnn.pt).

STEP 1 -- smoke test, about 1-2 minutes. Do this first:

    python multiseed_experiment.py --quick

STEP 2 -- if step 1 finished without an error, run the real thing:

    python multiseed_experiment.py

Fewer seeds, if time is short (three is still enough here):

    python multiseed_experiment.py --seeds 0 1 2

The full run is RESUMABLE: each seed's results are written to
results/multiseed_runs.json as soon as that seed finishes, so Ctrl+C or a
reboot costs you at most one seed. Rerunning the same command continues from
where it stopped. Delete that JSON file to force a clean restart.

Expected runtime: roughly 10-25 minutes per seed on CPU, so about 1-2 hours for
five seeds. Progress and a running time estimate are printed after each seed.
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

# ============================ CONFIG ========================================
# Command line:
#     python multiseed_experiment.py --quick        smoke test (1 seed, 1 epoch)
#     python multiseed_experiment.py                full run, seeds 0-4
#     python multiseed_experiment.py --seeds 0 1 2  full run, chosen seeds only
_ap = argparse.ArgumentParser(description="Multi-seed replication for the ECG study.")
_ap.add_argument("--quick", action="store_true",
                 help="~1-2 minute smoke test: 1 seed, 1 epoch each. "
                      "Writes to separate files; results are NOT publishable.")
_ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                 help="Random seeds to run (default: 0 1 2 3 4).")
_args = _ap.parse_args()

QUICK_TEST = _args.quick
SEEDS = _args.seeds
N_EPOCHS_CNN = 15                # identical to train_baseline_cnn.py
N_EPOCHS_AE = 20                 # identical to train_autoencoder.py
BATCH_SIZE = 64
LR = 1e-3
CNN_THRESHOLD = 0.5
RESULTS_JSON = "results/multiseed3_runs.json"
SCORES_DIR = "results/multiseed3_scores"
OLD_RUNS_JSON = "results/multiseed_runs.json"

if QUICK_TEST:
    SEEDS = [SEEDS[0]]
    N_EPOCHS_CNN = 1
    N_EPOCHS_AE = 1
    RESULTS_JSON = "results/_quicktest3_runs.json"
    SCORES_DIR = "results/_quicktest3_scores"
    print("*** QUICK_TEST MODE: 1 seed, 1 epoch each. ***")
    print("*** These numbers are a plumbing check only -- NOT publishable. ***")
    print("*** If this finishes without error, rerun WITHOUT --quick. ***")
else:
    print(f"Full run: seeds {SEEDS}, {N_EPOCHS_CNN} CNN epochs + {N_EPOCHS_AE} AE epochs per seed.")
    print("Tip: this is resumable -- Ctrl+C is safe, finished seeds are kept.")
# ============================================================================

os.makedirs("results", exist_ok=True)
os.makedirs(SCORES_DIR, exist_ok=True)
device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Architectures -- byte-for-byte the same as the original training scripts
# ---------------------------------------------------------------------------
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


class XYDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).unsqueeze(1).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class XDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X).unsqueeze(1).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]


# ---------------------------------------------------------------------------
# Data (loaded once; PTB-XL is memory-mapped so only lead II is materialised)
# ---------------------------------------------------------------------------
print("Loading data...")
_X = np.load("processed/ptbxl_X.npy", mmap_mode="r")
LEAD_II_IDX = 1
X_lead2 = np.ascontiguousarray(_X[:, :, LEAD_II_IDX])          # (21799, 1000)
del _X
y_is_normal = np.load("processed/ptbxl_y_is_normal.npy")
folds = np.load("processed/ptbxl_folds.npy")

train_mask = folds <= 8
val_mask = folds == 9
test_mask = folds == 10

y_anom = (~y_is_normal).astype(np.float32)

X_train, y_train = X_lead2[train_mask], y_anom[train_mask]
X_val, y_val = X_lead2[val_mask], y_anom[val_mask]
X_test, y_test = X_lead2[test_mask], y_anom[test_mask]
X_train_normal = X_lead2[train_mask & y_is_normal]

X_mit = np.load("processed/mitbih_X.npy")
y_mit = (~np.load("processed/mitbih_y_is_normal.npy")).astype(np.float32)

X_cha = np.load("processed/chapman_X.npy")
_cha_normal = np.load("processed/chapman_y_is_normal.npy")
y_cha = (~_cha_normal).astype(np.float32)
# Pre-specified sensitivity label (fixed BEFORE any Chapman result was seen):
# "broad normal" = diagnosis set is a non-empty subset of
# {sinus rhythm, sinus bradycardia, sinus tachycardia, sinus irregularity}.
_BROAD = {"426783006", "426177001", "427084000", "427393009"}
_dx = np.load("processed/chapman_dx.npy")
_cha_normal_broad = np.array([len(s) > 0 and set(s.split(";")) <= _BROAD for s in _dx])
y_cha_broad = (~_cha_normal_broad).astype(np.float32)
assert len(X_cha) == len(y_cha) == len(y_cha_broad)

print(f"  PTB-XL  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
print(f"  PTB-XL  autoencoder train set (normal only) = {len(X_train_normal)}")
print(f"  MIT-BIH cross-dataset set = {len(X_mit)}")
print(f"  Chapman cross-dataset set = {len(X_cha)}  "
      f"(abnormal strict {y_cha.mean():.3f}, broad {y_cha_broad.mean():.3f})")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader(ds, shuffle, seed, bs=BATCH_SIZE):
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(ds, batch_size=bs, shuffle=True, generator=g)
    return DataLoader(ds, batch_size=bs, shuffle=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_at(y, score, thr):
    pred = (score >= thr).astype(int)
    y = y.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"AUC": float(roc_auc_score(y, score)), "Accuracy": (tp + tn) / len(y),
            "Precision": prec, "Recall": rec, "F1": f1, "Specificity": spec,
            "BalancedAcc": 0.5 * (rec + spec),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


# ---------------------------------------------------------------------------
# One complete run
# ---------------------------------------------------------------------------
def run_seed(seed):
    t_start = time.time()
    print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")

    # ---------------- supervised baseline ----------------
    set_all_seeds(seed)
    cnn = ECG_CNN().to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(cnn.parameters(), lr=LR)
    tl = loader(XYDataset(X_train, y_train), True, seed)

    print(f"  [CNN] training {N_EPOCHS_CNN} epochs...")
    for ep in range(N_EPOCHS_CNN):
        t0 = time.time()
        cnn.train()
        tot = 0.0
        for xb, yb in tl:
            opt.zero_grad()
            loss = crit(cnn(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        dt = time.time() - t0
        print(f"    epoch {ep+1:2d}/{N_EPOCHS_CNN}  loss={tot/len(X_train):.4f}  ({dt:.0f}s)")
        if ep == 0:
            # rough projection so you know early whether this is 10 min or 40 min
            per_seed = dt * N_EPOCHS_CNN + dt * (len(X_train_normal) / len(X_train)) * N_EPOCHS_AE
            print(f"    [estimate] ~{per_seed/60:.0f} min per seed, "
                  f"~{per_seed*len(SEEDS)/60:.0f} min for all {len(SEEDS)} seeds")

    cnn.eval()

    def cnn_score(Xd):
        out = []
        with torch.no_grad():
            for xb in loader(XDataset(Xd), False, seed, bs=128):
                out.extend(torch.sigmoid(cnn(xb)).numpy())
        return np.array(out)

    cnn_ptb = cnn_score(X_test)
    cnn_mit = cnn_score(X_mit)
    cnn_cha = cnn_score(X_cha)

    # ---------------- unsupervised autoencoder ----------------
    set_all_seeds(seed + 10_000)     # independent stream, still deterministic
    ae = ECG_Autoencoder().to(device)
    crit_ae = nn.MSELoss()
    opt_ae = torch.optim.Adam(ae.parameters(), lr=LR)
    tl_ae = loader(XDataset(X_train_normal), True, seed + 10_000)

    print(f"  [AE ] training {N_EPOCHS_AE} epochs on {len(X_train_normal)} normal records...")
    for ep in range(N_EPOCHS_AE):
        t0 = time.time()
        ae.train()
        tot = 0.0
        for xb in tl_ae:
            opt_ae.zero_grad()
            loss = crit_ae(ae(xb), xb)
            loss.backward()
            opt_ae.step()
            tot += loss.item() * len(xb)
        print(f"    epoch {ep+1:2d}/{N_EPOCHS_AE}  recon_loss={tot/len(X_train_normal):.5f}  ({time.time()-t0:.0f}s)")

    ae.eval()

    def ae_score(Xd):
        out = []
        with torch.no_grad():
            for xb in loader(XDataset(Xd), False, seed, bs=128):
                rec = ae(xb)
                out.extend(((rec - xb) ** 2).mean(dim=[1, 2]).numpy())
        return np.array(out)

    # threshold from the PTB-XL validation set via Youden's J -- exactly as before
    ae_val = ae_score(X_val)
    fpr, tpr, thr = roc_curve(y_val.astype(int), ae_val)
    thr_ae = float(thr[np.argmax(tpr - fpr)])

    ae_ptb = ae_score(X_test)
    ae_mit = ae_score(X_mit)
    ae_cha = ae_score(X_cha)

    # ---------------- collect ----------------
    res = {
        "seed": seed,
        "ae_threshold": thr_ae,
        "cnn_ptbxl": metrics_at(y_test, cnn_ptb, CNN_THRESHOLD),
        "cnn_mitbih": metrics_at(y_mit, cnn_mit, CNN_THRESHOLD),
        "ae_ptbxl": metrics_at(y_test, ae_ptb, thr_ae),
        "ae_mitbih": metrics_at(y_mit, ae_mit, thr_ae),
        "cnn_chapman": metrics_at(y_cha, cnn_cha, CNN_THRESHOLD),
        "ae_chapman": metrics_at(y_cha, ae_cha, thr_ae),
        "cnn_chapman_broad_auc": float(roc_auc_score(y_cha_broad.astype(int), cnn_cha)),
        "ae_chapman_broad_auc": float(roc_auc_score(y_cha_broad.astype(int), ae_cha)),
        "minutes": (time.time() - t_start) / 60.0,
    }
    res["gap_cnn"] = res["cnn_ptbxl"]["AUC"] - res["cnn_mitbih"]["AUC"]
    res["gap_ae"] = res["ae_ptbxl"]["AUC"] - res["ae_mitbih"]["AUC"]
    res["gap_diff"] = res["gap_cnn"] - res["gap_ae"]
    res["gap_cnn_cha"] = res["cnn_ptbxl"]["AUC"] - res["cnn_chapman"]["AUC"]
    res["gap_ae_cha"] = res["ae_ptbxl"]["AUC"] - res["ae_chapman"]["AUC"]
    res["gap_diff_cha"] = res["gap_cnn_cha"] - res["gap_ae_cha"]

    np.savez(os.path.join(SCORES_DIR, f"seed{seed}.npz"),
             cnn_ptb=cnn_ptb, cnn_mit=cnn_mit, ae_ptb=ae_ptb, ae_mit=ae_mit,
             cnn_cha=cnn_cha, ae_cha=ae_cha, y_cha=y_cha, y_cha_broad=y_cha_broad,
             y_ptb=y_test, y_mit=y_mit, ae_threshold=np.array(thr_ae))

    print(f"  -> CNN  PTB-XL AUC={res['cnn_ptbxl']['AUC']:.4f}  MIT-BIH AUC={res['cnn_mitbih']['AUC']:.4f}"
          f"  gap={res['gap_cnn']:.4f}")
    print(f"  -> AE   PTB-XL AUC={res['ae_ptbxl']['AUC']:.4f}  MIT-BIH AUC={res['ae_mitbih']['AUC']:.4f}"
          f"  gap={res['gap_ae']:.4f}")
    print(f"  -> CNN  Chapman AUC={res['cnn_chapman']['AUC']:.4f}  gap={res['gap_cnn_cha']:.4f}")
    print(f"  -> AE   Chapman AUC={res['ae_chapman']['AUC']:.4f}  gap={res['gap_ae_cha']:.4f}")
    print(f"  -> gap difference = {res['gap_diff']:.4f}   ({res['minutes']:.1f} min)")
    return res


# ---------------------------------------------------------------------------
# Main loop (resumable)
# ---------------------------------------------------------------------------
def check_repro(r):
    """Compare this run's PTB-XL / MIT-BIH numbers with the original multi-seed run."""
    if QUICK_TEST or not os.path.exists(OLD_RUNS_JSON):
        return
    with open(OLD_RUNS_JSON, encoding="utf-8") as f:
        old = {o["seed"]: o for o in json.load(f)}
    o = old.get(r["seed"])
    if o is None:
        print("  [repro] no original run for this seed; cannot compare.")
        return
    pairs = [(r[k]["AUC"], o[k]["AUC"]) for k in ("cnn_ptbxl", "cnn_mitbih", "ae_ptbxl", "ae_mitbih")]
    pairs.append((r["ae_threshold"], o["ae_threshold"]))
    worst = max(abs(a - b) for a, b in pairs)
    if worst == 0.0:
        print("  [repro] PTB-XL and MIT-BIH numbers IDENTICAL to the original run (bit-for-bit).")
    else:
        print(f"  [repro] *** DRIFT: largest difference from the original run = {worst:.2e} ***")
        print("  [repro] *** Do not use this run until the cause is understood. ***")


runs = []
if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON, encoding="utf-8") as f:
        runs = json.load(f)
    done = sorted(r["seed"] for r in runs)
    print(f"\nResuming: seeds already completed = {done}")

for seed in SEEDS:
    if any(r["seed"] == seed for r in runs):
        print(f"\nSeed {seed} already done -- skipping.")
        continue
    runs.append(run_seed(seed))
    check_repro(runs[-1])
    runs.sort(key=lambda r: r["seed"])
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    remaining = len([s for s in SEEDS if not any(r["seed"] == s for r in runs)])
    if remaining:
        avg = np.mean([r["minutes"] for r in runs])
        print(f"\n  [{len(runs)}/{len(SEEDS)} seeds done]  estimated time remaining: "
              f"{avg*remaining:.0f} min")

# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------
report = []
def out(line=""):
    print(line)
    report.append(line)

runs = [r for r in runs if r["seed"] in SEEDS]
runs.sort(key=lambda r: r["seed"])
n = len(runs)

def col(path):
    a, b = path
    return np.array([r[a][b] if isinstance(r[a], dict) else r[a] for r in runs], dtype=float)

def flat(key):
    return np.array([r[key] for r in runs], dtype=float)

def msd(v):
    """mean, sd, and t-based 95% CI of the mean"""
    m = float(v.mean())
    if len(v) > 1:
        sd = float(v.std(ddof=1))
        h = float(stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)))
    else:
        sd, h = 0.0, 0.0
    return m, sd, (m - h, m + h)

out("=" * 78)
out("MULTI-SEED REPLICATION -- ECG cross-dataset anomaly detection")
out(f"seeds = {[r['seed'] for r in runs]}   (n = {n} independent trainings per model)")
out("Everything held fixed except the random seed.")
out("=" * 78)

out()
out("PER-SEED RESULTS")
out("-" * 78)
out(f"{'seed':>5} | {'CNN PTB':>8} {'CNN MIT':>8} {'CNN gap':>8} | "
    f"{'AE PTB':>8} {'AE MIT':>8} {'AE gap':>8} | {'gap diff':>8}")
out("-" * 78)
for r in runs:
    out(f"{r['seed']:>5} | {r['cnn_ptbxl']['AUC']:>8.4f} {r['cnn_mitbih']['AUC']:>8.4f} {r['gap_cnn']:>8.4f} | "
        f"{r['ae_ptbxl']['AUC']:>8.4f} {r['ae_mitbih']['AUC']:>8.4f} {r['gap_ae']:>8.4f} | {r['gap_diff']:>8.4f}")
out("-" * 78)

out()
out("ACROSS-SEED SUMMARY  (mean +/- SD [95% CI of the mean], min-max)")
out("-" * 78)

summary = {}
rows = [
    ("CNN  AUC  PTB-XL", col(("cnn_ptbxl", "AUC"))),
    ("CNN  AUC  MIT-BIH", col(("cnn_mitbih", "AUC"))),
    ("CNN  generalization gap", flat("gap_cnn")),
    ("AE   AUC  PTB-XL", col(("ae_ptbxl", "AUC"))),
    ("AE   AUC  MIT-BIH", col(("ae_mitbih", "AUC"))),
    ("AE   generalization gap", flat("gap_ae")),
    ("DIFFERENCE of gaps", flat("gap_diff")),
]
for name, v in rows:
    m, sd, (lo, hi) = msd(v)
    out(f"  {name:<26} {m:.4f} +/- {sd:.4f}  [{lo:.4f}, {hi:.4f}]   ({v.min():.4f} - {v.max():.4f})")
    summary[name.strip()] = {"mean": m, "sd": sd, "ci_low": lo, "ci_high": hi,
                             "min": float(v.min()), "max": float(v.max()),
                             "values": [float(x) for x in v]}

# secondary metrics, for the tables
out()
out("SECONDARY METRICS  (mean +/- SD across seeds)")
out("-" * 78)
for block, label in [("cnn_ptbxl", "CNN  / PTB-XL "), ("ae_ptbxl", "AE   / PTB-XL "),
                     ("cnn_mitbih", "CNN  / MIT-BIH"), ("ae_mitbih", "AE   / MIT-BIH")]:
    parts = []
    for m_name in ["Accuracy", "Precision", "Recall", "F1", "Specificity", "BalancedAcc"]:
        v = col((block, m_name))
        parts.append(f"{m_name}={v.mean():.3f}+/-{v.std(ddof=1) if n > 1 else 0:.3f}")
        summary[f"{block}.{m_name}"] = {"mean": float(v.mean()),
                                        "sd": float(v.std(ddof=1)) if n > 1 else 0.0}
    out(f"  {label}: " + "  ".join(parts))

# ---------------------------------------------------------------------------
# Significance testing across seeds
# ---------------------------------------------------------------------------
out()
out("=" * 78)
out("SIGNIFICANCE TESTS ACROSS SEEDS  (paired: same seed, both models)")
out("=" * 78)

g_c, g_a, g_d = flat("gap_cnn"), flat("gap_ae"), flat("gap_diff")

if n >= 2:
    t_stat, p_t = stats.ttest_rel(g_c, g_a)
    out(f"  Paired t-test on the generalization gap (CNN vs AE), n={n} seeds:")
    out(f"    t({n-1}) = {t_stat:.3f}   p = {p_t:.4g}")
    d = float(g_d.mean() / g_d.std(ddof=1)) if g_d.std(ddof=1) > 0 else float("inf")
    out(f"    Cohen's d_z = {d:.2f}  (mean difference / SD of differences)")
    summary["paired_t_gap"] = {"t": float(t_stat), "p": float(p_t), "df": n - 1, "cohen_dz": d}

    if n >= 5:
        try:
            w_stat, p_w = stats.wilcoxon(g_c, g_a)
            out(f"  Wilcoxon signed-rank (distribution-free): W = {w_stat:.1f}, p = {p_w:.4g}")
            out(f"    NOTE: with n={n} the smallest attainable two-sided p is {2/2**n:.4g};")
            out(f"    the test is reported for completeness, not as the primary evidence.")
            summary["wilcoxon_gap"] = {"W": float(w_stat), "p": float(p_w)}
        except Exception as e:
            out(f"  Wilcoxon not computed: {e}")

    # consistency of the ordering reversal
    rev = sum(1 for r in runs if r["cnn_ptbxl"]["AUC"] > r["ae_ptbxl"]["AUC"]
              and r["cnn_mitbih"]["AUC"] < r["ae_mitbih"]["AUC"])
    out()
    out(f"  Ordering reversal (CNN better in-domain AND worse cross-dataset)")
    out(f"    observed in {rev} / {n} seeds.")
    summary["reversal_count"] = {"reversed": rev, "total": n}

    pos = sum(1 for x in g_d if x > 0)
    out(f"  CNN gap larger than AE gap in {pos} / {n} seeds.")
    summary["gap_larger_count"] = {"count": pos, "total": n}

out()
out("=" * 78)
out("THIRD DATABASE: Chapman-Shaoxing/Ningbo (mild shift; clinical -> clinical)")
out("=" * 78)
out(f"{'seed':>5} | {'CNN CHA':>8} {'CNN gap':>8} | {'AE CHA':>8} {'AE gap':>8} | {'AE-CNN':>8}")
for r in runs:
    out(f"{r['seed']:>5} | {r['cnn_chapman']['AUC']:>8.4f} {r['gap_cnn_cha']:>8.4f} | "
        f"{r['ae_chapman']['AUC']:>8.4f} {r['gap_ae_cha']:>8.4f} | "
        f"{r['ae_chapman']['AUC'] - r['cnn_chapman']['AUC']:>8.4f}")
for name, v in [("CNN  AUC  Chapman", col(("cnn_chapman", "AUC"))),
                ("CNN  gap  Chapman", flat("gap_cnn_cha")),
                ("AE   AUC  Chapman", col(("ae_chapman", "AUC"))),
                ("AE   gap  Chapman", flat("gap_ae_cha")),
                ("DIFFERENCE of gaps (Chapman)", flat("gap_diff_cha")),
                ("AE - CNN target AUC (Chapman)", col(("ae_chapman", "AUC")) - col(("cnn_chapman", "AUC"))),
                ("CNN  AUC  Chapman, broad label", flat("cnn_chapman_broad_auc")),
                ("AE   AUC  Chapman, broad label", flat("ae_chapman_broad_auc"))]:
    m, sd, (lo, hi) = msd(v)
    out(f"  {name:<32} {m:.4f} +/- {sd:.4f}  [{lo:.4f}, {hi:.4f}]   ({v.min():.4f} - {v.max():.4f})")
    summary[name.strip()] = {"mean": m, "sd": sd, "ci_low": lo, "ci_high": hi,
                             "min": float(v.min()), "max": float(v.max()),
                             "values": [float(x) for x in v]}
if n >= 2:
    t3, p3 = stats.ttest_rel(flat("gap_cnn_cha"), flat("gap_ae_cha"))
    out(f"  Paired t-test on the Chapman gap (CNN vs AE), n={n}: t({n-1}) = {t3:.3f}  p = {p3:.4g}")
    summary["paired_t_gap_chapman"] = {"t": float(t3), "p": float(p3), "df": n - 1}
    d_cha = col(("ae_chapman", "AUC")) - col(("cnn_chapman", "AUC"))
    t4, p4 = stats.ttest_1samp(d_cha, 0.0)
    out(f"  AE - CNN on Chapman target AUC: mean {d_cha.mean():+.4f}, SD {d_cha.std(ddof=1):.4f}, "
        f"AE ahead in {int((d_cha > 0).sum())}/{n} seeds, t({n-1}) = {t4:.3f}  p = {p4:.4g}")
    summary["ae_minus_cnn_chapman"] = {"t": float(t4), "p": float(p4), "ahead": int((d_cha > 0).sum())}
    cs, ct = col(("cnn_ptbxl", "AUC")), col(("cnn_chapman", "AUC"))
    out(f"  Seed SD of CNN AUC: source {cs.std(ddof=1):.4f}, Chapman {ct.std(ddof=1):.4f} "
        f"(ratio {ct.std(ddof=1) / max(cs.std(ddof=1), 1e-12):.1f}x)")
    out(f"  Seed SD of CNN AUC: MIT-BIH {col(('cnn_mitbih', 'AUC')).std(ddof=1):.4f}")

# ---------------------------------------------------------------------------
# Ready-to-paste
# ---------------------------------------------------------------------------
out()
out("=" * 78)
out("READY-TO-PASTE TEXT FOR THE MANUSCRIPT")
out("=" * 78)
m_cp, s_cp, _ = msd(col(("cnn_ptbxl", "AUC")))
m_cm, s_cm, _ = msd(col(("cnn_mitbih", "AUC")))
m_ap, s_ap, _ = msd(col(("ae_ptbxl", "AUC")))
m_am, s_am, _ = msd(col(("ae_mitbih", "AUC")))
m_gc, s_gc, ci_gc = msd(g_c)
m_ga, s_ga, ci_ga = msd(g_a)
m_gd, s_gd, ci_gd = msd(g_d)

out(f"  Across {n} independent trainings differing only in random seed, the supervised")
out(f"  baseline reached AUC {m_cp:.3f} +/- {s_cp:.3f} on PTB-XL and {m_cm:.3f} +/- {s_cm:.3f} on MIT-BIH,")
out(f"  while the autoencoder reached {m_ap:.3f} +/- {s_ap:.3f} and {m_am:.3f} +/- {s_am:.3f} respectively.")
out(f"  The generalization gap was {m_gc:.3f} +/- {s_gc:.3f} for the supervised model and")
out(f"  {m_ga:.3f} +/- {s_ga:.3f} for the autoencoder; the difference of {m_gd:.3f} +/- {s_gd:.3f}")
_pos = summary.get("gap_larger_count", {}).get("count", 0)
_consistency = ("was in the same direction in every seed" if _pos == n
                else f"was in the same direction in {_pos} of {n} seeds")
out(f"  (95% CI [{ci_gd[0]:.3f}, {ci_gd[1]:.3f}]) {_consistency}")
if "paired_t_gap" in summary:
    out(f"  (paired t({n-1}) = {summary['paired_t_gap']['t']:.2f}, p = {summary['paired_t_gap']['p']:.3g}).")
else:
    out("  .")

with open("results/multiseed3_analysis.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
with open("results/multiseed3_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print()
print("Saved: results/multiseed3_analysis.txt")
print("Saved: results/multiseed3_summary.json")
print("Saved: results/multiseed3_runs.json  (per-seed raw results)")
print()
print(">>> Send me results/multiseed3_analysis.txt and I will fold the across-seed")
print(">>> numbers into the manuscript.")
