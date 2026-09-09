"""
Reverse-direction transfer: train on MIT-BIH, test on PTB-XL.

WHY THIS EXISTS
---------------
The manuscript currently transfers in one direction only (PTB-XL -> MIT-BIH).
The most predictable reviewer question is: "is the asymmetry you report a
property of the two learning paradigms, or an artefact of this particular
source/target pair?" A single sentence in the limitations section concedes the
point; running the experiment answers it.

This script mirrors the main experiment exactly, with source and target swapped:

    train on MIT-BIH  ->  test in-domain on MIT-BIH  ->  transfer to PTB-XL

Same architectures, same optimiser, same epoch counts, same Youden threshold
rule, same zero-adaptation protocol, same seeds. Only the direction changes.

If the supervised model again shows the larger generalization gap, the finding
is bidirectional and the paper's central claim is substantially strengthened.
If it does not, that is a genuine and interesting boundary condition, and it
belongs in the paper too. Report whichever you observe.

PATIENT-DISJOINT SPLITTING
--------------------------
MIT-BIH contains ~30-minute recordings from 47 subjects, so consecutive windows
from one recording are highly correlated. Splitting windows at random would put
the same patient in train and test and inflate the in-domain result badly. This
script therefore splits by RECORD, never by window, and requires
processed/mitbih_record_ids.npy (written by preprocess_mitbih.py). Records are
sorted by their abnormal fraction and dealt round-robin into train/val/test so
that the three splits have comparable class balance despite the small number of
records.

USAGE
-----
    python reverse_transfer.py --quick        # 1 seed, 1 epoch, ~2 min
    python reverse_transfer.py                # 5 seeds
    python reverse_transfer.py --seeds 0 1 2  # 3 seeds

MIT-BIH's training split is roughly a third the size of PTB-XL's, so each seed
is considerably faster than in multiseed_experiment.py. Resumable: finished
seeds are kept in results/reverse_runs.json.
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

# ============================= CONFIG =======================================
_ap = argparse.ArgumentParser(description="Reverse-direction transfer: MIT-BIH -> PTB-XL.")
_ap.add_argument("--quick", action="store_true",
                 help="1 seed, 1 epoch each. Plumbing check only, NOT publishable.")
_ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                 help="Random seeds to run (default: 0 1 2 3 4).")
_args = _ap.parse_args()

QUICK = _args.quick
SEEDS = [_args.seeds[0]] if QUICK else _args.seeds
N_EPOCHS_CNN = 1 if QUICK else 15
N_EPOCHS_AE = 1 if QUICK else 20
BATCH_SIZE, LR, CNN_THRESHOLD = 64, 1e-3, 0.5
RESULTS_JSON = "results/_quicktest_reverse_runs.json" if QUICK else "results/reverse_runs.json"

os.makedirs("results", exist_ok=True)
device = torch.device("cpu")

if QUICK:
    print("*** QUICK MODE: 1 seed, 1 epoch each -- plumbing check only. ***")
    print("*** If this finishes cleanly, rerun without --quick. ***")
# ============================================================================


# ---------------------------------------------------------------------------
# Architectures -- identical to the forward experiment
# ---------------------------------------------------------------------------
class ECG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(),
                                nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x):
        return self.fc(self.net(x)).squeeze(-1)


class ECG_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv1d(32, 16, kernel_size=5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv1d(16, 1, kernel_size=7, padding=3))

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
# Record-disjoint split of MIT-BIH
# ---------------------------------------------------------------------------
def record_disjoint_split(record_ids, y):
    """
    Deal records into train/val/test so that no recording appears in more than
    one split, and the three splits have comparable abnormal prevalence.

    Records are ordered by abnormal fraction and dealt round-robin
    (train, train, train, val, test, ...), which keeps prevalence balanced
    without any randomness -- the split is identical for every seed, so
    across-seed variation reflects training only.
    """
    recs = np.unique(record_ids)
    frac = np.array([y[record_ids == r].mean() for r in recs])
    order = np.argsort(frac)
    pattern = ["train", "train", "train", "val", "test"]
    assign = {}
    for i, idx in enumerate(order):
        assign[recs[idx]] = pattern[i % len(pattern)]
    split = np.array([assign[r] for r in record_ids])
    return split, assign


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
print("Loading data...")
X_mit = np.load("processed/mitbih_X.npy")
y_mit = (~np.load("processed/mitbih_y_is_normal.npy")).astype(np.float32)

REC_PATH = "processed/mitbih_record_ids.npy"
if not os.path.exists(REC_PATH):
    sys.exit(
        f"\nERROR: {REC_PATH} not found.\n\n"
        "This experiment needs a PATIENT-DISJOINT split of MIT-BIH. Splitting\n"
        "30-minute recordings at the window level would place the same patient\n"
        "in train and test and produce a meaningless in-domain result.\n\n"
        "preprocess_mitbih.py already writes this file -- re-run it:\n"
        "    python preprocess_mitbih.py\n")
record_ids = np.load(REC_PATH, allow_pickle=True).astype(str)

_X = np.load("processed/ptbxl_X.npy", mmap_mode="r")
X_ptb_all = np.ascontiguousarray(_X[:, :, 1])          # lead II
del _X
folds = np.load("processed/ptbxl_folds.npy")
y_ptb_all = (~np.load("processed/ptbxl_y_is_normal.npy")).astype(np.float32)
test_mask = folds == 10
X_ptb, y_ptb = X_ptb_all[test_mask], y_ptb_all[test_mask]

split, assign = record_disjoint_split(record_ids, y_mit)
tr, va, te = split == "train", split == "val", split == "test"
X_tr, y_tr = X_mit[tr], y_mit[tr]
X_va, y_va = X_mit[va], y_mit[va]
X_te, y_te = X_mit[te], y_mit[te]
X_tr_normal = X_mit[tr & (y_mit == 0)]

n_rec = {s: sum(1 for v in assign.values() if v == s) for s in ("train", "val", "test")}
print(f"  MIT-BIH records: {len(assign)} total -> "
      f"train {n_rec['train']}, val {n_rec['val']}, test {n_rec['test']} (disjoint)")
print(f"  MIT-BIH windows: train {tr.sum()} (abnormal {y_tr.mean():.3f}), "
      f"val {va.sum()} ({y_va.mean():.3f}), test {te.sum()} ({y_te.mean():.3f})")
print(f"  Autoencoder training set (normal only): {len(X_tr_normal)}")
print(f"  PTB-XL cross-dataset target (fold 10): {len(X_ptb)} "
      f"(abnormal {y_ptb.mean():.3f})")

if len(X_tr_normal) < 200:
    print("\n  WARNING: very few normal training windows; the autoencoder result "
          "will be unstable.")


def set_all_seeds(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def loader(ds, shuffle, seed, bs=BATCH_SIZE):
    if shuffle:
        g = torch.Generator(); g.manual_seed(seed)
        return DataLoader(ds, batch_size=bs, shuffle=True, generator=g)
    return DataLoader(ds, batch_size=bs, shuffle=False)


def metrics_at(y, score, thr):
    pred = (score >= thr).astype(int); y = y.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"AUC": float(roc_auc_score(y, score)), "Accuracy": (tp + tn) / len(y),
            "Precision": prec, "Recall": rec, "F1": f1, "Specificity": spec,
            "BalancedAcc": 0.5 * (rec + spec),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------
def run_seed(seed):
    t_start = time.time()
    print(f"\n{'='*70}\nSEED {seed}   (source = MIT-BIH, target = PTB-XL)\n{'='*70}")

    set_all_seeds(seed)
    cnn = ECG_CNN().to(device)
    crit, opt = nn.BCEWithLogitsLoss(), None
    opt = torch.optim.Adam(cnn.parameters(), lr=LR)
    tl = loader(XYDataset(X_tr, y_tr), True, seed)
    print(f"  [CNN] training {N_EPOCHS_CNN} epochs on {len(X_tr)} windows...")
    for ep in range(N_EPOCHS_CNN):
        t0 = time.time(); cnn.train(); tot = 0.0
        for xb, yb in tl:
            opt.zero_grad(); loss = crit(cnn(xb), yb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        dt = time.time() - t0
        print(f"    epoch {ep+1:2d}/{N_EPOCHS_CNN}  loss={tot/len(X_tr):.4f}  ({dt:.0f}s)")
        if ep == 0:
            per = dt * N_EPOCHS_CNN + dt * (len(X_tr_normal) / len(X_tr)) * N_EPOCHS_AE
            print(f"    [estimate] ~{per/60:.0f} min per seed, "
                  f"~{per*len(SEEDS)/60:.0f} min total")
    cnn.eval()

    def cnn_score(Xd):
        o = []
        with torch.no_grad():
            for xb in loader(XDataset(Xd), False, seed, bs=128):
                o.extend(torch.sigmoid(cnn(xb)).numpy())
        return np.array(o)

    cnn_in, cnn_out = cnn_score(X_te), cnn_score(X_ptb)

    set_all_seeds(seed + 10_000)
    ae = ECG_Autoencoder().to(device)
    crit_ae = nn.MSELoss()
    opt_ae = torch.optim.Adam(ae.parameters(), lr=LR)
    tl_ae = loader(XDataset(X_tr_normal), True, seed + 10_000)
    print(f"  [AE ] training {N_EPOCHS_AE} epochs on {len(X_tr_normal)} normal windows...")
    for ep in range(N_EPOCHS_AE):
        t0 = time.time(); ae.train(); tot = 0.0
        for xb in tl_ae:
            opt_ae.zero_grad(); loss = crit_ae(ae(xb), xb); loss.backward(); opt_ae.step()
            tot += loss.item() * len(xb)
        print(f"    epoch {ep+1:2d}/{N_EPOCHS_AE}  recon={tot/len(X_tr_normal):.5f}  "
              f"({time.time()-t0:.0f}s)")
    ae.eval()

    def ae_score(Xd):
        o = []
        with torch.no_grad():
            for xb in loader(XDataset(Xd), False, seed, bs=128):
                rec = ae(xb)
                o.extend(((rec - xb) ** 2).mean(dim=[1, 2]).numpy())
        return np.array(o)

    ae_val = ae_score(X_va)
    fpr, tpr, thr = roc_curve(y_va.astype(int), ae_val)
    thr_ae = float(thr[np.argmax(tpr - fpr)])
    ae_in, ae_out = ae_score(X_te), ae_score(X_ptb)

    res = {"seed": seed, "ae_threshold": thr_ae,
           "cnn_mitbih": metrics_at(y_te, cnn_in, CNN_THRESHOLD),
           "cnn_ptbxl": metrics_at(y_ptb, cnn_out, CNN_THRESHOLD),
           "ae_mitbih": metrics_at(y_te, ae_in, thr_ae),
           "ae_ptbxl": metrics_at(y_ptb, ae_out, thr_ae),
           "minutes": (time.time() - t_start) / 60.0}
    res["gap_cnn"] = res["cnn_mitbih"]["AUC"] - res["cnn_ptbxl"]["AUC"]
    res["gap_ae"] = res["ae_mitbih"]["AUC"] - res["ae_ptbxl"]["AUC"]
    res["gap_diff"] = res["gap_cnn"] - res["gap_ae"]

    print(f"  -> CNN  in-domain(MIT)={res['cnn_mitbih']['AUC']:.4f}  "
          f"cross(PTB)={res['cnn_ptbxl']['AUC']:.4f}  gap={res['gap_cnn']:.4f}")
    print(f"  -> AE   in-domain(MIT)={res['ae_mitbih']['AUC']:.4f}  "
          f"cross(PTB)={res['ae_ptbxl']['AUC']:.4f}  gap={res['gap_ae']:.4f}")
    print(f"  -> gap difference = {res['gap_diff']:.4f}  ({res['minutes']:.1f} min)")
    return res


# ---------------------------------------------------------------------------
# Main loop (resumable)
# ---------------------------------------------------------------------------
runs = []
if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON, encoding="utf-8") as f:
        runs = json.load(f)
    print(f"\nResuming: seeds already done = {sorted(r['seed'] for r in runs)}")

for seed in SEEDS:
    if any(r["seed"] == seed for r in runs):
        print(f"\nSeed {seed} already done -- skipping.")
        continue
    runs.append(run_seed(seed))
    runs.sort(key=lambda r: r["seed"])
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    left = len([s for s in SEEDS if not any(r["seed"] == s for r in runs)])
    if left:
        print(f"\n  [{len(runs)}/{len(SEEDS)} done]  ~"
              f"{np.mean([r['minutes'] for r in runs])*left:.0f} min remaining")

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
report = []
def out(line=""):
    print(line); report.append(line)

runs = sorted([r for r in runs if r["seed"] in SEEDS], key=lambda r: r["seed"])
n = len(runs)
flat = lambda k: np.array([r[k] for r in runs], dtype=float)
col = lambda b, m: np.array([r[b][m] for r in runs], dtype=float)

def msd(v):
    m = float(v.mean())
    if len(v) > 1:
        sd = float(v.std(ddof=1))
        h = float(stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)))
    else:
        sd, h = 0.0, 0.0
    return m, sd, (m - h, m + h)

out("=" * 78)
out("REVERSE-DIRECTION TRANSFER:  train on MIT-BIH  ->  test on PTB-XL")
out(f"seeds = {[r['seed'] for r in runs]}   (n = {n});  patient-disjoint MIT-BIH split")
out("=" * 78)
out()
out(f"{'seed':>5} | {'CNN MIT':>8} {'CNN PTB':>8} {'CNN gap':>8} | "
    f"{'AE MIT':>8} {'AE PTB':>8} {'AE gap':>8} | {'gap diff':>8}")
out("-" * 78)
for r in runs:
    out(f"{r['seed']:>5} | {r['cnn_mitbih']['AUC']:>8.4f} {r['cnn_ptbxl']['AUC']:>8.4f} "
        f"{r['gap_cnn']:>8.4f} | {r['ae_mitbih']['AUC']:>8.4f} {r['ae_ptbxl']['AUC']:>8.4f} "
        f"{r['gap_ae']:>8.4f} | {r['gap_diff']:>8.4f}")
out("-" * 78)
out()
out("ACROSS-SEED SUMMARY  (mean +/- SD [95% CI of the mean])")
out("-" * 78)
summary = {}
for name, v in [("CNN  AUC  MIT-BIH (in-domain)", col("cnn_mitbih", "AUC")),
                ("CNN  AUC  PTB-XL (cross)", col("cnn_ptbxl", "AUC")),
                ("CNN  generalization gap", flat("gap_cnn")),
                ("AE   AUC  MIT-BIH (in-domain)", col("ae_mitbih", "AUC")),
                ("AE   AUC  PTB-XL (cross)", col("ae_ptbxl", "AUC")),
                ("AE   generalization gap", flat("gap_ae")),
                ("DIFFERENCE of gaps", flat("gap_diff"))]:
    m, sd, (lo, hi) = msd(v)
    out(f"  {name:<32} {m:.4f} +/- {sd:.4f}  [{lo:.4f}, {hi:.4f}]")
    summary[name.strip()] = {"mean": m, "sd": sd, "ci_low": lo, "ci_high": hi,
                             "values": [float(x) for x in v]}

gc, ga = flat("gap_cnn"), flat("gap_ae")
if n >= 2:
    t, p = stats.ttest_rel(gc, ga)
    gd = flat("gap_diff")
    dz = float(gd.mean() / gd.std(ddof=1)) if gd.std(ddof=1) > 0 else float("inf")
    out()
    out("=" * 78)
    out("SIGNIFICANCE (paired across seeds)")
    out("=" * 78)
    out(f"  Paired t-test on the generalization gap: t({n-1}) = {t:.3f}, p = {p:.4g}")
    out(f"  Cohen's d_z = {dz:.2f}")
    out(f"  CNN gap larger than AE gap in {int((gd > 0).sum())} / {n} seeds.")
    summary["paired_t_gap"] = {"t": float(t), "p": float(p), "df": n - 1,
                               "cohen_dz": dz,
                               "cnn_larger": int((gd > 0).sum()), "n": n}

# ---------------------------------------------------------------------------
# Side-by-side with the forward direction
# ---------------------------------------------------------------------------
out()
out("=" * 78)
out("BIDIRECTIONAL COMPARISON")
out("=" * 78)
fwd_path = "results/multiseed_summary.json"
if os.path.exists(fwd_path):
    with open(fwd_path, encoding="utf-8") as f:
        fwd = json.load(f)
    try:
        f_cnn, f_ae = fwd["CNN  generalization gap"], fwd["AE   generalization gap"]
        r_cnn, r_ae = msd(gc), msd(ga)
        out(f"  {'Direction':<26} {'CNN gap':>16} {'AE gap':>16} {'difference':>13}")
        out(f"  {'PTB-XL -> MIT-BIH':<26} {f_cnn['mean']:>9.3f} +/- {f_cnn['sd']:.3f} "
            f"{fwd['AE   generalization gap']['mean']:>9.3f} +/- {f_ae['sd']:.3f} "
            f"{f_cnn['mean']-f_ae['mean']:>13.3f}")
        out(f"  {'MIT-BIH -> PTB-XL':<26} {r_cnn[0]:>9.3f} +/- {r_cnn[1]:.3f} "
            f"{r_ae[0]:>9.3f} +/- {r_ae[1]:.3f} "
            f"{r_cnn[0]-r_ae[0]:>13.3f}")
        out()
        same = (f_cnn["mean"] - f_ae["mean"] > 0) and (r_cnn[0] - r_ae[0] > 0)
        if same:
            out("  -> The supervised model shows the larger generalization gap in BOTH")
            out("     directions. The asymmetry is not an artefact of the source/target")
            out("     choice, and the paper's central claim can be stated as bidirectional.")
        else:
            out("  -> The asymmetry does NOT reproduce in the reverse direction. This is a")
            out("     real boundary condition, not a failure: report it, and restate the")
            out("     claim as specific to the PTB-XL -> MIT-BIH direction. Section 5.4")
            out("     should say so plainly rather than leaving the reader to infer it.")
        summary["bidirectional_consistent"] = bool(same)
    except KeyError:
        out("  (could not parse results/multiseed_summary.json)")
else:
    out("  (results/multiseed_summary.json not found -- run multiseed_experiment.py")
    out("   first to get the forward-direction numbers for comparison.)")

out()
out("CAVEAT TO CARRY INTO THE PAPER")
out("-" * 78)
out("  This direction is NOT symmetric with the forward one, and the manuscript")
out("  should say so. MIT-BIH offers ~47 subjects and a few thousand windows")
out("  against PTB-XL's 18,869 patients, so the reverse direction trains on far")
out("  less and far less diverse data. A larger gap here may reflect the smaller")
out("  training set as much as the direction of transfer. The comparison worth")
out("  making is between the TWO MODELS within this direction, not between the")
out("  absolute gap values across directions.")

txt = "results/_quicktest_reverse.txt" if QUICK else "results/reverse_transfer.txt"
js = "results/_quicktest_reverse.json" if QUICK else "results/reverse_transfer_summary.json"
with open(txt, "w", encoding="utf-8") as f:
    f.write("\n".join(report) + "\n")
with open(js, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print()
print(f"Saved: {txt}")
print(f"Saved: {js}")
print()
print(">>> Send me the .txt and I will add the reverse direction to the manuscript.")
