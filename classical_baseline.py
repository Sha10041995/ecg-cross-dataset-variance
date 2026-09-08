"""
Classical (non-deep) feature-based baseline for the ECG cross-dataset study.

WHY THIS EXISTS
---------------
Reviewers of a deep-learning paper almost always ask: "did you compare against a
simple, interpretable baseline?" This script answers that, and it does something
more useful for the paper's argument.

The manuscript claims (Section 5.1) that the supervised CNN degrades under domain
shift because it latches onto *domain-specific* rather than *pathology-specific*
cues. That claim is currently an interpretation, not a measurement. A classical
model built on hand-chosen physiological features -- heart rate, RR-interval
variability, QRS morphology, spectral content -- cannot exploit acquisition
shortcuts in the same way, because it only ever sees those features. So:

  * If the feature model's generalization gap is SMALL (like the autoencoder's),
    that supports the shortcut-learning explanation: the fragility belongs to
    what the CNN learned, not to the supervised objective as such.
  * If the feature model's gap is LARGE (like the CNN's), the explanation is
    weaker, and the degradation is better attributed to genuine distribution
    shift in the physiology/labels themselves.

Either outcome is publishable and both sharpen the paper. Report what you get.

WHAT IT DOES
------------
1. Extracts 27 interpretable features per 10-second single-lead window.
2. Trains logistic regression and random forest on the PTB-XL training folds.
3. Evaluates in-domain (PTB-XL fold 10) and cross-dataset (MIT-BIH), using the
   same zero-adaptation protocol as the deep models.
4. Reports the generalization gap for each, alongside the CNN and autoencoder
   numbers, plus random-forest feature importances.
5. Adds stratified-bootstrap 95% confidence intervals for every AUC and gap, and
   paired DeLong tests of each feature model against the deep models on MIT-BIH,
   reusing the cached deep-model scores in processed/analysis_scores.npz.

USAGE
-----
    python classical_baseline.py                 # full run
    python classical_baseline.py --quick         # 2000-sample subset, ~1 min

Runtime: feature extraction is the slow part (~3-8 min for 21,799 + 8,269
windows on CPU). Features are cached to processed/classical_features.npz, so a
second run is nearly instant.
"""
import os
import json
import time
import argparse

import numpy as np
from scipy import stats, signal as sps
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

N_BOOT = 2000

_ap = argparse.ArgumentParser(description="Classical feature baseline for the ECG study.")
_ap.add_argument("--quick", action="store_true",
                 help="Subsample to 2000 windows per dataset for a fast smoke test.")
_args = _ap.parse_args()

FS = 100                      # Hz, both datasets after preprocessing
WINDOW = 1000                 # samples (10 s)
SEED = 42
CACHE = "processed/classical_features.npz"
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # --- rhythm / HRV (from detected R-peaks) ---
    "n_beats", "heart_rate", "rr_mean", "rr_std", "rr_cv",
    "rr_rmssd", "rr_pnn50", "rr_min", "rr_max", "rr_range",
    "rr_median_abs_dev",
    # --- QRS morphology ---
    "r_amp_mean", "r_amp_std", "r_amp_cv",
    "beat_template_corr_mean", "beat_template_corr_min",
    "qrs_width_mean", "qrs_width_std",
    # --- global waveform statistics ---
    "sig_std", "sig_skew", "sig_kurtosis", "sig_iqr",
    "zero_cross_rate", "abs_diff_mean",
    # --- spectral ---
    "bp_0p5_4", "bp_4_15", "bp_15_40", "spectral_entropy",
]


def _rr_features(rpeaks, n):
    """Rhythm features from R-peak sample indices."""
    out = {}
    out["n_beats"] = float(len(rpeaks))
    if len(rpeaks) < 3:
        # Too few beats to characterise rhythm. Return neutral values; the
        # n_beats feature itself carries the information that this happened.
        for k in ["heart_rate", "rr_mean", "rr_std", "rr_cv", "rr_rmssd",
                  "rr_pnn50", "rr_min", "rr_max", "rr_range", "rr_median_abs_dev"]:
            out[k] = 0.0
        return out
    rr = np.diff(rpeaks) / FS                       # seconds
    out["heart_rate"] = float(60.0 / np.mean(rr)) if np.mean(rr) > 0 else 0.0
    out["rr_mean"] = float(np.mean(rr))
    out["rr_std"] = float(np.std(rr))
    out["rr_cv"] = float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else 0.0
    drr = np.diff(rr)
    out["rr_rmssd"] = float(np.sqrt(np.mean(drr ** 2))) if len(drr) else 0.0
    out["rr_pnn50"] = float(np.mean(np.abs(drr) > 0.05)) if len(drr) else 0.0
    out["rr_min"] = float(np.min(rr))
    out["rr_max"] = float(np.max(rr))
    out["rr_range"] = float(np.max(rr) - np.min(rr))
    out["rr_median_abs_dev"] = float(np.median(np.abs(rr - np.median(rr))))
    return out


def _morphology_features(x, rpeaks):
    """QRS amplitude/shape consistency across the beats in the window."""
    out = {}
    if len(rpeaks) < 2:
        for k in ["r_amp_mean", "r_amp_std", "r_amp_cv",
                  "beat_template_corr_mean", "beat_template_corr_min",
                  "qrs_width_mean", "qrs_width_std"]:
            out[k] = 0.0
        return out

    amps = x[rpeaks]
    out["r_amp_mean"] = float(np.mean(amps))
    out["r_amp_std"] = float(np.std(amps))
    out["r_amp_cv"] = float(np.std(amps) / (np.abs(np.mean(amps)) + 1e-8))

    # Beat-aligned windows: 250 ms before, 400 ms after each R peak
    pre, post = int(0.25 * FS), int(0.40 * FS)
    beats = [x[r - pre:r + post] for r in rpeaks
             if r - pre >= 0 and r + post <= len(x)]
    if len(beats) < 2:
        out["beat_template_corr_mean"] = 0.0
        out["beat_template_corr_min"] = 0.0
    else:
        B = np.vstack(beats)
        template = B.mean(axis=0)
        t_c = template - template.mean()
        t_n = np.linalg.norm(t_c) + 1e-12
        corrs = []
        for b in B:
            b_c = b - b.mean()
            corrs.append(float(np.dot(b_c, t_c) / (np.linalg.norm(b_c) * t_n + 1e-12)))
        # Low mean correlation = beats do not look alike = ectopy / arrhythmia
        out["beat_template_corr_mean"] = float(np.mean(corrs))
        out["beat_template_corr_min"] = float(np.min(corrs))

    # Crude QRS width: samples around each R peak exceeding 30% of its amplitude
    widths = []
    for r in rpeaks:
        lo, hi = max(0, r - int(0.10 * FS)), min(len(x), r + int(0.10 * FS))
        seg = np.abs(x[lo:hi] - np.median(x))
        if seg.size and seg.max() > 0:
            widths.append(float(np.sum(seg > 0.3 * seg.max()) / FS))
    out["qrs_width_mean"] = float(np.mean(widths)) if widths else 0.0
    out["qrs_width_std"] = float(np.std(widths)) if widths else 0.0
    return out


def _global_features(x):
    out = {}
    out["sig_std"] = float(np.std(x))
    out["sig_skew"] = float(stats.skew(x))
    out["sig_kurtosis"] = float(stats.kurtosis(x))
    out["sig_iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
    out["zero_cross_rate"] = float(np.mean(np.diff(np.signbit(x)) != 0))
    out["abs_diff_mean"] = float(np.mean(np.abs(np.diff(x))))
    return out


def _spectral_features(x):
    out = {}
    f, pxx = sps.welch(x, fs=FS, nperseg=min(256, len(x)))
    total = np.sum(pxx) + 1e-12
    for name, (lo, hi) in {"bp_0p5_4": (0.5, 4), "bp_4_15": (4, 15),
                           "bp_15_40": (15, 40)}.items():
        m = (f >= lo) & (f < hi)
        out[name] = float(np.sum(pxx[m]) / total)
    p = pxx / total
    p = p[p > 0]
    out["spectral_entropy"] = float(-np.sum(p * np.log(p)) / np.log(len(p))) if len(p) > 1 else 0.0
    return out


def detect_rpeaks(x):
    """
    Simple, dataset-agnostic R-peak detector (Pan-Tompkins style: derivative ->
    square -> moving-window integration -> adaptive threshold -> refractory).

    Deliberately not tuned per dataset: the point of this baseline is that it
    uses no domain-specific fitting. The identical function is applied to
    PTB-XL and MIT-BIH.

    The threshold is a fraction of the 95th percentile of the integration
    envelope, which tracks QRS amplitude while ignoring P and T waves. If the
    implied heart rate is physiologically implausible, the threshold is raised
    or lowered once through a small fixed ladder -- the same ladder for every
    window in both datasets.
    """
    d = np.diff(x, prepend=x[0])
    envelope = np.convolve(d ** 2, np.ones(max(1, int(0.12 * FS))) / max(1, int(0.12 * FS)),
                           mode="same")
    if envelope.max() <= 0:
        return np.array([], dtype=int)

    # Fix R-peak polarity once per window so that beats align consistently.
    positive = np.percentile(x, 99) >= abs(np.percentile(x, 1))

    ref = envelope[envelope > 0]
    base = np.percentile(ref, 95) if ref.size else envelope.max()
    n_sec = len(x) / FS
    best = None
    for frac in (0.35, 0.50, 0.22, 0.65, 0.15):
        peaks, _ = sps.find_peaks(envelope, height=frac * base,
                                  distance=int(0.30 * FS))   # 300 ms refractory
        hr = 60.0 * len(peaks) / n_sec
        if best is None:
            best = peaks
        if 30.0 <= hr <= 220.0:       # physiologically plausible
            best = peaks
            break
    peaks = best

    # Snap each detection to the dominant local extremum of the raw signal
    refined, half = [], int(0.05 * FS)
    for p in peaks:
        lo, hi = max(0, p - half), min(len(x), p + half)
        if hi > lo:
            seg = x[lo:hi]
            refined.append(lo + int(np.argmax(seg) if positive else np.argmin(seg)))
    return np.unique(np.array(refined, dtype=int))


def extract_features(x):
    rpeaks = detect_rpeaks(x)
    f = {}
    f.update(_rr_features(rpeaks, len(x)))
    f.update(_morphology_features(x, rpeaks))
    f.update(_global_features(x))
    f.update(_spectral_features(x))
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float64)


def extract_matrix(X, label=""):
    t0 = time.time()
    out = np.zeros((len(X), len(FEATURE_NAMES)), dtype=np.float64)
    for i, x in enumerate(X):
        out[i] = extract_features(np.asarray(x, dtype=np.float64))
        if label and (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"    {label}: {i+1}/{len(X)}  ({el:.0f}s elapsed, "
                  f"~{el/(i+1)*(len(X)-i-1):.0f}s left)")
    out[~np.isfinite(out)] = 0.0
    return out



# ---------------------------------------------------------------------------
# Uncertainty: stratified bootstrap and the DeLong test
# ---------------------------------------------------------------------------
def _strat_boot(y, n_boot, rng):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, pos.size, replace=True),
                              rng.choice(neg, neg.size, replace=True)])


def auc_ci(y, score, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = [roc_auc_score(y[i], score[i]) for i in _strat_boot(y, n_boot, rng)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def gap_ci(y_in, s_in, y_out, s_out, n_boot=N_BOOT, seed=SEED):
    r1 = np.random.default_rng(seed + 1)
    r2 = np.random.default_rng(seed + 2)
    a = list(_strat_boot(y_in, n_boot, r1))
    b = list(_strat_boot(y_out, n_boot, r2))
    vals = [roc_auc_score(y_in[i], s_in[i]) - roc_auc_score(y_out[j], s_out[j])
            for i, j in zip(a, b)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2


def delong_test(y, s1, s2):
    """Paired AUC comparison on identical samples (Sun & Xu fast DeLong)."""
    order = (-np.asarray(y, dtype=float)).argsort(kind="mergesort")
    m = int(np.sum(y))
    preds = np.vstack((s1, s2))[:, order]
    k, N = preds.shape
    n = N - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, N])
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    l = np.array([[1.0, -1.0]])
    var = float((l @ cov @ l.T)[0, 0])
    diff = float(aucs[0] - aucs[1])
    if not np.isfinite(var) or var <= 0:
        # Degenerate case: the two score vectors rank the samples identically,
        # so the variance of their AUC difference is zero and z is undefined.
        return diff, float("nan"), float("nan")
    z = diff / np.sqrt(var)
    return diff, float(z), float(2 * stats.norm.sf(abs(z)))


# ---------------------------------------------------------------------------
# Load data and build the feature matrices
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    _X = np.load("processed/ptbxl_X.npy", mmap_mode="r")
    X_ptb_all = np.ascontiguousarray(_X[:, :, 1])          # lead II
    del _X
    y_is_normal = np.load("processed/ptbxl_y_is_normal.npy")
    folds = np.load("processed/ptbxl_folds.npy")
    X_mit = np.load("processed/mitbih_X.npy")
    y_mit = (~np.load("processed/mitbih_y_is_normal.npy")).astype(int)
    y_ptb_all = (~y_is_normal).astype(int)

    if _args.quick:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(len(X_ptb_all), min(2000, len(X_ptb_all)), replace=False)
        X_ptb_all, y_ptb_all, folds = X_ptb_all[keep], y_ptb_all[keep], folds[keep]
        keep_m = rng.choice(len(X_mit), min(2000, len(X_mit)), replace=False)
        X_mit, y_mit = X_mit[keep_m], y_mit[keep_m]
        print("  QUICK MODE: 2000 windows per dataset (results NOT publishable)")

    use_cache = os.path.exists(CACHE) and not _args.quick
    if use_cache:
        print(f"Loading cached features from {CACHE}")
        d = np.load(CACHE)
        F_ptb, F_mit = d["F_ptb"], d["F_mit"]
        if len(F_ptb) != len(X_ptb_all) or len(F_mit) != len(X_mit):
            print("  Cache size mismatch -- recomputing.")
            use_cache = False
    if not use_cache:
        print(f"Extracting {len(FEATURE_NAMES)} features per window...")
        print("  PTB-XL:")
        F_ptb = extract_matrix(X_ptb_all, "PTB-XL")
        print("  MIT-BIH:")
        F_mit = extract_matrix(X_mit, "MIT-BIH")
        if not _args.quick:
            np.savez(CACHE, F_ptb=F_ptb, F_mit=F_mit)
            print(f"  Cached to {CACHE}")

    train_mask, test_mask = folds <= 8, folds == 10
    F_tr, y_tr = F_ptb[train_mask], y_ptb_all[train_mask]
    F_te, y_te = F_ptb[test_mask], y_ptb_all[test_mask]
    print(f"\nTrain {len(F_tr)} | PTB-XL test {len(F_te)} | MIT-BIH {len(F_mit)}")

    # -----------------------------------------------------------------------
    # Fit and evaluate
    # -----------------------------------------------------------------------
    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED)),
        "Random forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   random_state=SEED, n_jobs=-1)),
    }

    report, results = [], {}

    def out(line=""):
        print(line)
        report.append(line)

    out("=" * 74)
    out("CLASSICAL FEATURE BASELINE -- cross-dataset generalization")
    out(f"{len(FEATURE_NAMES)} interpretable features; trained on PTB-XL folds 1-8;")
    out("evaluated on PTB-XL fold 10 (in-domain) and MIT-BIH (cross-dataset),")
    out("with no adaptation -- the same protocol as the deep models.")
    out("=" * 74)

    scores = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(F_tr, y_tr)
        s_in = model.predict_proba(F_te)[:, 1]
        s_out = model.predict_proba(F_mit)[:, 1]
        scores[name] = (s_in, s_out)
        auc_in = roc_auc_score(y_te, s_in)
        auc_out = roc_auc_score(y_mit, s_out)
        gap = auc_in - auc_out
        ci_in = auc_ci(y_te, s_in)
        ci_out = auc_ci(y_mit, s_out)
        ci_gap = gap_ci(y_te, s_in, y_mit, s_out)
        out()
        out(f"{name}   ({time.time()-t0:.1f}s)")
        out(f"  PTB-XL  (in-domain)     AUC = {auc_in:.4f}  "
            f"95% CI [{ci_in[0]:.4f}, {ci_in[1]:.4f}]")
        out(f"  MIT-BIH (cross-dataset) AUC = {auc_out:.4f}  "
            f"95% CI [{ci_out[0]:.4f}, {ci_out[1]:.4f}]")
        out(f"  Generalization gap          = {gap:.4f}  "
            f"95% CI [{ci_gap[0]:.4f}, {ci_gap[1]:.4f}]   "
            f"({100*gap/auc_in:.1f}% relative)")
        results[name] = {"auc_in": float(auc_in), "auc_in_ci": list(ci_in),
                         "auc_out": float(auc_out), "auc_out_ci": list(ci_out),
                         "gap": float(gap), "gap_ci": list(ci_gap),
                         "relative_gap": float(gap / auc_in)}

    np.savez("processed/classical_scores.npz",
             lr_in=scores["Logistic regression"][0], lr_out=scores["Logistic regression"][1],
             rf_in=scores["Random forest"][0], rf_out=scores["Random forest"][1],
             y_in=y_te, y_out=y_mit)

    # ---- paired DeLong tests against the deep models on the SAME samples ----
    deep_cache = "processed/analysis_scores.npz"
    if os.path.exists(deep_cache) and not _args.quick:
        d = np.load(deep_cache)
        if len(d["y_mit"]) == len(y_mit) and len(d["y_ptb"]) == len(y_te):
            out()
            out("-" * 74)
            out("PAIRED DeLong TESTS (same samples; feature model vs deep model)")
            out("-" * 74)
            pairs = {"Supervised CNN": ("cnn_ptb", "cnn_mit"),
                     "Autoencoder": ("ae_ptb", "ae_mit")}
            results["delong"] = {}

            # the two feature models against each other, so that every pair in
            # the four-model comparison has been tested
            lr_i, lr_o = scores["Logistic regression"]
            rf_i, rf_o = scores["Random forest"]
            for dom, y, a, b in (("PTB-XL", y_te, lr_i, rf_i),
                                 ("MIT-BIH", y_mit, lr_o, rf_o)):
                diff, z, pv = delong_test(np.asarray(y), np.asarray(a), np.asarray(b))
                verdict = ("undefined (identical rankings)" if not np.isfinite(pv)
                           else "significant" if pv < 0.05 else "n.s.")
                out(f"  {'Logistic regression':<20} vs {'Random forest':<15} on {dom:<8} "
                    f"dAUC={diff:+.4f}  z={z:+7.3f}  p={pv:.3g}  ({verdict})")
                results["delong"][f"Logistic regression vs Random forest on {dom}"] = {
                    "auc_diff": diff, "z": z, "p": pv}
            for fname, (s_in, s_out) in scores.items():
                for dname, (kin, kout) in pairs.items():
                    for dom, y, a, b in (("PTB-XL", y_te, s_in, d[kin]),
                                         ("MIT-BIH", y_mit, s_out, d[kout])):
                        diff, z, p = delong_test(np.asarray(y), np.asarray(a), np.asarray(b))
                        sig = ("undefined (identical rankings)" if not np.isfinite(p)
                               else "significant" if p < 0.05 else "n.s.")
                        out(f"  {fname:<20} vs {dname:<15} on {dom:<8} "
                            f"dAUC={diff:+.4f}  z={z:+7.3f}  p={p:.3g}  ({sig})")
                        results["delong"][f"{fname} vs {dname} on {dom}"] = {
                            "auc_diff": diff, "z": z, "p": p}
        else:
            out()
            out("  (deep-model score cache has a different size -- DeLong tests skipped;")
            out("   re-run statistical_analysis.py so the cached scores match.)")
    elif not _args.quick:
        out()
        out("  (processed/analysis_scores.npz not found -- run statistical_analysis.py")
        out("   first to enable the paired DeLong tests against the deep models.)")

    # Feature importances from the random forest
    rf = models["Random forest"][-1]
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    out()
    out("-" * 74)
    out("Random-forest feature importances (top 12)")
    out("-" * 74)
    for i in order[:12]:
        out(f"  {FEATURE_NAMES[i]:<26} {imp[i]:.4f}")
    results["rf_importances"] = {FEATURE_NAMES[i]: float(imp[i]) for i in order}

    # -----------------------------------------------------------------------
    # Comparison against the deep models
    # -----------------------------------------------------------------------
    out()
    out("=" * 74)
    out("COMPARISON WITH THE DEEP MODELS  (generalization gap, AUC)")
    out("=" * 74)
    deep = {}
    ms_path = "results/multiseed_summary.json"
    if os.path.exists(ms_path):
        with open(ms_path, encoding="utf-8") as f:
            ms = json.load(f)
        try:
            deep["Supervised CNN (5-seed mean)"] = (
                ms["CNN  AUC  PTB-XL"]["mean"], ms["CNN  AUC  MIT-BIH"]["mean"])
            deep["Autoencoder (5-seed mean)"] = (
                ms["AE   AUC  PTB-XL"]["mean"], ms["AE   AUC  MIT-BIH"]["mean"])
        except KeyError:
            pass
    if not deep:
        out("  (results/multiseed_summary.json not found -- deep-model numbers")
        out("   omitted. Run multiseed_experiment.py first to fill this in.)")
    else:
        out(f"  {'Model':<32} {'PTB-XL':>8} {'MIT-BIH':>9} {'gap':>8} {'rel.':>7}")
        for nm, (a, b) in deep.items():
            out(f"  {nm:<32} {a:>8.3f} {b:>9.3f} {a-b:>8.3f} {100*(a-b)/a:>6.1f}%")
        for nm in models:                      # model names only, not the
            r = results[nm]                    # 'rf_importances'/'delong' keys
            out(f"  {nm:<32} {r['auc_in']:>8.3f} {r['auc_out']:>9.3f} "
                f"{r['gap']:>8.3f} {100*r['relative_gap']:>6.1f}%")

    out()
    out("HOW TO READ THIS")
    out("-" * 74)
    out("  A SMALL gap for the feature models supports the shortcut-learning")
    out("  explanation in Section 5.1: hand-chosen physiological features cannot")
    out("  encode acquisition-specific cues, so a model restricted to them keeps")
    out("  its performance. The CNN's fragility would then belong to what it")
    out("  learned, not to supervision itself.")
    out()
    out("  A LARGE gap for the feature models argues the other way: the shift")
    out("  between these datasets is substantive (physiology, label definitions,")
    out("  recording context), not merely a matter of exploitable shortcuts.")
    out()
    out("  Both outcomes belong in the paper. Report whichever you observe, and")
    out("  adjust the Section 5.1 wording to match rather than the reverse.")

    with open("results/classical_baseline.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    with open("results/classical_baseline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("Saved: results/classical_baseline.txt")
    print("Saved: results/classical_baseline.json")
    print()
    print(">>> Send me results/classical_baseline.txt and I will add this as a")
    print(">>> third model to the manuscript's tables and discussion.")


if __name__ == "__main__":
    main()
