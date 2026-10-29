"""
Classical feature models, third database.

Reuses classical_baseline.py (imported, not copied) for feature extraction,
bootstrap and DeLong, so the 28 features and the R-peak detector are exactly
the ones used for PTB-XL and MIT-BIH. Trains the same two models on the same
PTB-XL folds with the same settings and scores Chapman-Shaoxing/Ningbo.

Does NOT modify any existing file. Writes:
    processed/classical_features_chapman.npz   (feature cache)
    processed/classical_scores_chapman.npz
    results/classical_chapman.txt, results/classical_chapman.json

Run from the project root, after preprocess_chapman.py:
    python classical_3db.py

Takes about 10-20 minutes (feature extraction for 45,150 windows dominates).
"""
import os
import sys
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

sys.argv = sys.argv[:1]                  # classical_baseline parses argv at import
sys.path.insert(0, os.getcwd())
import classical_baseline as cb          # noqa: E402

SEED, N_BOOT = cb.SEED, cb.N_BOOT
CACHE_CHA = "processed/classical_features_chapman.npz"
BROAD = {"426783006", "426177001", "427084000", "427393009"}


def gap_ci_stream(y_in, s_in, y_out, s_out, n_boot=N_BOOT, seed=SEED):
    """Same estimator as cb.gap_ci, but streams the resamples (a 45k-window
    target would otherwise hold ~700 MB of index arrays)."""
    r1, r2 = np.random.default_rng(seed + 1), np.random.default_rng(seed + 2)
    ga, gb = cb._strat_boot(y_in, n_boot, r1), cb._strat_boot(y_out, n_boot, r2)
    vals = [roc_auc_score(y_in[i], s_in[i]) - roc_auc_score(y_out[j], s_out[j])
            for i, j in zip(ga, gb)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    print("Loading data...")
    X_cha = np.load("processed/chapman_X.npy")
    cha_normal = np.load("processed/chapman_y_is_normal.npy")
    y_cha = (~cha_normal).astype(int)
    dx = np.load("processed/chapman_dx.npy")
    cha_normal_broad = np.array([len(s) > 0 and set(s.split(";")) <= BROAD for s in dx])
    y_cha_broad = (~cha_normal_broad).astype(int)

    y_ptb_all = (~np.load("processed/ptbxl_y_is_normal.npy")).astype(int)
    folds = np.load("processed/ptbxl_folds.npy")
    y_mit = (~np.load("processed/mitbih_y_is_normal.npy")).astype(int)

    d = np.load(cb.CACHE)
    F_ptb, F_mit = d["F_ptb"], d["F_mit"]
    assert len(F_ptb) == len(y_ptb_all) and len(F_mit) == len(y_mit), "feature cache mismatch"

    if os.path.exists(CACHE_CHA) and len(np.load(CACHE_CHA)["F_cha"]) == len(X_cha):
        print(f"Loading cached Chapman features from {CACHE_CHA}")
        F_cha = np.load(CACHE_CHA)["F_cha"]
    else:
        print(f"Extracting {len(cb.FEATURE_NAMES)} features for {len(X_cha)} Chapman windows...")
        F_cha = cb.extract_matrix(X_cha, "Chapman")
        np.savez(CACHE_CHA, F_cha=F_cha)

    tr, te = folds <= 8, folds == 10
    F_tr, y_tr, F_te, y_te = F_ptb[tr], y_ptb_all[tr], F_ptb[te], y_ptb_all[te]

    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "Random forest": make_pipeline(
            StandardScaler(), RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
    }

    old = {}
    if os.path.exists("results/classical_baseline.json"):
        with open("results/classical_baseline.json", encoding="utf-8") as f:
            old = json.load(f)

    report, results = [], {}

    def out(line=""):
        print(line)
        report.append(line)

    out("=" * 76)
    out("CLASSICAL FEATURE MODELS -- three databases")
    out(f"trained on PTB-XL folds 1-8; targets: MIT-BIH ({len(F_mit)}), "
        f"Chapman-Shaoxing/Ningbo ({len(F_cha)})")
    out(f"Chapman abnormal prevalence: strict {y_cha.mean():.3f}, broad {y_cha_broad.mean():.3f}")
    out("=" * 76)

    scores = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(F_tr, y_tr)
        s_in = model.predict_proba(F_te)[:, 1]
        s_mit = model.predict_proba(F_mit)[:, 1]
        s_cha = model.predict_proba(F_cha)[:, 1]
        scores[name] = (s_in, s_mit, s_cha)

        a_in, a_mit = roc_auc_score(y_te, s_in), roc_auc_score(y_mit, s_mit)
        a_cha = roc_auc_score(y_cha, s_cha)
        a_cha_b = roc_auc_score(y_cha_broad, s_cha)

        out()
        out(f"{name}   ({time.time()-t0:.1f}s)")
        if name in old:
            same = (a_in == old[name]["auc_in"]) and (a_mit == old[name]["auc_out"])
            diff = max(abs(a_in - old[name]["auc_in"]), abs(a_mit - old[name]["auc_out"]))
            out("  [repro] PTB-XL and MIT-BIH AUC IDENTICAL to classical_baseline.py"
                if same else
                f"  [repro] *** DRIFT: largest difference from the original run = {diff:.2e} ***")
        ci_in, ci_mit = cb.auc_ci(y_te, s_in), cb.auc_ci(y_mit, s_mit)
        ci_cha = cb.auc_ci(y_cha, s_cha)
        ci_cha_b = cb.auc_ci(y_cha_broad, s_cha)
        gap_m, gap_c = a_in - a_mit, a_in - a_cha
        ci_gap_c = gap_ci_stream(y_te, s_in, y_cha, s_cha)
        ci_gap_m = gap_ci_stream(y_te, s_in, y_mit, s_mit)
        out(f"  PTB-XL   (in-domain)        AUC = {a_in:.4f}  [{ci_in[0]:.4f}, {ci_in[1]:.4f}]")
        out(f"  MIT-BIH  (severe shift)     AUC = {a_mit:.4f}  [{ci_mit[0]:.4f}, {ci_mit[1]:.4f}]"
            f"   gap {gap_m:.4f} [{ci_gap_m[0]:.4f}, {ci_gap_m[1]:.4f}]")
        out(f"  Chapman  (mild shift)       AUC = {a_cha:.4f}  [{ci_cha[0]:.4f}, {ci_cha[1]:.4f}]"
            f"   gap {gap_c:.4f} [{ci_gap_c[0]:.4f}, {ci_gap_c[1]:.4f}]  ({100*gap_c/a_in:.1f}% relative)")
        out(f"  Chapman, broad label        AUC = {a_cha_b:.4f}  [{ci_cha_b[0]:.4f}, {ci_cha_b[1]:.4f}]")
        results[name] = {"auc_in": float(a_in), "auc_mit": float(a_mit), "gap_mit": float(gap_m),
                         "gap_mit_ci": list(ci_gap_m), "auc_cha": float(a_cha),
                         "auc_cha_ci": list(ci_cha), "gap_cha": float(gap_c),
                         "gap_cha_ci": list(ci_gap_c), "relative_gap_cha": float(gap_c / a_in),
                         "auc_cha_broad": float(a_cha_b), "auc_cha_broad_ci": list(ci_cha_b)}

    out()
    out("-" * 76)
    out("PAIRED DeLong: logistic regression vs random forest on Chapman (strict label)")
    diff, z, p = cb.delong_test(y_cha, scores["Logistic regression"][2], scores["Random forest"][2])
    out(f"  dAUC(LR-RF) = {diff:+.4f}  z = {z:+.3f}  p = {p:.3g}")
    results["delong_lr_rf_chapman"] = {"auc_diff": diff, "z": z, "p": p}

    rf = models["Random forest"][-1]
    imp = rf.feature_importances_
    results["rf_importances"] = {cb.FEATURE_NAMES[i]: float(imp[i]) for i in np.argsort(imp)[::-1]}

    np.savez("processed/classical_scores_chapman.npz",
             lr_cha=scores["Logistic regression"][2], rf_cha=scores["Random forest"][2],
             y_cha=y_cha, y_cha_broad=y_cha_broad)
    with open("results/classical_chapman.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    with open("results/classical_chapman.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/classical_chapman.txt, results/classical_chapman.json")
    print(">>> Send me results/classical_chapman.txt")


if __name__ == "__main__":
    main()
