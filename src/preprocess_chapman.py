"""
Preprocessing of the Chapman-Shaoxing / Ningbo 12-lead database, as a third
domain for the cross-dataset study:

1. Take lead II (comparable to PTB-XL lead II and MIT-BIH MLII)
2. Resample 500 Hz -> 100 Hz
3. Bandpass filter + z-score normalization (identical to PTB-XL and MIT-BIH)
4. Each record is already a 10-second (1000-sample after resampling) window
5. Label per record: normal if the diagnosis set is exactly {sinus rhythm},
   abnormal otherwise

The label rule mirrors the PTB-XL one. There, a record counts as normal only
if its set of diagnostic superclasses is exactly {NORM}: a record carrying
NORM alongside anything else counts as abnormal. The same conservative rule
is applied here to the SNOMED CT codes in the header.

Download (2.3 GB zipped, 5.1 GB unpacked):
    https://physionet.org/content/ecg-arrhythmia/1.0.0/
Unpack so that the tree looks like:
    data/chapman/WFDBRecords/01/010/JS00001.hea
    data/chapman/WFDBRecords/01/010/JS00001.mat
    data/chapman/ConditionNames_SNOMED-CT.csv      (if the download includes it)

USAGE
    python preprocess_chapman.py --inspect     # read headers only, print labels
    python preprocess_chapman.py               # full run, writes processed/chapman_*.npy

RUN --inspect FIRST. It reads no signals, finishes in a couple of minutes, and
prints which SNOMED codes actually occur and how many records the normal rule
would select. The code for sinus rhythm is assumed to be 426783006; confirm
that against the printed table before trusting the full run. A wrong label
rule would corrupt everything downstream in a way no later check would catch.
"""
import argparse
import os
import glob
from collections import Counter

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm

data_dir = "data/chapman"
output_dir = "processed"

ORIGINAL_FS = 500
TARGET_FS = 100
WINDOW_SEC = 10
WINDOW_SAMPLES = TARGET_FS * WINDOW_SEC  # 1000, identical to PTB-XL and MIT-BIH

SINUS_RHYTHM_SNOMED = "426783006"


def bandpass_filter(signal, fs=100, lowcut=0.5, highcut=40, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def find_records(root):
    """Every .hea under the tree. The signal file is .mat in the PhysioNet
    release and .dat in some mirrors; wfdb resolves whichever the header
    names, so both are accepted and anything unreadable is counted as a skip."""
    heas = sorted(glob.glob(os.path.join(root, "**", "*.hea"), recursive=True))
    return [h[:-4] for h in heas]


def read_dx(header):
    """SNOMED CT codes from the 'Dx:' comment line of a WFDB header."""
    for c in header.comments:
        if c.strip().lower().startswith("dx:"):
            raw = c.split(":", 1)[1]
            return [d.strip() for d in raw.split(",") if d.strip()]
    return []


def load_snomed_names():
    """Optional: the mapping file shipped with the database, if it is there."""
    for name in ("ConditionNames_SNOMED-CT.csv", "ConditionNames_SNOMED-CT.CSV"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            import csv
            names = {}
            with open(path, newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    code = None
                    label = None
                    for k, v in row.items():
                        kl = (k or "").strip().lower()
                        if "snomed" in kl and "code" in kl or kl == "snomed_ct":
                            code = (v or "").strip()
                        if "full name" in kl or kl in ("name", "condition"):
                            label = (v or "").strip()
                    if code:
                        names[code] = label or ""
            return names
    return {}


def inspect(records, names):
    """Header-only pass. Reads no signals."""
    counter = Counter()
    sets = Counter()
    n_leadII = 0
    lengths = Counter()
    fs_seen = Counter()

    for path in tqdm(records, desc="headers"):
        try:
            h = wfdb.rdheader(path)
        except Exception:
            continue
        dx = read_dx(h)
        counter.update(dx)
        sets[frozenset(dx)] += 1
        if "II" in h.sig_name:
            n_leadII += 1
        lengths[h.sig_len] += 1
        fs_seen[h.fs] += 1

    print(f"\nRecords with a lead named 'II': {n_leadII} of {len(records)}")
    print(f"Sampling rates seen: {dict(fs_seen)}")
    print(f"Record lengths seen (samples): {dict(lengths.most_common(5))}")

    print("\n20 most frequent SNOMED codes")
    print(f"  {'code':<14} {'count':>7}  name")
    for code, n in counter.most_common(20):
        print(f"  {code:<14} {n:>7}  {names.get(code, '')}")

    only_sinus = sets.get(frozenset([SINUS_RHYTHM_SNOMED]), 0)
    with_sinus = sum(n for s, n in sets.items() if SINUS_RHYTHM_SNOMED in s)
    print(f"\nUnder the rule 'diagnosis set is exactly {{{SINUS_RHYTHM_SNOMED}}}':")
    print(f"  normal   {only_sinus}")
    print(f"  abnormal {len(records) - only_sinus}")
    print(f"\nFor comparison, records carrying {SINUS_RHYTHM_SNOMED} among other codes: "
          f"{with_sinus - only_sinus}")
    print("These count as abnormal, exactly as a PTB-XL record carrying NORM "
          "alongside another superclass does.")
    print(f"\nIf {SINUS_RHYTHM_SNOMED} is not the sinus-rhythm code in the table above, "
          "pass the right one with --normal-code and do not run the full pass until it matches.")


def run(records, normal_code):
    X, y_is_normal, record_ids, dx_strings = [], [], [], []
    skipped = Counter()

    for path in tqdm(records, desc="records"):
        try:
            h = wfdb.rdheader(path)
        except Exception:
            skipped["unreadable header"] += 1
            continue

        if "II" not in h.sig_name:
            skipped["no lead II"] += 1
            continue
        ch = h.sig_name.index("II")

        try:
            rec = wfdb.rdrecord(path, channels=[ch])
        except Exception:
            skipped["unreadable signal"] += 1
            continue

        signal = np.asarray(rec.p_signal[:, 0], dtype=np.float64)
        if not np.isfinite(signal).all():
            skipped["non-finite samples"] += 1
            continue

        # Resample to the common rate, then filter at that rate, as for MIT-BIH
        n_new = int(round(len(signal) * TARGET_FS / rec.fs))
        signal = resample(signal, n_new)

        if len(signal) < WINDOW_SAMPLES:
            skipped["shorter than 10 s"] += 1
            continue
        signal = signal[:WINDOW_SAMPLES]          # one 10-second window per record

        signal = bandpass_filter(signal, fs=TARGET_FS)

        std = signal.std()
        if std == 0:
            skipped["flat signal"] += 1
            continue
        signal = (signal - signal.mean()) / std

        dx = read_dx(h)
        if not dx:
            skipped["no Dx field"] += 1
            continue

        X.append(signal.astype(np.float32))
        y_is_normal.append(set(dx) == {normal_code})
        record_ids.append(os.path.basename(path))
        dx_strings.append(";".join(sorted(set(dx))))

    X = np.array(X, dtype=np.float32)
    y_is_normal = np.array(y_is_normal)

    print(f"\nTotal number of windows: {len(X)}")
    print(f"Normal windows: {y_is_normal.sum()}")
    print(f"Abnormal windows: {(~y_is_normal).sum()}")
    print(f"Abnormal prevalence: {(~y_is_normal).mean():.3f}")
    print(f"X shape: {X.shape}")
    if skipped:
        print("\nSkipped:")
        for reason, n in skipped.most_common():
            print(f"  {n:>7}  {reason}")

    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/chapman_X.npy", X)
    np.save(f"{output_dir}/chapman_y_is_normal.npy", y_is_normal)
    np.save(f"{output_dir}/chapman_record_ids.npy", np.array(record_ids))
    # Full diagnosis set per record, so any alternative label rule can be derived
    # later without another 9-minute header pass.
    np.save(f"{output_dir}/chapman_dx.npy", np.array(dx_strings))
    print(f"\nDone. Results saved to the '{output_dir}/' directory.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", action="store_true",
                    help="read headers only and print the label distribution")
    ap.add_argument("--normal-code", default=SINUS_RHYTHM_SNOMED,
                    help="SNOMED CT code that defines a normal record")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N records, for a smoke test")
    args = ap.parse_args()

    root = os.path.join(data_dir, "WFDBRecords")
    if not os.path.isdir(root):
        root = data_dir            # tolerate a flatter unpacking
    records = find_records(root)
    if not records:
        raise SystemExit(
            f"No .hea/.mat pairs found under {root}.\n"
            "Download the database from https://physionet.org/content/ecg-arrhythmia/1.0.0/\n"
            "and unpack it so that data/chapman/WFDBRecords/... exists.")
    print(f"Found {len(records)} records in total.")
    if args.limit:
        records = records[:args.limit]
        print(f"Limited to the first {len(records)}.")

    if args.inspect:
        inspect(records, load_snomed_names())
    else:
        run(records, args.normal_code)
