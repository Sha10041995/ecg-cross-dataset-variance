"""
Preprocessing of the MIT-BIH data, for comparison with PTB-XL:
1. Take the MLII channel (comparable to PTB-XL Lead II)
2. Resample 360 Hz -> 100 Hz
3. Bandpass filter + z-score normalization (identical to PTB-XL)
4. Split into 10-second (1000-sample) windows
5. Label per window: normal if all beats are 'N', abnormal otherwise
"""
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, resample
import os
from tqdm import tqdm

data_dir = "data/mitbih"
output_dir = "processed"
os.makedirs(output_dir, exist_ok=True)

ORIGINAL_FS = 360
TARGET_FS = 100
WINDOW_SEC = 10
WINDOW_SAMPLES = TARGET_FS * WINDOW_SEC  # 1000, identical to PTB-XL

# Collect all record names
records = sorted(set(f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.dat')))
print(f"Found {len(records)} records in total.")

def bandpass_filter(signal, fs=100, lowcut=0.5, highcut=40, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

X_windows = []
y_is_normal = []
record_ids = []

for rec_name in tqdm(records):
    rec_path = os.path.join(data_dir, rec_name)
    record = wfdb.rdrecord(rec_path)
    annotation = wfdb.rdann(rec_path, 'atr')

    # Locate the MLII channel
    if 'MLII' not in record.sig_name:
        continue  # some records have no MLII, so they are skipped
    ch_idx = record.sig_name.index('MLII')
    signal = record.p_signal[:, ch_idx]

    # Resample 360 Hz -> 100 Hz
    n_samples_new = int(len(signal) * TARGET_FS / ORIGINAL_FS)
    signal_resampled = resample(signal, n_samples_new)

    # Filtering
    signal_filtered = bandpass_filter(signal_resampled, fs=TARGET_FS)

    # The annotations are rescaled to the new sampling rate as well
    ann_samples_resampled = (np.array(annotation.sample) * TARGET_FS / ORIGINAL_FS).astype(int)
    ann_symbols = np.array(annotation.symbol)

    # Keep only beat symbols (not rhythm/signal-quality symbols such as +, ~, |, etc.)
    beat_symbols = set('NLRBAaJSVrFejnE/fQ?')
    valid_mask = np.array([s in beat_symbols for s in ann_symbols])
    ann_samples_resampled = ann_samples_resampled[valid_mask]
    ann_symbols = ann_symbols[valid_mask]

    # Split into non-overlapping 10-second windows
    n_windows = len(signal_filtered) // WINDOW_SAMPLES
    for w in range(n_windows):
        start = w * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES
        window = signal_filtered[start:end]

        # Normalization (z-score)
        mean, std = window.mean(), window.std()
        if std == 0:
            continue
        window_norm = (window - mean) / std

        # Find the beats that fall inside this window
        in_window = (ann_samples_resampled >= start) & (ann_samples_resampled < end)
        symbols_in_window = ann_symbols[in_window]

        if len(symbols_in_window) == 0:
            continue  # skip windows that contain no beats

        is_normal = np.all(symbols_in_window == 'N')

        X_windows.append(window_norm.astype(np.float32))
        y_is_normal.append(is_normal)
        record_ids.append(rec_name)

X_windows = np.array(X_windows, dtype=np.float32)
y_is_normal = np.array(y_is_normal)

print(f"\nTotal number of windows: {len(X_windows)}")
print(f"Normal windows: {y_is_normal.sum()}")
print(f"Abnormal windows: {(~y_is_normal).sum()}")
print(f"X shape: {X_windows.shape}")

np.save(f"{output_dir}/mitbih_X.npy", X_windows)
np.save(f"{output_dir}/mitbih_y_is_normal.npy", y_is_normal)
np.save(f"{output_dir}/mitbih_record_ids.npy", np.array(record_ids))

print(f"\nDone. Results saved to the '{output_dir}/' directory.")