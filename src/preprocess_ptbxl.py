"""
Preprocessing of the PTB-XL data:
1. Load all signals (the 100 Hz version, for faster processing)
2. Bandpass filtering (0.5-40 Hz) - removes baseline wander and high-frequency noise
3. Per-lead normalization (z-score)
4. Prepare the train/validation/test split (based on strat_fold)
5. Save the results to .npy files (for fast loading in later stages)
"""
import pandas as pd
import numpy as np
import wfdb
import ast
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import os

data_dir = "data/ptbxl"
output_dir = "processed"
os.makedirs(output_dir, exist_ok=True)

# --- 1. Load metadata ---
print("Loading metadata...")
df = pd.read_csv(f"{data_dir}/ptbxl_database.csv", index_col='ecg_id')
df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

scp_df = pd.read_csv(f"{data_dir}/scp_statements.csv", index_col=0)
scp_df = scp_df[scp_df.diagnostic == 1]

def aggregate_diagnostic(scp_codes):
    classes = []
    for code in scp_codes.keys():
        if code in scp_df.index:
            classes.append(scp_df.loc[code].diagnostic_class)
    return list(set(classes))

df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
df['is_normal'] = df['diagnostic_superclass'].apply(lambda x: x == ['NORM'])

# --- 2. Bandpass filter function ---
def bandpass_filter(signal, fs=100, lowcut=0.5, highcut=40, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)

# --- 3. Load and preprocess all signals ---
def load_and_process(row):
    path = f"{data_dir}/{row['filename_lr']}"
    record = wfdb.rdrecord(path)
    signal = record.p_signal  # shape: (1000, 12)

    # Filtering
    signal = bandpass_filter(signal, fs=100)

    # Per-lead z-score normalization
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std[std == 0] = 1e-8  # guard against division by zero
    signal = (signal - mean) / std

    return signal.astype(np.float32)

print(f"\nLoading and preprocessing {len(df)} signals...")
print("This may take several minutes...")

X = np.zeros((len(df), 1000, 12), dtype=np.float32)
for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
    X[i] = load_and_process(row)

y_is_normal = df['is_normal'].values
folds = df['strat_fold'].values

# --- 4. Train/val/test split (following the official recommendation) ---
train_mask = folds <= 8
val_mask = folds == 9
test_mask = folds == 10

print(f"\nTrain: {train_mask.sum()} records")
print(f"Validation: {val_mask.sum()} records")
print(f"Test: {test_mask.sum()} records")

# --- 5. Save ---
np.save(f"{output_dir}/ptbxl_X.npy", X)
np.save(f"{output_dir}/ptbxl_y_is_normal.npy", y_is_normal)
np.save(f"{output_dir}/ptbxl_folds.npy", folds)

print(f"\nDone. Results saved to the '{output_dir}/' directory:")
print(f"  ptbxl_X.npy - shape: {X.shape}")
print(f"  ptbxl_y_is_normal.npy - shape: {y_is_normal.shape}")
print(f"  ptbxl_folds.npy - shape: {folds.shape}")