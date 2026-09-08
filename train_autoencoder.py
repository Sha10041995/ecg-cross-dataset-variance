"""
Main proposal: unsupervised 1D convolutional autoencoder.
Trained on NORMAL (NORM) records only. At test time, anomalies are detected
through the reconstruction error (MSE) - a high error means an abnormal signal.
PTB-XL Lead II is used (to match MIT-BIH MLII).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_recall_fscore_support, confusion_matrix
import time

# --- 1. Load the data ---
print("Loading data...")
X = np.load("processed/ptbxl_X.npy")
y_is_normal = np.load("processed/ptbxl_y_is_normal.npy")
folds = np.load("processed/ptbxl_folds.npy")

LEAD_II_IDX = 1
X_lead2 = X[:, :, LEAD_II_IDX]

train_mask = folds <= 8
val_mask = folds == 9
test_mask = folds == 10

# The autoencoder is trained on normal samples ONLY
train_normal_mask = train_mask & y_is_normal
X_train_normal = X_lead2[train_normal_mask]

X_val, y_val_is_normal = X_lead2[val_mask], y_is_normal[val_mask]
X_test, y_test_is_normal = X_lead2[test_mask], y_is_normal[test_mask]

print(f"Train (normal only): {X_train_normal.shape}")
print(f"Validation (mixed): {X_val.shape}, normal fraction: {y_val_is_normal.mean():.3f}")
print(f"Test (mixed): {X_test.shape}, normal fraction: {y_test_is_normal.mean():.3f}")

# --- 2. Dataset ---
class ECGDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X).unsqueeze(1).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx]

train_ds = ECGDataset(X_train_normal)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# --- 3. Convolutional Autoencoder ---
class ECG_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 1000 -> 500 -> 250 -> 125
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        # Decoder: 125 -> 250 -> 500 -> 1000
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv1d(32, 16, kernel_size=5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Upsample(scale_factor=2), nn.Conv1d(16, 1, kernel_size=7, padding=3),
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

device = torch.device("cpu")
model = ECG_Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 4. Training ---
N_EPOCHS = 20
print(f"\nAutoencoder training started ({N_EPOCHS} epochs, CPU)...")

for epoch in range(N_EPOCHS):
    t0 = time.time()
    model.train()
    train_loss = 0
    for xb in train_loader:
        optimizer.zero_grad()
        recon = model(xb)
        loss = criterion(recon, xb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_ds)
    print(f"Epoch {epoch+1}/{N_EPOCHS} | train_recon_loss={train_loss:.5f} | {time.time()-t0:.1f}s")

# --- 5. Function for computing the reconstruction error ---
def compute_recon_errors(X_data):
    ds = ECGDataset(X_data)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    errors = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            recon = model(xb)
            err = ((recon - xb) ** 2).mean(dim=[1, 2])
            errors.extend(err.numpy())
    return np.array(errors)

print("\nComputing reconstruction errors on the validation set...")
val_errors = compute_recon_errors(X_val)
val_y_anomal = (~y_val_is_normal).astype(int)

# --- 6. Find the optimal threshold (via Youden's J statistic) ---
fpr, tpr, thresholds = roc_curve(val_y_anomal, val_errors)
val_auc = roc_auc_score(val_y_anomal, val_errors)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print(f"Validation AUC: {val_auc:.4f}")
print(f"Selected threshold: {best_threshold:.6f}")

# --- 7. Final evaluation on the test set ---
print("\n--- Final result on the test set ---")
test_errors = compute_recon_errors(X_test)
test_y_anomal = (~y_test_is_normal).astype(int)

test_auc = roc_auc_score(test_y_anomal, test_errors)
test_pred = (test_errors >= best_threshold).astype(int)

acc = accuracy_score(test_y_anomal, test_pred)
prec, rec, f1, _ = precision_recall_fscore_support(test_y_anomal, test_pred, average='binary')
cm = confusion_matrix(test_y_anomal, test_pred)

print(f"AUC:       {test_auc:.4f}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Confusion matrix:\n{cm}")

# --- 8. Save ---
torch.save(model.state_dict(), "autoencoder.pt")
np.save("processed/ae_threshold.npy", np.array([best_threshold]))
print("\nModel saved as 'autoencoder.pt' and the threshold as 'processed/ae_threshold.npy'.")