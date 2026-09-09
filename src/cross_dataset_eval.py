"""
Cross-dataset generalization test.
Both models trained on PTB-XL (the baseline CNN and the autoencoder) are
evaluated on MIT-BIH without retraining (an entirely new dataset, different
equipment and patients). This shows how well the models generalize in practice.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- 1. Load the MIT-BIH data ---
print("Loading MIT-BIH data...")
X_mit = np.load("processed/mitbih_X.npy")
y_mit_is_normal = np.load("processed/mitbih_y_is_normal.npy")
y_mit_anomal = (~y_mit_is_normal).astype(int)

print(f"MIT-BIH shape: {X_mit.shape}, normal fraction: {y_mit_is_normal.mean():.3f}")

class ECGDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X).unsqueeze(1).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx]

loader = DataLoader(ECGDataset(X_mit), batch_size=128, shuffle=False)

# --- 2. Model architectures (identical to the previous scripts) ---
class ECG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x = self.net(x)
        return self.fc(x).squeeze(-1)

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

# --- 3. Load the models ---
baseline = ECG_CNN()
baseline.load_state_dict(torch.load("baseline_cnn.pt", map_location="cpu"))
baseline.eval()

autoencoder = ECG_Autoencoder()
autoencoder.load_state_dict(torch.load("autoencoder.pt", map_location="cpu"))
autoencoder.eval()

ae_threshold = np.load("processed/ae_threshold.npy")[0]
print(f"Autoencoder threshold (from PTB-XL validation): {ae_threshold:.6f}")

# --- 4. Evaluate the baseline CNN on MIT-BIH ---
print("\n=== Baseline CNN -> MIT-BIH (cross-dataset) ===")
cnn_preds = []
with torch.no_grad():
    for xb in loader:
        logits = baseline(xb)
        cnn_preds.extend(torch.sigmoid(logits).numpy())
cnn_preds = np.array(cnn_preds)
cnn_pred_labels = (cnn_preds >= 0.5).astype(int)

cnn_auc = roc_auc_score(y_mit_anomal, cnn_preds)
cnn_acc = accuracy_score(y_mit_anomal, cnn_pred_labels)
cnn_prec, cnn_rec, cnn_f1, _ = precision_recall_fscore_support(y_mit_anomal, cnn_pred_labels, average='binary')
cnn_cm = confusion_matrix(y_mit_anomal, cnn_pred_labels)

print(f"AUC:       {cnn_auc:.4f}")
print(f"Accuracy:  {cnn_acc:.4f}")
print(f"Precision: {cnn_prec:.4f}")
print(f"Recall:    {cnn_rec:.4f}")
print(f"F1-score:  {cnn_f1:.4f}")
print(f"Confusion matrix:\n{cnn_cm}")

# --- 5. Evaluate the autoencoder on MIT-BIH ---
print("\n=== Autoencoder -> MIT-BIH (cross-dataset) ===")
ae_errors = []
with torch.no_grad():
    for xb in loader:
        recon = autoencoder(xb)
        err = ((recon - xb) ** 2).mean(dim=[1, 2])
        ae_errors.extend(err.numpy())
ae_errors = np.array(ae_errors)
ae_pred_labels = (ae_errors >= ae_threshold).astype(int)

ae_auc = roc_auc_score(y_mit_anomal, ae_errors)
ae_acc = accuracy_score(y_mit_anomal, ae_pred_labels)
ae_prec, ae_rec, ae_f1, _ = precision_recall_fscore_support(y_mit_anomal, ae_pred_labels, average='binary')
ae_cm = confusion_matrix(y_mit_anomal, ae_pred_labels)

print(f"AUC:       {ae_auc:.4f}")
print(f"Accuracy:  {ae_acc:.4f}")
print(f"Precision: {ae_prec:.4f}")
print(f"Recall:    {ae_rec:.4f}")
print(f"F1-score:  {ae_f1:.4f}")
print(f"Confusion matrix:\n{ae_cm}")

# --- 6. Summary table ---
print("\n=== SUMMARY: PTB-XL (in-domain) vs MIT-BIH (cross-dataset) ===")
print(f"{'Model':<20} {'PTB-XL AUC':<12} {'MIT-BIH AUC':<12} {'Diff':<10}")
print(f"{'Baseline CNN':<20} {'0.8869':<12} {cnn_auc:<12.4f} {0.8869-cnn_auc:<10.4f}")
print(f"{'Autoencoder':<20} {'0.7566':<12} {ae_auc:<12.4f} {0.7566-ae_auc:<10.4f}")