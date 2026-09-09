"""
Baseline: supervised 1D CNN classifier.
Binary normal/abnormal classification from the PTB-XL Lead II signal.
The official strat_fold split (1-8 train, 9 val, 10 test) is used.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import time

# --- 1. Load the data ---
print("Loading data...")
X = np.load("processed/ptbxl_X.npy")          # (21799, 1000, 12)
y_is_normal = np.load("processed/ptbxl_y_is_normal.npy")
folds = np.load("processed/ptbxl_folds.npy")

LEAD_II_IDX = 1  # ['I','II','III','AVR','AVL','AVF','V1'...] -> II = index 1
X_lead2 = X[:, :, LEAD_II_IDX]  # (21799, 1000)

# Label: 1 = abnormal, 0 = normal (the classifier takes the anomaly as the positive class)
y = (~y_is_normal).astype(np.float32)

train_mask = folds <= 8
val_mask = folds == 9
test_mask = folds == 10

X_train, y_train = X_lead2[train_mask], y[train_mask]
X_val, y_val = X_lead2[val_mask], y[val_mask]
X_test, y_test = X_lead2[test_mask], y[test_mask]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train abnormal fraction: {y_train.mean():.3f}")

# --- 2. Dataset and DataLoader ---
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).unsqueeze(1).float()  # (N, 1, 1000)
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = ECGDataset(X_train, y_train)
val_ds = ECGDataset(X_val, y_val)
test_ds = ECGDataset(X_test, y_test)

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# --- 3. 1D CNN model ---
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

device = torch.device("cpu")
model = ECG_CNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 4. Training loop ---
N_EPOCHS = 15
print(f"\nTraining started ({N_EPOCHS} epochs, CPU)...")

for epoch in range(N_EPOCHS):
    t0 = time.time()
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0
    val_preds, val_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * len(xb)
            val_preds.extend(torch.sigmoid(logits).numpy())
            val_true.extend(yb.numpy())
    val_loss /= len(val_ds)
    val_auc = roc_auc_score(val_true, val_preds)

    print(f"Epoch {epoch+1}/{N_EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_AUC={val_auc:.4f} | {time.time()-t0:.1f}s")

# --- 5. Final evaluation on the test set ---
print("\n--- Final result on the test set ---")
model.eval()
test_preds, test_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        test_preds.extend(torch.sigmoid(logits).numpy())
        test_true.extend(yb.numpy())

test_preds = np.array(test_preds)
test_true = np.array(test_true)
test_pred_labels = (test_preds >= 0.5).astype(int)

acc = accuracy_score(test_true, test_pred_labels)
prec, rec, f1, _ = precision_recall_fscore_support(test_true, test_pred_labels, average='binary')
auc = roc_auc_score(test_true, test_preds)
cm = confusion_matrix(test_true, test_pred_labels)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print(f"Confusion matrix:\n{cm}")

# --- 6. Save the model ---
torch.save(model.state_dict(), "baseline_cnn.pt")
print("\nModel saved as 'baseline_cnn.pt'.")