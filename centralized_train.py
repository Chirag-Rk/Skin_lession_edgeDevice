# centralized_train.py  — Step 4: Train on full dataset (baseline)

import os
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

from src.config import CFG, DEVICE, df, train_df, val_df
from src.dataset import SkinDataset, get_tf
from src.model import MobileNetAttentionModel
from src.train import train_one_epoch
from src.validate import validate

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS      = int(os.environ.get("EPOCHS", 5))
SAVE_CKPT   = True
CKPT_BEST   = "checkpoints/centralized_best.pt"
LOG_PATH    = "logs/centralized_metrics.json"

# ── Data ──────────────────────────────────────────────────────────────────────
train_loader = DataLoader(
    SkinDataset(train_df, get_tf("train")),
    batch_size=CFG["batch_size"],
    shuffle=True,
)
val_loader = DataLoader(
    SkinDataset(val_df, get_tf("val")),
    batch_size=CFG["batch_size"],
    shuffle=False,
)

# ── Model ─────────────────────────────────────────────────────────────────────
model     = MobileNetAttentionModel(num_classes=CFG["num_classes"]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training Loop ─────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  CENTRALIZED TRAINING — SKIN LESION CLASSIFICATION")
print(f"  Epochs: {EPOCHS}   Device: {DEVICE}")
print(f"  Train: {len(train_df)}   Val: {len(val_df)}")
print("═" * 60 + "\n")

epoch_logs = []
best_auc   = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    acc, auc, f1, val_loss = validate(model, val_loader, criterion)
    scheduler.step()

    auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
    print(f"Epoch {epoch:02d}/{EPOCHS}  "
          f"TrainLoss: {train_loss:.4f}  "
          f"ValLoss: {val_loss:.4f}  "
          f"Acc: {acc:.4f}  AUC: {auc_str}  F1: {f1:.4f}")

    entry = {
        "epoch":      epoch,
        "train_loss": round(float(train_loss), 4),
        "val_loss":   round(float(val_loss),   4),
        "accuracy":   round(float(acc),         4),
        "auc":        round(float(auc) if not np.isnan(auc) else 0.0, 4),
        "f1":         round(float(f1),           4),
    }
    epoch_logs.append(entry)

    # Save best model
    if not np.isnan(auc) and auc > best_auc and SAVE_CKPT:
        best_auc = auc
        torch.save(model.state_dict(), CKPT_BEST)
        print(f"  ✅ Best model saved (AUC: {best_auc:.4f})")

# ── Save logs ──────────────────────────────────────────────────────────────────
with open(LOG_PATH, "w") as f:
    json.dump({
        "mode":      "centralized",
        "timestamp": datetime.now().isoformat(),
        "epochs":    epoch_logs,
        "best_auc":  round(best_auc, 4),
    }, f, indent=2)

print(f"\n✅ Centralized training complete.")
print(f"   Best AUC : {best_auc:.4f}")
print(f"   Logs     : {LOG_PATH}")
print(f"   Checkpoint: {CKPT_BEST}")
