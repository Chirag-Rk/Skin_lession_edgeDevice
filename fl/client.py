# fl/client.py  — used in simulation mode

import flwr as fl
import torch
import torch.nn as nn
import numpy as np

from src.train import train_one_epoch
from src.validate import validate
from src.model import MobileNetAttentionModel   # ✅ correct model name
from src.config import CFG, DEVICE


class SkinClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, client_id=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.client_id = client_id

    # ── Send model weights to server ──────────────────────────────────────
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    # ── Receive global weights from server ────────────────────────────────
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    # ── Local training (1 epoch per round) ───────────────────────────────
    def fit(self, parameters, config):
        self.set_parameters(parameters)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=CFG["lr"])

        train_loss = train_one_epoch(
            self.model,
            self.train_loader,
            self.criterion,
            optimizer,
        )

        tag = f"[Client {self.client_id}]" if self.client_id is not None else "[Client]"
        print(f"{tag} Train Loss: {train_loss:.4f}")

        return self.get_parameters(config), len(self.train_loader.dataset), {
            "train_loss": float(train_loss)
        }

    # ── Validation ────────────────────────────────────────────────────────
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        acc, auc, f1, val_loss = validate(self.model, self.val_loader, self.criterion)

        tag = f"[Client {self.client_id}]" if self.client_id is not None else "[Client]"
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
        print(f"{tag} Val Acc: {acc:.4f}  AUC: {auc_str}  F1: {f1:.4f}  Loss: {val_loss:.4f}")

        safe_auc = float(auc) if not np.isnan(auc) else 0.0

        return float(val_loss), len(self.val_loader.dataset), {
            "accuracy": float(acc),
            "auc": safe_auc,
            "f1": float(f1),
        }