import torch
import torch.nn.functional as F
import numpy as np
from src.config import DEVICE
from src.utils import compute_metrics


@torch.no_grad()
def validate(model, loader, criterion=None):
    """
    Validate model on a data loader.
    Returns: (accuracy, auc, f1, avg_loss)
    """
    model.eval()

    all_labels = []
    all_probs = []
    total_loss = 0.0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        logits = model(imgs)

        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        probs = F.softmax(logits, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    acc, auc, f1 = compute_metrics(all_labels, all_probs)

    avg_loss = total_loss / total if total > 0 else 0.0

    return acc, auc, f1, avg_loss