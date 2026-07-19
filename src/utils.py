import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


# ======================
# MIXUP
# ======================
def mixup_data(x, y, alpha=0.4):
    """
    Applies Mixup augmentation
    Returns:
        mixed_x, y_a, y_b, lambda
    """

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)

    # Shuffle indices
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss calculation
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ======================
# OPTIONAL: TOGGLE MIXUP
# ======================
def apply_mixup(x, y, criterion, model, use_mixup=True, alpha=0.4):
    """
    Wrapper to optionally apply mixup
    """

    if use_mixup:
        x, y_a, y_b, lam = mixup_data(x, y, alpha)
        preds = model(x)
        loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
    else:
        preds = model(x)
        loss = criterion(preds, y)

    return preds, loss


# ======================
# METRICS
# ======================
def compute_metrics(all_labels, all_probs):
    """
    Computes Accuracy, F1, AUC
    """

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    num_classes = all_probs.shape[1]
    preds = all_probs.argmax(axis=1)

    acc = (preds == all_labels).mean()
    f1 = f1_score(all_labels, preds, average="macro", labels=np.arange(num_classes))

    # Handle edge case (single class)
    try:
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", labels=np.arange(num_classes))
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    return acc, auc, f1