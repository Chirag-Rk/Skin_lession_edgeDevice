import torch
from src.config import CFG, DEVICE
from src.utils import apply_mixup


def train_one_epoch(model, loader, criterion, optimizer):
    """Train model for one epoch. Returns average loss."""
    model.train()
    total_loss, total = 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        # Mixup augmentation
        preds, loss = apply_mixup(
            imgs,
            labels,
            criterion,
            model,
            use_mixup=True,
            alpha=0.4
        )

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CFG["grad_clip"]
        )

        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

    return total_loss / total