# gradcam.py  — Step 9: Grad-CAM visualization for model interpretability

import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.model import MobileNetAttentionModel
from src.config import CFG, DEVICE, df
from src.dataset import IMAGENET_MEAN, IMAGENET_STD

os.makedirs("plots/gradcam", exist_ok=True)

# ── Label names ────────────────────────────────────────────────────────────────
LABEL_NAMES = {
    0: "Melanocytic nevi (nv)",
    1: "Melanoma (mel)",
    2: "Basal cell carcinoma (bkl)",
    3: "Basal cell carcinoma (bcc)",
    4: "Actinic keratoses (akiec)",
    5: "Dermatofibroma (df)",
    6: "Vascular lesions (vasc)",
}


# =============================================================================
# GRAD-CAM IMPLEMENTATION
# =============================================================================
class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model, target_layer):
        self.model       = model
        self.target_layer = target_layer
        self.gradients   = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # GAP over spatial dimensions
        weights      = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam          = (weights * self.activations).sum(dim=1, keepdim=True)
        cam          = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = F.interpolate(
            cam,
            size=(CFG["image_size"], CFG["image_size"]),
            mode="bilinear",
            align_corners=False,
        )
        return cam.squeeze().cpu().numpy(), class_idx, output.softmax(dim=1).squeeze().detach().cpu().numpy()


def get_target_layer(model):
    """Get the last conv layer from MobileNetV3 backbone."""
    # MobileNetV3 last conv block
    try:
        return model.backbone.blocks[-1][-1].conv
    except Exception:
        # Fallback to last Sequential block
        return list(model.backbone.children())[-2]


# =============================================================================
# VISUALIZE
# =============================================================================

preprocess = transforms.Compose([
    transforms.Resize((CFG["image_size"], CFG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def denorm(tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def visualize_gradcam(model, image_paths, true_labels, n=6, save_path="plots/gradcam/gradcam_grid.png"):
    target_layer = get_target_layer(model)
    gradcam      = GradCAM(model, target_layer)

    n = min(n, len(image_paths))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle("Grad-CAM Visualization — Skin Lesion Classification",
                 fontsize=14, fontweight="bold")

    for i in range(n):
        path      = image_paths[i]
        true_lbl  = true_labels[i]

        img_pil   = Image.open(path).convert("RGB")
        img_orig  = np.array(img_pil.resize((CFG["image_size"], CFG["image_size"]))) / 255.0

        inp       = preprocess(img_pil).unsqueeze(0).to(DEVICE)
        cam, pred_idx, probs = gradcam.generate(inp)

        # Overlay heatmap
        heatmap   = cm.jet(cam)[:, :, :3]
        overlay   = 0.5 * img_orig + 0.5 * heatmap
        overlay   = np.clip(overlay, 0, 1)

        # Plot
        axes[i, 0].imshow(img_orig)
        axes[i, 0].set_title(f"Original\nTrue: {LABEL_NAMES[true_lbl]}", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cam, cmap="jet")
        axes[i, 1].set_title(f"Grad-CAM Heatmap\nPred: {LABEL_NAMES[pred_idx]}", fontsize=8)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Overlay  (conf: {probs[pred_idx]:.2%})", fontsize=8)
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Grad-CAM saved to {save_path}")
    plt.show()  # Display heatmaps directly on screen
    return save_path


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/centralized_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--n",          type=int, default=6,  help="Number of images")
    parser.add_argument("--seed",       type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Load model
    model = MobileNetAttentionModel(num_classes=CFG["num_classes"]).to(DEVICE)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        print(f"✅ Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"⚠️  No checkpoint found at {args.checkpoint}. Using random weights.")

    # Sample images (one per class if possible)
    random.seed(args.seed)
    samples = []
    for cls_id in range(CFG["num_classes"]):
        cls_rows = df[df["label"] == cls_id]
        if len(cls_rows) > 0:
            row = cls_rows.sample(1, random_state=args.seed).iloc[0]
            samples.append((row["path"], cls_id))

    # Fill remaining slots randomly
    while len(samples) < args.n:
        row = df.sample(1, random_state=random.randint(0, 9999)).iloc[0]
        samples.append((row["path"], int(row["label"])))

    samples   = samples[:args.n]
    paths     = [s[0] for s in samples]
    labels    = [s[1] for s in samples]

    visualize_gradcam(model, paths, labels, n=args.n)
