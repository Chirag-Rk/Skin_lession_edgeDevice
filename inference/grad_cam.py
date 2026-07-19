# inference/grad_cam.py - Grad-CAM processor for single image explanation

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.cm as cm

from src.config import CFG
from src.dataset import IMAGENET_MEAN, IMAGENET_STD

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for MobileNetAttentionModel.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # If target_layer is not specified, try to find the last conv layer from the backbone
        if target_layer is None:
            target_layer = self._get_target_layer(model)
            
        self.target_layer = target_layer
        self._register_hooks()

    def _get_target_layer(self, model):
        """Get the last conv layer from MobileNetV3 backbone."""
        try:
            # timm mobilenetv3_large_100 structure: backbone.blocks[-1][-1].conv
            return model.backbone.blocks[-1][-1].conv
        except Exception:
            try:
                # Fallback: find the last convolution block
                conv_layers = []
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        conv_layers.append(module)
                if conv_layers:
                    return conv_layers[-1]
            except Exception:
                pass
            # Fallback to last child of backbone
            return list(model.backbone.children())[-2]

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_pil, class_idx, device):
        """
        Generates Grad-CAM heatmap and blended overlay.
        """
        self.model.eval()
        
        # Preprocess
        preprocess = transforms.Compose([
            transforms.Resize((CFG["image_size"], CFG["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        input_tensor = preprocess(image_pil).unsqueeze(0).to(device)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # GAP over spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
            
        # Interpolate to input size
        cam = F.interpolate(
            cam,
            size=(CFG["image_size"], CFG["image_size"]),
            mode="bilinear",
            align_corners=False,
        )
        
        cam_np = cam.squeeze().cpu().numpy()
        
        # Format outputs as PIL Images
        img_resized = image_pil.resize((CFG["image_size"], CFG["image_size"]))
        img_np = np.array(img_resized) / 255.0
        
        # Apply colormap (jet) to CAM
        heatmap_rgba = cm.jet(cam_np)  # shape (H, W, 4)
        heatmap_rgb = heatmap_rgba[:, :, :3]  # shape (H, W, 3)
        
        # Blend overlay (0.5 original, 0.5 heatmap)
        overlay_np = 0.5 * img_np + 0.5 * heatmap_rgb
        overlay_np = np.clip(overlay_np, 0.0, 1.0)
        
        # Convert back to PIL images
        heatmap_img = Image.fromarray((heatmap_rgb * 255).astype(np.uint8))
        overlay_img = Image.fromarray((overlay_np * 255).astype(np.uint8))
        
        return img_resized, heatmap_img, overlay_img
