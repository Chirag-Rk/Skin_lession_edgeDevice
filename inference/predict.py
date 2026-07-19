# inference/predict.py - Inference engine for Jetson Nano skin cancer classification

import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import MobileNetAttentionModel
from src.config import CFG, DEVICE
from src.dataset import IMAGENET_MEAN, IMAGENET_STD

LABEL_NAMES = {
    0: "Melanocytic nevi (nv)",
    1: "Melanoma (mel)",
    2: "Benign keratosis-like lesions (bkl)",
    3: "Basal cell carcinoma (bcc)",
    4: "Actinic keratoses and intraepithelial carcinoma (akiec)",
    5: "Dermatofibroma (df)",
    6: "Vascular lesions (vasc)",
}

# Short codes for labeling
LABEL_CODES = {
    0: "nv",
    1: "mel",
    2: "bkl",
    3: "bcc",
    4: "akiec",
    5: "df",
    6: "vasc"
}

class SkinClassifier:
    """
    Manages loading the MobileNetV3 model checkpoint and performing inference.
    """
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device if device else DEVICE
        self.model = MobileNetAttentionModel(
            num_classes=CFG["num_classes"],
            pretrained=False, # Do not try to download weights from HF at run-time
            use_attention=True
        ).to(self.device)
        
        # Determine checkpoint path
        if not checkpoint_path:
            # Look for common paths
            possible_paths = [
                "checkpoints/centralized_best.pt",
                "checkpoints/best_model.pt",
                os.path.join(CFG["checkpoint_dir"], "centralized_best.pt"),
                os.path.join(CFG["checkpoint_dir"], "best_model.pt"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                print(f"[Inference] Loaded model weights from {checkpoint_path} onto {self.device}")
                self.loaded = True
            except Exception as e:
                print(f"[Inference] Error loading weights from {checkpoint_path}: {e}")
                self.loaded = False
        else:
            print(f"[Inference] Warning: No checkpoint found at {checkpoint_path or 'default paths'}. Using random weights.")
            self.loaded = False
            
        self.model.eval()
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((CFG["image_size"], CFG["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def predict(self, image_pil):
        """
        Classifies an input PIL Image.
        Returns:
            dict containing:
                - class_idx (int)
                - label_code (str)
                - label_name (str)
                - confidence (float)
                - probabilities (dict of label_name: float)
                - latency_ms (float)
        """
        start_time = time.time()
        
        # Preprocess
        img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1).squeeze(0)
            
        latency = (time.time() - start_time) * 1000.0
        
        # Postprocess
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        
        # Probabilities dictionary
        probs_dict = {
            LABEL_NAMES[i]: float(probs[i].item())
            for i in range(len(probs))
        }
        
        return {
            "class_idx": pred_idx,
            "label_code": LABEL_CODES[pred_idx],
            "label_name": LABEL_NAMES[pred_idx],
            "confidence": confidence,
            "probabilities": probs_dict,
            "latency_ms": latency
        }
