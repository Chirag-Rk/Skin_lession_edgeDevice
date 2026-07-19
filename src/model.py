import torch
import torch.nn as nn

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class MobileNetAttentionModel(nn.Module):
    """
    MobileNetV3-Large backbone + channel-wise attention + classifier head.

    Args:
        num_classes (int): Number of output classes. Default: 7
        pretrained (bool): Load ImageNet pretrained weights for backbone.
                           Set False to skip HuggingFace download. Default: True
        use_attention (bool): Toggle attention block. Default: True
    """

    def __init__(self, num_classes=7, pretrained=True, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.feat_dim = 1280

        # ── Backbone ─────────────────────────────────────────────────────────
        if TIMM_AVAILABLE:
            try:
                self.backbone = timm.create_model(
                    "mobilenetv3_large_100",
                    pretrained=pretrained,
                    num_classes=0          # remove timm head
                )
            except Exception as e:
                print(f"[Model] Warning: pretrained load failed ({e}). Using random weights.")
                self.backbone = timm.create_model(
                    "mobilenetv3_large_100",
                    pretrained=False,
                    num_classes=0
                )
        else:
            raise ImportError("timm is required. Run: pip install timm")

        # ── Attention ─────────────────────────────────────────────────────────
        if self.use_attention:
            self.attn = AttentionBlock(self.feat_dim)

        # ── Classifier ───────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)              # (B, 1280)

        if self.use_attention:
            features = self.attn(features)       # channel-wise weighting

        return self.classifier(features)         # (B, num_classes)