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
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class MobileNetAttentionModel(nn.Module):
    """
    MobileNetV3-Small + Attention Model
    Optimized for NVIDIA Jetson Nano (4GB)

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Load ImageNet pretrained weights.
        use_attention (bool): Enable/Disable Attention Block.
    """

    def __init__(
        self,
        num_classes=7,
        pretrained=True,
        use_attention=True
    ):
        super().__init__()

        self.use_attention = use_attention

        # ------------------------------------------------------------
        # Backbone
        # ------------------------------------------------------------
        if not TIMM_AVAILABLE:
            raise ImportError("Please install timm:\npip install timm")

        try:
            self.backbone = timm.create_model(
                "mobilenetv3_small_100",
                pretrained=pretrained,
                num_classes=0
            )
        except Exception as e:
            print(f"[Warning] Pretrained weights could not be loaded: {e}")
            print("[Info] Using randomly initialized weights.")

            self.backbone = timm.create_model(
                "mobilenetv3_small_100",
                pretrained=False,
                num_classes=0
            )

        # Automatically detect output feature dimension
        self.feat_dim = self.backbone.num_features

        # ------------------------------------------------------------
        # Attention
        # ------------------------------------------------------------
        if self.use_attention:
            self.attn = AttentionBlock(self.feat_dim)

        # ------------------------------------------------------------
        # Lightweight Classifier
        # ------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        # Feature extraction
        features = self.backbone(x)

        # Attention
        if self.use_attention:
            features = self.attn(features)

        # Classification
        output = self.classifier(features)

        return output


if __name__ == "__main__":

    model = MobileNetAttentionModel()

    x = torch.randn(1, 3, 128, 128)

    y = model(x)

    print(model)

    print("\nOutput Shape:", y.shape)

    print("Feature Dimension:", model.feat_dim)