from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
from src.config import CFG

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            img = Image.open(row["path"]).convert("RGB")
        except Exception as e:
            # Handle corrupted images safely
            img = Image.new("RGB", (CFG["image_size"], CFG["image_size"]))

        label = int(row["label"])

        if self.transform:
            img = self.transform(img)

        return img, label


# ImageNet normalization (IMPORTANT for timm models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_tf(mode):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((CFG["image_size"], CFG["image_size"])),

            # Augmentations (safe for medical images)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),  # slightly reduced

            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03
            ),

            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),   # reduced for realism
                scale=(0.95, 1.05)
            ),

            transforms.ToTensor(),

            # VERY IMPORTANT
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    else:
        return transforms.Compose([
            transforms.Resize((CFG["image_size"], CFG["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])