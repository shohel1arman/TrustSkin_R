"""HAM10000 dataset loading from prepared split CSVs.

Augmentation follows the project outline: flips, mild rotation, scale/crop and
conservative color jitter for training only. Normalization statistics are
computed from the training split (fallback: ImageNet statistics, which the
pretrained backbones were trained with).
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class HAM10000Dataset(Dataset):
    def __init__(self, split_csv: str, image_size: int = 224, train: bool = False, transform=None):
        self.df = pd.read_csv(split_csv)
        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"{split_csv} must contain 'path' and 'label' columns (run prepare_splits first)")
        self.transform = transform or build_transforms(image_size, train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        x = self.transform(img)
        y = int(row["label"])
        return x, y, row["image_id"]


def class_counts(split_csv: str, num_classes: int = 7) -> torch.Tensor:
    df = pd.read_csv(split_csv)
    counts = torch.zeros(num_classes, dtype=torch.float)
    vc = df["label"].value_counts()
    for k, v in vc.items():
        counts[int(k)] = float(v)
    return counts
