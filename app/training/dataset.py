"""
Triplet dataset for fin re-identification training.

Expected directory layout:

    data/
      individual_A/
        photo1.jpg
        photo2.jpg
      individual_B/
        photo1.jpg
      ...

Each subdirectory name is treated as the identity label.
A triplet is (anchor, positive, negative) where anchor and positive share
the same identity, and negative comes from a different identity.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

_val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


class FinTripletDataset(Dataset):
    """
    Yields (anchor, positive, negative) image triples.
    At least 2 images per identity are required to form valid triplets.
    """

    def __init__(self, data_dir: str | Path, train: bool = True) -> None:
        data_dir = Path(data_dir)
        self.transform = _train_transform if train else _val_transform

        # Build identity → [image paths] map
        self.identity_to_paths: dict[str, list[Path]] = {}
        for identity_dir in sorted(data_dir.iterdir()):
            if not identity_dir.is_dir():
                continue
            images = [
                p for p in identity_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
            if len(images) >= 2:
                self.identity_to_paths[identity_dir.name] = images

        self.identities = list(self.identity_to_paths.keys())
        if len(self.identities) < 2:
            raise ValueError(
                f"Need at least 2 identities with ≥2 images each. "
                f"Found {len(self.identities)} valid identities in {data_dir}"
            )

        # Flat list of (identity, path) for indexing
        self.samples: list[tuple[str, Path]] = [
            (identity, path)
            for identity, paths in self.identity_to_paths.items()
            for path in paths
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_identity, anchor_path = self.samples[idx]

        # Positive: different image of the same individual
        positive_path = random.choice(
            [p for p in self.identity_to_paths[anchor_identity] if p != anchor_path]
        )

        # Negative: image from a different individual
        neg_identity = random.choice(
            [i for i in self.identities if i != anchor_identity]
        )
        negative_path = random.choice(self.identity_to_paths[neg_identity])

        anchor   = self._load(anchor_path)
        positive = self._load(positive_path)
        negative = self._load(negative_path)

        return anchor, positive, negative

    def _load(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)
