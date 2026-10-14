"""Datasets and augmentation for training/evaluation (requires torch + torchvision)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from app.capture import load_image_file
from app.classify.preprocess import IMG_SIZE, RESIZE_TO, preprocess, resize_center_crop
from app.labels import Labels

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_folder(root: Path, labels: Labels) -> list[tuple[Path, int]]:
    """<root>/<class_key>/*.jpg -> [(path, class_id)]; unknown folders are ignored."""
    items = []
    for info in labels:
        d = root / info.key
        if d.is_dir():
            items += [(p, info.id) for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return items


def load_eval_arrays(items: list[tuple[Path, int]], threads: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Decode + resize/centre-crop once (same as the app's ``preprocess`` before normalisation).

    Returns uint8 N x 224 x 224 x 3 and int targets; normalisation happens on the device.
    """
    def one(path: Path) -> np.ndarray:
        return resize_center_crop(load_image_file(path))

    with ThreadPoolExecutor(threads) as pool:
        x = np.stack(list(pool.map(one, [p for p, _ in items])))
    return x, np.array([t for _, t in items])


def train_transform() -> v2.Compose:
    """Augmentations aimed at the webcam domain gap: framing, lighting, blur, compression."""
    return v2.Compose(
        [
            v2.RandomRotation(30, fill=0),
            v2.RandomResizedCrop(IMG_SIZE, scale=(0.45, 1.0), ratio=(0.8, 1.25), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.35, hue=0.03),
            v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
            v2.RandomApply([v2.JPEG((35, 95))], p=0.3),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.12), value=0),
        ]
    )  # stays uint8: 4x less data between worker processes; normalised on the device


class LeafFolderDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], train: bool) -> None:
        self.items = items
        self.train = train
        self.tf = train_transform() if train else None

    def __len__(self) -> int:
        return len(self.items)

    @property
    def targets(self) -> np.ndarray:
        return np.array([t for _, t in self.items])

    def __getitem__(self, i: int):
        path, target = self.items[i]
        img = load_image_file(path)
        if self.train:
            img = resize_center_crop(img, RESIZE_TO, RESIZE_TO)  # normalise size before augmenting
            return self.tf(torch.from_numpy(img).permute(2, 0, 1).contiguous()), target  # uint8 CHW
        return torch.from_numpy(preprocess(img)), target  # identical to the app's inference path
