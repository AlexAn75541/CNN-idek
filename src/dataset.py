"""Self-contained dataset loader, stratified 60/20/20 split, and Slide 29 transforms."""

import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    augment: bool = True,
    img_size: Tuple[int, int] = (224, 224),
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, val_transform) as specified in Slide 29.

    train_tf:
        Resize((224, 224)), RandomHorizontalFlip(), RandomRotation(12),
        ColorJitter(brightness=0.2), ToTensor(), Normalize(mean, std)
    val_tf:
        Resize((224, 224)), ToTensor(), Normalize(mean, std)
    """
    val_tf = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    if augment:
        train_tf = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        # Slide 32 ablation study: disable augmentation
        train_tf = val_tf

    return train_tf, val_tf


def generate_synthetic_flowers(
    output_dir: str = "./data/flowers5",
    samples_per_class: int = 60,
    seed: int = 42,
) -> List[str]:
    """Generate a deterministic 5-class flower dataset for 100% offline reproducibility.

    Classes:
        0: daisy (white petals, yellow center)
        1: dandelion (dense yellow radial petals)
        2: rose (concentric red petals, dark center)
        3: sunflower (dark brown center, large yellow petals)
        4: tulip (purple/violet cup-shaped petals)
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    os.makedirs(output_dir, exist_ok=True)

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        for i in range(samples_per_class):
            img_path = os.path.join(cls_dir, f"{cls_name}_{i:03d}.jpg")
            if os.path.exists(img_path):
                continue

            # Base background: textured green garden
            bg_color = (
                int(rng.randint(25, 45)),
                int(rng.randint(90, 140)),
                int(rng.randint(20, 50)),
            )
            img = Image.new("RGB", (256, 256), color=bg_color)
            draw = ImageDraw.Draw(img)

            # Background foliage noise
            for _ in range(12):
                x1 = rng.randint(0, 240)
                y1 = rng.randint(0, 240)
                r_len = rng.randint(15, 60)
                leaf_col = (
                    int(rng.randint(20, 60)),
                    int(rng.randint(100, 160)),
                    int(rng.randint(20, 70)),
                )
                draw.ellipse([x1, y1, x1 + r_len, y1 + r_len // 2], fill=leaf_col)

            # Center position with slight jitter
            cx = 128 + rng.randint(-12, 12)
            cy = 128 + rng.randint(-12, 12)

            if cls_name == "daisy":
                # White petals + yellow center
                num_petals = rng.randint(12, 18)
                petal_len = rng.randint(45, 60)
                for p in range(num_petals):
                    angle = (2 * math.pi / num_petals) * p + rng.uniform(-0.1, 0.1)
                    px = cx + math.cos(angle) * petal_len
                    py = cy + math.sin(angle) * petal_len
                    p_shade = rng.randint(235, 255)
                    draw.line([(cx, cy), (px, py)], fill=(p_shade, p_shade, p_shade), width=12)
                # Center
                c_rad = rng.randint(18, 24)
                draw.ellipse(
                    [cx - c_rad, cy - c_rad, cx + c_rad, cy + c_rad],
                    fill=(rng.randint(235, 255), rng.randint(190, 220), 10),
                )

            elif cls_name == "dandelion":
                # Dense yellow radial tufts
                num_tufts = rng.randint(35, 55)
                for _ in range(num_tufts):
                    angle = rng.uniform(0, 2 * math.pi)
                    dist = rng.uniform(15, 55)
                    px = cx + math.cos(angle) * dist
                    py = cy + math.sin(angle) * dist
                    col = (rng.randint(230, 255), rng.randint(190, 225), rng.randint(0, 30))
                    draw.line([(cx, cy), (px, py)], fill=col, width=rng.randint(4, 7))

            elif cls_name == "rose":
                # Concentric layered red petals
                for r in range(55, 10, -8):
                    red_shade = int(rng.randint(160, 230) * (r / 55.0))
                    col = (red_shade, rng.randint(10, 30), rng.randint(20, 45))
                    ox = rng.randint(-5, 5)
                    oy = rng.randint(-5, 5)
                    draw.ellipse([cx - r + ox, cy - r + oy, cx + r + ox, cy + r + oy], fill=col)

            elif cls_name == "sunflower":
                # Large brown center + bright yellow outer petals
                num_petals = rng.randint(18, 26)
                for p in range(num_petals):
                    angle = (2 * math.pi / num_petals) * p
                    px = cx + math.cos(angle) * 65
                    py = cy + math.sin(angle) * 65
                    col = (rng.randint(240, 255), rng.randint(180, 205), 0)
                    draw.polygon(
                        [(cx, cy), (px - 8, py), (px + 8, py), (px, py + 12)],
                        fill=col,
                    )
                c_rad = rng.randint(28, 36)
                draw.ellipse(
                    [cx - c_rad, cy - c_rad, cx + c_rad, cy + c_rad],
                    fill=(rng.randint(60, 85), rng.randint(30, 50), 15),
                )

            elif cls_name == "tulip":
                # Cup-shaped magenta/purple petals
                col = (rng.randint(170, 220), rng.randint(20, 60), rng.randint(140, 190))
                # Stem
                draw.line([(cx, cy + 20), (cx, 250)], fill=(30, 120, 30), width=10)
                # Cup
                draw.ellipse([cx - 35, cy - 45, cx + 35, cy + 35], fill=col)
                draw.polygon(
                    [(cx - 35, cy - 20), (cx, cy - 65), (cx + 35, cy - 20)],
                    fill=col,
                )

            # Gentle gaussian blur to simulate optical depth of field
            img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
            img.save(img_path, format="JPEG", quality=90)

    return class_names


class ImageDataset(Dataset):
    """Dataset wrapping a list of (image_path, label) pairs with independent transforms."""

    def __init__(self, samples: List[Tuple[str, int]], class_names: List[str], transform=None):
        self.samples = samples
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_dataset(
    data_dir: Optional[str] = None,
    n_classes: int = 5,
    samples_per_class: int = 60,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Load dataset sample paths and class names.

    Defaults to `./data/flowers5`, generating it deterministically if absent.
    """
    target_dir = data_dir or "./data/flowers5"
    if not os.path.exists(target_dir) or len(os.listdir(target_dir)) < n_classes:
        generate_synthetic_flowers(target_dir, samples_per_class=samples_per_class, seed=seed)

    # Discover classes sorted alphabetically for consistent index mapping
    subdirs = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
    class_names = subdirs[:n_classes]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    samples: List[Tuple[str, int]] = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for c in class_names:
        c_dir = os.path.join(target_dir, c)
        for fname in sorted(os.listdir(c_dir)):
            if os.path.splitext(fname)[1].lower() in valid_exts:
                samples.append((os.path.join(c_dir, fname), class_to_idx[c]))

    return samples, class_names


def stratified_split(
    samples: List[Tuple[str, int]],
    class_names: List[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    augment_train: bool = True,
) -> Tuple[ImageDataset, ImageDataset, ImageDataset]:
    """Stratified 60/20/20 train/val/test split before fitting or augmentation."""
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-5), "Ratios must sum to 1.0"

    labels = [s[1] for s in samples]
    train_samples, holdout_samples, train_lbl, holdout_lbl = train_test_split(
        samples,
        labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Split remaining 40% into 20% val and 20% test (50/50 of holdout)
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        holdout_samples,
        train_size=relative_val_ratio,
        stratify=holdout_lbl,
        random_state=seed,
    )

    train_tf, val_tf = get_transforms(augment=augment_train)

    train_dataset = ImageDataset(train_samples, class_names, transform=train_tf)
    val_dataset = ImageDataset(val_samples, class_names, transform=val_tf)
    test_dataset = ImageDataset(test_samples, class_names, transform=val_tf)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: ImageDataset,
    val_dataset: ImageDataset,
    test_dataset: ImageDataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create CPU-optimized PyTorch DataLoaders with pin_memory=False."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader
