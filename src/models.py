"""Models module: Majority baseline, ScratchCNN, and ResNet-18 Phase 1/2 transfer learning."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and frozen parameters in a PyTorch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
    }


class MajorityClassBaseline:
    """Trivial majority-class classifier establishing performance floor (Slide 54 Step 5)."""

    def __init__(self, n_classes: int = 5):
        self.n_classes = n_classes
        self.majority_class: int = 0
        self.class_priors: Optional[np.ndarray] = None

    def fit(self, labels: Union[List[int], np.ndarray, torch.Tensor]) -> "MajorityClassBaseline":
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        labels = np.asarray(labels)

        counts = np.bincount(labels, minlength=self.n_classes)
        self.majority_class = int(np.argmax(counts))
        self.class_priors = counts / counts.sum()
        return self

    def fit_loader(self, dataloader) -> "MajorityClassBaseline":
        all_labels = []
        for _, y in dataloader:
            all_labels.extend(y.tolist())
        return self.fit(all_labels)

    def predict(self, num_samples: int) -> np.ndarray:
        return np.full((num_samples,), self.majority_class, dtype=int)

    def predict_proba(self, num_samples: int) -> np.ndarray:
        probs = np.zeros((num_samples, self.n_classes), dtype=float)
        probs[:, self.majority_class] = 1.0
        return probs

    def evaluate(self, dataloader) -> Dict[str, float]:
        all_targets = []
        for _, y in dataloader:
            all_targets.extend(y.tolist())
        targets = np.array(all_targets)
        preds = self.predict(len(targets))
        accuracy = float(np.mean(preds == targets))
        return {
            "accuracy": accuracy,
            "majority_class": self.majority_class,
            "n_samples": len(targets),
        }


class ScratchCNN(nn.Module):
    """Custom lightweight CNN trained from scratch (Slides 22 & 27).

    Architecture:
        Block 1: Conv2d(3, 32, 3, pad=1) -> ReLU -> MaxPool2d(2, 2)
        Block 2: Conv2d(32, 64, 3, pad=1) -> ReLU -> MaxPool2d(2, 2)
        AdaptiveAvgPool2d(4, 4) -> Flatten -> Linear(1024, 128) -> ReLU -> Dropout(0.3) -> Linear(128, n_classes)
    """

    def __init__(
        self,
        n_classes: int = 5,
        input_size: Tuple[int, int] = (224, 224),
        use_adaptive_pool: bool = True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.use_adaptive_pool = use_adaptive_pool

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if use_adaptive_pool:
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            flatten_dim = 64 * 4 * 4
        else:
            self.pool = nn.Identity()
            h_out = input_size[0] // 4
            w_out = input_size[1] // 4
            flatten_dim = 64 * h_out * w_out

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, n_classes),
        )

        # Convert to channels_last for AVX2/AVX-512 SIMD vectorization
        self.to(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and not x.is_contiguous(memory_format=torch.channels_last):
            x = x.to(memory_format=torch.channels_last)
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class ResNet18Transfer(nn.Module):
    """Two-Phase ResNet-18 Transfer Learning Model (Slides 30 & 31).

    Phase 1: Linear Probing. Entire backbone frozen (torch.no_grad), only fc is trained (~2,565 params).
    Phase 2: Fine-Tuning. layer4 and fc are unfrozen and trained.
    """

    def __init__(self, n_classes: int = 5):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze all base parameters
        for p in base.parameters():
            p.requires_grad = False

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, n_classes)

        self.phase = 1
        self.to(memory_format=torch.channels_last)

    def set_phase(self, phase: int) -> None:
        """Switch between Phase 1 (linear probe) and Phase 2 (layer4 unfrozen)."""
        self.phase = phase
        if phase == 1:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.fc.parameters():
                p.requires_grad = True
        elif phase == 2:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.layer4.parameters():
                p.requires_grad = True
            for p in self.fc.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and not x.is_contiguous(memory_format=torch.channels_last):
            x = x.to(memory_format=torch.channels_last)

        if self.training and self.phase == 1:
            # Phase 1: skip autograd tracking for frozen backbone
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

        elif self.training and self.phase == 2:
            # Phase 2: skip autograd tracking for layers 1..3, train layer4 + fc
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

        else:
            # Eval / Inference mode
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x


def build_resnet18_transfer(n_classes: int = 5) -> ResNet18Transfer:
    """Build ResNet-18 Phase 1 (Frozen Backbone / Linear Probe) as specified in Slide 30."""
    model = ResNet18Transfer(n_classes=n_classes)
    counts = count_parameters(model)
    expected_trainable = 512 * n_classes + n_classes
    assert counts["trainable_params"] == expected_trainable, (
        f"Expected {expected_trainable} trainable parameters, got {counts['trainable_params']}"
    )
    return model


def unfreeze_layer4(model: nn.Module) -> nn.Module:
    """Unfreeze model.layer4 for Transfer Learning Phase 2 (Slide 31)."""
    if hasattr(model, "set_phase"):
        model.set_phase(2)
    else:
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    return model
