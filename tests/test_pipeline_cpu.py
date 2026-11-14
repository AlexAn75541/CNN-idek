"""Smoke and integration tests for CPU pipeline, channels_last, and transfer learning."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from src.cpu_runtime import configure_cpu_runtime, get_device, inspect_cpu_simd, seed_everything
from src.dataset import get_transforms, load_dataset, stratified_split
from src.metrics import compute_confusion_matrix
from src.models import (
    MajorityClassBaseline,
    ScratchCNN,
    build_resnet18_transfer,
    count_parameters,
    unfreeze_layer4,
)


def test_cpu_simd_runtime():
    """Verify thread configuration, SIMD detection, and device is CPU."""
    threads = configure_cpu_runtime(num_threads=4)
    assert threads == 4
    assert torch.get_num_threads() == 4

    info = inspect_cpu_simd()
    assert info["avx2"] is True or info["avx512"] is True
    assert get_device() == torch.device("cpu")


def test_seed_everything_determinism():
    """Verify deterministic seed invariance across numpy and torch."""
    seed_everything(42)
    t1 = torch.randn(10)
    n1 = np.random.randn(10)

    seed_everything(42)
    t2 = torch.randn(10)
    n2 = np.random.randn(10)

    assert torch.equal(t1, t2)
    np.testing.assert_array_equal(n1, n2)


def test_channels_last_memory_format():
    """Verify 4D tensors convert cleanly to channels_last for oneDNN."""
    x = torch.randn(4, 3, 224, 224)
    x_nhwc = x.to(memory_format=torch.channels_last)
    assert x_nhwc.is_contiguous(memory_format=torch.channels_last)

    model = ScratchCNN(n_classes=5)
    with torch.inference_mode():
        y = model(x_nhwc)
    assert y.shape == (4, 5)


def test_resnet18_phase1_and_phase2_parameters():
    """Verify parameter counts for Phase 1 (linear probe) and Phase 2 (layer4 unfreeze)."""
    model = build_resnet18_transfer(n_classes=5)
    counts1 = count_parameters(model)
    # Exactly 512 * 5 + 5 = 2,565 trainable parameters
    assert counts1["trainable_params"] == 2565
    assert counts1["frozen_params"] > 11_000_000

    # Unfreeze layer4
    unfreeze_layer4(model)
    counts2 = count_parameters(model)
    assert counts2["trainable_params"] > 8_000_000
    assert counts2["frozen_params"] > 2_000_000

    # Verify earlier layers remain frozen
    assert model.conv1.weight.requires_grad is False
    assert model.layer1[0].conv1.weight.requires_grad is False
    assert model.layer2[0].conv1.weight.requires_grad is False
    assert model.layer3[0].conv1.weight.requires_grad is False
    # layer4 and fc are trainable
    assert model.layer4[0].conv1.weight.requires_grad is True
    assert model.fc.weight.requires_grad is True


def test_majority_baseline_classifier():
    """Verify majority baseline predicts mode class and achieves exact accuracy."""
    base = MajorityClassBaseline(n_classes=5)
    labels = np.array([0, 1, 1, 1, 2, 3, 4])
    base.fit(labels)
    assert base.majority_class == 1

    preds = base.predict(num_samples=7)
    assert np.all(preds == 1)
    acc = np.mean(preds == labels)
    assert np.isclose(acc, 3 / 7)


def test_compute_confusion_matrix_top2():
    """Verify confusion matrix and extraction of top-2 confused pairs."""
    targets = np.array([0, 0, 1, 1, 2, 2, 3, 4])
    preds = np.array([0, 1, 1, 2, 2, 0, 3, 0])
    class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

    cm, top_2 = compute_confusion_matrix(targets, preds, class_names)
    assert cm.shape == (5, 5)
    assert len(top_2) <= 2
    # Verify count > 0 in top_2
    for pair in top_2:
        assert pair["count"] > 0
        assert pair["true_class"] != pair["pred_class"]
