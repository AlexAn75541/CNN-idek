"""Unit tests for CNN manual math, dimensions, and parameter savings (Slides 9-20)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from src.cnn_math import (
    compare_dense_vs_conv_params,
    conv2d_output_size,
    manual_convolve2d,
    verify_hand_convolution,
)


def test_conv2d_output_size_formula():
    """Slide 16 checkpoint: 32x32 input, 5x5 kernel, padding=0, stride=1 -> 28x28."""
    h_out, w_out = conv2d_output_size(input_size=32, kernel_size=5, padding=0, stride=1)
    assert (h_out, w_out) == (28, 28)

    # PyTorch ground truth check
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
    x = torch.zeros(1, 3, 32, 32)
    y = conv(x)
    assert y.shape == (1, 16, 28, 28)


def test_conv2d_output_size_with_padding_and_stride():
    """Verify formula with non-zero padding and stride."""
    # 224x224 input, 7x7 kernel, stride=2, padding=3 (ResNet conv1) -> 112x112
    h_out, w_out = conv2d_output_size(input_size=224, kernel_size=7, padding=3, stride=2)
    assert (h_out, w_out) == (112, 112)

    # Asymmetric input/kernel
    h_out, w_out = conv2d_output_size(
        input_size=(64, 32), kernel_size=(3, 5), padding=(1, 2), stride=(2, 1)
    )
    # (64 - 3 + 2)/2 + 1 = 63/2 + 1 = 31 + 1 = 32
    # (32 - 5 + 4)/1 + 1 = 31 + 1 = 32
    assert (h_out, w_out) == (32, 32)


def test_verify_hand_convolution_slide_9():
    """Verify Slide 9 hand-worked vertical edge filter."""
    result = verify_hand_convolution()
    assert result["is_correct"] is True
    expected = np.array(
        [
            [-4.0, 4.0, -1.0],
            [-6.0, 1.0, 3.0],
            [-3.0, -4.0, 5.0],
        ]
    )
    np.testing.assert_allclose(result["numpy_output"], expected)
    np.testing.assert_allclose(result["torch_output"], expected)


def test_compare_dense_vs_conv_params_slide_20():
    """Verify Slide 20 parameter comparison on 32x32x3 input."""
    res = compare_dense_vs_conv_params(
        input_shape=(3, 32, 32),
        dense_units=1024,
        conv_filters=32,
        conv_kernel_size=3,
        include_bias=True,
    )
    # Dense: 3*32*32 * 1024 = 3,145,728 weights
    assert res["dense_weights_only"] == 3145728
    # Conv2d: 32 * (3 * 3 * 3) + 32 = 864 + 32 = 896 parameters
    assert res["conv_params"] == 896
    # Savings ratio ~ 3510x (3,145,728 / 896 = 3510.85)
    assert round(res["param_savings_ratio"]) == 3511
    assert round(res["total_savings_ratio"]) == 3512
