"""CNN Theoretical Math Verification & Parameter Savings Module (Slides 9-20)."""

import math
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_output_size(
    input_size: Union[int, Tuple[int, int]],
    kernel_size: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
) -> Tuple[int, int]:
    """Calculate 2D convolution output spatial dimensions (Slides 10 & 17).

    Formula:
        output = floor((input - filter + 2 * padding) / stride) + 1
    """
    if isinstance(input_size, int):
        h_in, w_in = input_size, input_size
    else:
        h_in, w_in = input_size

    if isinstance(kernel_size, int):
        k_h, k_w = kernel_size, kernel_size
    else:
        k_h, k_w = kernel_size

    if isinstance(padding, int):
        p_h, p_w = padding, padding
    else:
        p_h, p_w = padding

    if isinstance(stride, int):
        s_h, s_w = stride, stride
    else:
        s_h, s_w = stride

    h_out = math.floor((h_in - k_h + 2 * p_h) / s_h) + 1
    w_out = math.floor((w_in - k_w + 2 * p_w) / s_w) + 1
    return h_out, w_out


def manual_convolve2d(
    input_matrix: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
) -> np.ndarray:
    """Perform a pure NumPy 2D correlation/convolution without padding."""
    h_in, w_in = input_matrix.shape
    k_h, k_w = kernel.shape
    h_out = math.floor((h_in - k_h) / stride) + 1
    w_out = math.floor((w_in - k_w) / stride) + 1

    output = np.zeros((h_out, w_out), dtype=np.float32)
    for i in range(h_out):
        for j in range(w_out):
            r_start = i * stride
            c_start = j * stride
            patch = input_matrix[r_start : r_start + k_h, c_start : c_start + k_w]
            output[i, j] = np.sum(patch * kernel)
    return output


def verify_hand_convolution(
    input_matrix: np.ndarray | None = None,
    kernel: np.ndarray | None = None,
) -> dict:
    """Verify hand-worked vertical edge filter convolution from Slide 9.

    Kernel: 3x3 vertical edge detector [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    Expected output:
        [[-4,  4, -1],
         [-6,  1,  3],
         [-3, -4,  5]]
    """
    if kernel is None:
        kernel = np.array(
            [
                [1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )

    if input_matrix is None:
        input_matrix = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 6.0, 5.0, 5.0],
                [2.0, 10.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 4.0, 0.0],
                [0.0, 0.0, 3.0, 10.0, 0.0],
            ],
            dtype=np.float32,
        )

    expected_output = np.array(
        [
            [-4.0, 4.0, -1.0],
            [-6.0, 1.0, 3.0],
            [-3.0, -4.0, 5.0],
        ],
        dtype=np.float32,
    )

    numpy_output = manual_convolve2d(input_matrix, kernel, stride=1)

    # PyTorch verification
    t_in = torch.from_numpy(input_matrix).unsqueeze(0).unsqueeze(0)
    t_k = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
    torch_output = F.conv2d(t_in, t_k, stride=1, padding=0).squeeze().numpy()

    numpy_match = np.allclose(numpy_output, expected_output)
    torch_match = np.allclose(torch_output, expected_output)

    return {
        "input_matrix": input_matrix,
        "kernel": kernel,
        "expected_output": expected_output,
        "numpy_output": numpy_output,
        "torch_output": torch_output,
        "is_correct": bool(numpy_match and torch_match),
    }


def compare_dense_vs_conv_params(
    input_shape: Tuple[int, int, int] = (3, 32, 32),
    dense_units: int = 1024,
    conv_filters: int = 32,
    conv_kernel_size: int = 3,
    include_bias: bool = True,
) -> dict:
    """Compare parameter counts of Dense vs. Conv2d layer (Slide 20).

    Dense: (3 * 32 * 32) * 1024 = 3,145,728 weights (+ 1,024 bias)
    Conv2d: 32 filters * (3 * 3 * 3) = 864 weights (+ 32 bias = 896 parameters)
    """
    in_channels, in_h, in_w = input_shape
    flat_dim = in_channels * in_h * in_w

    dense_weights = flat_dim * dense_units
    dense_biases = dense_units if include_bias else 0
    dense_total = dense_weights + dense_biases

    conv_weights = conv_filters * (in_channels * conv_kernel_size * conv_kernel_size)
    conv_biases = conv_filters if include_bias else 0
    conv_total = conv_weights + conv_biases

    weight_savings_ratio = dense_weights / conv_total
    total_savings_ratio = dense_total / conv_total

    return {
        "input_shape": input_shape,
        "dense_units": dense_units,
        "dense_params": dense_total,
        "dense_weights_only": dense_weights,
        "conv_filters": conv_filters,
        "conv_kernel_size": conv_kernel_size,
        "conv_params": conv_total,
        "conv_weights_only": conv_weights,
        "param_savings_ratio": weight_savings_ratio,
        "total_savings_ratio": total_savings_ratio,
    }
