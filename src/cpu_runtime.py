"""CPU SIMD Runtime, Thread Pinning, and Deterministic Setup for EE300148."""

import os
import random
import sys
import psutil

# Step 1: Thread control before importing torch or numpy
DEFAULT_PHYSICAL_CORES = psutil.cpu_count(logical=False) or 10
_num_threads = int(os.environ.get("OMP_NUM_THREADS", str(DEFAULT_PHYSICAL_CORES)))
os.environ["OMP_NUM_THREADS"] = str(_num_threads)
os.environ["MKL_NUM_THREADS"] = str(_num_threads)

import numpy as np
import torch

# Enable oneDNN (MKL-DNN) backend
if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = True

torch.set_num_threads(_num_threads)


def configure_cpu_runtime(num_threads: int | None = None) -> int:
    """Configure OpenMP/MKL threads and torch thread pools to physical core count."""
    global _num_threads
    if num_threads is None:
        num_threads = psutil.cpu_count(logical=False) or DEFAULT_PHYSICAL_CORES

    _num_threads = int(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(_num_threads)
    torch.set_num_threads(_num_threads)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True
    return _num_threads


def get_device() -> torch.device:
    """Return pure CPU device to ensure 100% zero-CUDA compliance."""
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def inspect_cpu_simd() -> dict:
    """Inspect and report CPU SIMD ISA features (AVX2, AVX-512, oneDNN/MKL-DNN)."""
    cfg_text = torch.__config__.show()
    has_avx2 = "PERF_WITH_AVX2=1" in cfg_text or "AVX2" in cfg_text
    has_avx512 = "AVX512" in cfg_text
    has_mkldnn = "USE_MKLDNN=ON" in cfg_text or "MKL-DNN" in cfg_text

    return {
        "avx2": has_avx2,
        "avx512": has_avx512,
        "mkldnn": has_mkldnn,
        "num_threads": torch.get_num_threads(),
        "physical_cores": psutil.cpu_count(logical=False) or DEFAULT_PHYSICAL_CORES,
        "logical_cores": psutil.cpu_count(logical=True) or DEFAULT_PHYSICAL_CORES,
        "raw_config": cfg_text,
    }


if __name__ == "__main__":
    info = inspect_cpu_simd()
    print("CPU Runtime Diagnostics:")
    print(f"  Physical Cores: {info['physical_cores']}")
    print(f"  Logical Cores:  {info['logical_cores']}")
    print(f"  Active Threads: {info['num_threads']}")
    print(f"  AVX2 Support:   {info['avx2']}")
    print(f"  AVX-512 Support:{info['avx512']}")
    print(f"  oneDNN Enabled: {info['mkldnn']}")
