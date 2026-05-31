from __future__ import annotations

import os
import sys
from typing import Any


def detect_compute_capabilities() -> dict[str, Any]:
    """Return backend compute capability without forcing CUDA allocation."""

    torch_available = False
    cuda_available = False
    torch_version = None
    torch_cuda_version = None
    gpu_name = None
    gpu_total_memory_gb = 0.0
    gpu_compute_capability = None

    try:
        import torch  # type: ignore

        torch_available = True
        torch_version = str(torch.__version__)
        torch_cuda_version = str(torch.version.cuda) if torch.version.cuda else None
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = str(props.name)
            gpu_total_memory_gb = float(props.total_memory / 1024**3)
            gpu_compute_capability = f"{props.major}.{props.minor}"
    except Exception:
        torch_available = False
        cuda_available = False

    cpu_workers_default = int(os.environ.get("SPAD_CPU_WORKERS", "8"))
    return {
        "python_executable": sys.executable,
        "cpu_workers_default": cpu_workers_default,
        "torch_available": torch_available,
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_total_memory_gb": gpu_total_memory_gb,
        "gpu_compute_capability": gpu_compute_capability,
        "recommended_backend": "cuda" if cuda_available else "cpu",
        "notes": [
            "Use CPU workers for branching control flow and small scenes.",
            "Use CUDA for large batched photon-rate cubes, Poisson sampling, and Monte Carlo sweeps.",
        ],
    }
