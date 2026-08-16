"""Capture the reproducibility-critical environment/hardware description required
by the assignment (Python/PyTorch/tsl/CUDA versions, GPU, CPU, RAM, OS, seed).

Run as ``python -m code.utils.env_report`` after the venv is set up. Writes
``environment.json`` at the project root and prints a human-readable summary.
This file is quoted verbatim in the report's Experimental Setup section --
nothing about hardware/software versions is hand-typed there.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _safe_import_version(module_name: str, attr: str = "__version__"):
    try:
        mod = __import__(module_name)
        return getattr(mod, attr, "unknown")
    except ImportError:
        return None


def _cpu_name() -> str:
    try:
        out = subprocess.run(
            ["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=10
        )
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip() and "Name" not in l]
        if lines:
            return lines[0]
    except Exception:
        pass
    return platform.processor() or "unknown"


def _total_ram_gb() -> float | str:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=15,
            )
            total_bytes = int(out.stdout.strip())
            return round(total_bytes / (1024 ** 3), 1)
        except Exception:
            return "unknown"


def build_report(seed: int) -> dict:
    import torch

    cuda_available = torch.cuda.is_available()

    report = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "tsl_version": _safe_import_version("tsl"),
        "torch_geometric_version": _safe_import_version("torch_geometric"),
        "pytorch_lightning_version": _safe_import_version("pytorch_lightning"),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else "None (CPU-only training)",
        "cpu": _cpu_name(),
        "ram_gb": _total_ram_gb(),
        "operating_system": f"{platform.system()} {platform.release()} ({platform.version()})",
        "random_seed": seed,
    }
    return report


def main(seed: int = 42) -> None:
    report = build_report(seed)
    out_path = _ROOT / "environment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Environment report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
