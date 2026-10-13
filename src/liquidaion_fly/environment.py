"""Report actual hardware rather than assuming the target machine is present."""
from __future__ import annotations
import importlib.metadata
import os
import platform
import sys
import shutil
import subprocess
from pathlib import Path
import torch


def _ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):  # Windows has no sysconf
        return None


def _peak_rss_bytes() -> int | None:
    try:
        import resource  # POSIX only
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # macOS reports bytes, Linux kilobytes


def inspect_environment() -> dict:
    ram = _ram_bytes()
    cpu_info = {}
    cpu_path = Path("/proc/cpuinfo")
    if cpu_path.exists():
        for line in cpu_path.read_text().splitlines():
            if ":" in line:
                key, value = (part.strip() for part in line.split(":", 1))
                if key in {"model name", "flags"} and key not in cpu_info:
                    cpu_info[key] = value
    memory_info = {}
    memory_path = Path("/proc/meminfo")
    if memory_path.exists():
        for line in memory_path.read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value, _ = line.split()
                memory_info[key.rstrip(":")] = int(value) * 1024
    gpu = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True, check=False)
        gpu = result.stdout.strip() if result.returncode == 0 else None
    return {"os": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "cpu": platform.processor(), "cpu_capabilities": cpu_info,
            "system_memory": memory_info, "ram_bytes": ram,
            "disk_free_bytes": shutil.disk_usage(Path.cwd()).free, "nvidia_smi": gpu,
            "torch_cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda,
            "packages": {name: importlib.metadata.version(name) for name in
                         ["torch", "transformers", "numpy", "scipy", "pandas"]},
            "peak_process_rss_bytes": _peak_rss_bytes()}
