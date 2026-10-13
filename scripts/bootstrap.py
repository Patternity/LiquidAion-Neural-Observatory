"""First-run setup and launch for a downloaded LiquidAion release (Python standard library only).

Started by the platform launchers (LiquidAion-Windows.bat, liquidaion-linux.sh,
LiquidAion-macOS.command) through uv, which also provides Python 3.12. On the first run it
creates a private environment in .liquidaion-env/ next to this checkout and installs PyTorch
2.8.0 and the pinned dependencies from constraints.txt. Later runs start immediately.

  NVIDIA GPU (Windows, Linux)  CUDA 12.8 build (includes 4-bit NF4 support for the 4B profile)
  AMD GPU (Linux)              ROCm 6.4 build; PyTorch exposes it through the same torch.cuda API.
                               Untested by the authors; no NF4 (the 4B profile is unavailable).
  otherwise, and on macOS      CPU build

    launcher [--cpu | --cuda | --rocm] [--check] [--reinstall] [Observatory arguments ...]

--cpu / --cuda / --rocm  override the automatic choice      --reinstall  rebuild the environment
--check         install, verify and run the test suite headless instead of opening the GUI
Model weights are downloaded by Hugging Face on first use (Qwen3-0.6B: about 1.5 GB).
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".liquidaion-env"
MARKER = ENV / "liquidaion-install.json"
TORCH = "2.8.0"
INDEX = {"cuda": "https://download.pytorch.org/whl/cu128", "rocm": "https://download.pytorch.org/whl/rocm6.4",
         "cpu": "https://download.pytorch.org/whl/cpu"}
NEEDED_GIB = {"cuda": 9, "rocm": 12, "cpu": 3}  # environment sizes plus headroom
GPU_MODES = {"cuda", "rocm"}


def say(message: str) -> None:
    print(f"[LiquidAion] {message}", flush=True)


def fail(message: str) -> None:
    say(f"ERROR: {message}")
    sys.exit(2)


def nvidia_present() -> bool:
    if sys.platform == "darwin" or not shutil.which("nvidia-smi"):
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def amd_present() -> bool:
    """A ROCm-capable AMD GPU on Linux (rocminfo lists a gfx agent)."""
    if not sys.platform.startswith("linux") or not shutil.which("rocminfo"):
        return False
    result = subprocess.run(["rocminfo"], capture_output=True, text=True)
    return result.returncode == 0 and "gfx" in result.stdout


def detect_mode() -> str:
    if nvidia_present():
        return "cuda"
    if amd_present():
        return "rocm"
    if os.name == "nt" and _windows_has_amd():
        say("An AMD GPU was found, but PyTorch 2.8 has no official AMD (ROCm) build for Windows; using the CPU.")
    return "cpu"


def _windows_has_amd() -> bool:
    result = subprocess.run(["powershell", "-NoProfile", "-Command",
                             "(Get-CimInstance Win32_VideoController).Name"], capture_output=True, text=True)
    return result.returncode == 0 and any(word in result.stdout for word in ("AMD", "Radeon"))


def uv_env() -> dict[str, str]:
    """Large wheels (PyTorch, NVIDIA/ROCm libraries) need generous timeouts on slow connections."""
    env = dict(os.environ)
    env.setdefault("UV_HTTP_TIMEOUT", "900")
    env.setdefault("UV_HTTP_RETRIES", "10")
    env.setdefault("UV_CONCURRENT_DOWNLOADS", "4")
    return env


def env_python() -> Path:
    return ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(command: list[str], **kwargs) -> None:
    say("$ " + " ".join(str(c) for c in command))
    if subprocess.run([str(c) for c in command], **kwargs).returncode != 0:
        fail(f"command failed: {command[0]} {command[1] if len(command) > 1 else ''}")


def install(uv: str, mode: str) -> None:
    free = shutil.disk_usage(ROOT).free / 2**30
    if free < NEEDED_GIB[mode]:
        fail(f"Only {free:.1f} GiB free in {ROOT}; the {mode} environment needs about {NEEDED_GIB[mode]} GiB.")
    if ENV.exists():
        shutil.rmtree(ENV)
    say(f"Creating the {mode.upper()} environment in {ENV} (first run only; several GB are downloaded).")
    run([uv, "venv", "--python", "3.12", str(ENV)], env=uv_env())
    python = str(env_python())
    torch = [uv, "pip", "install", "--python", python, f"torch=={TORCH}"]
    if sys.platform != "darwin":  # macOS uses the regular PyPI build (CPU; MPS is not used by LiquidAion)
        torch += ["--index-url", INDEX[mode]]
    run(torch, env=uv_env())
    # bitsandbytes NF4 (4B profile) is installed for NVIDIA only.
    extras = "desktop,test,quant" if mode == "cuda" else "desktop,test"
    run([uv, "pip", "install", "--python", python, "-c", str(ROOT / "constraints.txt"), "-e", f"{ROOT}[{extras}]"],
        env=uv_env())
    verify(mode)
    MARKER.write_text(json.dumps({"mode": mode, "torch": TORCH, "platform": platform.platform(),
                                  "python": "3.12"}, indent=2) + "\n")


def verify(mode: str) -> None:
    check = ("import torch, transformers, PySide6, pyvista, pyvistaqt; "
             "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")
    result = subprocess.run([str(env_python()), "-c", check], capture_output=True, text=True, env=runtime_env())
    if result.returncode != 0:
        fail("the environment cannot import its dependencies:\n" + result.stderr[-2000:])
    say(result.stdout.strip())
    if mode == "cuda" and "cuda True" not in result.stdout:
        fail("PyTorch was installed with CUDA but cannot use the GPU (driver too old for CUDA 12.8?). "
             "Update the NVIDIA driver, or start with --cpu.")
    if mode == "rocm" and "cuda True" not in result.stdout:
        fail("PyTorch was installed with ROCm 6.4 but cannot use the GPU (unsupported card, or the ROCm "
             "driver/kernel module is missing; HSA_OVERRIDE_GFX_VERSION may help consumer cards). "
             "Start with --cpu to use the processor.")


def runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["QT_API"] = "pyside6"
    env["PYTHONUTF8"] = "1"  # all project files are UTF-8; Windows would otherwise use a legacy code page
    return env


def main() -> int:
    args = sys.argv[1:]
    flags = {flag for flag in ("--cpu", "--cuda", "--rocm", "--check", "--reinstall") if flag in args}
    passthrough = [a for a in args if a not in {"--cuda", "--rocm", "--check", "--reinstall"}]
    if len(flags & {"--cpu", "--cuda", "--rocm"}) > 1:
        fail("use only one of --cpu, --cuda, --rocm")
    forced = next((f[2:] for f in ("--cpu", "--cuda", "--rocm") if f in flags), None)
    mode = forced or detect_mode()
    if mode in GPU_MODES and sys.platform == "darwin":
        fail("GPU builds are not available on macOS; LiquidAion runs on the CPU there.")
    if mode == "rocm" and not sys.platform.startswith("linux"):
        fail("PyTorch ROCm builds exist only for Linux.")
    uv = os.environ.get("LIQUIDAION_UV") or shutil.which("uv")
    if not uv:
        fail("uv was not found; start LiquidAion through its launcher script.")
    installed = json.loads(MARKER.read_text()) if MARKER.exists() else {}
    if "--reinstall" in flags or installed.get("mode") != mode or installed.get("torch") != TORCH \
            or not env_python().exists():
        install(uv, mode)
    say(f"Mode: {mode.upper()}, environment {ENV}")
    env = runtime_env()
    python = str(env_python())
    device = ["--cpu"] if mode == "cpu" and "--cpu" not in passthrough else []
    if "--check" in flags:
        if mode in GPU_MODES:
            say("GPU check only; the test suite itself runs on the CPU.")
        run([python, "-m", "liquidaion_fly.observatory", "--prepare-only", *device], cwd=ROOT, env=env)
        env["QT_QPA_PLATFORM"] = "offscreen"
        return subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT,
                              env=env).returncode
    say("Starting the Observatory (the first launch downloads Qwen3-0.6B, about 1.5 GB).")
    return subprocess.run([python, "-m", "liquidaion_fly.observatory", *device, *passthrough], cwd=ROOT,
                          env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
