#!/usr/bin/env bash
# Create or verify the project environment, then start the Observatory.
#   ./scripts/launch_observatory.sh            CUDA GUI (default)
#   ./scripts/launch_observatory.sh --cpu      explicit CPU validation
#   ./scripts/launch_observatory.sh --replay F offline replay, no model weights
# Any other arguments are passed to `python -m liquidaion_fly.observatory`.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

torch_version="2.8.0"
cuda_index="https://download.pytorch.org/whl/cu128"
cpu_index="https://download.pytorch.org/whl/cpu"
mode="cuda"
if [[ " $* " == *" --cpu "* || " $* " == *" --replay "* ]]; then
  mode="cpu"
fi

fail() { echo "launch_observatory: $*" >&2; exit 2; }

free_gib() {
  local path="$1"
  while [[ ! -e "$path" ]]; do path="$(dirname -- "$path")"; done
  df -Pk -- "$path" | awk 'NR == 2 { printf "%d", $4 / 1048576 }'
}

if [[ -n "${LIQUIDAION_PYTHON:-}" ]]; then
  runtime_python="$LIQUIDAION_PYTHON"
elif [[ -x "$project_root/.venv/bin/python" ]]; then
  runtime_python="$project_root/.venv/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
  runtime_python="$(command -v python3.12)"
else
  runtime_python="$(command -v python3 || true)"
fi
[[ -n "$runtime_python" ]] || fail "Python 3.12 or newer was not found."
"$runtime_python" -c 'import sys; assert sys.version_info >= (3, 12)' 2>/dev/null \
  || fail "Python 3.12 or newer is required ($runtime_python is older)."

# Exit 0 when every pinned dependency is installed; exit 3 when only the PyTorch
# build mismatches the requested mode (a CPU wheel in CUDA mode or vice versa).
dependencies_ok() {
  "$runtime_python" - "$mode" <<'PY' >/dev/null 2>&1
import importlib.metadata
import sys
import tomllib
from packaging.requirements import Requirement
import torch, transformers, scipy, PySide6, pyvista, pyvistaqt
with open("pyproject.toml", "rb") as source:
    project = tomllib.load(source)["project"]
for spec in project["dependencies"] + project["optional-dependencies"]["desktop"]:
    requirement = Requirement(spec)
    assert importlib.metadata.version(requirement.name) in requirement.specifier
# "2.8.0+cpu" satisfies "torch==2.8.0", so check the build explicitly.
if sys.argv[1] == "cuda" and torch.version.cuda is None:
    sys.exit(3)
PY
}

status=0
dependencies_ok || status=$?
if [[ $status -ne 0 ]]; then
  venv="$project_root/.venv"
  if [[ "$mode" == "cuda" ]]; then
    # Measured on Linux x86_64: CUDA venv ~8.1 GB (4.1 GB NVIDIA libraries), Qwen weights 1.5 GB.
    need_venv=9; index="$cuda_index"
  else
    need_venv=3; index="$cpu_index"
  fi
  available="$(free_gib "$venv")"
  if (( available < need_venv )); then
    fail "Only ${available} GiB free at $venv; the $mode environment needs about ${need_venv} GiB. \
Free disk space or point LIQUIDAION_PYTHON at an existing interpreter with the dependencies."
  fi
  if [[ ! -x "$venv/bin/python" ]]; then
    echo "Creating $venv"
    "$runtime_python" -m venv "$venv"
  fi
  runtime_python="$venv/bin/python"
  # --no-cache-dir avoids keeping a second copy of the multi-GB wheels in ~/.cache/pip.
  if [[ $status -eq 3 ]]; then
    echo "Replacing the CPU-only PyTorch build with the CUDA build ($cuda_index)."
  else
    echo "Installing PyTorch $torch_version ($mode build) and project dependencies."
  fi
  "$runtime_python" -m pip install --no-cache-dir --upgrade pip
  "$runtime_python" -m pip install --no-cache-dir --force-reinstall "torch==$torch_version" --index-url "$index"
  "$runtime_python" -m pip install --no-cache-dir -e '.[desktop,test]'
  status=0
  dependencies_ok || status=$?
  [[ $status -eq 0 ]] || fail "Dependencies are still incomplete after installation (status $status)."
fi

if [[ " $* " != *" --replay "* ]]; then
  hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
  if [[ ! -d "$hf_home/hub/models--Qwen--Qwen3-0.6B" && " $* " != *" --model-path "* ]]; then
    available="$(free_gib "$hf_home")"
    if (( available < 2 )); then
      fail "Qwen3-0.6B (1.5 GB) must be downloaded to $hf_home but only ${available} GiB is free."
    fi
    echo "Qwen3-0.6B weights (1.5 GB) will be downloaded once into $hf_home."
  fi
fi

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export QT_API=pyside6
exec "$runtime_python" -m liquidaion_fly.observatory "$@"
