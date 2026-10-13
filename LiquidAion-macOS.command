#!/usr/bin/env bash
# LiquidAion Neural Observatory: download-and-run launcher for macOS (double-click in Finder).
# macOS has no CUDA: LiquidAion runs on the CPU (the 4B NF4 profile is unavailable).
# First run: installs uv (pinned) and Python 3.12 into .tools/, then PyTorch and dependencies into
# .liquidaion-env/ (CUDA build if an NVIDIA GPU is visible, else CPU). Options: see scripts/bootstrap.py.
set -uo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UV_VERSION="0.12.17"
export UV_PYTHON_INSTALL_DIR="$root/.tools/python" UV_CACHE_DIR="$root/.tools/cache"
uv="$root/.tools/uv"
if [[ ! -x "$uv" ]]; then
  echo "[LiquidAion] Installing uv $UV_VERSION into $root/.tools (first run only)"
  command -v curl >/dev/null || { echo "[LiquidAion] curl is required for the first run" >&2; exit 2; }
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$root/.tools" UV_NO_MODIFY_PATH=1 sh || exit 2
  [[ -x "$uv" ]] || uv="$root/.tools/bin/uv"
fi
export LIQUIDAION_UV="$uv"
cd "$root"
"$uv" run --no-project --python 3.12 python "$root/scripts/bootstrap.py" "$@"
status=$?
echo
[[ -t 0 ]] && read -r -p "[LiquidAion] Finished (exit code $status). Press Enter to close this window. "
exit $status
