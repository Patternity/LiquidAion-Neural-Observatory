#!/usr/bin/env bash
# Reproduce the "Infinite Nekrasov" continuous-generation experiments (RTX 3070 Ti: ~1 h).
# Every run: seed "Однажды в студёную зимнюю пору", 128-token window, greedy decoding,
# EOS suppressed identically in all conditions (each suppression is recorded).
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python="${LIQUIDAION_PYTHON:-$project_root/.venv/bin/python}"
out="${1:-results/continuous}"
run() { PYTHONPATH=src "$python" -m liquidaion_fly.continuous --window 128 --eos suppress "$@"; }

run --output "$out/phase1-smoke" --tokens 256 --conditions clean,fly --eos stop --checkpoints 128
run --output "$out/raw-s0.25-f0.25" --tokens 5000 --conditions clean,fly,fly_reset,random --reset-intervals 128,512
run --output "$out/raw-s0.50-f0.25" --tokens 5000 --conditions clean,fly,fly_reset,random --reset-intervals 512 --strength 0.50
run --output "$out/raw-s0.70-f0.45" --tokens 5000 --conditions clean,fly,fly_reset,random --reset-intervals 512 --strength 0.70 --feedback 0.45
run --output "$out/chat-s0.25-f0.25" --tokens 5000 --prompt-policy chat_prefix --conditions clean,fly,fly_reset,random --reset-intervals 128,512
run --output "$out/chat-s0.70-f0.45" --tokens 5000 --prompt-policy chat_prefix --conditions clean,fly,fly_reset,random --reset-intervals 512 --strength 0.70 --feedback 0.45

# Stochastic decoding (added after greedy runs collapsed into token loops). Same seed in every
# condition of a series: common random numbers, order-independent. No repetition penalty.
for seed in 42 43 44; do
  run --output "$out/sample-t1.0-p0.95-seed$seed" --tokens 5000 --decoding sample --temperature 1.0 --top-p 0.95 \
      --sampling-seed "$seed" --conditions clean,fly,fly_reset,random --reset-intervals 512
done
