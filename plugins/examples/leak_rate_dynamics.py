"""Leak Rate Dynamics: reservoir memory, steering geometry and influence on Qwen at several leak rates.

Run from Experiments -> Run Python Script... Edit the settings below first. Nothing in the
engine changes: the plugin sets the leak rate (as the RESERVOIR DYNAMICS control does), runs
read-only steering probes and, optionally, idle reservoir steps. See docs/RESEARCH_PLUGINS.md.
Output: results/plugins/leak-rate-dynamics/<time>/
"""
from liquidaion_fly.leak_dynamics import run_leak_dynamics

MODEL = None                       # profile key, e.g. "qwen3-4b-base"; None keeps the loaded model
BYPASSES = [(2, 2), (34, 34)]      # None keeps the current bypasses; 34 needs the 36-layer 4B model
STEERING_STRENGTH = 0.90           # configured strength (the applied size is measured separately)
FEEDBACK_STRENGTH = 0.65
LEAK_RATES = [0.005, 0.02, 0.10, 0.20, 0.50]

MODE = "forced_tokens"             # "forced_tokens" (controlled) or "free_generation" (Fly's own greedy tokens)
PROMPT = "Однажды в студёную зимнюю пору"
MAX_TOKENS = 128                   # measurement tokens per leak rate
FORCED_TOKENS = None               # token ids; None = Clean's greedy continuation (computed once, reused)

WARMUP_TOKENS = 0                  # > 0: a fixed forced warm-up sequence before the measurement
WARMUP_PROMPT = None               # None = PROMPT
IDLE_STEPS = 0                     # > 0: idle reservoir updates after the measurement (not tokens)

RESET_BETWEEN_RUNS = True          # new session (zero reservoir) before every leak rate
COUNTERFACTUAL = True              # per-token Fly-off run on a copy of the Fly cache
SAVE_VECTORS = True
SAVE_PLOTS = True


def run(api):
    api.set_output_name("leak-rate-dynamics")
    if MODEL:
        api.load_model(MODEL)
    if BYPASSES is not None:
        api.set_bypasses(BYPASSES)
    api.set_coupling(strength=STEERING_STRENGTH, feedback=FEEDBACK_STRENGTH, steering_enabled=True,
                     feedback_enabled=True)
    config = api.get_config()
    api.log(f"{config['model']['display']} · {config['coupling']['coupling_mode']} "
            f"{config['coupling']['active_bypasses']} · {config['coupling']['reservoir_updates_per_token']} "
            "reservoir update(s) per token")
    result = run_leak_dynamics(
        api, LEAK_RATES, PROMPT, MAX_TOKENS, mode="forced" if MODE == "forced_tokens" else "free",
        forced_tokens=FORCED_TOKENS, warmup_prompt=WARMUP_PROMPT, warmup_tokens=WARMUP_TOKENS,
        idle_steps=IDLE_STEPS, reset_between_runs=RESET_BETWEEN_RUNS, counterfactual=COUNTERFACTUAL,
        save_vectors=SAVE_VECTORS, save_plots=SAVE_PLOTS)
    api.log(f"Saved to {result['output']}")
