"""Steering dynamics probe: is the reservoir's steering direction changing from token to token?

For each leak rate below, from a zero reservoir and the SAME token sequence, this records per
token: reservoir states, the 64-d readout, the steering vector and the applied delta, their
norms and cosines (consecutive and to a reference), and Clean / Fly / Fly-off next-token
distributions. Nothing in the coupling is changed; see docs/STEERING_DIAGNOSTICS.md.

Edit the settings, then run it from Experiments -> Run Python Script... Output:
results/plugins/steering_dynamics_probe/<time>/leak-<value>/ (+ comparison.md).
"""

MODEL_PROFILE = None            # e.g. "qwen3-4b-base"; None keeps the loaded model
BYPASSES = [(2, 2), (34, 34)]   # None keeps the current bypasses
STRENGTH = 0.90
FEEDBACK = 0.65
LEAK_RATES = [0.005, 0.200]     # e.g. [0.005, 0.020, 0.100, 0.200, 0.500]
PROMPT = "Однажды в студёную зимнюю пору"
MAX_TOKENS = 48
FORCED_TOKENS = None            # list of token ids; None = Clean's greedy continuation (computed once)
COUNTERFACTUAL = True           # also run each token uncoupled on a copy of the Fly cache


def run(api):
    if MODEL_PROFILE:
        api.load_model(MODEL_PROFILE)
    if BYPASSES is not None:
        api.set_bypasses(BYPASSES)
    api.set_coupling(strength=STRENGTH, feedback=FEEDBACK, steering_enabled=True, feedback_enabled=True)
    tokens = FORCED_TOKENS
    rows = []
    for leak in LEAK_RATES:
        api.check_cancelled()
        api.new_session()  # zero reservoir, empty histories and caches
        api.set_coupling(leak_rate=leak)
        result = api.steering_probe(PROMPT, MAX_TOKENS, name=f"leak-{leak:.3f}", forced_tokens=tokens,
                                    counterfactual=COUNTERFACTUAL)
        tokens = result["metadata"]["token_ids"]  # every leak rate sees the same sequence
        s = result["summary"]
        rows.append((leak, s))
        api.log(f"leak {leak:.3f}: steering cos(prev) median {s['steering_cos_prev'].get('median', float('nan')):.4f}, "
                f"cos(ref) last {s['steering_cos_ref'].get('last', float('nan')):.4f}, "
                f"KL(Clean||Fly) median {s['kl_clean_fly'].get('median', float('nan')):.3g} -> {result['output']}")
    lines = ["| leak | state cos prev | steering cos prev | steering cos ref (last) | orth ratio | "
             "KL Clean‖Fly | KL immediate | top-1 changed |", "|---|---|---|---|---|---|---|---|"]
    for leak, s in rows:
        def m(key, field="median"):
            return f"{s[key][field]:.4g}" if s[key].get("n") else "n/a"
        lines.append(f"| {leak:.3f} | {m('state_cos_prev')} | {m('steering_cos_prev')} | {m('steering_cos_ref', 'last')} | "
                     f"{m('steering_orth_ratio')} | {m('kl_clean_fly')} | {m('kl_off_fly_immediate')} | "
                     f"{s['top1_changed']['count']}/{s['top1_changed']['of']} |")
    (api.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    api.log("\n".join(lines))
