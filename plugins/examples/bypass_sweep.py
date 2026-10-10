"""Sweep the capture layer of a single bypass (capture -> 10) on one prompt.

Run it from the Observatory: Experiments -> Run Python Script... The script is trusted local
code executed with your user permissions. Results go to results/plugins/bypass_sweep/<time>/.
"""

PROMPT = "Ты в курсе, что ты немного муха дрозофила?"
STEER_LAYER = 10


def run(api):
    model = api.get_config()["model"]
    api.log(f"Model: {model['display']} ({model['num_hidden_layers']} layers)")
    for capture in range(1, 7):
        api.check_cancelled()  # Stop requested -> the script ends here
        api.new_session()  # empty histories and caches, zero reservoir; weights stay loaded
        coupling = api.set_bypasses([(capture, STEER_LAYER)])
        result = api.generate(prompt=PROMPT, max_tokens=128, mode="fly")
        reservoir = api.get_reservoir_state()
        api.save_result(name=f"bypass_{capture}_{STEER_LAYER}", result=result,
                        extra={"coupling_mode": coupling["coupling_mode"], "final_reservoir_rms": reservoir["rms"]})
        api.log(f"{capture} -> {STEER_LAYER}: {len(result.token_ids)} tokens, stop {result.stop_reason}: "
                f"{result.text[:80]!r}")
