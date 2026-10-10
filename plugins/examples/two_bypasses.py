"""Clean and Fly on one prompt with two simultaneous bypasses (Synchronous Multi-Bypass)."""


def run(api):
    api.new_session()
    coupling = api.set_bypasses([(2, 10), (20, 5)])
    api.log(f"{coupling['coupling_mode']}: {coupling['active_bypasses']}, "
            f"{coupling['reservoir_updates_per_token']} reservoir update per token")
    results = api.generate(prompt="Однажды в студёную зимнюю пору", max_tokens=64, mode="both")
    api.save_result(name="two_bypasses", result=results)
    for branch, result in results.items():
        api.log(f"{branch}: {result.text[:100]!r}")
