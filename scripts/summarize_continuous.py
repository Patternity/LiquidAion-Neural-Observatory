"""Collect every continuous-generation series under results/continuous into SUMMARY.json and SUMMARY.md."""
import json
from pathlib import Path

root = Path("results/continuous")
rows = []
for comparison in sorted(root.glob("*/comparison.json")) + sorted(root.glob("sampling-validation/*/comparison.json")):
    series = str(comparison.parent.relative_to(root))
    data = json.loads(comparison.read_text())
    for row in data["conditions"]:
        meta = json.loads((comparison.parent / row["condition"] / "metadata.json").read_text())
        decoding = meta["decoding"] if isinstance(meta["decoding"], dict) else {"method": "greedy"}
        kl = row["checkpoint_kl_retained_vs_reset"]
        rows.append({"series": series, "condition": row["condition"], "tokens": row["generated_tokens"],
                     "prompt_policy": meta["config"].get("prompt_policy", "raw"), "strength": meta["config"]["strength"],
                     "feedback": meta["config"]["feedback"], "decoding": decoding["method"],
                     "temperature": decoding.get("temperature"), "top_p": decoding.get("top_p"),
                     "seed": decoding.get("sampling_seed"), "tokens_per_second": round(row["tokens_per_second"], 1),
                     "peak_gpu_gib": round(row["peak_gpu_allocated_bytes"] / 2**30, 2) if row["peak_gpu_allocated_bytes"] else None,
                     "distinct_1": round(row["distinct_1"], 3), "distinct_2": round(row["distinct_2"], 3),
                     "loop_intervals": row["repetition_intervals"]["count"],
                     "loop_tokens": row["repetition_intervals"]["tokens"],
                     "loops_recovered": row["repetition_intervals"]["recovered"],
                     "first_loop_start": row["repetition_intervals"]["list"][0]["start"] if row["repetition_intervals"]["list"] else None,
                     "eos_suppressed": row["eos_suppressed"],
                     "first_divergence_from_clean": data["first_divergence_from_clean"].get(row["condition"]),
                     "rms_period": (row["rms_autocorrelation_peak"] or {}).get("lag"),
                     "kl_128": kl.get("128"), "kl_512": kl.get("512"), "kl_4096": kl.get("4096"),
                     "probe_integrity": row["probe_integrity"]})
(root / "SUMMARY.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
keys = ["series", "condition", "tokens", "distinct_2", "loop_tokens", "loops_recovered", "first_loop_start",
        "first_divergence_from_clean", "rms_period", "kl_128", "kl_512", "kl_4096", "tokens_per_second", "peak_gpu_gib"]
fmt = lambda v: "—" if v is None else (f"{v:.1e}" if isinstance(v, float) and 0 < abs(v) < 1e-2 else str(v))
lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
lines += ["| " + " | ".join(fmt(row[key]) for key in keys) + " |" for row in rows]
(root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
