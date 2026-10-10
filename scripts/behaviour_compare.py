"""Exact comparison of two behaviour captures (scripts/behaviour_capture.py); wall-clock fields are ignored."""
import json
import sys

IGNORED = {"elapsed", "seconds", "tokens_per_second", "generate_pair_256_tokens_s", "continuous_fly_tok_s",
           "timestamp_utc"}


def diff(x, y, path, out):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k in IGNORED:
                continue
            if k not in x or k not in y:
                out.append(f"{path}/{k}: only in {'reference' if k in x else 'migrated'}")
                continue
            diff(x[k], y[k], f"{path}/{k}", out)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            out.append(f"{path}: length {len(x)} vs {len(y)}")
            return
        for i, (u, v) in enumerate(zip(x, y)):
            diff(u, v, f"{path}[{i}]", out)
    elif x != y and not (x != x and y != y):
        out.append(f"{path}: {x!r} vs {y!r}")


a, b = (json.load(open(p)) for p in sys.argv[1:3])
out = []
diff(a, b, "", out)
print("IDENTICAL (exact equality, timing ignored)" if not out else f"{len(out)} differences")
for line in out[:30]:
    print(line)
