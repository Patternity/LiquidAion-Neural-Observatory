"""Write MANIFEST.sha256.json: SHA-256 of every bundled data and configuration file.

Source code is identified by the git commit; this manifest pins the research inputs.
Run from the repository root after any intentional data change.
"""
import hashlib
import json
from pathlib import Path

ROOTS = ("data", "configs")


def manifest(root: Path = Path(".")) -> dict[str, str]:
    files = sorted(p for base in ROOTS for p in (root / base).rglob("*") if p.is_file())
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


if __name__ == "__main__":
    Path("MANIFEST.sha256.json").write_text(json.dumps(manifest(), indent=2) + "\n")
