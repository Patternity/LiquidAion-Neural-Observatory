"""Bundled research inputs must match the published hash manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_data_and_configs_match_manifest():
    manifest = json.loads((ROOT / "MANIFEST.sha256.json").read_text())
    assert any(path.startswith("data/cx512/") for path in manifest)
    for path, expected in manifest.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    on_disk = {p.relative_to(ROOT).as_posix() for base in ("data", "configs")
               for p in (ROOT / base).rglob("*") if p.is_file()}
    assert on_disk == set(manifest), "Unlisted or missing data/config files"
