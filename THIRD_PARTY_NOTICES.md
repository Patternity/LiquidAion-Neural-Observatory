# Third-party notices

The original LiquidAion source code is MIT-licensed (see `LICENSE`). The following bundled
or required third-party materials keep their own terms. The MIT license does not apply to
them.

| Material | Location | Rights holder / source | License | Notes |
|---|---|---|---|---|
| FlyWire connectivity subset (512 neurons) | `data/cx512/` | FlyWire Consortium; FAFB materialization 783 via the public Codex export | CC BY-NC 4.0 | Adapted: pairs aggregated over neuropils and an induced subset selected. See `data/LICENSE.md`, `data/cx512/provenance.json`, `docs/DATA.md`. |
| FlyWire coordinates and skeletons | `data/anatomy/positions.csv`, `marked_positions.csv`, `coordinates.csv.gz`, `skeletons/` | FlyWire Consortium / Codex | CC BY-NC 4.0 | Subset of the published files; hashes in `data/anatomy/manifest.json`. |
| Brain reference meshes | `data/anatomy/FLYWIRE.ply`, `FLYWIRE_whole_brain.ply` | navis-org/navis-flybrains, commit `273333c8d8bf5adeebebd274e554621462e388bd` | GPL-3.0 | Copied unchanged. The full license is in `data/anatomy/LICENSE-navis-flybrains`. |
| Recorded reservoir states and sessions | `results/`, `publication/` | Derived from the FlyWire subset | CC BY-NC 4.0 (conservatively retained) | Measurements produced by this software. |
| Qwen3-0.6B weights and tokenizer | not bundled; downloaded from Hugging Face | Alibaba Cloud / Qwen Team | Apache-2.0 | Revision `c1899de289a04d12100db370d81485cdf75e47ca`. |
| Python dependencies | installed by pip | respective authors | various (BSD, MIT, Apache-2.0, LGPL-3.0 for PySide6) | Not bundled. |

Required attribution for FlyWire data: the FlyWire Consortium, Dorkenwald et al. (2024),
Schlegel et al. (2024), and the contributors credited at https://flywire.ai/guidelines.
Because of the CC BY-NC 4.0 data, a complete copy of this repository, including `data/`, may
be used for **non-commercial purposes only**. The code alone is MIT-licensed.
