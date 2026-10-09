# Contributing

LiquidAion Neural Observatory is experimental research software. Issues and pull requests
are welcome.

## Ground rules

- **Do not fabricate data.** Neural activity, anatomy and model outputs must come from real
  computation. Never use synthetic placeholders that look like measurements.
- **Keep claims measurable.** Changes to documentation or results must state what was
  measured, on which hardware, and with which settings. Do not describe the system as
  conscious, sentient or "thinking".
- **Do not hide effects.** Do not add decoding changes such as repetition penalties or
  sampling to the scientific paths. Greedy decoding is the reproducible baseline.
- **Respect data licenses.** FlyWire-derived data are CC BY-NC 4.0 and the anatomy meshes are
  GPL-3.0. Keep their notices and provenance files with any copy.

## Development setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --no-cache-dir -r requirements-cpu.txt
pip install --no-cache-dir --no-deps -e .
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

To run tests that use the genuine Qwen3 chat template, cache the tokenizer once (about 11 MB):

```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca')"
```

To run the real-model tests (downloads 1.5 GB of weights; uses CUDA when available):

```bash
LIQUIDAION_REAL_MODEL=1 QT_QPA_PLATFORM=offscreen python -m pytest tests/test_real_model_regression.py -s
```

## Pull requests

- Keep changes focused, and match the style of the surrounding code. Comments must be in
  English.
- Add or update tests. Tests with tiny random-weight transformers check correctness only and
  must be labeled as such.
- If behavior that affects results changes, update `CHANGELOG.md` and the relevant document
  in `docs/`.
- Report hardware honestly. Do not present CPU or headless validation as GPU validation.
