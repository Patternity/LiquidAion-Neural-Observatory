# LiquidAion Fly

A minimal, real coupling between a frozen local Qwen3-0.6B transformer and a sparse recurrent network derived from FlyWire connectivity. This repository contains the exact 512-neuron subset, source hashes, code, tests and recorded CPU experiments.

## What is biological, and what is engineered?

The directed neuron-to-neuron edges and synapse counts are biological reconstruction data. The dynamics, projections, time constants and interfaces are engineered. Neurotransmitter predictions are retained as metadata but are not converted into excitation/inhibition signs. This is a connectome-derived rate reservoir, not a faithful fly-brain simulation.

We select CX-annotated neurons, take the largest strongly connected component, rank by total within-component synapse count, induce the top 512 nodes, and retain the largest resulting strongly connected component. The packaged graph has 512 neurons, 16,731 directed edges and 310,999 synapses. All retained edges come from the downloaded source. External connections are omitted; this truncation changes the circuit.

The central complex is a defensible starting circuit because recurrent computation and internal-state-dependent behavior have been studied there. This does not establish that our selected subset preserves those functions, or that its topology is better for language-model control. See [data provenance](DATA.md).

## Verified CPU setup

Python 3.12 is required. From the extracted repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
python -m liquidaion_fly.cli inspect
python -m pytest -q
python -m liquidaion_fly.cli run --config configs/cpu.json --output results/reproduction
python -m liquidaion_fly.cli visualize --results results/reproduction
```

The model is downloaded from Hugging Face at the revision fixed in the configuration. Approximately 1.5 GB of pretrained model weights must be downloaded; CPU float32 model parameters use more memory. The recorded environment, process peak RAM and timings are in `results/cpu/summary.json`. The initial environment had about 9.7 GiB RAM and no NVIDIA GPU.

You can also use an existing local model directory:

```bash
python -m liquidaion_fly.cli run --model-path /absolute/path/to/qwen3-0.6b
```

The local directory must contain the pretrained Qwen3 tokenizer and weights. The recorded run used exactly the revision in `configs/cpu.json`; local-directory contents are additionally documented in `docs/MODEL.json`.

For NVIDIA execution, create an environment with a CUDA-enabled PyTorch 2.8 build suitable for your driver, then use `configs/cuda.json`. Start with `inspect`: it reports the actual card, free/total VRAM, driver and PyTorch CUDA support. RTX 3070 Ti execution has **not** been validated in this environment. An explicit CUDA request fails if CUDA is unavailable; there is no silent fallback or mock inference.

## Reproduce the source extraction

The authentic subset is bundled, so full-source acquisition is optional:

```bash
python -m liquidaion_fly.cli prepare --raw data/raw --output data/reextracted --neurons 512
```

This downloads three public compressed CSV exports, totaling about 53 MB. It streams the connectivity table and only retains CX-to-CX rows. It does not download EM images, morphology, or the full synapse-coordinate dataset. This route still reads the whole compressed neuron-edge export; it is not a server-side subset query. Authentication was not required for these public objects at retrieval time.

Codex exports and annotations can change. Check their recorded SHA-256 hashes against `data/cx512/provenance.json` before calling a new extraction an exact reproduction. A materialization number alone is not sufficient to freeze current Codex annotations or synapse prediction updates.

## Dynamics and intervention

For state `x`, compressed input `u` and row-normalized directed connectivity `W`:

```python
x_next = (1 - dt / tau) * x + (dt / tau) * tanh(W @ x + input_strength * input_matrix @ u)
```

`W[post, pre]` transports presynaptic activity to a postsynaptic node. Absolute row sums are bounded by gain 0.9. With `0 < dt/tau <= 1`, tanh keeps states bounded. These settings also make zero-input dynamics contractive: memory decays, rather than persisting forever or forming a biological attractor. Simulation time is dimensionless; it is not calibrated to physiological seconds.

Layer indices are zero-based. The last token's layer-6 output is RMS-normalized and projected into 64 tanh features. One reservoir step receives those features. The current rate state passes through two seeded fixed projections to a layer-14 residual direction. Its RMS is capped at one. The injected residual is scaled by the original token's activation RMS and configured strength. Only the last token is modified during prefill; each subsequent cached pass modifies its single new token.

After intervention, the layer-14 activation is encoded and supplies an additional reservoir update with configurable feedback strength. It affects future inference steps. Every prompt starts a fresh LLM text context and KV cache; reservoir state only resets when explicitly requested. `Reservoir.advance(steps=...)` advances zero-input idle dynamics without running the LLM. `snapshot`, `restore` and `reset` are independent of textual context. Snapshot identity includes graph, input projection and dynamics.

No LLM weights or adapters are trained in this milestone. The fixed projections demonstrate coupling; they do not provide a validated semantic steering direction.

## Recorded experiments

`run` executes three prompts with deterministic greedy decoding and a fixed seed:

- Baseline pretrained model, without hooks.
- Disabled intervention, with observation and dynamics active.
- Static steering, freezing the first observed reservoir state.
- Random directed topology with matching node count, edge count, raw weight multiset and self-loop count.
- Authentic FlyWire topology.
- Repeated FlyWire run, zero-strength check and explicit post-intervention feedback ablation.
- Prompt A, zero-input idle interval, then prompt B with retained vs reset reservoir state; restoring the snapshot repeats the retained-state condition.

Each comparison uses the **same baseline token stream** under teacher forcing. KL divergence and logit RMS differences are measured on identical contexts. Separate free-generation examples are descriptive and may diverge in tokens; their logits are not used as paired measurements.

The JSON traces record layer, token position, prefill/decode sequence length, activation RMS, actual injected residual RMS, compressed features and simulation time. NPZ files contain actual per-step reservoir states and neuron IDs. Full vocabulary logits are compared during execution, then represented by hashes and metrics rather than dumped by default.

Control limitations matter: randomization changes degrees and SCC structure; row normalization alters its effective weight distribution. Static steering has a different evolving residual magnitude from dynamic steering. One seed and three prompts cannot establish topology superiority, coherence gains, category separation or emotion. There is no task-calibrated adapter or blinded output-quality evaluation.

## Visualization and results

Open `results/cpu/dashboard.html` with its companion PNGs in the same folder. It replays actual saved states and switches recorded conditions. Its slider selects a recorded inference step; it does not apply live steering. The heatmap's neuron order and PCA geometry have no biological spatial interpretation.

See `docs/RESULTS.md` for verified numerical findings. The CPU causal-pathway milestone can be verified here. The RTX 3070 Ti results are in `docs/GPU_ACCEPTANCE.md`. The project makes no consciousness or subjective-emotion claims.

## Licenses and attribution

Original implementation: MIT, see `LICENSE`. FlyWire-derived data and corresponding redistributed state artifacts are subject to the source CC BY-NC 4.0 terms, see `data/LICENSE.md`. This distinction is not a relicensing of biological data. Qwen3 pretrained weights are Apache 2.0 and are downloaded separately. Credit Dorkenwald et al. (2024), Schlegel et al. (2024), relevant additional annotation/synapse contributors and Codex as detailed in `docs/DATA.md`. Check source-specific terms before commercial reuse.
