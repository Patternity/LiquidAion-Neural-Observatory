# Continuous generation with a sliding window ("Infinite Nekrasov")

**Question.** What happens when the language model has to forget its own distant text,
while the coupled reservoir keeps evolving across the whole generation?

The experiment generates thousands of tokens from one seed. Qwen3-0.6B sees only the
latest `window` tokens (default 128). The FlyWire reservoir is never reset, except in the
periodic-reset control. Nothing is inserted into the input during the run. The full text
goes to a log on disk and is never fed back to the model.

## Method

### Prompt policy

- **`raw`** (default). The seed is tokenized as plain text with `add_special_tokens=False`.
  No chat template, BOS or instruction is used. Every model input is exactly the most recent
  `window` token ids from seed + generated text.
- **`chat_prefix`** (option). A fixed prefix is pinned in front of the window and never
  evicted: one user turn holding the instruction «Продолжи текст.», followed by the assistant
  header. Only the open assistant text slides, so eviction cannot produce a malformed
  template. The prefix does not count toward `window`.

Both policies are recorded in each run's `metadata.json`, together with the exact prefix and
seed token ids.

### Bounded inference

At every step, the full window is recomputed with `use_cache=False` and positions
`0..len-1`. Qwen3 uses rotary position embeddings, so attention scores depend only on
relative positions. Rebasing positions to 0 is therefore equivalent to attending over the
last `window` tokens.

A truncated KV cache would **not** be equivalent. The keys and values of retained tokens
were computed while the evicted tokens were still visible, so discarded text would leak back
in. The code refuses to run if the forward pass returns a cache. Memory does not depend on
run length: no KV cache is kept, and the engine keeps only the bounded window.

### Reservoir updates

There is **one forward pass per generated token**. The layer-6 capture hook reads only the
last position, and the layer-14 steering hook modifies only the last position. Recomputing
the window therefore advances the reservoir:

- exactly once from capture, and
- once from feedback (when feedback > 0),

per token, however long the window is. This is checked on every step, with `reservoir_advances`
recorded in telemetry, and asserted in the tests.

**Difference from chat mode.** In the cached chat mode, the steered activations of earlier
tokens persist in the KV cache. Here, earlier positions are recomputed without steering on
every step. The only carriers of past intervention are therefore the generated tokens and
the reservoir state. This suits the question being asked.

### Conditions

All conditions share the seed, window, model weights, greedy decoding, EOS policy and seed 42.

| Name | Meaning |
|---|---|
| `clean` | No hooks, no reservoir. |
| `fly` | Persistent reservoir, never reset. |
| `fly_reset_N` | The reservoir is set to zero before generated tokens N, 2N, 3N, … Reset steps are recorded. |
| `random` | Degree-unmatched random directed graph with matched node count, edge count, weight multiset and self-loops (the existing control). Persistent. |

### EOS policy

- **`stop`** (normal). End-of-text (`<|im_end|>` or `<|endoftext|>`) ends the run.
- **`suppress`** (forced continuation). When EOS would be the argmax, the next best token is
  taken instead. Every such step is recorded as `eos_suppressed` in the telemetry and in the
  summary. This changes the decoding procedure. It is applied identically to all compared
  conditions.

### Checkpoint probes (identical suffix, different reservoir state)

At steps 128, 256, 512, 1024, 2048 and 4096, the run stores its current window and reservoir
state. Once the next 16 tokens exist, those same 16 tokens are scored three ways, on
**cloned** reservoirs:

- **retained** — the saved state;
- **reset** — a zero state;
- **clean** — no hooks.

Reported metrics are mean KL(reset‖retained), logit RMS difference and argmax disagreement
over the 16 forced steps.

Two integrity checks run on every probe:

- the main reservoir must be unchanged afterwards;
- the retained probe's first argmax must equal the token the main run actually generated
  (`probe_integrity`).

In `fly_reset_N`, a checkpoint that coincides with a reset has a zero retained state, so its
KL is 0 by construction.

### Telemetry on disk

Each condition directory contains:

| File | Content |
|---|---|
| `metadata.json` | Config, model revision, graph hashes, tokenization and inference policy, seed and prefix token ids |
| `telemetry.jsonl` | One row per token: step, token id, decoded fragment, window length and start, reservoir RMS and norm, ‖Δh‖/‖h‖, steering norm, feedback input norm, reservoir advances, reset and EOS-suppression flags, elapsed time |
| `text.txt` | Seed + generated text, streamed |
| `token_ids.json` | Seed and generated token ids |
| `states.npz` | Full 512-dimensional reservoir state every 64 tokens and at every checkpoint |
| `summary.json` | Stop reason, throughput, peak GPU memory, window shifts, resets, EOS events, probes |

`liquidaion_fly.continuous.load_run()` reads a directory back. `comparison.json` and the PNG
plots compare the conditions of one series.

## Reproduce

```bash
./scripts/run_continuous_experiments.sh                       # all series, ~1 h on an RTX 3070 Ti
PYTHONPATH=src .venv/bin/python -m liquidaion_fly.continuous --output results/continuous/my-run \
    --tokens 5000 --window 128 --conditions clean,fly,fly_reset,random --reset-intervals 512 \
    --strength 0.25 --feedback 0.25 --eos suppress [--prompt-policy chat_prefix]
```

In the GUI, open **EXPERIMENTS → Continuous generation · sliding window…**. Runs started
there are saved under `results/continuous/gui/`.

## Results (RTX 3070 Ti, 2026-10-10)

All runs used the seed «Однажды в студёную зимнюю пору», a 128-token window, 5000 generated
tokens (unless noted) and seed 42 projections. EOS was suppressed in all long runs; no EOS
was ever the selected token, so `eos_suppressed = 0` everywhere. The full per-run table is in
`results/continuous/SUMMARY.md` / `SUMMARY.json`. Every run directory passed an integrity
check: telemetry steps are contiguous, telemetry tokens equal `token_ids.json`, and the text
equals seed + decoded tokens.

### Engineering measurements

| Item | Measured |
|---|---|
| Peak GPU memory | **1.16 GiB** in every run, from 256 to 5000 tokens: constant, so the context is bounded |
| Throughput (alone on the GPU) | Clean 50–53 tok/s, Fly 43–48 tok/s (greedy); about 30–35 tok/s when sampling with three processes sharing the GPU |
| Model input length | Never above 128 body tokens (`max_window_seen` = 128); 4885 window shifts per 5000-token run |
| Reservoir updates | Exactly 2.0 per generated token with feedback (capture + feedback), 0 for Clean |
| Probe integrity | Retained-state probe logits equal the main run's logits bit for bit at every checkpoint; the main reservoir is unchanged by probes |

### Greedy decoding: immediate collapse

| Series | What happened |
|---|---|
| raw, strength 0.25 / 0.50 / 0.70 (feedback 0.25 / 0.25 / 0.45) | Every condition, Clean included, produced the **identical** 5000 tokens: «…в саду сидели птицы, и вдруг вдруг вдруг…» from token 9 onward. No condition ever diverged from Clean. |
| chat_prefix, 0.25 / 0.25 | Clean wrote about 100 coherent tokens, then looped «😊. 😄.». Fly diverged from Clean at token 15 and settled into a single **91-token paragraph** repeated to the end (the "creepypasta" below). |
| chat_prefix, 0.70 / 0.45 | Fly diverged at token 9 and looped from the start of the measurement. |

Same-suffix effect of retained versus zero state inside these loops: KL 1e-7 to 2e-5 per
token, argmax unchanged at every checkpoint.

### Sampling (T=1.0, top-p 0.95, three seeds)

All four conditions share the random-number stream of their seed.

| Seed | Clean d2 / loop tokens | Fly d2 / loop tokens | Fly reset 512 d2 / loop tokens | Random d2 / loop tokens | First divergence of Fly from Clean |
|---|---|---|---|---|---|
| 42 | 0.083 / 4488 | 0.237 / 3464 | 0.257 / 3464 | 0.259 / 2952 | token 2 |
| 43 | 0.376 / 1928 (2 loops, both recovered) | 0.117 / 4488 | 0.119 / 4488 | 0.212 / 2440 (1 recovered) | token 0 |
| 44 | 0.311 / 2952 | 0.197 / 3464 | 0.109 / 4488 | 0.247 / 3464 | token 1 |

`d2` is distinct bigrams over all generated tokens. Loop tokens are tokens in 512-token
blocks where more than half of the positions repeat the token one period earlier
(period ≤ 256).

At T=0.8 / top-p 0.9 (1000 tokens), Clean looped within 256–512 tokens for both seeds tested.

### Answers to the research questions

1. **Can Clean generate 5000 diverse tokens in a bounded context?** Not with greedy decoding:
   it collapses within the first few tokens. With sampling at T=1.0 it writes diverse,
   surreal text for 500–2000 tokens. In all three seeds it then falls into a token loop that
   covers 39–90% of the run. Only once (seed 43) did it leave a loop on its own.
2. **Can Fly do the same?** To the same degree. Diverse text for 500–1500 tokens, then a loop
   covering 69–90% of the run. The reservoir never stopped evolving, but it did not prevent
   or break the loops.
3. **Does Fly influence token selection?** Yes.
   - With common random numbers, Fly diverges from Clean within the first 0–2 tokens when
     sampling, and at token 15 with the chat prefix under greedy decoding.
   - Resetting the reservoir changes the tokens 2–4 steps after the first reset (tokens
     514–516).
   - The per-token effect on the distribution is small: same-suffix KL between retained and
     zero state is 6e-5 to 5e-4 before a loop forms. Inside a loop it falls to 1e-7.
4. **Differences between Clean, Fly persistent and Fly reset?** None that are consistent over
   three seeds. Each condition is best in some seed and worst in another. Persistent and
   reset runs share their text until the first reset, and then usually end in loops of the
   same kind.
5. **Collapse?** Yes, under every configuration tested. With greedy decoding it is immediate.
   With sampling it is delayed but reached in 12 of 12 five-thousand-token runs. Loops arise
   from Qwen's own continuation once the 128-token window is filled with repeated text. The
   window then contains nothing else, and the dynamics become self-sustaining.
6. **Most useful long-running neural dynamics?** Sampling at T=1.0 / top-p 0.95 gives
   500–2000 tokens of irregular, input-driven reservoir activity before lock-in. Once the
   text locks into a cycle, the reservoir **phase-locks** to it:
   - its RMS turns into a narrow periodic band whose period equals the text cycle (2, 5 or
     91 tokens);
   - after each reset it returns to the same band within a few dozen tokens
     (`publication/continuous/reservoir-dynamics.png`).

   This is a measured synchronization of the coupled system. Whether it is an attractor in
   the strict sense is untested.

### Steering-layer sensitivity (optional check)

Four contexts × 16 forced tokens, with the same recorded reservoir state and the same
relative intervention (about 4.8% of the hidden-state RMS at every layer). Output in
`results/continuous/layer-sensitivity/`.

| Steering layer | 8 | 12 | 14 (default) | 18 | 22 | 26 |
|---|---|---|---|---|---|---|
| Mean KL(Clean‖Fly) | 4.5e-4 | 7.5e-4 | 9.7e-4 | 1.4e-3 | 3.0e-3 | 6.6e-3 |
| Argmax disagreement | 1.6% | 1.6% | 0% | 3.1% | 4.7% | 3.1% |

For the same relative perturbation, later layers move the output distribution more: about 7×
at layer 26 versus 14. This is a small sample (64 steps). The default was not changed, and
two-layer steering was not tested.

### Excerpts (token offsets into the generated text)

- **Sampling seed 42, Fly, tokens 0–120:** «…мне стало лёгкое тепло от солнца и я начал носить в
  себя купюлярный воротник, потом в другой день пришла бешеная ларигова гравюра на миля…»
- **Same run, Fly, token 1400:** «…# поправка с учетом проверки безопасности…» (pseudo-code
  and specifications).
- **Same run, Fly, from token about 1650 to 5000:** «## заключение\n\n## заключение\n\n…»
- **Sampling seed 42, Clean, token 300:** «Let me count: 1-2-3-4-5-6-7-8-9-10-…». It counts to 899
  and repeats «-899-899-899» to the end.
- **chat_prefix 0.25/0.25, Fly:** one 91-token paragraph («Меня ждала тень, и я сел на берег,
  чтобы увидеть её. Тень, как тёплый дождь, проникала в моё сердце…») repeated about 50
  times; the reservoir RMS repeats with the same 91-token period.

### Limitations

- One model (0.6B), one seed text, one window size (128), three sampling seeds. Differences
  between conditions are within seed-to-seed variation.
- The 512-token blocks of the loop detector place loop onsets only to the nearest block.
- With common random numbers, Clean and Fly diverge almost immediately when sampling. After
  that, long-run comparisons are between different trajectories. The controlled measure is the
  same-suffix probe.
- The throughput figures for sampled runs were measured while several runs shared the GPU.

## Live intervention: injection, external neural cue and leak rate

These features let an operator perturb a running stream. They add **no** system prompt, chat
template, role token or instruction. The model cannot tell, from the text, that a token came
from a human.

### How injection modifies the stream

Operator text is tokenized exactly as typed, with `tokenizer(text, add_special_tokens=False)`.
Nothing is trimmed, prefixed or summarized. Note that a leading space is part of the text. The
tokens are teacher-forced into the **same** token stream that the sliding window reads.

1. The worker applies queued interventions only at a token boundary. The GUI thread never
   touches the model or the reservoir: `ContinuousRun.inject()` and `request_leak()` only put
   requests on thread-safe queues. A new injection starts only after the previous one has been
   fully inserted.
2. For each injected token, one forward pass runs over the current window, which processes
   the newest token exactly as in autonomous generation. The sampled token is then replaced by
   the injected one. No random number is drawn on forced steps.
3. After the last injected token, autonomous generation resumes. The next forward pass
   processes that token once, as it would any other. Every stream token is therefore processed
   by exactly one forward pass, with one capture update and one feedback update. Nothing is
   skipped and nothing is processed twice.
4. The reservoir is never reset by an injection. Injected tokens do not count toward
   `total_tokens`, which counts model-generated tokens.

If the run is stopped, every unfinished intervention is recorded:

| Status | Meaning |
|---|---|
| `partial_cancelled` | Insertion had started; the count of inserted tokens is kept. |
| `not_inserted_cancelled` | The injection was still queued. |
| `scheduled_injection_not_reached` | A scheduled injection whose position was never reached. |

### Provenance categories (display and telemetry only)

| Category | Definition |
|---|---|
| `autonomous` | A model token generated from a window that contains no injected token. |
| `external` | An injected token. |
| `post_injection` | A model token generated from a window (the actual model input) that contains at least one injected token. |

Window membership is tracked per position (`window_sources`). The category is decided by token
counts, not by time or by what the text means. The first model token generated after the last
injected token has been evicted is `autonomous` again. That eviction is recorded as the event
`injection_left_window`, together with its stream position and a full state snapshot.
Consecutive or overlapping injections are handled per injection id.

The GUI colors the chronological text:

- grey — autonomous;
- orange — external injection;
- blue — post-injection.

The colors never enter the model input.

### External neural cue (optional)

- **Target.** The 16 nodes of `data/cx512/nodes.csv` annotated `nt_type = DA` are selected by
  row. Their root IDs are recorded, and indices equal graph order because `load_graph` builds
  the graph in `nodes.csv` order. The control target is 16 non-DA nodes drawn with seed 42.
  The graph and its edges are unchanged.
- **Mechanism.** The cue applies while the newest window token is an injected one, that is,
  while an injected token is being processed. It adds `cue_strength` to the forcing of those
  neurons in the capture update:

  ```text
  x ← (1 − leak)·x + leak·tanh(W·x + input_strength·U·features + cue)
  ```

  It is an input to the recurrent dynamics. It never overwrites the state, and it stops
  immediately when autonomous tokens are processed again. `cue_strength = 0` and Off are
  bitwise identical. The default 0.5 is about 1.6× the measured per-neuron standard deviation of
  the linguistic forcing (0.32), so it is of the same order and does not saturate tanh.
- **What it is not.** It is not dopamine release or a reward signal. The FlyWire
  neurotransmitter labels are predictions (confidence 0.41–0.73 for these 16 nodes), and the
  model has no synaptic signs, neuromodulation or plasticity. The cue is a physically
  distinguishable, anatomically motivated input whose consequences can be measured.

### Leak rate

The leak `dt/tau` of the update equation can be set:

- at start, via `leak_rate` / `--leak`;
- live, with the slider and spin box (logarithmic, 0.005–0.5) in the Continuous Generation
  window and in the main window's **RESERVOIR DYNAMICS** panel.

A change takes effect at the next update and keeps the current state:

- **smaller** leak: slower state change and potentially longer persistence;
- **larger** leak: faster response to the current input.

Recurrence means the network's real decay is not set by this coefficient alone, so no
"memory duration" is displayed. Each change is recorded with its old value, new value, stream
position and simulation time. The reservoir fingerprint includes the dynamics, so a recorded
state is never restored under other dynamics:

- a classic session restores its own leak;
- recordings without `leak_rate` restore at 0.2;
- a state recorded under a mismatching leak is rejected.

Changing the leak in the classic GUI discards the Fly KV cache.

### Time base

| Clock | Meaning |
|---|---|
| Reservoir updates | Two per processed token (capture + feedback); one per idle tick. |
| Simulation time | Updates × `dt` (0.2). Dimensionless, not seconds. |
| Wall time | Depends on GPU speed. It never enters the dynamics: the classic idle timer fires every 250 ms, but each tick is exactly one update, so experiments do not depend on wall-clock speed. |

The selected-neuron plot shows telemetry samples, which are not uniform in time; the GUI says
so under the plot.

### Reproducing interventions

`summary.json` records each injection's text and `start_position`, and each `leak_changed`
event's position. Passing them back as `injections=((position, text), …)` and
`leak_schedule=((position, leak), …)`, or as `--inject POSITION:TEXT` and
`--leak-at POSITION:LEAK`, reproduces the run exactly.

This was verified for the two GUI acceptance sessions: 723 identical stream tokens and a
bitwise-identical final reservoir state, one of them with a live leak change. Baseline
compatibility also holds:

- with leak 0.2, cue off and no injections, the published continuous runs (greedy smoke and
  sampled seed 42) reproduce token for token;
- the golden «ты немножко муха?» test passes unchanged on CUDA and CPU.

### Result: identical injection with and without the DA cue

Setup:

- `scripts/injection_experiment.py`; output in `results/injection/da-cue-0.5/`;
- sampled stream (T=1.0, top-p 0.95, seed 42), window 128;
- « Из-за деревьев неожиданно появился огромный зелёный мамонт.» (23 tokens) inserted at
  stream position 250;
- cue strength 0.5.

Before the injection, the three runs are identical by construction.

| | cue off | DA cue | random 16 (control) |
|---|---|---|---|
| Mean signed DA-group state before → during injection | −0.13 → −0.13 | −0.13 → **+0.07** (peak +0.15) | −0.13 → −0.12 |
| DA-group state 10–20 tokens after the cue stopped | back to the cue-off course | back to the cue-off course | — |
| State difference from cue off at end of injection (L2, whole / DA group) | 0 / 0 | 1.14 / 0.96 | 1.11 / 0.07 |
| KL on identical text, state at end of injection vs cue-off state | 0 | 1.3e-5 | 1.2e-5 |
| Same, at the moment the injection left the window (position 401) | 0 | 2.3e-5 | 2.7e-5 |
| First token that differs from cue off | — | 308 | 338 |

The cue is a real, targeted input. It drives the measured DA-group activity across zero, while
the size-matched random group leaves the DA neurons almost untouched. After the external text
leaves Qwen's window, state differences remain (L2 ≈ 2.7). By then the generated text has
diverged too, so part of that difference comes from different tokens.

The **direct effect on next-token distributions on identical text** is small (KL ≈ 1–3e-5)
and the same for the DA and random groups at this strength. **No DA-specific effect on the
model output is shown.** One injection, one seed, one strength: these are observations, not
evidence of memory, understanding or learning.

The GUI acceptance run (`scripts/gui_acceptance_injection.py`, 22/22 checks) used the real
model and widgets. The DA group moved from −0.125 before to +0.097 on average during the cue.
In classic mode, the idle RMS decay factor per update was 0.952 at leak 0.2 and 0.988 at 0.02.
