# Generation limits, context isolation and attractor diagnostics

Measured on 2026-10-09 with the real pretrained Qwen3-0.6B (revision `c1899de2…`, float16), the
authentic 512-neuron FlyWire CX graph and an RTX 3070 Ti (driver 595, PyTorch 2.8.0+cu128).
Decoding is greedy argmax throughout. No repetition penalty, sampling or prompt change was
applied anywhere.

## 1. Generation limits

| Setting | Default | Supported |
|---|---|---|
| `max_new_tokens` (one answer) | **512** (was 64) | 1–4096, and at most context − 64 in the GUI |
| `max_context_tokens` (history + message + answer) | **4096** (was 2048) | GUI 512–32768; must not exceed the model's 40960 positions; about 5k is the practical limit on an 8 GB GPU |

- Clean and Fly always use the same limit within one turn.
- Every result reports `stop_reason`: `eos`, `max_new_tokens`, `cancelled` or
  `cancelled_before_start`. It also reports `truncated`, `prompt_tokens`, `context_limit`,
  `history_dropped_messages` and `context_near_limit`.
- Generation stops on `<|im_end|>` **and** `<|endoftext|>`; the previous code ignored the second.
- The GUI marks truncated answers **TRUNCATED** and shows a context-use line. It saves both
  limits with `QSettings` and sends them to the engine when it starts.
- In the 10-turn conversation protocol below, Clean answers ran from 2 to 314 tokens, and no
  Clean answer reached 512.
- Eager attention memory grows with the square of the prompt length. Measured prefill peaks
  on this card were 3.66 GiB at 4096 tokens and 4.93 GiB at 5120. Prefills of 6144 and 8192
  tokens ran out of memory, with the desktop already using about 1.5 GB. The default of 4096
  is safe; above about 5k it is not.
- A pair of branches is atomic. If Fly fails after Clean has finished (for example, out of GPU
  memory), both histories are rolled back. The error message suggests lowering Context.

## 2. Context corruption or leakage: investigation

### What was traced

I traced every step of the request lifecycle:

1. GUI submission.
2. The per-branch `ChatSession`.
3. `apply_chat_template` with `enable_thinking=False`.
4. Tokenization.
5. Trimming.
6. The forward pass with `DynamicCache`.
7. Hook attach and detach for each forward.
8. Decoding.
9. Assistant storage.
10. Construction of the next turn.

The new opt-in debug log (`--debug-log summary|full`) records the actual tokenized input of every
generation. Tests compare that logged input with an independently rendered expectation.

### Defects found and fixed

| # | Defect | Effect | Fix |
|---|---|---|---|
| 1 | An oversized prompt was cut from the left with `ids[:, -budget:]`. | `<|im_start|>user` and the start of the latest message were silently dropped, leaving a malformed template. | The latest message is never cut. `ContextOverflowError` is raised with token counts, and history and caches are left untouched. |
| 2 | Trimming overwrote `session.messages` before generation. | History was lost permanently and silently, even when generation failed later. | Trimming affects only the model input. History is committed only after the turn finishes. Each trim is reported and recorded (`history_trimmed`). |
| 3 | A cancel during Clean meant Fly never received the user turn. | The Fly history drifted out of alignment with the GUI, so later turns showed Fly a different conversation. | Both branches record the user turn. A branch that never ran gets an empty assistant reply marked `cancelled_before_start`. |
| 4 | Only `<|im_end|>` stopped generation. | An `<|endoftext|>` from the model would run on to the limit. | Generation stops on all ids in `generation_config.eos_token_id`. |
| 5 | There was no stop reason. | A reply cut at 64 tokens could not be told apart from a finished one. | `stop_reason` and `truncated` are reported in results, telemetry, recordings, the debug log and the GUI. |
| 6 | Clean forwards did not check for hooks. | A leaked hook would have changed Clean output without any error. | A Clean (uncoupled) forward raises if any forward hook is present. |
| 7 | A Fly failure after Clean had finished left Clean's turn committed. | The histories went out of alignment. | Both histories are rolled back atomically, and OOM becomes an actionable error. |

None of these defects occurs in the multi-turn scenario that was reported, as long as nothing is
cancelled and no single message overflows the context. They were real defects, but **they do not
explain the reported repetition**.

### Evidence that each branch receives its intended context

- **Tiny-model tests with the genuine Qwen3 tokenizer and template**
  (`tests/test_context_isolation.py`, 16 tests).
  - Across the four test prompts (КВАРЦ / 7×8 / столица Франции / «Тебе не хватает токенов на
    ответ.»), each branch's logged input token ids equal the tokenization of an independently
    rendered `history + latest message`.
  - The input ends with `<|im_start|>user\n{latest}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.
  - The input contains exactly that branch's earlier replies.
  - Clean token ids are bit-identical whether Fly runs at strength 0 or at strength 1 with
    feedback 1.
  - Every Clean forward runs with zero hooks and every Fly forward with exactly two.
  - Running Clean leaves the Fly KV cache untouched.
  - Reusing the cache gives the same tokens as recomputing the whole prompt.
  - A truncated reply is followed by `<|im_end|>\n<|im_start|>user\n<new>`.
  - Trimming drops only whole oldest exchanges and keeps the session history.
- **The real model** (`tests/test_real_model_regression.py`, opt-in, passed on CUDA).
  - Each branch's logged input ends with the latest prompt and contains all of that branch's
    earlier replies.
  - Clean answers «КВАРЦ», «…56.» and «…**Париж**.».
  - With coupling disabled, Fly is token-identical to Clean on every turn.
- **The attractor diagnostic.**
  - `fly_off` (hooks attached, strength 0, feedback 0) is token-identical to Clean in all 10 turns.
  - Greedy reruns are token-identical, both for Clean and for `steer_fb_0.50`.
- **KV-cache finding.** The Qwen3 template renders historical assistant turns *without* the empty
  `<think>\n\n</think>\n\n` block. The cached prefix therefore never matches the next turn's
  prompt, and the cache is discarded and recomputed (`discard_prefix_mismatch`). That is correct,
  but it means caches are in practice never carried across turns, so there is no cross-turn cache
  state that could leak.

### What the reported symptom actually was

Before any change, I replayed the conversation with the old code. On «Тебе не хватает токенов на
ответ.», **both Clean and Fly** answered with a near-verbatim copy of their reply to the first
question.

- The dumped model input was correct: the latest message was at the end and all boundaries were
  intact.
- In a fresh conversation, the same message gets a sensible reply.

This is the small model copying from its own context in response to an ambiguous meta-message,
and the Clean branch is just as susceptible. Which branch copies depends on the exact history.
In the final run, Clean did not copy, but `steer_0.50` reproduced the reported pattern on that
same turn: «Коннектом плодовой мушки — это …».

## 3. Attractor diagnostic

`python -m liquidaion_fly.attractor` produces `attractor_diagnostic.json`, CSV tables and four
plots in `results/attractor/`. It also exposes a quick mode. The full run took 588 s on the RTX
3070 Ti.

### Conditions

All conditions start from the same zero reservoir state, seed 42 and identical prompts:

- `clean`: no hooks.
- `fly_off`: hooks attached, strength 0, feedback 0.
- `steer_s`: steering only, s ∈ {0.05, 0.10, 0.25, 0.50}.
- `steer_fb_s`: steering plus feedback 0.25, for the same values of s.
- `steer_fb_reset_s`: the reservoir is reset before every turn, s ∈ {0.25, 0.50}.
- `random_fb_s`: a degree-matched random topology, s ∈ {0.25, 0.50}.

The run uses `max_new_tokens=512` and a context setting of 8192. The longest prompt was 1475
tokens and nothing was trimmed. The default is now 4096 (see the memory note in section 1).

### Protocols

- **Conversation.** Ten unrelated prompts in one chat (connectome, modes, КВАРЦ, 7×8, Paris,
  «не хватает токенов», translation, insect legs, 2+2).
- **Sensitivity.** First-token distributions for six different probe prompts. They are taken
  after each condition's own conversation, and separately on a fresh context compared with Clean.
- **Repeated prompt / turn map.** One prompt is asked repeatedly. The history is fresh each time,
  but the reservoir persists. This makes the turn-to-turn map state → answer → state iterable.
  It is swept over 11 strengths × 2 feedback values × 2 prompts, with 8 iterations each.
- **Recovery.** The probes are asked after the `steer_fb_0.50` conversation, under each of the
  recovery variants listed in section D.

### A. Repetition (10-turn conversation)

| Condition | Turns copying an earlier reply (>0.8 char-3-gram cosine) | Probe accuracy | Truncated turns | Turns token-identical to Clean | Mean relative intervention |
|---|---|---|---|---|---|
| `clean` | 0 | 0.67 | 0 | — | 0 |
| `fly_off` | 0 | 0.67 | 0 | 10/10 | 0 |
| `steer_0.05` | 1 | 0.67 | 0 | 7/10 | 0.013 |
| `steer_0.25` | 1 | 0.67 | 0 | 6/10 | 0.065 |
| `steer_0.50` | 1 | 0.67 | 1 | 5/10 | 0.127 |
| `steer_fb_0.25` | 0 | 0.67 | 1 | 7/10 | 0.047 |
| `steer_fb_0.50` | 0 | 0.50 | 0 | 5/10 | 0.095 |
| `random_fb_0.25` | 2 | 0.67 | 0 | 7/10 | 0.043 |

The full table is in `conversation_summary.csv`. The copying episodes are all of one kind: the
reply to the related turn 1, or the meta-message on turn 6, copies the turn-0 reply. There is
**no monotonic dependence on strength**. Feedback does not increase copying, and the random
topology shows the same behavior. Mean similarity between replies to unrelated prompts stays at
0.05–0.09 everywhere, including Clean at 0.066.

### B. Sensitivity to input

- Mean pairwise first-token JS divergence between different prompts after each conversation is
  0.682–0.687 nats for every condition (Clean 0.685).
- The latest message still determines the output distribution almost entirely. There is no loss
  of input sensitivity.
- On a fresh context, the per-token effect of the intervention on the first-token distribution is
  KL(Clean‖Fly) = 7e-6 at 0.05, 4e-5 at 0.25 and 1.3e-4 at 0.50. It changed none of the six
  argmax tokens.
- Greedy decoding amplifies these tiny per-token differences into different continuations over
  tens of tokens: only 5–7 of 10 turns stay token-identical to Clean.

### C. Internal dynamics

- With identical repeated input, the reservoir alone converges geometrically to a fixed point.
  In `fly_off`, the end-state step distances are 0.17 → 0.016 → 0.0019 → 0.00027. That is the
  expected fading-memory contraction (gain 0.9), not an emergent property.
- Feedback lowers the steady reservoir RMS from about 0.25 to about 0.18 and keeps it bounded.
  There is no runaway (`dynamics.png`).
- Layer-6 activation similarity across turns is 0.77–0.79 in every condition, including `fly_off`.
  Unrelated prompts do not converge toward a shared activation pattern.
- **Turn map.** At low coupling, every orbit is a fixed point. Above about 0.25, the closed loop
  sometimes settles into a **deterministic cycle**. Fixed points still dominate overall: 29 of 44
  orbits.

| Prompt «Опиши одним абзацем, что ты видишь за окном.» | Orbit (answer lengths over 8 turns) | Regime |
|---|---|---|
| s=0.25, fb=0.25 | 38, 38, 38, … | fixed point |
| s=0.25, fb=0 | 38, 38, 77, 38, 77, 38, 77, 38 | period 2 |
| s=0.30, fb=0.25 | 38, 38, **512**, 38, 38, **512**, 38, 38 | period 3 |
| s=0.40, fb=0.25 | 46, **512**, 46, **512**, … | period 2 |
| s=0.50, fb=0.25 | 46, 89, 30, 89, 30, 89, 30, 89 | period 2 (also reproduced in 2/2 standalone runs, end-state correlation 0.844) |
| s=0.35–0.50, fb=0 | e.g. 46, 89, 90, 46, 89, 30, 89, 31 | no period within 8 turns |

The 512-token answers on the period-2 and period-3 orbits degenerate into «…зеленый зеленый
зеленый…» up to the limit. This is ordinary LLM token looping, but the reservoir state on the
cycle triggers it periodically. Cycles occur both with and without explicit feedback, because the
loop also closes through the generated tokens: reservoir → steering → tokens → layer-6 capture →
reservoir. The second prompt («Придумай короткое название…») stays at fixed points or near-fixed
alternations with equal lengths. The regime map is irregular, not a clean bifurcation in strength.

### D. Recovery (after the `steer_fb_0.50` conversation)

| Variant | Probes answered | Replies copying an earlier conversation reply |
|---|---|---|
| as is | 3/6 | 2 |
| steering off | 3/6 | 3 |
| feedback off | 3/6 | 2 |
| reservoir reset | 3/6 | 2 |
| fresh conversation (state kept) | 4/6 | 4 |
| fresh conversation + reset | 4/6 | 4 |
| **Clean model given the Fly history** | 3/6 | 3 |
| Clean with its own history | 4/6 | 2 |

The Clean model, with no hooks at all, reproduces the Fly branch's errors when it is given the Fly
history («Кварт.» instead of «КВАРЦ», «1 нога»). Turning off steering or feedback, or resetting the
reservoir, changes nothing. Only replacing the history helps. The higher copy count in the
"fresh conversation" rows is benign. The probes repeat questions that conversation had already
answered, so a correct fresh answer resembles the earlier reply.

## 4. Confirmed findings and open hypotheses

**Confirmed (measured here):**

1. The latest user message is delivered correctly to both branches every turn. Branch histories
   and KV caches are separate, and Clean inference never runs through Fly hooks. No context or
   cache leakage was found in the reported scenario.
2. Seven real context-handling defects existed and are fixed: left-truncation, history loss,
   cancel misalignment, the missing `<|endoftext|>` stop, no stop reason, no hook guard, and a
   non-atomic pair on failure. They did not cause the reported repetition.
3. Copying an earlier reply is a property of Qwen3-0.6B given a particular history. It occurs in
   Clean too. It persists in a coupled branch because the copied text is now part of that
   branch's history (recovery table), not because of the live reservoir state.
4. The intervention's direct per-token effect is small (KL ≤ 1.3e-4 at 0.50), but greedy decoding
   amplifies it into different histories. So the coupling decides *which* history a branch builds,
   and that history then drives the repetition.
5. The closed loop (reservoir, steering, tokens, capture) forms a discrete turn-to-turn map. It has
   fixed points, and at s ≥ 0.25 it also has deterministic period-2 and period-3 cycles, some of
   which periodically produce degenerate looping replies. These orbits are reproducible from
   identical initial conditions.

**Open (consistent with the data, not established):**

- Whether these cycles are attractors in the dynamical-systems sense with nontrivial basins. We
  observed convergence from one initial state (zero) per configuration over at most 8 iterations.
  Basins, stability under perturbation and longer periods are untested.
- Whether the turn-map regime depends on FlyWire topology specifically. The random-topology control
  shows similar conversation-level behavior; the turn map was swept only for the FlyWire graph.
- Whether a stronger coupling or a longer horizon produces input-insensitive collapse. Up to
  s = 0.50 and 10 turns, input sensitivity is unchanged.
- Generalization beyond two turn-map prompts, one seed and Qwen3-0.6B.

## 5. Proposed GUI integration (not implemented)

The diagnostic runs for about 10 minutes and holds the GPU. That is too long to run on the GUI
worker without a progress and cancel UI, and that UI is outside the scope of this release. The
proposal is a small addition:

1. Add an "Attractor · turn map" button in **EXPERIMENTS**. It runs `Diagnostic.turn_map` for the
   current strength and feedback and the prompt in the input box, with 8 iterations, on a temporary
   reservoir. This mirrors how demos A/B/C protect the live state. It should take about 20 s.
2. Show the regime, the answer lengths and hashes per iteration, and the step distances in the
   result panel. Stream the temporary states to the anatomy view, as demo C already does.
3. Record the result as a `turn_map` event so replay shows it. The worker path already supports
   cancellation at token boundaries.

## 6. Reproduction

```bash
# All software tests (random tiny weights + genuine Qwen3 tokenizer/template)
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
# Real-model multi-turn regression (CUDA if available)
LIQUIDAION_REAL_MODEL=1 QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_real_model_regression.py -s
# Full attractor diagnostic (~10 min on RTX 3070 Ti) or quick variant (~3 min)
PYTHONPATH=src .venv/bin/python -m liquidaion_fly.attractor --output results/attractor
PYTHONPATH=src .venv/bin/python -m liquidaion_fly.attractor --quick --output results/attractor-quick
# GUI with inference logging
./scripts/launch_observatory.sh --debug-log summary
```

## 7. Limitations

- Probe accuracy uses simple keyword matching on six questions, and text similarity is
  char-3-gram cosine. Both are coarse.
- The conversation protocol uses one prompt order and one seed. Conversation-level differences of
  one turn between conditions are within what ordinary greedy divergence produces.
- Recordings still store a telemetry snapshot and the cumulative text for every token. With
  512-token answers, a session reaches the 10,000-event cap after a few turns. The earliest events
  are then dropped (`dropped_events`), although the final histories and state are still saved.
- The KV cache is effectively never reused across turns with the Qwen3 template, so each turn
  re-prefills the whole history. This is correct, but it costs time on long contexts.
- Eager attention limits prompts on an 8 GB GPU to about 5k tokens. SDPA or flash attention would
  remove this limit, but would change numerics compared with earlier recorded results, so it was
  not switched.
