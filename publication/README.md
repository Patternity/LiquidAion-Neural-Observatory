# Publication assets

All materials here come from real runs of the application: the pretrained Qwen3-0.6B
(revision `c1899de2…`), an RTX 3070 Ti, and greedy decoding. Every directory includes the
exact parameters (`capture.json`) and the full recording (`session.json`). A recording can
be replayed with `./scripts/launch_observatory.sh --replay <dir>/session.json`. Nothing was
drawn or edited by hand.

| File | Content |
|---|---|
| `hero/dashboard.png` | Full application window (1920×1200): two-turn conversation, default coupling (steering 0.25, feedback 0.25). |
| `hero/anatomy-state.png` | Anatomy view at the end of the hero session. |
| `hero/activity.gif` | Recorded Fly-branch reservoir states, followed by idle decay, replayed through the anatomy view. Every frame is labeled with its token index and state RMS. |
| `habr-fly-question/` | The "are you a fly?" exchange at steering 0.7, feedback 0.45, from a zeroed reservoir: window screenshot, anatomy, animation, recording. |
| `habr-fly-question/original-session-excerpt.json` | Excerpt of the owner's original live session in which the exchange was first seen, published with the owner's approval. Messages, settings and final state are included; per-token telemetry is omitted. |
| `architecture.svg` / `architecture.png` | Simplified architecture diagram. |

To recapture:
`QT_QPA_PLATFORM=offscreen PYTHONPATH=src .venv/bin/python scripts/capture_publication_assets.py --output publication/hero --prompt "…" [--strength S --feedback F]`.

## Experiments in one paragraph each

- **A, coupling (Clean vs Fly).** Clean generates a continuation. Fly is then forced through
  exactly the same tokens, starting from the same reservoir state. The difference between
  their next-token distributions measures the intervention, with no confound from different
  text.
- **B, topology (FlyWire vs random).** The same procedure, but with a random directed graph
  that matches node count, edge count, raw weights and self-loops. This tests whether the
  specific biological wiring matters. In this release, effects are of the same order for
  both graphs.
- **C, memory (persistent vs reset).** One prompt drives the reservoir; ten idle steps
  follow. A second prompt is then evaluated in a fresh text context, once with the retained
  state and once with a zeroed state, on identical tokens. The difference is memory carried
  only by the network state.

## Controlled results (RTX 3070 Ti, default coupling, `results/cuda/summary.json`)

| Prompt 0, 32 tokens | ‖Δh‖/‖h‖ at layer 14 | Mean KL | Argmax disagreement |
|---|---:|---:|---:|
| hooks, strength 0 | 0 | 0 | 0% |
| static state | 1.41% | 4.8e-4 | 0% |
| random topology | 3.82% | 3.2e-3 | 3.1% |
| FlyWire | 4.15% | 5.1e-3 | 6.3% |

Other results:

- **C, memory.** The retained state changed the logits of prompt B (KL 7.1e-4). Restoring a
  snapshot reproduced the logits bitwise.
- **Repeat run.** The whole experiment repeated bitwise identically.
- **Attractor diagnostic.** See `docs/CONTEXT_AND_ATTRACTOR.md`.

## Examples of Clean vs Fly divergence

1. Hero session, turn 2 (default coupling):
   - Clean: "The sun rises over the summer morning, casting golden light across the fields,
     illuminating the vibrant colors of the **plants and the surrounding environment**."
   - Fly: "…illuminating the vibrant colors of the **flowers and the leaves**."
2. Paired CLI run, prompt 0 (default coupling, 32-token limit):
   - Clean: "…scattered by the Earth's atmosphere in the form of **Rayleigh scattering**"
   - Fly: "…scattered **more efficiently** by the Earth's atmosphere than other colors."
3. The "are you a fly?" exchange (steering 0.7, feedback 0.45):
   - Clean: «Привет! Я не муха, а человек. …»
   - Fly: «Привет! Меня зовут, например, Муха. А ещё я — это муха. …»

These are illustrations. Different text is not evidence of understanding, identity or
improved ability. The controlled measure is the same-prefix logit difference.

## Suggested figures for an article

1. `hero/dashboard.png`: what the application looks like.
2. `architecture.png`: how the coupling works.
3. `hero/activity.gif`: the network's actual activity during generation.
4. The controlled results table above, with the caveat about topology.
5. `habr-fly-question/dashboard.png`: the anecdote, labeled with its settings.
6. `results/attractor/turn_map.png`: fixed points and cycles of the closed loop.

Licensing: these images and recordings contain states derived from FlyWire data
(CC BY-NC 4.0). Credit the FlyWire Consortium when reproducing them.
