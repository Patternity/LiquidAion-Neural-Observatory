# Verified CPU results

Status: `cpu_pathway_verified`. Qwen3-0.6B pretrained weights, fixed revision
`c1899de289a04d12100db370d81485cdf75e47ca`, CPU float32, four PyTorch threads.
No GPU, cloud inference, random-weight model substitute or fabricated activity
was used for these research results. Tiny randomly initialized transformers
appear only in isolated software tests and are clearly labeled there.

## Paired comparisons

Prompts 0/1/2 concern the sky, thanking a friend, and responding to criticism.
All conditions see the same baseline token prefix at every measured step.
Layer 6 observes; layer 14 steers. Configured strength is 0.25, not a guarantee
that 25% of the activation changes: the decoded direction has smaller RMS.

| Prompt | Condition | Mean hidden residual / original RMS | Logit RMS difference | Mean KL (baseline to condition) | Argmax disagreement |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | none | 0.0000% | 0.00000000 | 0 | 0.000% |
| 0 | static | 1.4138% | 0.04309216 | 0.00043897775 | 0.000% |
| 0 | random | 3.8211% | 0.10901067 | 0.003089365 | 3.125% |
| 0 | fly | 4.1527% | 0.12837034 | 0.0050553763 | 6.250% |
| 1 | none | 0.0000% | 0.00000000 | 0 | 0.000% |
| 1 | static | 1.3936% | 0.03364016 | 0.00041170544 | 0.000% |
| 1 | random | 3.3813% | 0.07910653 | 0.0023345314 | 0.000% |
| 1 | fly | 3.4848% | 0.08605853 | 0.0028298711 | 0.000% |
| 2 | none | 0.0000% | 0.00000000 | 0 | 0.000% |
| 2 | static | 1.4059% | 0.03637051 | 0.00033022849 | 0.000% |
| 2 | random | 3.1717% | 0.07050107 | 0.0015115604 | 0.000% |
| 2 | fly | 3.3256% | 0.07859974 | 0.0019706299 | 0.000% |

The disabled path and zero-strength path exactly match the uninstrumented
baseline. Repeating the authentic topology condition gives bitwise-identical
logits and recorded states in this environment. Other devices and dependency
versions may require numerical tolerances rather than bitwise equivalence.

The authentic-vs-random same-context comparison on prompt 0 has logit RMS
change 0.06913105 and mean KL
0.0018255133.
Their first-token KL is zero: from a zero recurrent state, the first input
update does not yet use graph recurrence. Subsequent graph-dependent updates
cause divergence. This timing is consistent with the implemented causal path.

Explicit post-intervention feedback on/off changes logit RMS by
0.04514460. This ablation removes the
extra layer-14 feedback update. The ordinary layer-6 observation remains
active, so it is not a completely decoupled one-way system.

## Persistence without text history

The model receives prompt B with fresh text and a fresh inference cache in
both retained and reset conditions. Only recurrent state differs. After prompt
A and ten zero-input steps (two dimensionless time units), state RMS declines
from 0.15363322 to 0.07235635.
Retaining that state rather than resetting it changes prompt-B logit RMS by
0.05498189, with mean KL
0.00054789598. Greedy argmax does not change in this
comparison. Restoring the state snapshot reproduces the retained-condition
logits bit for bit. These dynamics have fading memory, not indefinite memory.

## Free generation and quality

One of three greedy output sequences changed; two did not. The sky response
contains a factual error in both conditions (the model refers to a substance
called blue-violet light), and the baseline response reaches the 32-token cap.
Therefore changed wording is not evidence of improved quality. Thank-you and
criticism-response outputs are identical. No semantic objective was trained.

## Hardware and checks

The full experiment took 184.50 seconds, including model loading.
Peak process RSS was 4.667 GiB.
Measured per-condition throughput varies with output length and warm-up; it is
not a benchmark of topology speed. All 21 software tests passed. Actual runtime
versions and hardware reports accompany the results. The final code also
records additional startup hardware details and live-vs-steering state RMS;
these additive telemetry fields were verified separately after the full run.

## Acceptance boundaries

| Requirement | Evidence / status |
| --- | --- |
| Authentic directed fly connectivity | Bundled 512-node export, exact hashes and source IDs |
| Stateful recurrent simulation | Saved trajectories, reset/idle/snapshot tests |
| Pretrained local model inference | Genuine downloaded Qwen3 weights and generated text |
| Internal capture and intervention | Real layer hooks, measured residual and downstream logits |
| Enabled vs disabled causal comparison | Paired same-context measurements, exact zero-strength equality |
| Bidirectional feedback | Layer-6 input plus modified layer-14 feedback; measured ablation |
| Cross-prompt neural-state persistence | Fresh text/cache, retained vs reset effect, exact restore |
| Control implementations | Static and matched-budget random topology; limited scientific interpretation |
| Visualization | Real trajectory PNGs and recorded-state HTML viewer |
| Target RTX 3070 Ti execution | Done later; see `results/cuda/` and `docs/GPU_ACCEPTANCE.md` |
| Topology superiority / semantic control | Not established |

This completes the causal CPU prototype. It does not satisfy the separate
requirement to validate target GPU execution, establish an advantage of the
biological graph, or infer consciousness or subjective emotion.
