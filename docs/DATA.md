# Source evaluation and provenance

## Selected route

FlyWire FAFB connectivity materialization 783, via the publicly accessible
Codex export bucket. `classification.csv.gz` supplies IDs, CX class, flow and
side; `neurons.csv.gz` supplies neurotransmitter predictions and confidence;
`connections.csv.gz` supplies directed pre/post root IDs, neuropil, synapse
counts and neurotransmitter metadata. The exact URLs, SHA-256 hashes,
retrieval UTC time, byte sizes and graph selection statistics are recorded in
`data/cx512/provenance.json`.

The URLs were actually fetched without an account or authentication token.
The public source URL pattern is implemented by the upstream Codex project:
https://github.com/murthylab/codex
https://github.com/murthylab/codex/blob/main/codex/data/local_data_loader.py

This route was chosen instead of CAVE querying because it avoids an access
credential and large coordinate-level synapse downloads. It downloads the
compressed edge export and streams filtering locally. Smaller authenticated
queries are possible with fafbseg, but were not needed here:
https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_connectivity.html

## Alternative sources considered

The FlyWire annotation repository provides classifications, types and other
metadata; it is not itself the neuron-to-neuron connectivity export:
https://github.com/flyconnectome/flywire_annotations
It points to the Zenodo whole-brain connection/synapse deposit:
https://zenodo.org/records/10676866
We did not download the large feather or per-synapse tables. The public Codex
CSV export is sufficient for this induced graph. No inferred FlyCircuit
connectivity or synthetic graph substitutes were used in the FlyWire condition.

The ESA project studies FlyWire-derived reservoirs for time-series prediction:
https://www.esa.int/gsp/ACT/projects/fly_connectome/
The published project is prior work supporting the motivation to test a
connectome reservoir. Its task, objective and comparisons differ from our
transformer-intervention demonstration. Our sparse rate implementation is
original and does not purport to reproduce that study's results.

## Circuit choice

CX has literature supporting recurrent processing and internal-state-dependent
navigation. Hulse et al.'s CX study concerns a different connectome, hemibrain;
it motivates the circuit class, not the authenticity of our FlyWire graph:
https://doi.org/10.7554/eLife.66039

The extracted graph's strong connectivity is verified directly, not assumed
from that literature. Top-degree selection biases toward highly connected
neurons and omits many types and external pathways. There is no claim that
the 512 selected nodes constitute a complete functional subcircuit. Mushroom
body circuitry remains an alternative for future learning-focused experiments;
we did not assume it would form the best small recurrent graph.

## Release and licensing caveats

FlyWire's official guidelines identify public release 783 and CC BY-NC 4.0:
https://flywire.ai/guidelines

Codex annotations evolve independently of root-ID materialization. Its current
about page also distinguishes newer synapse predictions from originally
released ones: https://codex.flywire.ai/about_flywire
Consequently, we label this **Codex export of materialization 783 retrieved at
the recorded date**, not an exact copy of the 2024 publication's synapse table.
The current export does not embed a separate prediction-version identifier;
this is an unresolved provenance detail. Byte hashes and bundled subset keep
the executed experiment reproducible. Predictions of neurotransmitter class
are metadata, not physiology. No sign is inferred from synapse counts.

For a public paper, verify all source-specific additional annotation and newer
synapse-prediction citations. Annotation repository version >=3.2 requests
Berg, Schlegel, Tastekin, Matsliah and Dorkenwald, but our files are Codex
exports and are not assumed to equal that repository's latest tag.

## Core citations

- Dorkenwald et al. (2024), "Neuronal wiring diagram of an adult brain", Nature. https://doi.org/10.1038/s41586-024-07558-y
- Schlegel et al. (2024), "Whole-brain annotation and multi-connectome cell typing of Drosophila", Nature. https://doi.org/10.1038/s41586-024-07686-5
- Hulse et al. (2021), "A connectome of the Drosophila central complex reveals network motifs suitable for flexible navigation and context-dependent action selection", eLife. https://doi.org/10.7554/eLife.66039
- Codex resource citation: https://doi.org/10.13140/RG.2.2.35928.67844
- Qwen3 Technical Report (2025): https://arxiv.org/abs/2505.09388

References above are credited for their actual contributions, not endorsements
of this project or evidence of consciousness, emotional experience, or topology
superiority.
