# Anatomical reference and exact neuron mapping

The desktop reference uses **real public FlyWire geometry**. All spatial arrays
remain in **FlyWire / FAFB14.1 nanometers**, without centering, projection,
mirroring, random placement, a force-directed layout, or nonrigid warping.
Camera motion and common display scaling do not change anatomical relationships.

## Neuron markers

`data/anatomy/positions.csv` has one entry for every root ID in
`data/cx512/nodes.csv`, in exactly the same order. It uses the first published
community-marked coordinate for that root in the Codex materialization-783
`coordinates.csv.gz` export. Root IDs and supervoxel IDs are stored as decimal
strings, never lossy floating-point numbers. Additional source positions are
retained in `marked_positions.csv`; the marker is **not a cell-body centroid**.
A missing authentic coordinate stops acquisition rather than inventing a point.

Source URL:
https://storage.googleapis.com/flywire-data/codex/data/fafb/783/coordinates.csv.gz

The public Codex loader describes these as community-marked positions and calls
`nanometer_to_flywire_coordinates(pos)` when passing them into Neuroglancer.
Thus the CSV is already in nanometers; the viewer's 4 x 4 x 40 nm voxel display
must not be applied again to these coordinates.

Primary source:
https://github.com/murthylab/codex/blob/3477b6cfccaf3bf976e6af0d40513ad221311a36/codex/data/neuron_data_initializer.py

## Compatible brain references

The pinned upstream navis-flybrains commit is
`273333c8d8bf5adeebebd274e554621462e388bd`.

- `FLYWIRE.ply` is a neuropil envelope based on automatically detected synapse
  density, 2 micrometer bins, marching cubes, and upstream cleaning, smoothing
  and downsampling. It is an anatomical reference envelope, not a neuron mesh.
- `FLYWIRE_whole_brain.ply` is a tissue envelope generated from the Google FAFB
  EM tissue mask, followed by upstream processing and transformation.

Both are **already transformed from FAFB14 into FlyWire space** upstream.
The upstream FLYWIRE template metadata specifies nanometers. We therefore apply
an identity transform and use the `FLYWIRE` meshes, **not untransformed FAFB14**.
The registered tissue geometry has 48,995 vertices and 98,402 faces; the neuropil
geometry has 25,047 vertices and 50,416 faces. Files are bundled byte-for-byte,
not generated locally. The raw geometry and positions share the same origin,
axis ordering, units and registration.

Primary sources:
https://github.com/navis-org/navis-flybrains/blob/273333c8d8bf5adeebebd274e554621462e388bd/flybrains/templates.py
https://github.com/navis-org/navis-flybrains/blob/273333c8d8bf5adeebebd274e554621462e388bd/flybrains/data/template_meta.json
https://github.com/navis-org/navis-flybrains/blob/273333c8d8bf5adeebebd274e554621462e388bd/flybrains/meshes/FLYWIRE_whole_brain.md

## Actual skeleton context

The first 16 graph IDs, a deterministic context subset rather than a whole-brain
reconstruction, are requested from the public materialization-783 skeleton
server. Successful files retain every original vertex, edge and radius in
compressed NPZ. This is a **lossless format conversion**, with no simplification,
repositioning or inferred branches. The input Neuroglancer format has an
identity transform and nanometer units, as declared in fafbseg's source.
Optional failures are recorded in the manifest; they never create substitute
skeletons. All 512 graph neuron markers still require genuine coordinates.

Public URL pattern:
https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783/ROOT_ID

Primary source:
https://github.com/navis-org/fafbseg-py/blob/d0da95123ee606e204ae2c702e7bc78538646fbd/fafbseg/flywire/skeletonize.py

`Anatomy.read_skeleton(root_id)` returns `(vertices, edges)` with original node
indices. Each NPZ additionally has `radius` in nm. Original download SHA-256,
cache SHA-256, vertex counts and edge counts are recorded per root.

## API and acquisition

```python
from pathlib import Path
from liquidaion_fly.anatomy import Anatomy, ensure_anatomy

anatomy = ensure_anatomy(Path("data/cx512"), Path("data/anatomy"))
assert anatomy.units == "nm"
assert anatomy.coordinate_space == "FLYWIRE/FAFB14.1"
# Mapping is exactly the graph node order.
print(anatomy.neuron_ids[0], anatomy.positions[0])
# PyVista supports these original PLY files directly.
print(anatomy.meshes["whole_brain"], anatomy.meshes["neuropil"])
vertices, edges = anatomy.read_skeleton(anatomy.neuron_ids[0])
```

Offline loading is `Anatomy.load(path)` or `load_anatomy(path)`. Both verify all
geometry cache checksums and the stored index mapping. `ensure_anatomy` also
checks the current `nodes.csv` hash and exact neuron order. Public downloads
need no token or account. Cache acquisition is atomic and never treats a partial
file as a valid result.

```bash
python -m liquidaion_fly.anatomy --graph data/cx512 --cache data/anatomy
```

The manifest records exact URL, pinned mesh commit, retrieval UTC, source
hashes, cache hashes, units, registration, graph hash, geometry semantics and
optional skeleton failures. AABBs are useful sanity checks, not proof of spatial
registration or that a point is inside the neuropil. Registration is supported
by the primary sources above. Marker connections are abstract graph edges
between marked locations; they are not reconstructed axonal trajectories.

## Attribution and restrictions

FlyWire data are CC BY-NC 4.0, with attribution and noncommercial restrictions:
https://flywire.ai/guidelines
https://creativecommons.org/licenses/by-nc/4.0/

navis-flybrains distributes the reference meshes in its GPL-3.0 repository.
The original GPL license is retained as `data/anatomy/LICENSE-navis-flybrains`.
This project does not relicense the downloaded reference files as its own code.
The FlyWire restriction continues to apply to biological data and derivatives.

- Zheng et al. (2018), Cell, DOI: 10.1016/j.cell.2018.06.019.
- Dorkenwald et al. (2022), Nature Methods, DOI: 10.1038/s41592-021-01330-0.
- Dorkenwald et al. (2024), Nature, DOI: 10.1038/s41586-024-07558-y.
- Schlegel et al. (2024), Nature, DOI: 10.1038/s41586-024-07686-5.
- Buhmann et al., automatic synaptic partner detection,
  DOI: 10.1101/2019.12.12.874172, for the reference neuropil mesh construction.

The animation colors are computed reservoir activity, not recorded biological
activity or validated electrophysiology. Anatomical authenticity does not
establish faithful physiological simulation.
