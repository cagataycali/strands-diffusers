# 3D Mesh

<p align="center">
  <img src="../../assets/mesh_render.png" width="300" alt="generated 3D mesh"/>
</p>

Mesh generation with ShapE and other 3D pipelines. Mesh outputs (`verts` +
`faces`) are exported to `.ply` / `.obj` (or `.npz` as a lossless fallback) - never
serialized to an opaque repr string.

<img class="sd-anim sd-anim--sm" style="max-width:240px" src="../../assets/anim/m_mesh.svg" alt="a rotating 3D wireframe mesh" />

```python
from strands_diffusers import use_diffusers

use_diffusers(
    action="run",
    pipeline="ShapEPipeline",
    model="openai/shap-e",
    parameters={"prompt": "a shark", "guidance_scale": 15.0},
)
# -> artifacts: ['/tmp/strands_diffusers/mesh_*.ply']
```

The serialized result carries the geometry summary:

```json
{ "type": "mesh", "num_verts": 12000, "num_faces": 24000,
  "path": "/tmp/strands_diffusers/mesh_*.ply" }
```

The render above is a real `verts`/`faces` mesh passed straight through the same
serializer the pipelines use - the geometry round-trips to disk losslessly, then
matplotlib draws the triangles shaded by surface normal.

## Find a 3D pipeline

```python
use_diffusers(action="modalities")["data"]["3d"]   # ['ShapEPipeline', ...]
```
