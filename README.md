# strands-diffusers

<p align="center">
  <img src="docs/assets/anim/banner.svg" alt="strands-diffusers — one tool, 300+ diffusion pipelines, every modality" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/cagataycali/awesome-strands-agents"><img alt="Awesome Strands Agents" src="https://img.shields.io/badge/Awesome-Strands%20Agents-00FF77?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjkwIiBoZWlnaHQ9IjQ2MyIgdmlld0JveD0iMCAwIDI5MCA0NjMiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik05Ny4yOTAyIDUyLjc4ODRDODUuMDY3NCA0OS4xNjY3IDcyLjIyMzQgNTYuMTM4OSA2OC42MDE3IDY4LjM2MTZDNjQuOTgwMSA4MC41ODQzIDcxLjk1MjQgOTMuNDI4MyA4NC4xNzQ5IDk3LjA1MDFMMjM1LjExNyAxMzkuNzc1QzI0NS4yMjMgMTQyLjc2OSAyNDYuMzU3IDE1Ni42MjggMjM2Ljg3NCAxNjEuMjI2TDMyLjU0NiAyNjAuMjkxQy0xNC45NDM5IDI4My4zMTYgLTkuMTYxMDcgMzUyLjc0IDQxLjQ4MzUgMzY3LjU5MUwxODkuNTUxIDQxMS4wMDlMMTkwLjEyNSA0MTEuMTY5QzIwMi4xODMgNDE0LjM3NiAyMTQuNjY1IDQwNy4zOTYgMjE4LjE5NiAzOTUuMzU1QzIyMS43ODQgMzgzLjEyMiAyMTQuNzc0IDM3MC4yOTYgMjAyLjU0MSAzNjYuNzA5TDU0LjQ3MzggMzIzLjI5MUM0NC4zNDQ3IDMyMC4zMjEgNDMuMTg3OSAzMDYuNDM2IDUyLjY4NTcgMzAxLjgzMUwyNTcuMDE0IDIwMi43NjZDMzA0LjQzMiAxNzkuNzc2IDI5OC43NTggMTEwLjQ4MyAyNDguMjMzIDk1LjUxMkw5Ny4yOTAyIDUyLjc4ODRaIiBmaWxsPSIjRkZGRkZGIi8+CjxwYXRoIGQ9Ik0yNTkuMTQ3IDAuOTgxODEyQzI3MS4zODkgLTIuNTc0OTggMjg0LjE5NyA0LjQ2NTcxIDI4Ny43NTQgMTYuNzA3NEMyOTEuMzExIDI4Ljk0OTIgMjg0LjI3IDQxLjc1NyAyNzIuMDI4IDQ1LjMxMzhMNzEuMTcyNyAxMDMuNjcxQzQwLjcxNDIgMTEyLjUyMSAzNy4xOTc2IDE1NC4yNjIgNjUuNzQ1OSAxNjguMDgzTDI0MS4zNDMgMjUzLjA5M0MzMDcuODcyIDI4NS4zMDIgMjk5Ljc5NCAzODIuNTQ2IDIyOC44NjIgNDAzLjMzNkwzMC40MDQxIDQ2MS41MDJDMTguMTcwNyA0NjUuMDg4IDUuMzQ3MDggNDU4LjA3OCAxLjc2MTUzIDQ0NS44NDRDLTEuODIzOSA0MzMuNjExIDUuMTg2MzcgNDIwLjc4NyAxNy40MTk3IDQxNy4yMDJMMjE1Ljg3OCAzNTkuMDM1QzI0Ni4yNzcgMzUwLjEyNSAyNDkuNzM5IDMwOC40NDkgMjIxLjIyNiAyOTQuNjQ1TDQ1LjYyOTcgMjA5LjYzNUMtMjAuOTgzNCAxNzcuMzg2IC0xMi43NzcyIDc5Ljk4OTMgNTguMjkyOCA1OS4zNDAyTDI1OS4xNDcgMC45ODE4MTJaIiBmaWxsPSIjRkZGRkZGIi8+Cjwvc3ZnPgo=&logoColor=white&labelColor=15151a"/></a>
</p>


**The universal entrypoint to HuggingFace `diffusers` for Strands agents.**
One tool — `use_diffusers` — wraps the whole library with zero hardcoding:
discover and run any of its 300+ pipelines across every modality. It's a *visual*
library, so here's what it actually produces — every asset below is **real
model output**, not a placeholder:

<table>
  <tr>
    <td align="center" width="25%">
      <b>text → image</b><br/>
      <img src="docs/assets/text_to_image.png" width="200"/><br/>
      <sub>any of 108 image pipelines</sub>
    </td>
    <td align="center" width="25%">
      <b>text → video</b><br/>
      <img src="docs/assets/text_to_video.gif" width="200"/><br/>
      <sub>LTX · Wan · CogVideoX · Hunyuan</sub>
    </td>
    <td align="center" width="25%">
      <b>robot actions</b> 🤖<br/>
      <img src="docs/assets/cosmos_world.gif" width="200"/><br/>
      <sub>Cosmos WFM: world video + actions</sub>
    </td>
    <td align="center" width="25%">
      <b>text → audio</b><br/>
      <img src="docs/assets/text_to_audio.png" width="200"/><br/>
      <sub>StableAudio · AudioLDM2</sub>
    </td>
  </tr>
</table>

```
text / image / video / robot-state  IN
image / video / audio / actions / 3d  OUT
```

The registry is built at runtime from `diffusers._import_structure`, so new
pipelines are supported automatically with no code change. Same philosophy as
`use_aws`, `use_lerobot`, and `use_transformers`: **discover, don't hardcode.**

<table>
  <tr>
    <td align="center" width="50%">
      <b>3D mesh</b><br/>
      <img src="docs/assets/mesh_render.png" width="200"/><br/>
      <sub>ShapE - verts/faces to .ply</sub>
    </td>
    <td align="center" width="50%">
      <b>audio</b> (<a href="docs/assets/text_to_audio.wav">hear the .wav</a>)<br/>
      <img src="docs/assets/text_to_audio.png" width="300"/><br/>
      <sub>StableAudio - waveform to .wav</sub>
    </td>
  </tr>
</table>

## 100% coverage, zero hardcoding

<p align="center">
  <img src="docs/assets/modality_coverage.png" width="640"/>
</p>

Every pipeline, model, and scheduler diffusers ships is reachable through one
tool. When diffusers adds a new pipeline, `use_diffusers` exposes it immediately.

## Physical-AI: world-foundation models with action outputs

<p align="center">
  <img src="docs/assets/cosmos_world.gif" width="360" alt="Cosmos world rollout"/>
</p>

<table>
  <tr>
    <td align="center"><img src="docs/assets/rollout_policy_1.gif" width="220"/><br/><sub>"Put the pot to the left of the purple item."</sub></td>
    <td align="center"><img src="docs/assets/rollout_policy_2.gif" width="220"/><br/><sub>"Pick up the cloth and place it in the bowl."</sub></td>
    <td align="center"><img src="docs/assets/rollout_policy_4.gif" width="220"/><br/><sub>"Open the drawer and place the spoon inside."</sub></td>
  </tr>
</table>

Same robot, same first observation — **different task prompt → different imagined
world and different predicted actions.** Five real rollouts + all three Cosmos
action modes in the [WFM gallery](https://cagataycali.github.io/strands-diffusers/wfm/).


This is the headline. A Cosmos action-policy rollout predicts both a future world
**video** and the **robot action chunk** that produces it. One
`use_diffusers(action="run", ...)` returns a `.mp4` world video, a `.json` action
chunk (normalized `[-1, 1]`, shape `[num_chunks, T, action_dim]`), and optional
`.wav` sound — and you can *see* the motion:

<table>
  <tr>
    <td align="center"><b>time-series</b> (every dim, gripper highlighted)<br/><img src="docs/assets/cosmos_action_timeseries.png" width="380"/></td>
    <td align="center"><b>end-effector path</b> (dims 0–2)<br/><img src="docs/assets/cosmos_action_trajectory.png" width="300"/></td>
  </tr>
</table>

Verified end-to-end on NVIDIA Thor (`nvidia/Cosmos3-Nano`, bf16/cuda): one call
produced a world video `(17, 480, 640, 3)` and an action chunk `(1, 16, 10)`. See
[`examples/cosmos_action_policy.py`](examples/cosmos_action_policy.py).

## Install

```bash
pip install -e .
pip install -e ".[video,audio]"   # mp4 export, wav I/O
```

## Quick start

```python
from strands import Agent
from strands_diffusers import use_diffusers

agent = Agent(tools=[use_diffusers])
agent("Generate an image of a robot arm in a kitchen")
agent("Run a Cosmos action-policy rollout on robot.mp4 and give me the actions")
```

Direct:

```python
use_diffusers(action="run", pipeline="StableDiffusionPipeline",
              model="stabilityai/stable-diffusion-2-1",
              parameters={"prompt": "a robot arm in a kitchen"})
# -> {"artifacts": ["/tmp/strands_diffusers/image_*.png"]}
```

## Two layers

`run` loads a pipeline via `from_pretrained` and calls it; inputs are coerced
(path / URL / base64 to PIL / video), outputs auto-saved and returned by path.

`call` resolves and calls any diffusers class, function, or method (schedulers,
VAEs, `CosmosActionCondition`, utils). `cached:key` references resolve to live
objects; `"**"` unpacks a cached mapping into kwargs.

```python
use_diffusers(action="call", target="CosmosActionCondition",
              parameters={"mode": "policy", "video": "robot.mp4"}, cache_key="cond")
use_diffusers(action="run", pipeline="Cosmos3OmniPipeline", model="nvidia/Cosmos3-Nano",
              parameters={"prompt": "...", "action": "cached:cond"},
              dtype="bfloat16", device="cuda")
```

## Discovery

| action | returns |
|---|---|
| `pipelines` / `models` / `schedulers` | classes + derived modality |
| `tasks` / `modalities` / `wfm` | task maps / modality groups / world-foundation models |
| `pipeline_info` / `inspect` | signature + docs |
| `visualize` | action chunk to plots + animation |
| `cache` / `clear_cache` | manage loaded pipelines |

## Architecture

```
core/registry.py  zero-hardcode taxonomy from diffusers._import_structure
core/engine.py    load/cache pipelines, auto device+dtype
core/io.py        coerce inputs; serialize video/image/audio/action/mesh
core/viz.py       render robot action chunks to plots + animation
tools/use_diffusers.py  the single @tool: run + call + discovery
```

## Testing

```bash
pip install -e ".[video,audio,dev]"
pytest tests/ -q          # unit tests, no GPU, no downloads
python examples/smoke.py  # E2E gate on tiny fixtures
```

Every visual in this README and the [docs](https://cagataycali.github.io/strands-diffusers/)
is produced by real `use_diffusers` calls — regenerate them with:

```bash
python examples/generate_docs_assets.py
```

## Docs

📖 **[cagataycali.github.io/strands-diffusers](https://cagataycali.github.io/strands-diffusers/)**
— quickstart, full gallery (images / video / audio / actions / 3D), the
world-foundation-model story, discovery, and the two-layer design.

MIT
