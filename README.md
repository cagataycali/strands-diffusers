# strands-diffusers

The universal entrypoint to HuggingFace `diffusers` for Strands agents. One tool,
`use_diffusers`, wraps the entire library with zero hardcoding: discover and run
any of its 300+ pipelines across every modality.

```
text / image / video / robot-state  IN
image / video / audio / actions / 3d  OUT
```

The registry is built at runtime from `diffusers._import_structure`, so new
pipelines (e.g. a fresh Cosmos world-foundation model) are supported automatically
with no code change. Same philosophy as `use_aws` (boto3), `use_lerobot` (lerobot),
and `use_transformers` (the transformers task taxonomy): discover, don't hardcode.

## Physical-AI focus: world-foundation models with action outputs

The headline use-case is NVIDIA Cosmos and other world-foundation models (WFMs).
A Cosmos 3 action-policy rollout predicts both a future world **video** and the
**robot action chunk** that produces it. One `use_diffusers(action="run", ...)`
returns:

- a playable world video (`.mp4`)
- the predicted action chunk, normalized to `[-1, 1]` (`.json`, shape
  `[num_chunks, T, action_dim]`)
- optional synchronized sound (`.wav`)

Verified end-to-end on NVIDIA Thor (`nvidia/Cosmos3-Nano`, bf16/cuda): one call
produced a world video `(17, 480, 640, 3)` and an action chunk `(1, 16, 10)`.
See [`examples/cosmos_action_policy.py`](examples/cosmos_action_policy.py).

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

Or drive it directly:

```python
from strands_diffusers import use_diffusers

use_diffusers(action="run", pipeline="StableDiffusionPipeline",
              model="stabilityai/stable-diffusion-2-1",
              parameters={"prompt": "a robot arm in a kitchen",
                          "num_inference_steps": 25})
```

## Two layers

**`run`** loads a pipeline class via `from_pretrained` and calls it. Inputs are
coerced (paths / URLs / base64 to PIL / video); outputs (image / video / audio /
action / mesh) are auto-saved and returned by path.

**`call`** resolves and calls any diffusers class, function, or method:
schedulers, VAEs, `CosmosActionCondition`, `utils.export_to_video`, or a cached
pipeline's method. `cached:key` references resolve to live objects; the `"**"`
key unpacks a cached mapping into kwargs.

```python
# Build a Cosmos action condition, cache it, then run an action-policy rollout.
use_diffusers(action="call", target="CosmosActionCondition",
              parameters={"mode": "policy", "chunk_size": 16,
                          "domain_name": "bridge_orig_lerobot",
                          "video": "robot.mp4"}, cache_key="cond")

use_diffusers(action="run", pipeline="Cosmos3OmniPipeline", model="nvidia/Cosmos3-Nano",
              parameters={"prompt": "Put the pot to the left of the cup.",
                          "action": "cached:cond", "num_inference_steps": 30},
              dtype="bfloat16", device="cuda")
```

## Discovery

| action | returns |
|---|---|
| `pipelines` | all pipeline classes + derived modality |
| `models` / `schedulers` | every model / scheduler class |
| `tasks` | AutoPipeline task to class maps |
| `modalities` | pipelines grouped by modality |
| `wfm` | world-foundation / action-capable pipelines |
| `pipeline_info` / `inspect` | signature + docs of a pipeline / anything |
| `visualize` | render an action chunk to plots + animation |
| `cache` / `clear_cache` | manage loaded pipelines |

## Architecture

```
strands_diffusers/
├── core/
│   ├── registry.py   # zero-hardcode taxonomy from diffusers._import_structure
│   ├── engine.py     # load/cache pipelines, auto device+dtype
│   ├── io.py         # coerce inputs; serialize video/image/audio/action/mesh
│   └── viz.py        # render robot action chunks to plots + animation
└── tools/
    └── use_diffusers.py   # the single @tool: run + call + discovery
```

## Testing

```bash
pip install -e ".[video,audio,dev]"
pytest tests/ -q          # unit tests, no GPU, no model downloads
python examples/smoke.py  # E2E gate on tiny fixtures
```

## License

MIT
