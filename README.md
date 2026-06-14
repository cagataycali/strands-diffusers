# 🎨 strands-diffusers

**The universal entrypoint to HuggingFace `diffusers` for Strands agents — 100%
pipeline & modality coverage, zero hardcoding.**

Just like [`use_aws`](https://github.com/strands-agents) wraps boto3,
[`use_lerobot`](https://github.com/cagataycali) wraps lerobot, and
[`use_transformers`](https://github.com/cagataycali/strands-transformers) wraps the
transformers task taxonomy, **`use_diffusers`** wraps the *entire* diffusers
library behind a single tool. Discover, don't hardcode: the registry is built at
runtime from `diffusers._import_structure`, so when diffusers ships a new pipeline
(say, a fresh Cosmos world-foundation model), strands-diffusers supports it
**automatically — no code change required**.

```
text / image / video / robot-state  IN
image / video / audio / ACTIONS      OUT   — natively.
```

## 🌍 Physical-AI focus: world-foundation models with action outputs

The headline use-case is **NVIDIA Cosmos** and other world-foundation models
(WFMs). A Cosmos 3 *action-policy* rollout doesn't just generate a plausible
future video — it predicts the **robot action chunk** that produces it. A single
`use_diffusers(action="run", ...)` call returns BOTH:

- a playable world **video** (`.mp4`)
- the predicted **action** chunk in model-normalized action space (`.json`,
  shape `[num_chunks, T, action_dim]`)
- (optionally) synchronized **sound** (`.wav`)

— all surfaced as artifact paths, ready to hand to a robot controller or the user.

> **Verified end-to-end** on NVIDIA Thor (diffusers `0.39.0.dev0`, `nvidia/Cosmos3-Nano`,
> bf16/cuda): one `use_diffusers(action="run", pipeline="Cosmos3OmniPipeline", ...)`
> call produced a world video `(17, 480, 640, 3)` **and** a robot action chunk
> `(1, 16, 10)` = `(num_chunks, T, action_dim)`, normalized to `[-1, 1]`.
> See [`examples/cosmos_action_policy.py`](examples/cosmos_action_policy.py) and
> [`examples/SETUP_COSMOS.md`](examples/SETUP_COSMOS.md).

## Install

```bash
pip install -e .
# optional extras:
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

# text → image
use_diffusers(
    action="run",
    pipeline="StableDiffusionPipeline",
    model="stabilityai/stable-diffusion-2-1",
    parameters={"prompt": "a robot arm in a kitchen", "num_inference_steps": 25},
)
```

## Two layers

### 1. `run` — high-level pipeline runner

Loads a pipeline class via `from_pretrained` and calls it. Inputs are coerced
(paths / URLs / base64 → PIL / video); outputs (image / video / audio / action)
are auto-saved and returned by path.

```python
use_diffusers(action="run", pipeline="WanPipeline", model="...",
              parameters={"prompt": "...", "num_frames": 81}, fps=16)
```

### 2. `call` — low-level dynamic dispatch

Resolve & call *any* diffusers class / function / method — schedulers, VAEs,
`CosmosActionCondition`, `utils.export_to_video`, or a cached pipeline's method.
`cached:key` references resolve to live objects; the `"**"` key unpacks a cached
mapping into kwargs (the `pipe(**inputs)` pattern).

```python
# Build a Cosmos action condition, cache it, then run an action-policy rollout.
use_diffusers(action="call", target="CosmosActionCondition",
              parameters={"mode": "policy", "chunk_size": 16,
                          "domain_name": "bridge_orig_lerobot",
                          "resolution_tier": 480, "video": "robot.mp4",
                          "view_point": "ego_view"},
              cache_key="act_cond")

use_diffusers(action="run", pipeline="Cosmos3OmniPipeline", model="nvidia/Cosmos3-Nano",
              parameters={"prompt": "Put the pot to the left of the purple item.",
                          "action": "cached:act_cond", "fps": 5,
                          "num_inference_steps": 30, "guidance_scale": 1.0,
                          "use_system_prompt": False},
              dtype="bfloat16", device="cuda")
# → artifacts: cosmos_world.mp4  +  action chunk .json  ([1, 16, action_dim])
```

## Discovery (the agent never guesses)

| action | what it returns |
|---|---|
| `pipelines` | all 300+ pipeline classes + derived modality |
| `models` | every model class (VAEs, transformers, controlnets) |
| `schedulers` | every scheduler class |
| `tasks` | diffusers' `AutoPipeline` task → `{family: class}` maps |
| `modalities` | pipelines grouped by modality (image / video / world / audio) |
| `wfm` | world-foundation / action-capable pipelines (Cosmos, Wan, Hunyuan) |
| `pipeline_info` | modality + `__call__` signature for one pipeline class |
| `inspect` | signature + docstring of any target |
| `cache` / `clear_cache` | manage loaded pipelines (free GPU memory) |

## Architecture

```
strands_diffusers/
├── core/
│   ├── registry.py   # zero-hardcode taxonomy from diffusers._import_structure
│   ├── engine.py     # load/cache pipelines, auto device+dtype
│   └── io.py         # coerce inputs; serialize video/image/audio/ACTION outputs
└── tools/
    └── use_diffusers.py   # the single @tool: run + call + discovery
```

## License

MIT
