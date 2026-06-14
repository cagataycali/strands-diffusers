# How it works

`use_diffusers` exposes the entire diffusers library through exactly two
execution layers — a high-level `run` and a low-level `call` — plus the
[discovery](discovery.md) actions.

<img class="sd-anim sd-anim--sm" src="assets/anim/denoise.svg"
     alt="diffusion: a field of noise resolves into a clean sample" />

## `run` — high-level pipelines

Give it a `pipeline` class name, a `model` repo, and `parameters`. It loads (and
caches) the pipeline via `from_pretrained`, coerces inputs (path / URL / base64 →
PIL / video), runs it, and serializes **every** output to an artifact path.

```python
from strands_diffusers import use_diffusers

r = use_diffusers(
    action="run",
    pipeline="StableDiffusionPipeline",
    model="stabilityai/stable-diffusion-2-1",
    parameters={"prompt": "a robot arm in a kitchen", "num_inference_steps": 25},
)
print(r["artifacts"])   # -> ['/tmp/strands_diffusers/image_*.png']
```

Outputs auto-save by modality:

| output | artifact |
|---|---|
| image | `.png` |
| video | `.mp4` (imageio fallback, gif last resort) |
| audio | `.wav` (sample rate read from the model) |
| **action** | `.json` (normalized `[-1, 1]`, full chunk + metadata) |
| 3D mesh | `.ply` / `.obj` (`.npz` lossless fallback) |

![text to image](assets/text_to_image.png){ width="256" }

## `call` — low-level dynamic dispatch

Resolve and call **any** diffusers class, function, or method: schedulers, VAEs,
`CosmosActionCondition`, `utils.export_to_video`, or a method on a cached
pipeline. This is the escape hatch that reaches everything `run` doesn't.

```python
# call a utility function
use_diffusers(action="call", target="utils.export_to_video",
              parameters={"video_frames": "cached:frames", "fps": 16})

# construct an object and stash it
use_diffusers(action="call", target="CosmosActionCondition",
              parameters={"mode": "policy", "video": "robot.mp4"},
              cache_key="cond")
```

### Cached references

`cache_key` stashes a constructed object; `cached:key` feeds it back into a later
call. `{"**": "cached:key"}` unpacks a cached mapping into kwargs. This is how the
Cosmos example builds an action condition and threads it into the pipeline run:

```python
use_diffusers(action="call", target="CosmosActionCondition",
              parameters={"mode": "policy", "video": "robot.mp4"}, cache_key="cond")

use_diffusers(action="run", pipeline="Cosmos3OmniPipeline",
              model="nvidia/Cosmos3-Nano",
              parameters={"prompt": "...", "action": "cached:cond"},
              dtype="bfloat16", device="cuda")
```

## When to use which

| you want to… | layer |
|---|---|
| generate an image/video/audio/action from a known pipeline | `run` |
| swap a scheduler, call a VAE, run a util, build a condition object | `call` |
| chain a constructed object into a later run | `call` + `cache_key` / `cached:` |
| find out what's available | [find a pipeline](discovery.md) |

## Architecture

```mermaid
flowchart TB
    Agent([Strands Agent]) --> Tool

    subgraph Tool["use_diffusers · the single @tool"]
        direction TB
        Run["run<br/><small>high-level pipelines</small>"]
        Call["call<br/><small>dynamic dispatch</small>"]
        Disco["discovery<br/><small>pipelines · wfm · inspect</small>"]
    end

    Tool --> Registry["core/registry.py<br/><small>zero-hardcode taxonomy from<br/>diffusers._import_structure</small>"]
    Tool --> Engine["core/engine.py<br/><small>load · cache · auto device+dtype</small>"]
    Tool --> IO["core/io.py<br/><small>coerce inputs · serialize<br/>video/image/audio/action/mesh</small>"]
    Tool --> Viz["core/viz.py<br/><small>render action chunks<br/>to plots + animation</small>"]

    Registry -.discovers.-> Diffusers[(HuggingFace<br/>diffusers)]
    Engine -.from_pretrained.-> Diffusers
```
