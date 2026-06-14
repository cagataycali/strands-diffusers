# strands-diffusers examples

All examples import the real `use_diffusers` tool and run **real diffusion
inference** (no mocks). The image/video examples use tiny HF test fixtures so they
run fast on any machine — swap `model` for a full checkpoint to get real quality.

| example | what it shows | path used | model |
|---|---|---|---|
| `text_to_image.py` | text to image, image to png artifact | `run` | tiny-stable-diffusion-pipe |
| `text_to_video.py` | text to video, video to mp4 artifact | `run` | tiny-random-ltx-video |
| `text_to_audio.py` | audio generation, audio to wav artifact | `run` | tiny DanceDiffusion (built in-process) |
| `cosmos_action_policy.py` | WFM action-policy: video + robot action out | `call` + `run` | nvidia/Cosmos3-Nano |
| `smoke.py` | E2E gate (discovery + img + video + audio + action) | all | tiny fixtures |

## run vs call

- **`run`** — high-level. Give it a `pipeline` class name + `model` repo +
  `parameters`. It loads (and caches) the pipeline, coerces inputs, runs it, and
  serializes every output (image/video/audio/**action**) to an artifact path.

- **`call`** — low-level dynamic dispatch. Resolve & call *any* diffusers class,
  function, or method: schedulers, VAEs, `CosmosActionCondition`,
  `utils.export_to_video`, or a cached pipeline's method. Use `cache_key` to stash
  a constructed object and `cached:key` (or `{"**": "cached:key"}`) to feed it
  back into a later call. This is how the Cosmos example builds an action
  condition and threads it into the pipeline run.

## The action payload (why this library exists)

World-foundation models like NVIDIA Cosmos 3 emit a `Cosmos3OmniPipelineOutput`
with `video`, optional `sound`, and **`action`** (a `list[torch.Tensor]`, each a
normalized action chunk `[T, action_dim]`). `core/io.py` serializes:

- `video`  → `.mp4`  (via `diffusers.utils.export_to_video`, imageio fallback, gif last resort)
- `sound`  → `.wav`  (soundfile or stdlib `wave`)
- `action` → `.json` (full nested list + `chunk_shape` / `num_chunks` metadata)

So one `use_diffusers(action="run", ...)` hands the agent both a playable world and
a robot-ready action vector.

## Cosmos3OmniPipeline availability

`Cosmos3OmniPipeline` ships in **diffusers from source** (>0.38). `use_diffusers`
resolves pipeline classes dynamically, so the moment your diffusers has it, the
example works unchanged:

```bash
pip install 'git+https://github.com/huggingface/diffusers'
python examples/cosmos_action_policy.py
```

On older diffusers the example degrades gracefully and still lists the
action-capable WFM pipelines available now (Cosmos2*, CosmosVideoToWorld, Wan, …).
