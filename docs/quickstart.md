# Quickstart

## Install

```bash
pip install -e .
pip install -e ".[video,audio]"   # mp4 export, wav I/O
```

## Use it through an agent

```python
from strands import Agent
from strands_diffusers import use_diffusers

agent = Agent(tools=[use_diffusers])
agent("Generate an image of a robot arm in a kitchen")
agent("Run a Cosmos action-policy rollout on robot.mp4 and give me the actions")
```

The agent discovers what is available (it never guesses), picks a pipeline, runs
it, and hands you back artifact paths.

## Or drive it directly

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

Outputs are auto-saved and returned by path:

![text to image](assets/text_to_image.png){ width="256" }

## Discover, don't guess

```python
use_diffusers(action="pipelines")     # all pipeline classes + modality
use_diffusers(action="modalities")    # grouped by modality
use_diffusers(action="wfm")           # world-foundation / action models
use_diffusers(action="pipeline_info", target="Cosmos3OmniPipeline")
```

## Test it

```bash
pip install -e ".[video,audio,dev]"
pytest tests/ -q          # unit tests, no GPU, no downloads
python examples/smoke.py  # E2E gate on tiny fixtures
```

## Regenerate the docs gallery

Every visual in these docs is produced by real `use_diffusers` calls:

```bash
python examples/generate_docs_assets.py   # writes docs/assets/*
```
