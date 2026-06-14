# Images

![text to image](../assets/text_to_image.png){ width="320" }

<sub>Real SD-Turbo output (4 steps) — "a robot arm at a sunlit kitchen counter".</sub>

<img class="sd-anim sd-anim--sm" style="max-width:280px" src="../../assets/anim/m_image.svg" alt="an image tile resolving from noise" />

Text to image with any of the 100+ image pipelines (Stable Diffusion, SDXL, Flux,
Kandinsky, PixArt, Sana, ...).

```python
from strands_diffusers import use_diffusers

use_diffusers(
    action="run",
    pipeline="StableDiffusionPipeline",
    model="stabilityai/stable-diffusion-2-1",
    parameters={"prompt": "a robot arm in a kitchen", "num_inference_steps": 25},
)
# -> artifacts: ['/tmp/strands_diffusers/image_*.png']
```

## Image to image

Pass an input image (path, URL, or base64) - it is coerced to PIL automatically.

```python
use_diffusers(
    action="run",
    pipeline="StableDiffusionImg2ImgPipeline",
    model="stabilityai/stable-diffusion-2-1",
    parameters={"prompt": "make it watercolor", "image": "input.png",
                "strength": 0.6},
)
```

## Inpainting

```python
use_diffusers(
    action="run",
    pipeline="StableDiffusionInpaintPipeline",
    model="runwayml/stable-diffusion-inpainting",
    parameters={"prompt": "a cat", "image": "photo.png", "mask_image": "mask.png"},
)
```

## Batched generation

`num_images_per_prompt > 1` saves every image with a collision-free name.

```python
use_diffusers(action="run", pipeline="StableDiffusionPipeline", model="...",
              parameters={"prompt": "robot", "num_images_per_prompt": 4})
# -> 4 distinct .png artifacts
```

## Find an image pipeline

```python
use_diffusers(action="modalities")["data"]["image"]   # 100+ classes
```
