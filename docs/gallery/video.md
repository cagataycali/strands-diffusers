# Video

![text to video](../assets/text_to_video.gif){ width="320" }

<sub>Real AnimateDiff-Lightning output (4 steps, SD1.5 base) — "a robot arm picking up a red cube".</sub>

<img class="sd-anim sd-anim--sm" src="../../assets/anim/m_video.svg" alt="video frames exported to mp4" />

Text to video with LTX, Wan, CogVideoX, HunyuanVideo, Mochi, and more. Output
frames are exported to `.mp4` automatically (imageio fallback, gif last resort).

```python
from strands_diffusers import use_diffusers

use_diffusers(
    action="run",
    pipeline="LTXPipeline",
    model="Lightricks/LTX-Video",
    parameters={"prompt": "a robot arm moving a cube", "num_frames": 81},
    fps=16,
)
# -> artifacts: ['/tmp/strands_diffusers/video_*.mp4']
```

The serializer normalizes whatever shape the pipeline returns - `list[PIL]`,
`[T, H, W, C]`, `[T, C, H, W]`, or batched `[B, T, H, W, C]` - into a clean mp4.

## Image to video

```python
use_diffusers(
    action="run",
    pipeline="WanImageToVideoPipeline",
    model="Wan-AI/Wan2.1-I2V-14B",
    parameters={"image": "first_frame.png", "prompt": "camera pans right",
                "num_frames": 81},
    fps=16,
)
```

## Find a video pipeline

```python
use_diffusers(action="modalities")["data"]["video"]            # architecture-named
use_diffusers(action="modalities")["data"]["image-to-video"]   # i2v transitions
use_diffusers(action="modalities")["data"]["text-to-video"]    # t2v transitions
```
