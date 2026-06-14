"""Text-to-video via use_diffusers (run path) — real diffusion, E2E.

Exercises the video→mp4 output serializer with a tiny LTX-Video fixture. Swap
`pipeline`/`model` for full models (WanPipeline, CogVideoXPipeline,
HunyuanVideoPipeline, ...) — discovery via use_diffusers(action="wfm"/"modalities").

Run directly:   python examples/text_to_video.py
"""

from strands_diffusers import use_diffusers

MODEL = "katuni4ka/tiny-random-ltx-video"


def generate(prompt: str = "a robot arm moving a cube"):
    return use_diffusers(
        action="run",
        pipeline="LTXPipeline",
        model=MODEL,
        parameters={
            "prompt": prompt,
            "num_frames": 9,
            "num_inference_steps": 2,
            "height": 32,
            "width": 32,
            "output_type": "pil",
        },
        dtype="float32",
        fps=8,
        save_artifacts=True,
        label="text-to-video",
    )


if __name__ == "__main__":
    r = generate()
    print("status:", r["status"])
    print(r["content"][0]["text"][:800])
    print("artifacts:", r.get("artifacts"))
