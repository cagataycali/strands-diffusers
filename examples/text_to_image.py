"""Text-to-image via use_diffusers (run path) — real diffusion, E2E.

Uses a tiny Stable Diffusion fixture so it runs fast on any machine. The same
call works for full models (stabilityai/stable-diffusion-2-1, SDXL, etc.) — just
swap `model`.

Run directly:   python examples/text_to_image.py
"""

from strands_diffusers import use_diffusers

MODEL = "hf-internal-testing/tiny-stable-diffusion-pipe"  # swap for a real SD repo


def generate(prompt: str = "a robot arm in a kitchen"):
    return use_diffusers(
        action="run",
        pipeline="StableDiffusionPipeline",
        model=MODEL,
        parameters={
            "prompt": prompt,
            "num_inference_steps": 4,
            "output_type": "pil",
            # tiny fixture ships no safety-checker weights → disable at load
            "from_pretrained_kwargs": {"safety_checker": None},
        },
        dtype="float32",
        save_artifacts=True,
        label="text-to-image",
    )


if __name__ == "__main__":
    r = generate()
    print("status:", r["status"])
    print(r["content"][0]["text"][:800])
