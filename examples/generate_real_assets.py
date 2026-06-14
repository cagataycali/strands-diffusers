"""Generate the HERO docs assets from REAL model checkpoints (GPU).

generate_docs_assets.py uses tiny fixtures so the gallery is fast and
deterministic on any machine. THIS script regenerates the same asset names from
real, full-quality checkpoints — what you see in the README/docs hero.

Run on a CUDA GPU (the Cosmos step needs ~33GB weights):

    # Cosmos3OmniPipeline ships in diffusers-from-source (>0.38):
    pip install 'git+https://github.com/huggingface/diffusers' --no-deps --target /tmp/dmain
    PYTHONPATH=/tmp/dmain python examples/generate_real_assets.py

Each step is independent and guarded — if a model/checkpoint isn't reachable, it
prints why and moves on, so a partial run still upgrades whatever it can.

Asset name           model                              modality
-------------------  ---------------------------------  -------------------------
text_to_image.png    stabilityai/sd-turbo               text → image (4 steps)
text_to_video.gif    ByteDance/AnimateDiff-Lightning    text → video (4 steps)
text_to_audio.wav    cvssp/audioldm-s-full-v2           text → audio
cosmos_world.gif     nvidia/Cosmos3-Nano                WFM: world video
cosmos_action_*.png  nvidia/Cosmos3-Nano                WFM: robot action chunk
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from strands_diffusers import use_diffusers, registry
from strands_diffusers.core import io

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def _gif_from_mp4(mp4: str, name: str, size, hold_last: int = 0, duration: int = 150):
    import imageio.v3 as iio
    from PIL import Image

    frames = list(iio.imread(mp4))
    imgs = [Image.fromarray(f).convert("RGB").resize(size, Image.BICUBIC) for f in frames]
    imgs[0].save(ASSETS / name, save_all=True,
                 append_images=imgs[1:] + [imgs[-1]] * hold_last,
                 duration=duration, loop=0, optimize=True)
    print(f"  -> docs/assets/{name} ({len(frames)} frames)")


def real_image():
    print("[1/4] text-to-image — stabilityai/sd-turbo (real, 4 steps)")
    r = use_diffusers(action="run", pipeline="AutoPipelineForText2Image",
                      model="stabilityai/sd-turbo",
                      parameters={"prompt": "a sleek 7-DOF robot arm working at a sunlit "
                                            "kitchen counter, photorealistic, cinematic lighting",
                                  "num_inference_steps": 4, "guidance_scale": 0.0,
                                  "height": 512, "width": 512},
                      dtype="float16", device="cuda", save_artifacts=True)
    if r["status"] == "success" and r.get("artifacts"):
        shutil.copy(r["artifacts"][0], ASSETS / "text_to_image.png")
        print("  -> docs/assets/text_to_image.png")
    else:
        print("  skipped:", r["content"][0]["text"][:160])


def real_video():
    print("[2/4] text-to-video — AnimateDiff-Lightning (real, 4 steps, SD1.5 base)")
    try:
        import torch
        from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        device, dtype, step = "cuda", torch.float16, 4
        adapter = MotionAdapter().to(device, dtype)
        adapter.load_state_dict(load_file(hf_hub_download(
            "ByteDance/AnimateDiff-Lightning",
            f"animatediff_lightning_{step}step_diffusers.safetensors"), device=device))
        pipe = AnimateDiffPipeline.from_pretrained(
            "emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=dtype).to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
        frames = pipe(prompt="a robot arm smoothly picking up a red cube on a table, cinematic",
                      guidance_scale=1.0, num_inference_steps=step).frames[0]
        out = io._serialize_video(frames, [], True, {"fps": 12})
        shutil.copy(out["path"], ASSETS / "text_to_video.mp4")
        _gif_from_mp4(out["path"], "text_to_video.gif", (400, 225), duration=120)
    except Exception as e:
        print("  skipped:", str(e)[:160])


def real_audio():
    print("[3/4] text-to-audio — cvssp/audioldm-s-full-v2 (real)")
    r = use_diffusers(action="run", pipeline="AudioLDMPipeline",
                      model="cvssp/audioldm-s-full-v2",
                      parameters={"prompt": "a warm mellow lo-fi hip hop beat with soft piano "
                                            "and vinyl crackle",
                                  "num_inference_steps": 50, "audio_length_in_s": 8.0},
                      dtype="float16", device="cuda", save_artifacts=True)
    if r["status"] == "success" and r.get("artifacts"):
        wav = next((a for a in r["artifacts"] if a.endswith(".wav")), None)
        if wav:
            shutil.copy(wav, ASSETS / "text_to_audio.wav")
            from examples.generate_docs_assets import _plot_waveform
            _plot_waveform(str(ASSETS / "text_to_audio.wav"), "text_to_audio.png")
            print("  -> docs/assets/text_to_audio.wav + .png")
    else:
        print("  skipped:", r["content"][0]["text"][:160])


def real_cosmos():
    print("[4/4] WFM — nvidia/Cosmos3-Nano action-policy rollout (world video + actions)")
    try:
        registry.resolve_attr("Cosmos3OmniPipeline")
    except Exception:
        print("  skipped: Cosmos3OmniPipeline not importable. Install diffusers-from-source:")
        print("    pip install 'git+https://github.com/huggingface/diffusers' --no-deps --target /tmp/dmain")
        print("    PYTHONPATH=/tmp/dmain python examples/generate_real_assets.py")
        return

    from examples.cosmos_action_policy import run_action_policy
    r = run_action_policy()
    if r["status"] != "success":
        print("  skipped:", r["content"][0]["text"][:160])
        return
    arts = r.get("artifacts", [])
    mp4 = next((a for a in arts if a.endswith(".mp4")), None)
    js = next((a for a in arts if a.endswith(".json")), None)
    if mp4:
        shutil.copy(mp4, ASSETS / "cosmos_world.mp4")
        _gif_from_mp4(mp4, "cosmos_world.gif", (360, 270), hold_last=3, duration=200)
    if js:
        shutil.copy(js, ASSETS / "cosmos_action_chunk.json")
        chunk = json.load(open(js))
        chunk = chunk["action"] if isinstance(chunk, dict) and "action" in chunk else chunk
        v = use_diffusers(action="visualize", inputs=chunk,
                          parameters={"save_prefix": "cosmos_real_action", "fps": 8})
        for a in v.get("artifacts", []):
            if "timeseries" in a:
                shutil.copy(a, ASSETS / "cosmos_action_timeseries.png")
            elif "trajectory" in a:
                shutil.copy(a, ASSETS / "cosmos_action_trajectory.png")
            elif "animation" in a and a.endswith(".mp4"):
                shutil.copy(a, ASSETS / "cosmos_action_animation.mp4")
                _gif_from_mp4(a, "cosmos_action_animation.gif", (360, 270), duration=160)
        print("  -> docs/assets/cosmos_world.* + cosmos_action_*")


def main():
    print(f"Generating REAL hero assets into {ASSETS}\n")
    real_image()
    real_video()
    real_audio()
    real_cosmos()
    print("\nDone. Hero assets regenerated from real checkpoints.")


if __name__ == "__main__":
    main()
