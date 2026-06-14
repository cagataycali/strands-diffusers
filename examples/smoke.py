"""Fast E2E smoke test for strands-diffusers — no big downloads.

Exercises discovery + a real tiny image diffusion + a real tiny video diffusion +
the action-output serializer. Exits non-zero on any failure so it can gate CI.

Run:   python examples/smoke.py
"""

import sys

from strands_diffusers import use_diffusers
from strands_diffusers.core import io


def check(name, cond):
    print(f"  {'✅' if cond else '❌'} {name}")
    return cond


def main() -> int:
    ok = True

    # 1. discovery
    r = use_diffusers(action="pipelines")
    ok &= check(f"pipelines discovery ({len(r['data'])} pipelines)",
                r["status"] == "success" and len(r["data"]) > 100)

    r = use_diffusers(action="wfm")
    ok &= check("wfm detector finds Cosmos",
                any("Cosmos" in p for p in r["data"]))

    r = use_diffusers(action="modalities")
    ok &= check("modalities grouping",
                r["status"] == "success" and "video-to-world" in r["data"])

    r = use_diffusers(action="pipeline_info", target="StableDiffusionPipeline")
    ok &= check("pipeline_info has call_params",
                r["status"] == "success" and "call_params" in r["data"])

    r = use_diffusers(action="inspect", target="utils.export_to_video")
    ok &= check("inspect resolves util fn", r["status"] == "success")

    # 2. real tiny image diffusion → image artifact
    r = use_diffusers(
        action="run", pipeline="StableDiffusionPipeline",
        model="hf-internal-testing/tiny-stable-diffusion-pipe",
        parameters={"prompt": "robot", "num_inference_steps": 2, "output_type": "pil",
                    "from_pretrained_kwargs": {"safety_checker": None}},
        dtype="float32", save_artifacts=True)
    ok &= check("text-to-image E2E → image artifact",
                r["status"] == "success" and r.get("artifacts")
                and r["artifacts"][0].endswith(".png"))

    # 3. real tiny video diffusion → mp4 artifact
    r = use_diffusers(
        action="run", pipeline="LTXPipeline",
        model="katuni4ka/tiny-random-ltx-video",
        parameters={"prompt": "robot", "num_frames": 9, "num_inference_steps": 2,
                    "height": 32, "width": 32, "output_type": "pil"},
        dtype="float32", fps=8, save_artifacts=True)
    ok &= check("text-to-video E2E → video artifact",
                r["status"] == "success" and r.get("artifacts")
                and any(a.endswith((".mp4", ".gif")) for a in r["artifacts"]))

    # 3b. real tiny audio diffusion → wav artifact (built from components, no download)
    import tempfile
    from pathlib import Path
    import torch
    from diffusers import DanceDiffusionPipeline, IPNDMScheduler, UNet1DModel
    fixture = Path(tempfile.gettempdir()) / "smoke-dance-diffusion"
    if not (fixture / "model_index.json").exists():
        torch.manual_seed(0)
        unet = UNet1DModel(
            sample_size=2048, sample_rate=16000, in_channels=2, out_channels=2,
            extra_in_channels=64, time_embedding_type="fourier",
            use_timestep_embedding=False, flip_sin_to_cos=True,
            block_out_channels=(32, 32, 64), mid_block_type="UNetMidBlock1D",
            down_block_types=("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"))
        DanceDiffusionPipeline(unet=unet, scheduler=IPNDMScheduler()).save_pretrained(str(fixture))
    r = use_diffusers(
        action="run", pipeline="DanceDiffusionPipeline", model=str(fixture),
        parameters={"num_inference_steps": 2, "audio_length_in_s": 0.05},
        dtype="float32", save_artifacts=True)
    ok &= check("text-to-audio E2E → wav artifact",
                r["status"] == "success" and r.get("artifacts")
                and any(a.endswith(".wav") for a in r["artifacts"]))

    # 4. action-output serializer (the WFM payload) — no model needed
    import numpy as np
    action = [np.random.randn(16, 7)]   # one chunk [T=16, action_dim=7]
    out = io.serialize_output(
        type("Cosmos3OmniPipelineOutput", (), {"action": action, "video": None,
                                               "sound": None})(),
        save_artifacts=True)
    res = out["result"]
    ok &= check("action serializer → json artifact, shape [1,16,7]",
                res.get("action", {}).get("chunk_shape") == [16, 7]
                and any(a.endswith(".json") for a in out.get("artifacts", [])))

    # 5. action visualization → plots + animation artifacts
    r = use_diffusers(action="visualize",
                      inputs=[[[0.1 * i, -0.1 * i, 0.05, 0, 0, 0, 1.0]
                               for i in range(16)]],
                      parameters={"save_prefix": "smoke_action", "fps": 5})
    arts = r.get("artifacts", [])
    ok &= check("action visualization → timeseries + trajectory + animation",
                r["status"] == "success"
                and any(a.endswith("timeseries_" + a.rsplit("_", 1)[1]) or "timeseries" in a for a in arts)
                and any("animation" in a for a in arts)
                and any("trajectory" in a for a in arts))

    print("\n" + ("🎉 ALL SMOKE CHECKS PASSED" if ok else "💥 SMOKE FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
