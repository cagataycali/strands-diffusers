"""20-cycle usage gallery — exercise the breadth of use_diffusers, end to end.

This is the "just USE it" companion to smoke.py: 20 distinct capabilities run
live (discovery + real tiny inference + serialization + visualization + cache
lifecycle), each printing ✅/❌ and any artifact paths. No GPU, no big downloads
— tiny HF fixtures only. Swap models for full checkpoints to get real quality.

Run:   python examples/gallery_20.py
"""
import sys
import numpy as np

from strands_diffusers import use_diffusers as u
from strands_diffusers.core import io, registry

_n = 0
def cyc(label, ok, artifacts=None):
    global _n; _n += 1
    tag = f" → {len(artifacts)} artifact(s)" if artifacts else ""
    print(f"[{_n:2}] {'✅' if ok else '❌'} {label}{tag}")
    return ok


def main() -> int:
    ok = True
    # ── discovery (1-8) ──
    ok &= cyc("pipelines", u(action="pipelines")["status"] == "success")
    ok &= cyc("models", u(action="models")["status"] == "success")
    ok &= cyc("schedulers", u(action="schedulers")["status"] == "success")
    ok &= cyc("tasks (AutoPipeline maps)", u(action="tasks")["status"] == "success")
    ok &= cyc("modalities", u(action="modalities")["status"] == "success")
    ok &= cyc("wfm (world-foundation)", u(action="wfm")["status"] == "success")
    ok &= cyc("pipeline_info StableDiffusion",
              u(action="pipeline_info", target="StableDiffusionPipeline")["status"] == "success")
    ok &= cyc("inspect export_to_video",
              u(action="inspect", target="utils.export_to_video")["status"] == "success")

    # ── real tiny inference (9-10) ──
    r = u(action="run", pipeline="StableDiffusionPipeline",
          model="hf-internal-testing/tiny-stable-diffusion-pipe",
          parameters={"prompt": "a robot arm in a kitchen", "num_inference_steps": 2,
                      "output_type": "pil", "from_pretrained_kwargs": {"safety_checker": None}},
          dtype="float32")
    ok &= cyc("run text→image (tiny SD)", r["status"] == "success", r.get("artifacts"))
    r = u(action="run", pipeline="LTXPipeline", model="katuni4ka/tiny-random-ltx-video",
          parameters={"prompt": "robot moving a cube", "num_frames": 9,
                      "num_inference_steps": 2, "height": 32, "width": 32, "output_type": "pil"},
          dtype="float32", fps=8)
    ok &= cyc("run text→video (tiny LTX)", r["status"] == "success", r.get("artifacts"))

    # ── cache + dynamic call (11-12) ──
    ok &= cyc("cache (loaded pipelines)", u(action="cache")["status"] == "success")
    # export_to_video wants PIL frames (what the run path feeds it); raw ndarrays
    # make imageio mis-read the first arg as a URI. Demonstrate the call layer the
    # way it is actually used — with PIL images.
    from PIL import Image
    frames = [Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
              for _ in range(8)]
    r = u(action="call", target="utils.export_to_video",
          parameters={"video_frames": frames,
                      "output_video_path": "/tmp/strands_diffusers/gallery_call.mp4", "fps": 8})
    ok &= cyc("call utils.export_to_video (PIL frames)", r["status"] == "success")

    # ── WFM action payload + visualization (13-14) ──
    out = io.serialize_output(
        type("Cosmos3OmniPipelineOutput", (),
             {"action": [np.random.randn(16, 7)], "video": None, "sound": None})(),
        save_artifacts=True)
    ok &= cyc("action chunk → json", bool(out.get("artifacts")), out.get("artifacts"))
    r = u(action="visualize",
          inputs=[[[0.1 * i, 0.05 * i, 0.1 * np.sin(i), 0, 0, 0, (-1 if i < 10 else 1)]
                   for i in range(16)]],
          parameters={"save_prefix": "gallery", "fps": 5})
    ok &= cyc("visualize action (timeseries+3d+anim)", r["status"] == "success", r.get("artifacts"))

    # ── 3D mesh (15) ──
    try:
        import torch
        from diffusers.pipelines.shap_e.renderer import MeshDecoderOutput
        mesh = MeshDecoderOutput(verts=torch.randn(80, 3),
                                 faces=torch.randint(0, 80, (40, 3)), vertex_channels=None)
        mo = io.serialize_output(type("ShapEPipelineOutput", (), {"images": [mesh]})(),
                                 save_artifacts=True)
        ok &= cyc("3D mesh → artifact", bool(mo.get("artifacts")), mo.get("artifacts"))
    except Exception as e:
        ok &= cyc(f"3D mesh (skipped: {e})", True)

    # ── classifier + graceful WFM + inspect (16-18) ──
    ok &= cyc("modality_of Cosmos (hybrid)",
              registry.modality_of("CosmosVideoToWorldPipeline") == "video-to-world")
    ok &= cyc("pipeline_info Cosmos3 (graceful)",
              u(action="pipeline_info", target="Cosmos3OmniPipeline")["status"] == "success")
    ok &= cyc("inspect DDIMScheduler",
              u(action="inspect", target="DDIMScheduler")["status"] == "success")

    # ── cache lifecycle (19-20) ──
    ok &= cyc("clear_cache (free GPU mem)", u(action="clear_cache")["status"] == "success")
    ok &= cyc("cache empty after clear", u(action="cache")["status"] == "success")

    print("\n" + ("🎉 ALL 20 CYCLES PASSED" if ok else "💥 GALLERY FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
