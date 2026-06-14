"""Generate a GALLERY of real Cosmos3-Nano world-foundation rollouts (GPU).

The world-foundation-model story is the reason strands-diffusers exists, so the
docs show a *gallery* of real rollouts — not one clip. This script drives
nvidia/Cosmos3-Nano through use_diffusers in the policy action-mode with several
task prompts, plus a text-to-world rollout, and writes the gifs/mp4s the WFM docs
page embeds.

Cosmos3OmniPipeline ships in diffusers-from-source (>0.38):

    pip install 'git+https://github.com/huggingface/diffusers' --no-deps --target /tmp/dmain
    PYTHONPATH=/tmp/dmain python examples/generate_wfm_rollouts.py

Needs a CUDA GPU and ~33GB of Cosmos3-Nano weights. Each rollout is independent
and cache is cleared between them, so a partial run still upgrades what it can.
"""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from strands_diffusers import use_diffusers, registry
from strands_diffusers.core import io

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
REPO = "nvidia/Cosmos3-Nano"

# A bridge-domain observation video shipped by NVIDIA (coerced from URL → frames).
OBS_VIDEO = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/"
    "assets/cosmos3/inputs/action/bridge_20260501_0.mp4"
)

# (prompt, output asset stem) — same observation, different intent.
POLICY_TASKS = [
    ("Put the pot to the left of the purple item.", "rollout_policy_1"),
    ("Pick up the cloth and place it in the bowl.", "rollout_policy_2"),
    ("Move the gripper toward the metal pan and grasp the handle.", "rollout_policy_3"),
    ("Open the drawer and place the spoon inside.", "rollout_policy_4"),
    ("Wipe the surface with the cloth in a circular motion.", "rollout_policy_5"),
]


def _gif(mp4: str, name: str, size=(360, 270), dur=200, hold=3):
    import imageio.v3 as iio
    from PIL import Image

    frames = list(iio.imread(mp4))
    imgs = [Image.fromarray(f).convert("RGB").resize(size, Image.BICUBIC)
            .quantize(colors=128, method=Image.FASTOCTREE) for f in frames]
    imgs[0].save(ASSETS / name, save_all=True,
                 append_images=imgs[1:] + [imgs[-1]] * hold, duration=dur,
                 loop=0, optimize=True)
    print(f"  -> docs/assets/{name} ({len(frames)} frames, "
          f"{(ASSETS / name).stat().st_size // 1024} KB)")


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    use_diffusers(action="clear_cache")


def policy_rollout(prompt: str, stem: str):
    print(f"[policy] {stem}: {prompt}")
    use_diffusers(action="call", target="CosmosActionCondition",
                  parameters={"mode": "policy", "chunk_size": 16,
                              "domain_name": "bridge_orig_lerobot",
                              "resolution_tier": 480, "video": OBS_VIDEO,
                              "view_point": "ego_view"},
                  cache_key="cond")
    r = use_diffusers(action="run", pipeline="Cosmos3OmniPipeline", model=REPO,
                      parameters={"prompt": prompt, "action": "cached:cond", "fps": 5,
                                  "num_inference_steps": 30, "guidance_scale": 1.0,
                                  "use_system_prompt": False,
                                  "from_pretrained_kwargs": {"enable_safety_checker": False}},
                      dtype="bfloat16", device="cuda", fps=5, save_artifacts=True)
    if r["status"] != "success":
        print("  skipped:", r["content"][0]["text"][:160])
        return
    arts = r.get("artifacts", [])
    mp4 = next((a for a in arts if a.endswith(".mp4")), None)
    js = next((a for a in arts if a.endswith(".json")), None)
    if mp4:
        shutil.copy(mp4, ASSETS / f"{stem}.mp4")
        _gif(mp4, f"{stem}.gif")
    if js:
        shutil.copy(js, ASSETS / f"{stem}_action.json")
    _free()


def text_to_world():
    print("[t2v] text-to-world (no action conditioning)")
    r = use_diffusers(action="run", pipeline="Cosmos3OmniPipeline", model=REPO,
                      parameters={"prompt": "A robot arm cleaning a white plate at a "
                                            "kitchen sink, photorealistic, smooth cinematic motion",
                                  "num_frames": 17, "fps": 10, "num_inference_steps": 35,
                                  "guidance_scale": 7.0, "use_system_prompt": False,
                                  "from_pretrained_kwargs": {"enable_safety_checker": False}},
                      dtype="bfloat16", device="cuda", fps=10, save_artifacts=True)
    if r["status"] != "success":
        print("  skipped:", r["content"][0]["text"][:160])
        return
    mp4 = next((a for a in r.get("artifacts", []) if a.endswith(".mp4")), None)
    if mp4:
        shutil.copy(mp4, ASSETS / "rollout_t2v.mp4")
        _gif(mp4, "rollout_t2v.gif", size=(400, 225), dur=160, hold=2)
    _free()


def inverse_dynamics(input_video: str, stem: str):
    """Observed video -> reconstruct the world + infer the action chunk."""
    print(f"[inverse_dynamics] {stem}")
    use_diffusers(action="call", target="CosmosActionCondition",
                  parameters={"mode": "inverse_dynamics", "chunk_size": 16,
                              "domain_name": "bridge_orig_lerobot",
                              "resolution_tier": 480, "video": input_video,
                              "view_point": "ego_view"},
                  cache_key="idcond")
    r = use_diffusers(action="run", pipeline="Cosmos3OmniPipeline", model=REPO,
                      parameters={"prompt": "Infer the executed actions.",
                                  "action": "cached:idcond", "fps": 5,
                                  "num_inference_steps": 35, "guidance_scale": 1.0,
                                  "use_system_prompt": False,
                                  "from_pretrained_kwargs": {"enable_safety_checker": False}},
                      dtype="bfloat16", device="cuda", fps=5, save_artifacts=True)
    if r["status"] != "success":
        print("  skipped:", r["content"][0]["text"][:160]); return
    arts = r.get("artifacts", [])
    mp4 = next((a for a in arts if a.endswith(".mp4")), None)
    js = next((a for a in arts if a.endswith(".json")), None)
    if mp4:
        shutil.copy(mp4, ASSETS / f"{stem}.mp4")
        _gif(mp4, f"{stem}.gif")
    if js:
        shutil.copy(js, ASSETS / f"{stem}_action.json")
    _free()


def forward_dynamics(first_frame: str, raw_actions, meta: dict, stem: str):
    """First frame + a known action sequence -> imagined world rollout.

    NOTE: build the condition and run WITHOUT clearing the cache in between —
    cached:fd must survive until the run consumes it.
    """
    print(f"[forward_dynamics] {stem}")
    use_diffusers(action="call", target="CosmosActionCondition",
                  parameters={"mode": "forward_dynamics",
                              "chunk_size": int(meta["action_chunk_size"]),
                              "domain_name": meta["domain_name"],
                              "resolution_tier": int(meta["image_size"]),
                              "image": first_frame,
                              "raw_actions": raw_actions,
                              "view_point": meta["view_point"]},
                  cache_key="fdcond")
    r = use_diffusers(action="run", pipeline="Cosmos3OmniPipeline", model=REPO,
                      parameters={"prompt": meta["prompt"], "action": "cached:fdcond",
                                  "fps": int(meta["fps"]), "num_inference_steps": 35,
                                  "guidance_scale": 1.0, "use_system_prompt": False,
                                  "from_pretrained_kwargs": {"enable_safety_checker": False}},
                      dtype="bfloat16", device="cuda", fps=int(meta["fps"]),
                      save_artifacts=True)
    if r["status"] != "success":
        print("  skipped:", r["content"][0]["text"][:160]); return
    mp4 = next((a for a in r.get("artifacts", []) if a.endswith(".mp4")), None)
    if mp4:
        shutil.copy(mp4, ASSETS / f"{stem}.mp4")
        _gif(mp4, f"{stem}.gif")
    _free()


def main():
    try:
        registry.resolve_attr("Cosmos3OmniPipeline")
    except Exception:
        print("Cosmos3OmniPipeline not importable. Install diffusers-from-source:")
        print("  pip install 'git+https://github.com/huggingface/diffusers' "
              "--no-deps --target /tmp/dmain")
        print("  PYTHONPATH=/tmp/dmain python examples/generate_wfm_rollouts.py")
        return

    print(f"Generating WFM rollout gallery into {ASSETS}\n")
    for prompt, stem in POLICY_TASKS:
        policy_rollout(prompt, stem)
    text_to_world()

    # All three action modes need NVIDIA-shipped example inputs bundled with the
    # Cosmos3-Nano weights. Resolve them from the local HF snapshot if present.
    import json as _json
    from huggingface_hub import try_to_load_from_cache
    snap_assets = None
    cached = try_to_load_from_cache(REPO, "assets/example_action_id_av_0_input.mp4")
    if isinstance(cached, str):
        snap_assets = Path(cached).parent
    if snap_assets and snap_assets.exists():
        inverse_dynamics(str(snap_assets / "example_action_id_av_0_input.mp4"),
                         "rollout_id_av0")
        inverse_dynamics(str(snap_assets / "example_action_id_av_1_input.mp4"),
                         "rollout_id_av1")
        fd_meta_path = snap_assets / "example_action_fd_agibotworld_action_chunks.json"
        if fd_meta_path.exists():
            meta = _json.load(open(fd_meta_path))
            forward_dynamics(str(snap_assets / "example_action_fd_agibotworld_first_frame.png"),
                             meta["action_chunks"], meta, "rollout_fd_agibot")
    else:
        print("(skipping FD/ID — Cosmos3-Nano example assets not in local cache)")

    print("\nDone. WFM rollout gallery regenerated from real Cosmos3-Nano.")


if __name__ == "__main__":
    main()
