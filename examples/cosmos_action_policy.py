"""Cosmos 3 action-policy world-foundation rollout via use_diffusers — robot ACTIONS out.

This is the headline strands-diffusers use-case. NVIDIA Cosmos 3 is a unified world
foundation model (WFM) for Physical AI. An *action-policy* run predicts future
video AND the robot action chunk that produces it — from a first observation frame,
a task description, and an embodiment domain.

The output `Cosmos3OmniPipelineOutput` carries:
  • video  → exported to .mp4   (the imagined world rollout)
  • sound  → exported to .wav   (optional, sound-capable checkpoints)
  • action → exported to .json  (the model-normalized action chunk the robot runs)

We drive it entirely through use_diffusers:
  1. build a CosmosActionCondition (low-level `call`, cached)
  2. run the pipeline with action=cached:cond (`run`)
  3. io serializes the action tensor list → JSON artifact, shape [num_chunks, T, action_dim]

NOTE: Cosmos3OmniPipeline ships in diffusers *from source* (>0.38). If your
installed diffusers doesn't expose it, this example prints how to enable it and
exits cleanly — the use_diffusers tool itself is version-agnostic (it resolves the
class dynamically, so it works the moment your diffusers has it).

Reference: https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos3

VERIFIED E2E (NVIDIA Thor, diffusers 0.39.0.dev0, Cosmos3-Nano bf16/cuda):
  • world video  → .mp4  shape (17, 480, 640, 3)
  • robot action → .json shape (1, 16, 10)  = (num_chunks, T, action_dim),
                   normalized action space, values in [-1, 1].
  Both produced by ONE use_diffusers(action="run", ...) call.

Run directly (needs a GPU + ~33GB Cosmos3-Nano weights):
    python examples/cosmos_action_policy.py
"""

from strands_diffusers import use_diffusers, registry

REPO = "nvidia/Cosmos3-Nano"
ACTION_VIDEO = (
    "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/"
    "assets/cosmos3/inputs/action/bridge_20260501_0.mp4"
)


def cosmos_available() -> bool:
    """Cosmos3OmniPipeline is only in diffusers-from-source (>0.38)."""
    try:
        registry.resolve_attr("Cosmos3OmniPipeline")
        return True
    except Exception:
        return False


def run_action_policy(prompt: str = "Put the pot to the left of the purple item."):
    # 1. Build the action condition (mode='policy' conditions on the first frame).
    use_diffusers(
        action="call",
        target="CosmosActionCondition",
        parameters={
            "mode": "policy",
            "chunk_size": 16,
            "domain_name": "bridge_orig_lerobot",
            "resolution_tier": 480,
            "video": ACTION_VIDEO,      # coerced path/URL → frames
            "view_point": "ego_view",
        },
        cache_key="act_cond",
    )

    # 2. Run the omni pipeline. Action runs pass conditioning via `action=`, NOT the
    #    top-level image/height/width args. Returns video + (optional) sound + action.
    return use_diffusers(
        action="run",
        pipeline="Cosmos3OmniPipeline",
        model=REPO,
        parameters={
            "prompt": prompt,
            "action": "cached:act_cond",     # cached CosmosActionCondition
            "fps": 5,
            "num_inference_steps": 30,
            "guidance_scale": 1.0,
            "use_system_prompt": False,
            # skip the heavy guardrail download for this demo run
            "from_pretrained_kwargs": {"enable_safety_checker": False},
        },
        dtype="bfloat16",
        device="cuda",
        fps=5,
        save_artifacts=True,
        label="Cosmos3 action-policy rollout",
    )


if __name__ == "__main__":
    if not cosmos_available():
        print("⚠️  Cosmos3OmniPipeline not in this diffusers build.")
        print("   Enable with:  pip install 'git+https://github.com/huggingface/diffusers'")
        print("   Then re-run — use_diffusers resolves the class dynamically, no code change.")
        # Still prove the discovery layer surfaces the action-WFM family:
        wfm = use_diffusers(action="wfm")
        print("\nworld-foundation pipelines available now:")
        print(wfm["content"][0]["text"][:600])
        raise SystemExit(0)

    r = run_action_policy()
    print("status:", r["status"])
    print(r["content"][0]["text"][:1500])
    print("\nartifacts (video .mp4 + action .json):", r.get("artifacts"))
