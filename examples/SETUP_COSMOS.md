# Running real Cosmos 3 (action-policy) with strands-diffusers

`Cosmos3OmniPipeline` + `CosmosActionCondition` ship in **diffusers from source**
(>0.38). `use_diffusers` resolves pipeline classes dynamically, so no code change
is needed once they're importable.

## Option A — install from source into your env

```bash
pip install 'git+https://github.com/huggingface/diffusers'
python examples/cosmos_action_policy.py
```

## Option B — side-load from source (don't disturb a pinned diffusers)

Install only the diffusers source tree (no deps) and prepend it to `PYTHONPATH`,
reusing your existing torch / transformers:

```bash
pip install 'git+https://github.com/huggingface/diffusers' --no-deps --target /tmp/dmain
PYTHONPATH=/tmp/dmain python examples/cosmos_action_policy.py
```

This is exactly how the example above was verified end-to-end.

## What you get

A single `use_diffusers(action="run", pipeline="Cosmos3OmniPipeline", ...)` returns
a `Cosmos3OmniPipelineOutput` that `core/io.py` serializes to artifacts:

```
📎 artifacts:
  • /tmp/strands_diffusers/video_*.mp4    # world rollout  (17, 480, 640, 3)
  • /tmp/strands_diffusers/action_*.json  # robot actions  (1, 16, 10)
```

The action JSON is the model-normalized action chunk `[num_chunks, T, action_dim]`
(values in `[-1, 1]`) — feed it straight to your embodiment's un-normalizer /
controller. Pick `domain_name` to match your robot (e.g. `bridge_orig_lerobot`).

## Action modes (see the Cosmos 3 docs)

- `policy` — predict future video **and** actions from the first frame + task.
- `forward_dynamics` — roll out video from a first frame + a given `raw_actions` seq.
- `inverse_dynamics` — infer the actions connecting the frames of a conditioning video.

Build the condition with `use_diffusers(action="call",
target="CosmosActionCondition", parameters={...}, cache_key="cond")` then pass
`parameters={"action": "cached:cond"}` to the run.

## Notes

- ~33 GB of weights for `nvidia/Cosmos3-Nano`; needs a CUDA GPU.
- The default NVIDIA guardrail (`cosmos_guardrail`) is on under the model license.
  These demos pass `enable_safety_checker=False` for development; keep it enabled
  for anything public-facing.
