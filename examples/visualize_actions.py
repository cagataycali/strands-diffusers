"""See robot ACTIONS, don't just read numbers — visualize a WFM action chunk.

A Cosmos action-policy run returns an action chunk `[num_chunks, T, action_dim]`
in normalized action space. This example turns that opaque tensor into:

  1. timeseries.png  — every action dim over time, gripper channel highlighted
  2. trajectory.png  — the 3D end-effector path (dims 0-2 integrated as deltas)
  3. animation.mp4   — a playhead sweeping the action curves, side-by-side with the
                       generated world video frames and the moving 3D path

Two ways to use it:

A) Visualize an action you already generated (its .json artifact):

    use_diffusers(action="visualize",
                  target="/tmp/strands_diffusers/action_XXXX.json",
                  parameters={"world_video": "/tmp/strands_diffusers/video_XXXX.mp4"})

B) Pass a raw action inline (nested list / serialized dict):

    use_diffusers(action="visualize", inputs=[[[...16x7...]]])

Run directly (uses a synthetic action if no real artifact is given):
    python examples/visualize_actions.py [path/to/action.json] [path/to/world.mp4]
"""

import sys

import numpy as np

from strands_diffusers import use_diffusers


def synthetic_action(T: int = 16, dim: int = 7):
    """A smooth reach-grasp-lift action: xyz arc + gripper close near the end."""
    t = np.linspace(0, 1, T)
    a = np.zeros((1, T, dim), dtype=float)
    a[0, :, 0] = 0.08 * np.sin(np.pi * t)        # x delta
    a[0, :, 1] = 0.05 * t                         # y delta (drift forward)
    a[0, :, 2] = 0.10 * np.sin(np.pi * t) - 0.02  # z delta (down then up)
    for d in range(3, dim - 1):
        a[0, :, d] = 0.03 * np.sin(2 * np.pi * t + d)  # wrist joints
    a[0, :, -1] = np.where(t > 0.6, 1.0, -1.0)    # gripper: open → closed
    return a.tolist()


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else None
    world = sys.argv[2] if len(sys.argv) > 2 else None

    params = {"save_prefix": "demo_action", "fps": 5}
    if world:
        params["world_video"] = world

    if action:
        r = use_diffusers(action="visualize", target=action, parameters=params)
    else:
        print("No action .json given — visualizing a synthetic reach-grasp-lift.")
        r = use_diffusers(action="visualize", inputs=synthetic_action(), parameters=params)

    print("status:", r["status"])
    print(r["content"][0]["text"][:900])


if __name__ == "__main__":
    main()
