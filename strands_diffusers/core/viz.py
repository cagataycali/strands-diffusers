"""Visualize robot ACTION chunks — turn raw action tensors into something you can SEE.

A world-foundation model (Cosmos) emits an action chunk of shape
`[num_chunks, T, action_dim]` in normalized action space. Numbers alone are
opaque, so this renders them three ways:

1. time-series   — every action dimension plotted over the chunk's timesteps
                   (joint / delta curves), with the gripper channel highlighted.
2. trajectory    — if the first 3 dims look like an end-effector position/delta,
                   the cumulative 3D path of the gripper through space.
3. animation     — an mp4/gif that sweeps a playhead across the time-series so you
                   can watch the action unfold, optionally side-by-side with the
                   generated world video frames.

All outputs are written to ARTIFACT_DIR and returned as paths.

Design notes:
- We DON'T hardcode an embodiment. action_dim varies (7-DoF arm, 10-DoF, etc.).
  We label dims generically and treat the LAST dim as the gripper by convention
  (override with gripper_index=None to disable). The first 3 dims are *optionally*
  interpreted as an end-effector position for the 3D path (interpret_xyz).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional

from strands_diffusers.core.io import ARTIFACT_DIR


def _as_chunks(action: Any):
    """Normalize an action payload to a numpy array [num_chunks, T, action_dim]."""
    import numpy as np

    # Accept: serialized dict {"data": [...]}, raw nested list, tensor, ndarray,
    # OR a JSON string (LLM tool-calls serialize list inputs to a string).
    if isinstance(action, str):
        import json
        try:
            action = json.loads(action)
        except (ValueError, TypeError):
            raise ValueError(
                "action string is not valid JSON; pass a nested list "
                "[num_chunks, T, action_dim], a .json path, or cached:key")
    if isinstance(action, dict):
        action = action.get("data", action)
    a = np.asarray(action, dtype=float)
    if a.ndim == 1:          # [action_dim] → one chunk, one step
        a = a[None, None, :]
    elif a.ndim == 2:        # [T, action_dim] → one chunk
        a = a[None, :, :]
    return a


def visualize_action(
    action: Any,
    save_prefix: str = "action",
    interpret_xyz: bool = True,
    gripper_index: Optional[int] = -1,
    cumulative_xyz: bool = True,
    world_video: Optional[str] = None,
    fps: int = 5,
    dim_labels: Optional[List[str]] = None,
) -> dict:
    """Render an action chunk to plots + an animation. Returns artifact paths.

    Args:
        action: action payload (serialized dict, nested list, ndarray, or tensor),
                shape [num_chunks, T, action_dim] (lower ranks are promoted).
        save_prefix: filename prefix for artifacts.
        interpret_xyz: if action_dim >= 3, draw a 3D path from dims 0-2.
        gripper_index: dim treated as gripper (highlighted); None to disable.
        cumulative_xyz: treat xyz dims as deltas and integrate into a path; if
                        False, plot them as absolute positions.
        world_video: optional path to the generated world .mp4 to show beside the
                     action animation (frame-synced as best as frame counts allow).
        fps: animation frames-per-second.
        dim_labels: optional human labels per dimension.

    Returns:
        {"artifacts": [...], "summary": {...}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    chunks = _as_chunks(action)
    num_chunks, T, dim = chunks.shape
    # Flatten chunks end-to-end into a single timeline for plotting.
    flat = chunks.reshape(num_chunks * T, dim)
    steps = np.arange(flat.shape[0])

    labels = dim_labels or [f"dim{i}" for i in range(dim)]
    g_idx = (gripper_index % dim) if (gripper_index is not None) else None

    artifacts: List[str] = []
    from strands_diffusers.core.io import _stamp
    ts = _stamp()

    # ── 1. time-series ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(dim):
        is_grip = (i == g_idx)
        ax.plot(steps, flat[:, i],
                label=("gripper" if is_grip else labels[i]),
                lw=2.4 if is_grip else 1.3,
                color="black" if is_grip else None,
                ls="--" if is_grip else "-",
                alpha=1.0 if is_grip else 0.85)
    for c in range(1, num_chunks):
        ax.axvline(c * T - 0.5, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel("timestep")
    ax.set_ylabel("normalized action value")
    ax.set_title(f"Action chunk — {num_chunks}×{T} steps × {dim} dims")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    p_ts = ARTIFACT_DIR / f"{save_prefix}_timeseries_{ts}.png"
    fig.tight_layout()
    fig.savefig(p_ts, dpi=110)
    plt.close(fig)
    artifacts.append(str(p_ts))

    # ── 2. 3D end-effector path ─────────────────────────────────────
    path_xyz = None
    if interpret_xyz and dim >= 3:
        xyz = flat[:, :3].copy()
        if cumulative_xyz:
            xyz = np.cumsum(xyz, axis=0)  # treat as deltas → integrated trajectory
        path_xyz = xyz
        fig = plt.figure(figsize=(6, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "-o", ms=3, lw=1.5)
        ax.scatter(*xyz[0], c="green", s=60, label="start")
        ax.scatter(*xyz[-1], c="red", s=60, label="end")
        # mark gripper close/open events along the path if we have a gripper channel
        if g_idx is not None:
            g = flat[:, g_idx]
            thr = (g.max() + g.min()) / 2.0
            closed = g > thr
            if closed.any():
                ax.scatter(xyz[closed, 0], xyz[closed, 1], xyz[closed, 2],
                           c="orange", s=18, alpha=0.7, label="gripper engaged")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title("End-effector path"
                     + (" (∫ deltas)" if cumulative_xyz else " (absolute)"))
        ax.legend(fontsize=8)
        p_traj = ARTIFACT_DIR / f"{save_prefix}_trajectory_{ts}.png"
        fig.tight_layout()
        fig.savefig(p_traj, dpi=110)
        plt.close(fig)
        artifacts.append(str(p_traj))

    # ── 3. animation (playhead sweep, optional world video beside it) ──
    anim_path = _animate(flat, labels, g_idx, path_xyz, world_video, fps,
                         save_prefix, ts)
    if anim_path:
        artifacts.append(anim_path)

    summary = {
        "num_chunks": int(num_chunks),
        "timesteps_per_chunk": int(T),
        "action_dim": int(dim),
        "gripper_index": g_idx,
        "value_range": [float(flat.min()), float(flat.max())],
        "has_3d_path": path_xyz is not None,
    }
    return {"artifacts": artifacts, "summary": summary}


def _animate(flat, labels, g_idx, path_xyz, world_video, fps, prefix, ts):
    """Build an mp4/gif sweeping a playhead across the action curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = flat.shape[0]
    dim = flat.shape[1]

    # Optionally load world video frames to show alongside.
    world_frames = None
    if world_video:
        try:
            import imageio.v3 as iio
            world_frames = iio.imread(world_video)  # [F,H,W,C]
        except Exception:
            world_frames = None

    have_world = world_frames is not None and len(world_frames) > 0
    have_3d = path_xyz is not None

    ncols = 1 + int(have_world) + int(have_3d)
    frames_out = []

    for k in range(n):
        fig = plt.figure(figsize=(5 * ncols, 4.2))
        col = 1

        # panel A: action curves with playhead
        axc = fig.add_subplot(1, ncols, col); col += 1
        for i in range(dim):
            is_grip = (i == g_idx)
            axc.plot(flat[:, i], lw=2.2 if is_grip else 1.0,
                     color="black" if is_grip else None,
                     ls="--" if is_grip else "-", alpha=0.9 if is_grip else 0.7)
        axc.axvline(k, color="red", lw=2)
        axc.set_title("action")
        axc.set_xlabel("timestep"); axc.grid(alpha=0.2)

        # panel B: world frame (synced by proportion)
        if have_world:
            axw = fig.add_subplot(1, ncols, col); col += 1
            fi = min(int(k / max(n - 1, 1) * (len(world_frames) - 1)),
                     len(world_frames) - 1)
            axw.imshow(world_frames[fi])
            axw.set_title(f"world frame {fi}")
            axw.axis("off")

        # panel C: 3D path with current point
        if have_3d:
            ax3 = fig.add_subplot(1, ncols, col, projection="3d"); col += 1
            ax3.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2],
                     "-", lw=1, alpha=0.5)
            ax3.scatter(*path_xyz[k], c="red", s=50)
            ax3.set_title("end-effector"); ax3.set_xticks([]); ax3.set_yticks([])
            ax3.set_zticks([])

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames_out.append(buf.reshape(h, w, 4)[..., :3].copy())
        plt.close(fig)

    if not frames_out:
        return None

    out_mp4 = ARTIFACT_DIR / f"{prefix}_animation_{ts}.mp4"
    try:
        import imageio.v3 as iio
        iio.imwrite(str(out_mp4), np.stack(frames_out), fps=fps, codec="libx264")
        return str(out_mp4)
    except Exception:
        pass
    try:
        from PIL import Image
        out_gif = ARTIFACT_DIR / f"{prefix}_animation_{ts}.gif"
        imgs = [Image.fromarray(f) for f in frames_out]
        imgs[0].save(str(out_gif), save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(fps, 1)), loop=0)
        return str(out_gif)
    except Exception:
        return None
