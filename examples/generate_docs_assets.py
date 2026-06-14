"""Generate every visual asset used by the docs and README - reproducibly.

strands-diffusers is a visual library: images, videos, robot-action plots and
animations. The docs should SHOW those outputs, not just describe them. This
script runs real use_diffusers operations and writes the results into
docs/assets/ so the documentation always reflects what the tool actually does.

Run:   python examples/generate_docs_assets.py
Output: docs/assets/*.png  *.gif  *.mp4  *.json

Everything here uses tiny fixtures or synthetic actions - no GPU, no big
downloads - so the gallery is fast and deterministic to regenerate.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from strands_diffusers import use_diffusers
from strands_diffusers.core import io

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def _latest(prefix: str, ext: str) -> Path | None:
    src = io.ARTIFACT_DIR
    hits = sorted(src.glob(f"{prefix}*{ext}"))
    return hits[-1] if hits else None


def _adopt(artifact: str, name: str) -> str:
    """Copy a produced artifact into docs/assets under a stable name."""
    dst = ASSETS / name
    shutil.copy(artifact, dst)
    print(f"  -> {dst.relative_to(ASSETS.parent.parent)}")
    return str(dst)


def _gif_from_frames(frames, name: str, fps: int = 8) -> str:
    from PIL import Image

    imgs = [Image.fromarray(f).convert("RGB").resize((256, 256), Image.NEAREST)
            for f in frames]
    dst = ASSETS / name
    imgs[0].save(dst, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"  -> {dst.relative_to(ASSETS.parent.parent)}")
    return str(dst)


def gen_image():
    print("[1/7] text-to-image (StableDiffusion, tiny fixture)")
    r = use_diffusers(
        action="run", pipeline="StableDiffusionPipeline",
        model="hf-internal-testing/tiny-stable-diffusion-pipe",
        parameters={"prompt": "a robot arm in a kitchen", "num_inference_steps": 4,
                    "height": 256, "width": 256, "output_type": "pil",
                    "from_pretrained_kwargs": {"safety_checker": None}},
        dtype="float32", save_artifacts=True)
    if r["status"] == "success" and r.get("artifacts"):
        _adopt(r["artifacts"][0], "text_to_image.png")


def gen_video():
    print("[2/7] text-to-video (LTX, tiny fixture) -> gif preview")
    r = use_diffusers(
        action="run", pipeline="LTXPipeline",
        model="katuni4ka/tiny-random-ltx-video",
        parameters={"prompt": "a robot arm moving a cube", "num_frames": 16,
                    "num_inference_steps": 2, "height": 64, "width": 64,
                    "output_type": "pil"},
        dtype="float32", fps=8, save_artifacts=True)
    if r["status"] == "success" and r.get("artifacts"):
        mp4 = next((a for a in r["artifacts"] if a.endswith(".mp4")), None)
        if mp4:
            _adopt(mp4, "text_to_video.mp4")
            try:
                import imageio.v3 as iio
                frames = iio.imread(mp4)
                _gif_from_frames(list(frames), "text_to_video.gif", fps=8)
            except Exception as e:
                print(f"     (gif skipped: {e})")


def _reach_grasp_lift(T: int = 24, dim: int = 7):
    """A smooth, legible reach-grasp-lift action chunk (xyz arc + gripper)."""
    t = np.linspace(0, 1, T)
    a = np.zeros((1, T, dim))
    a[0, :, 0] = 0.10 * np.sin(np.pi * t)            # x: reach out and back
    a[0, :, 1] = 0.06 * t                            # y: drift forward
    a[0, :, 2] = 0.12 * np.sin(np.pi * t) - 0.02     # z: down then lift
    for d in range(3, dim - 1):
        a[0, :, d] = 0.04 * np.sin(2 * np.pi * t + d)  # wrist joints
    a[0, :, -1] = np.where(t > 0.55, 1.0, -1.0)      # gripper: open -> closed
    return a.tolist()


def gen_action_viz():
    print("[3/7] robot action visualization (timeseries + 3D path + animation)")
    r = use_diffusers(action="visualize", inputs=_reach_grasp_lift(),
                      parameters={"save_prefix": "hero_action", "fps": 8})
    if r["status"] != "success":
        print("     FAILED:", r["content"][0]["text"][:120])
        return
    for a in r.get("artifacts", []):
        if "timeseries" in a:
            _adopt(a, "cosmos_action_timeseries.png")
        elif "trajectory" in a:
            _adopt(a, "cosmos_action_trajectory.png")
        elif "animation" in a and a.endswith(".mp4"):
            _adopt(a, "cosmos_action_animation.mp4")
            try:
                import imageio.v3 as iio
                frames = iio.imread(a)
                step = max(1, len(frames) // 24)
                _gif_from_frames(list(frames)[::step], "cosmos_action_animation.gif", fps=6)
            except Exception as e:
                print(f"     (gif skipped: {e})")


def gen_action_json():
    print("[4/7] WFM action chunk -> json artifact (the robot payload)")
    out = io.serialize_output(
        type("Cosmos3OmniPipelineOutput", (),
             {"action": [np.asarray(_reach_grasp_lift()[0])],
              "video": None, "sound": None})(),
        save_artifacts=True)
    arts = out.get("artifacts", [])
    js = next((a for a in arts if a.endswith(".json")), None)
    if js:
        _adopt(js, "cosmos_action_chunk.json")


def gen_audio():
    print("[5/7] text-to-audio -> waveforms")
    import tempfile, shutil, wave
    import torch

    # Real gallery clips: AudioLDM (downloads cvssp/audioldm-s-full-v2 once, runs on GPU
    # if available). Falls back silently to the tiny offline fixture below if it can't run.
    gallery = [
        ("audio_dog",    "A dog barking in a backyard"),
        ("audio_rain",   "Heavy rain and rolling thunder"),
        ("audio_birds",  "Birds chirping in a forest at dawn"),
        ("audio_typing", "Mechanical keyboard typing quickly"),
    ]
    have_cuda = torch.cuda.is_available()
    made_gallery = False
    for slug, prompt in gallery:
        try:
            r = use_diffusers(action="run", pipeline="AudioLDMPipeline",
                              model="cvssp/audioldm-s-full-v2",
                              parameters={"prompt": prompt, "audio_length_in_s": 5.0,
                                          "num_inference_steps": 15},
                              dtype="float16" if have_cuda else "float32",
                              device="cuda" if have_cuda else "cpu",
                              save_artifacts=True)
            if r.get("status") == "success" and r.get("artifacts"):
                wav = next((a for a in r["artifacts"] if a.endswith(".wav")), None)
                if wav:
                    _adopt(wav, f"{slug}.wav")
                    _plot_waveform(wav, f"{slug}.png", title=f'“{prompt}”')
                    made_gallery = True
        except Exception as e:
            print(f"  (skip {slug}: {type(e).__name__})")

    # Offline fallback / always-on tiny sample for README + index hero.
    fixture = Path(tempfile.gettempdir()) / "docs-dance-diffusion"
    if not (fixture / "model_index.json").exists():
        from diffusers import DanceDiffusionPipeline, IPNDMScheduler, UNet1DModel
        torch.manual_seed(0)
        unet = UNet1DModel(
            sample_size=2048, sample_rate=16000, in_channels=2, out_channels=2,
            extra_in_channels=64, time_embedding_type="fourier",
            use_timestep_embedding=False, flip_sin_to_cos=True,
            block_out_channels=(32, 32, 64), mid_block_type="UNetMidBlock1D",
            down_block_types=("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"))
        DanceDiffusionPipeline(unet=unet, scheduler=IPNDMScheduler()).save_pretrained(str(fixture))
    r = use_diffusers(action="run", pipeline="DanceDiffusionPipeline", model=str(fixture),
                      parameters={"num_inference_steps": 4, "audio_length_in_s": 0.2},
                      dtype="float32", save_artifacts=True)
    if r["status"] == "success" and r.get("artifacts"):
        wav = next((a for a in r["artifacts"] if a.endswith(".wav")), None)
        if wav:
            _adopt(wav, "text_to_audio.wav")
            _plot_waveform(wav, "text_to_audio.png")


def _plot_waveform(wav_path: str, name: str, title: str = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wave

    with wave.open(wav_path, "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(float)
    t = np.arange(len(data)) / sr
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.plot(t, data, lw=0.6, color="#7c3aed")
    ax.set_xlabel("seconds")
    ax.set_ylabel("amplitude")
    ax.set_title(f"{title} \u2014 {sr} Hz" if title else f"generated audio - {sr} Hz")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=110)
    plt.close(fig)
    print(f"  -> {(ASSETS / name).relative_to(ASSETS.parent.parent)}")



def _synthetic_shape():
    """A low-poly 'shape' mesh (verts/faces) - the ShapE aesthetic, offline.

    Built with trimesh, then lightly deformed so it reads as a generated object
    rather than a textbook primitive.
    """
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    v = np.asarray(m.vertices, dtype=float)
    # gentle, smooth deformation -> organic 'blob' silhouette
    r = np.linalg.norm(v, axis=1, keepdims=True)
    v = v * (1.0 + 0.18 * np.sin(3 * v[:, 0:1]) * np.cos(3 * v[:, 1:2]))
    v[:, 2] *= 1.25  # stretch vertically a touch
    return v, np.asarray(m.faces)


class _MeshOut:
    """Minimal stand-in for a diffusers MeshDecoderOutput (verts + faces)."""
    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces


def _render_mesh_png(verts, faces, name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tris = verts[faces]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    # shade by face-normal z so the form reads in 3D
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nz = n[:, 2] / (np.linalg.norm(n, axis=1) + 1e-9)
    shade = 0.45 + 0.55 * (nz - nz.min()) / (np.ptp(nz) + 1e-9)
    colors = np.zeros((len(tris), 4))
    colors[:, 0] = 0.36 * shade + 0.10   # purple-ish (#7c3aed family)
    colors[:, 1] = 0.20 * shade
    colors[:, 2] = 0.55 * shade + 0.25
    colors[:, 3] = 1.0
    coll = Poly3DCollection(tris, facecolors=colors, edgecolors=(1, 1, 1, 0.06), linewidths=0.2)
    ax.add_collection3d(coll)
    lim = float(np.abs(verts).max()) * 0.8
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1.25))
    ax.view_init(elev=18, azim=35)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(ASSETS / name, dpi=120, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  -> {(ASSETS / name).relative_to(ASSETS.parent.parent)}")


def gen_mesh():
    print("[7/7] 3D mesh (synthetic verts/faces -> real .ply via serializer -> render)")
    verts, faces = _synthetic_shape()
    # Route through the REAL serializer so the artifact is honest (.ply/.obj/.npz).
    out = io.serialize_output(_MeshOut(verts, faces), save_artifacts=True)
    arts = out.get("artifacts", [])
    mesh_path = next((a for a in arts if a.endswith((".ply", ".obj", ".npz"))), None)
    if mesh_path:
        print(f"     serialized -> {Path(mesh_path).name} "
              f"(verts={out.get('num_verts')}, faces={out.get('num_faces')})")
    _render_mesh_png(verts, faces, "mesh_render.png")


def gen_modality_chart():
    print("[6/7] modality coverage chart (discovery, no model)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = use_diffusers(action="modalities")["data"]
    items = sorted(((k, len(v)) for k, v in groups.items()), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    counts = [c for _, c in items]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, counts, color="#6366f1")
    for b, c in zip(bars, counts):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, str(c),
                va="center", fontsize=9)
    total = sum(counts)
    ax.set_title(f"diffusers pipeline coverage - {total} pipelines across "
                 f"{len(labels)} modalities (zero hardcoding)")
    ax.set_xlabel("pipeline count")
    fig.tight_layout()
    fig.savefig(ASSETS / "modality_coverage.png", dpi=110)
    plt.close(fig)
    print(f"  -> docs/assets/modality_coverage.png")


def main():
    print(f"Generating docs assets into {ASSETS}\n")
    gen_image()
    gen_video()
    gen_action_viz()
    gen_action_json()
    gen_audio()
    gen_modality_chart()
    gen_mesh()
    print("\nDone. Assets:")
    for p in sorted(ASSETS.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
