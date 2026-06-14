"""Audio generation via use_diffusers (run path) - real diffusion, .wav out.

Exercises the audio output serializer end to end. Audio pipelines on the Hub are
large, so this builds a tiny DanceDiffusion fixture from components in-process
(no download) and runs it - proving the run -> AudioPipelineOutput -> .wav path.

Swap pipeline/model for a full checkpoint (AudioLDM2Pipeline, MusicLDMPipeline,
StableAudioPipeline, ...) - discover them with use_diffusers(action="modalities").

Run directly:   python examples/text_to_audio.py
"""

import tempfile
from pathlib import Path

from strands_diffusers import use_diffusers

FIXTURE = Path(tempfile.gettempdir()) / "tiny-dance-diffusion"


def _build_fixture() -> str:
    """Construct a tiny DanceDiffusion pipeline and save it locally."""
    import torch
    from diffusers import DanceDiffusionPipeline, IPNDMScheduler, UNet1DModel

    if (FIXTURE / "model_index.json").exists():
        return str(FIXTURE)

    torch.manual_seed(0)
    b0 = 32
    unet = UNet1DModel(
        sample_size=2048,
        sample_rate=16000,
        in_channels=2,
        out_channels=2,
        # fourier time-embedding concatenates 2*block_out_channels[0] extra channels
        extra_in_channels=2 * b0,
        time_embedding_type="fourier",
        use_timestep_embedding=False,
        flip_sin_to_cos=True,
        block_out_channels=(b0, b0, 64),
        mid_block_type="UNetMidBlock1D",
        down_block_types=("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
        up_block_types=("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
    )
    pipe = DanceDiffusionPipeline(unet=unet, scheduler=IPNDMScheduler())
    pipe.save_pretrained(str(FIXTURE))
    return str(FIXTURE)


def generate():
    model = _build_fixture()
    return use_diffusers(
        action="run",
        pipeline="DanceDiffusionPipeline",
        model=model,
        parameters={"num_inference_steps": 4, "audio_length_in_s": 0.1},
        dtype="float32",
        save_artifacts=True,
        label="text-to-audio",
    )


if __name__ == "__main__":
    r = generate()
    print("status:", r["status"])
    print(r["content"][0]["text"][:800])
    print("artifacts:", r.get("artifacts"))
