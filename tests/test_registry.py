"""Golden modality assertions — catch classifier regressions (incl. the 2 I shipped)."""
import pytest
from strands_diffusers.core import registry as r

# (pipeline, expected_modality) — covers each rule class + the two I mislabeled.
GOLDEN = [
    ("StableDiffusionPipeline", "image"),
    ("StableDiffusionXLPipeline", "image"),
    ("FluxPipeline", "image"),
    ("FluxImg2ImgPipeline", "image-to-image"),
    ("FluxFillPipeline", "inpainting"),
    ("Kandinsky5I2VPipeline", "image-to-video"),   # I once mislabeled -> image
    ("Kandinsky5T2VPipeline", "text-to-video"),
    ("CosmosVideoToWorldPipeline", "video-to-world"),
    ("CosmosTextToWorldPipeline", "text-to-video"),  # text-conditioned; WFM via world_foundation_models()
    ("AudioLDM2Pipeline", "audio"),
    ("StableDiffusionUpscalePipeline", "upscaling"),
    ("StableDiffusionInpaintPipeline", "inpainting"),
]

@pytest.mark.parametrize("name,expected", GOLDEN)
def test_modality(name, expected):
    assert r.modality_of(name) == expected, f"{name} -> {r.modality_of(name)} != {expected}"

def test_no_video_pipeline_labeled_image():
    """Regression guard: a *VideoTo*/I2V/T2V pipe must never land in a still-image bucket."""
    bad = []
    for p in r.pipelines():
        m = r.modality_of(p)
        is_video = any(k in p for k in ("I2V","T2V","V2V","VideoToWorld","TextToVideo","ImageToVideo"))
        if is_video and m in ("image","image-to-image","text-to-image","inpainting"):
            bad.append((p,m))
    assert not bad, f"video pipelines mislabeled as still-image: {bad}"

def test_coverage_counts():
    assert len(r.pipelines()) > 200
    assert len(r.world_foundation_models()) >= 10
    other = [p for p in r.pipelines() if r.modality_of(p)=="other"]
    assert len(other) < 60, f"too many unclassified: {len(other)}"
