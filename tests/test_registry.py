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


def test_pipeline_info_soft_error_for_from_source_wfm():
    """A known from-source WFM class degrades gracefully, not like a typo."""
    info = r.pipeline_info("Cosmos3OmniPipeline")
    assert info is not None, "from-source WFM should not return None (-> hard error)"
    assert info["available"] is False
    assert "from-source" in info["note"] or "diffusers" in info["note"]
    # a genuine typo still returns None (-> hard error, correctly)
    assert r.pipeline_info("DoesNotExistPipeline") is None


def test_shape_3d_modality():
    assert r.modality_of("ShapEPipeline") == "3d"
    assert r.modality_of("ShapEImg2ImgPipeline") in ("3d", "image-to-image")


def test_hybrid_doc_fallback_shrinks_other_safely():
    """Doc fallback (use_doc=True) must rescue 'other' WITHOUT mislabeling any
    video/WFM pipeline into a still-image bucket."""
    name_other = [p for p in r.pipelines() if r.modality_of(p) == "other"]
    hybrid_other = [p for p in r.pipelines() if r.modality_of(p, use_doc=True) == "other"]
    assert len(hybrid_other) < len(name_other), "doc fallback should shrink 'other'"
    # the WFM/video guard must still hold under the hybrid classifier
    for p in r.pipelines():
        m = r.modality_of(p, use_doc=True)
        if any(k in p for k in ("I2V", "T2V", "V2V", "VideoToWorld",
                                "TextToVideo", "ImageToVideo")):
            assert m not in ("image", "image-to-image", "text-to-image", "inpainting"), \
                f"{p} mislabeled {m} by hybrid"


def test_modality_of_default_is_name_only_deterministic():
    """Default modality_of must NOT import classes (fast, deterministic for tests)."""
    assert r.modality_of("StableDiffusionPipeline") == "image"
    # a name-only 'other' stays 'other' without use_doc
    assert r.modality_of("LDMPipeline") == "other"
    assert r.modality_of("LDMPipeline", use_doc=True) == "image"
