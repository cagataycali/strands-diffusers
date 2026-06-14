"""The library's RAISON D'ÊTRE: robot ACTION serialization from a WFM output.
Mocks a Cosmos3OmniPipelineOutput (video+sound+action) — no GPU/weights needed."""
import json, numpy as np, pytest
from strands_diffusers.core import io


def _mock_output(action, video=None, sound=None):
    return type("Cosmos3OmniPipelineOutput", (),
                {"action": action, "video": video, "sound": sound})()


def test_action_single_chunk_shape():
    out = io.serialize_output(_mock_output([np.random.randn(16, 10)]), save_artifacts=True)
    act = out["result"]["action"]
    assert act["type"] == "action"
    assert act["chunk_shape"] == [16, 10]
    assert act["num_chunks"] == 1
    assert any(a.endswith(".json") for a in out["artifacts"])
    # round-trip the json artifact
    p = [a for a in out["artifacts"] if a.endswith(".json")][0]
    data = json.load(open(p))
    assert np.asarray(data).shape == (1, 16, 10)


def test_action_multichunk():
    out = io.serialize_output(_mock_output([np.random.randn(16, 7) for _ in range(3)]),
                              save_artifacts=True)
    act = out["result"]["action"]
    assert act["num_chunks"] == 3 and act["chunk_shape"] == [16, 7]


def test_action_bf16_torch_tensor():
    """bf16 tensors must be upcast to f32 before numpy (the real Cosmos dtype)."""
    torch = pytest.importorskip("torch")
    t = torch.randn(16, 10, dtype=torch.bfloat16)
    out = io.serialize_output(_mock_output([t]), save_artifacts=True)
    assert out["result"]["action"]["chunk_shape"] == [16, 10]


def test_triple_video_sound_action(tmp_path):
    """video → mp4/gif, sound → wav, action → json, all in one output."""
    frames = [np.random.randint(0,255,(32,32,3),dtype=np.uint8) for _ in range(10)]
    out = io.serialize_output(
        _mock_output([np.random.randn(16,7)], video=frames,
                     sound=np.random.randn(16000).astype(np.float32)),
        save_artifacts=True, fps=5)
    arts = out["artifacts"]
    assert any(a.endswith((".mp4",".gif")) for a in arts), "no video artifact"
    assert any(a.endswith(".wav") for a in arts), "no audio artifact"
    assert any(a.endswith(".json") for a in arts), "no action artifact"


def test_action_values_preserved():
    """Serialized values must equal the input (no lossy clipping on actions)."""
    a = np.array([[0.5,-0.5,1.0,-1.0,0.123,0,0]], dtype=np.float64)  # [1,7]
    out = io.serialize_output(_mock_output([a]), save_artifacts=False)
    got = np.asarray(out["result"]["action"]["data"][0])
    np.testing.assert_allclose(got, a, rtol=1e-5)


def test_mesh_serialization_no_data_loss():
    """ShapE-style mesh output (verts/faces) must export to .ply/.obj/.npz, not
    serialize to an opaque repr string (the silent 3D data-loss bug)."""
    # Use the REAL diffusers mesh output class (its name ends with 'Output', which
    # an earlier naive mock missed — that path must still route to mesh export).
    import torch
    from diffusers.pipelines.shap_e.renderer import MeshDecoderOutput
    meshes = [MeshDecoderOutput(verts=torch.randn(100, 3),
                                faces=torch.randint(0, 100, (50, 3)),
                                vertex_channels=None) for _ in range(2)]
    out = io.serialize_output(
        type("ShapEPipelineOutput", (), {"images": meshes})(),
        save_artifacts=True)
    arts = out.get("artifacts", [])
    assert any(a.endswith((".ply", ".obj", ".npz")) for a in arts), \
        f"mesh produced no 3D artifact: {arts}"
    # the serialized result must NOT contain an opaque object repr
    s = str(out["result"])
    assert "object at 0x" not in s, "mesh fell through to repr string (data loss)"


def test_modular_pipeline_run_gives_actionable_error():
    """run path must reject Modular pipelines with guidance, not silently fail."""
    from strands_diffusers import use_diffusers
    r = use_diffusers(action="run", pipeline="FluxModularPipeline",
                      model="black-forest-labs/FLUX.1-schnell",
                      parameters={"prompt": "x"})
    assert r["status"] == "error"
    txt = r["content"][0]["text"]
    assert "Modular" in txt and ("load_components" in txt or "action='call'" in txt)


def test_stereo_audio_downmix_both_layouts():
    """Stereo must down-mix over the CHANNEL axis (short), preserving the time
    axis — for channels-first [C,N] AND channels-last [N,C]. Regression: [N,2]
    once collapsed to 2 samples (averaged across time)."""
    N = 16000
    for shape in [(2, N), (N, 2), (1, N), (N,)]:
        out = io.serialize_output(
            type("AudioPipelineOutput", (), {"audio": np.random.randn(*shape)})(),
            save_artifacts=False)
        samples = out["result"]["audio"]["samples"]
        assert samples == N, f"{shape} -> {samples} samples (expected {N})"


def test_visualize_accepts_json_string_action():
    """LLM tool-calls serialize list inputs to a JSON STRING. visualize must parse
    it (via inputs OR target), not crash in np.asarray(str). Regression: an agent
    needed 11 failed tool calls because inputs arrived stringified."""
    from strands_diffusers import use_diffusers
    js = "[[[0.1,0,0,0,0,0,-1],[0.5,0,0,0,0,0,1]]]"
    for via in ("inputs", "target"):
        r = use_diffusers(action="visualize", parameters={"fps": 5}, **{via: js})
        assert r["status"] == "success", f"{via}: {r['content'][0]['text'][:120]}"
        assert len(r.get("artifacts", [])) >= 2


def test_parameters_accepts_json_string():
    """LLM tool-calls may serialize `parameters` to a JSON string. The tool must
    parse it (not crash in dict(str)), and reject true garbage with a clear error."""
    from strands_diffusers import use_diffusers
    # discovery action needs no model; just exercise the parameters parsing path
    r = use_diffusers(action="inspect", target="utils.export_to_video",
                      parameters='{"unused": 1}')
    assert r["status"] == "success"
    bad = use_diffusers(action="inspect", target="utils.export_to_video",
                        parameters="not json at all")
    assert bad["status"] == "error" and "parameters" in bad["content"][0]["text"]


def test_cached_reference_threading():
    """The flagship Cosmos pattern: call->cache a condition, then thread it into a
    later call's parameters as 'cached:key'. Must resolve to the LIVE object at
    every nesting level (dict / list / scalar) and via the '**' unpack key."""
    import importlib
    from strands_diffusers.core import engine
    m = importlib.import_module("strands_diffusers.tools.use_diffusers")
    sentinel = object()
    engine._CACHE["cond"] = sentinel
    assert m._coerce_param({"action": "cached:cond"})["action"] is sentinel
    assert m._coerce_param(["cached:cond", "x"])[0] is sentinel
    assert m._coerce_param("cached:cond") is sentinel
    engine._CACHE["bundle"] = {"a": 1, "b": 2}
    assert m._resolve_target("cached:bundle") == {"a": 1, "b": 2}
    engine.cache_clear("cond"); engine.cache_clear("bundle")


def test_run_inputs_does_not_json_parse_strings():
    """DELIBERATE BOUNDARY: unlike `visualize` (inputs must be an action array),
    run/call `inputs` is a rarely-used positional convenience where a string is
    far more likely a PROMPT than a serialized array. It must NOT be auto-JSON-
    parsed — structured args belong in `parameters` (which IS parsed). This test
    documents+guards that boundary so a future 'fix' doesn't mangle prompts."""
    import importlib
    m = importlib.import_module("strands_diffusers.tools.use_diffusers")
    assert m._coerce_param("a robot in a kitchen") == "a robot in a kitchen"
    # a JSON-looking string is preserved verbatim (not parsed into a list)
    assert m._coerce_param("[[1,2,3]]") == "[[1,2,3]]"
    # real lists still pass through untouched
    assert m._coerce_param([[1, 2, 3]]) == [[1, 2, 3]]


def test_output_path_not_coerced_idempotent(tmp_path):
    """An EXISTING output-path string must not be coerced into loaded media.
    Regression: a 2nd call with the same output_video_path failed because
    coerce_input loaded the existing .mp4 as frames. Found via the usage gallery."""
    import importlib
    m = importlib.import_module("strands_diffusers.tools.use_diffusers")
    out = str(tmp_path / "o.mp4")
    open(out, "wb").write(b"fake-existing")  # path now exists
    kw = m._coerce_kwargs({"video_frames": "ignore", "output_video_path": out})
    assert kw["output_video_path"] == out, "output path must stay a string, not be loaded"


def test_artifact_filenames_collision_free():
    """Millisecond timestamps collide in tight loops / batched generation, silently
    overwriting artifacts. _stamp() appends an atomic counter. Regression: 30 rapid
    saves once yielded only 6 unique paths."""
    from PIL import Image
    paths = []
    for _ in range(40):
        out = io.serialize_output(
            type("ImagePipelineOutput", (), {"images": [Image.new("RGB", (8, 8))]})(),
            save_artifacts=True)
        paths += out.get("artifacts", [])
    assert len(paths) == len(set(paths)) == 40, f"collisions: {len(paths)-len(set(paths))}"


def test_batched_images_each_saved():
    """A single output with num_images_per_prompt>1 must persist EVERY image."""
    from PIL import Image
    out = io.serialize_output(
        type("ImagePipelineOutput", (),
             {"images": [Image.new("RGB", (8, 8), (i * 40, 0, 0)) for i in range(4)]})(),
        save_artifacts=True)
    arts = [a for a in out.get("artifacts", []) if a.endswith(".png")]
    assert len(arts) == 4 and len(set(arts)) == 4, f"batch lost images: {arts}"


def test_sample_rate_inferred_from_backbone():
    """Audio sample-rate must be read from the generative backbone (unet/transformer)
    when the pipeline has no dedicated audio component. Regression: DanceDiffusion
    stores sample_rate on its UNet1D, so .wav was silently written at the 16000
    default instead of the model's real rate."""
    from strands_diffusers.tools.use_diffusers import _infer_sample_rate

    unet = type("UNet", (), {"config": type("Cfg", (), {"sample_rate": 22050})()})()
    pipe = type("DanceDiffusionPipeline", (), {"unet": unet})()
    assert _infer_sample_rate(pipe) == 22050
    # transformer backbone with sampling_rate also works
    tr = type("T", (), {"config": type("Cfg", (), {"sampling_rate": 44100})()})()
    assert _infer_sample_rate(type("P", (), {"transformer": tr})()) == 44100
    # no audio hints anywhere → default
    assert _infer_sample_rate(type("P", (), {})()) == 16000


def test_looks_like_case_sensitive_path(tmp_path):
    """Paths with uppercase dirs (e.g. HF snapshot '.../Cosmos3-Nano/...') must
    still coerce. Regression: _looks_like lowercased the whole path before the
    os.path.exists() check, breaking case-sensitive filesystems."""
    # a real file in a mixed-case directory
    d = tmp_path / "Cosmos3-Nano" / "assets"
    d.mkdir(parents=True)
    f = d / "Clip_INPUT.MP4"
    f.write_bytes(b"\x00")  # contents irrelevant; we only test path classification
    assert io._looks_like(str(f), (".mp4", ".mov")) is True
    # extension match must be case-insensitive
    assert io._looks_like(str(f), (".mp4",)) is True
    # a nonexistent local path is not a media file
    assert io._looks_like(str(tmp_path / "Nope.mp4"), (".mp4",)) is False
    # urls are matched without existence
    assert io._looks_like("https://x.com/A/Video.MP4", (".mp4",)) is True
