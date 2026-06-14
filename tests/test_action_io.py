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
