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
