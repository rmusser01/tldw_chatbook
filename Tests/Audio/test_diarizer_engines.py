import numpy as np
import pytest

from tldw_chatbook.Audio import diarizer_worker as w


def test_parse_args_defaults_to_speechbrain_and_reads_engine():
    assert w._parse_args([]) == (0, "speechbrain")
    assert w._parse_args(["--start-id", "7", "--engine", "onnx"]) == (7, "onnx")
    with pytest.raises(ValueError):
        w._parse_args(["--engine", "bogus"])


def test_map_final_clusters_reuses_matches_and_mints_past_start_id():
    # Deviation from the brief (see task-1-report.md): the brief's fixture had
    # a SECOND live centroid ("S2": [0.0, 1.0]) which -- verified against the
    # real, untouched `reconcile()` -- greedily claims F1 in the very first
    # (threshold-blind) matching pass whenever every live slot has a final to
    # fill (`reconcile(live, final, 0.25) == {"F0": "S1", "F1": "S2"}`, not a
    # mint). One live centroid makes F1 a genuine surplus no live id claims
    # within threshold, which is what "unmatched -> minted" actually needs.
    live = {"S1": np.array([1.0, 0.0])}
    finals = [("F0", np.array([0.99, 0.01]), 4.0), ("F1", np.array([0.7, 0.7]), 2.0)]
    out = {}
    mapping = w._map_final_clusters(finals, live, threshold=0.25, start_id=5, out_centroids=out)
    assert mapping["F0"] == "S1"
    assert mapping["F1"] == "S6"          # unmatched -> minted past start_id
    assert set(out) == {"S1", "S6"} and abs(np.linalg.norm(out["S1"]) - 1.0) < 1e-6


def test_speechbrain_engine_module_imports_without_torch(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)   # importing torch would raise
    import importlib
    mod = importlib.import_module("tldw_chatbook.Audio.diarizer_engine_speechbrain")
    assert mod.MODEL_ID == "speechbrain/spkrec-ecapa-voxceleb@unpinned"
    assert callable(mod.load)


def test_both_engine_loads_annotate_the_shared_clusterer_and_return_type():
    """Qodo 3: `main()` reaches an engine through `importlib.import_module(
    ENGINES[name]).load(live, max_speakers)`, so nothing but these
    annotations can catch an engine handed the wrong clusterer, or returning
    something that is not a `LoadedEngine`, before it is selected at runtime.

    They must stay FORWARD references: both modules carry `from __future__
    import annotations` and import the two names only under `TYPE_CHECKING`,
    so the worker's pinned import graph (`test_meeting_import_safety.py`)
    does not change -- which is why this asserts the annotation STRINGS.
    """
    import inspect

    from tldw_chatbook.Audio import diarizer_engine_onnx as onnx
    from tldw_chatbook.Audio import diarizer_engine_speechbrain as sb

    for mod in (onnx, sb):
        sig = inspect.signature(mod.load)
        assert sig.parameters["live"].annotation == "OnlineClusterer", mod.__name__
        assert sig.parameters["max_speakers"].annotation == "int", mod.__name__
        assert sig.return_annotation == "LoadedEngine", mod.__name__


def test_loaded_engine_defaults_its_live_threshold_to_the_ecapa_tuned_value():
    """Task 8 (31827): the live clusterer's threshold becomes per-engine.

    0.25 is the value every engine used before this field existed, so an
    engine that does not set one (the SpeechBrain engine) keeps it.
    """
    loaded = w.LoadedEngine(lambda pcm, sr: [1.0], lambda *a: ([], {}), "fake@1")
    assert loaded.live_threshold == 0.25


def test_main_builds_the_clusterer_at_the_engines_live_threshold(monkeypatch):
    """The measured reason for this field (task-8-context, controller smoke):
    ECAPA's 0.25 minted S1..S8 in 30 s on a titanet_small stream, so each
    engine has to be able to hand `main()` its own tuned threshold."""
    import io
    import sys
    import types

    seen = {}

    def fake_load(live, max_speakers):
        seen["live"] = live
        seen["max_speakers"] = max_speakers
        return w.LoadedEngine(lambda pcm, sr: [1.0], lambda *a: ([], {}), "fake@1", live_threshold=0.55)

    monkeypatch.setitem(sys.modules, "tldw_chatbook.Audio.fake_engine_for_test", types.SimpleNamespace(load=fake_load))
    monkeypatch.setitem(w.ENGINES, "fake", "tldw_chatbook.Audio.fake_engine_for_test")
    monkeypatch.setattr(sys, "argv", ["worker", "--engine", "fake"])
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(b"")))
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(buffer=io.BytesIO()))

    assert w.main() == 0
    assert seen["live"].threshold == 0.55


# ---- the shared span reader (task 9: 31827) --------------------------------

def _wav(path, *, seconds=4.0, sr=16000, channels=1, width=2, value=None):
    """A WAV whose sample at index i is `value(i)` -- so a read of a span can
    be checked to have started where it claims."""
    import struct
    import wave

    frames = int(seconds * sr)
    value = value or (lambda i: 1000 + (i // sr) * 1000)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels); f.setsampwidth(width); f.setframerate(sr)
        pack = "<h" if width == 2 else "<b"
        f.writeframes(b"".join(struct.pack(pack, value(i) if width == 2 else 1) * channels for i in range(frames)))


def test_read_pcm16_span_reads_only_the_requested_span(tmp_path):
    """The span offsets are the whole point: the Stop pass asks for
    `[start, end)` of a meeting recording, and reading from 0 instead would
    label the wrong audio."""
    wav = tmp_path / "m.wav"
    _wav(wav, seconds=4.0)                      # second n holds (n+1)*1000

    samples, sr = w.read_pcm16_span(str(wav), 1.0, 3.0)
    assert sr == 16000
    assert len(samples) == 32000                # 2 s, not the whole 4 s file
    assert samples[0] == pytest.approx(2000 / 32768.0)      # starts at 1.0 s
    assert samples[-1] == pytest.approx(3000 / 32768.0)     # ends before 3.0 s

    # A falsy `end_s` means "to the end of the file" (the Stop pass's default).
    tail, _ = w.read_pcm16_span(str(wav), 3.0, 0.0)
    assert len(tail) == 16000 and tail[0] == pytest.approx(4000 / 32768.0)


@pytest.mark.parametrize("kwargs", [{"sr": 8000}, {"channels": 2}, {"width": 1}])
def test_read_pcm16_span_refuses_anything_but_mono_16k_pcm16(tmp_path, kwargs):
    """No resampling, no channel mixing (the meeting pipeline is 16 kHz mono
    end to end). A refused file becomes a framed `ERROR diarize ValueError`
    and a skipped Stop pass -- never silently mis-scaled audio."""
    wav = tmp_path / "m.wav"
    _wav(wav, seconds=1.0, **kwargs)
    with pytest.raises(ValueError, match="unsupported wav"):
        w.read_pcm16_span(str(wav), 0.0, 1.0)


def test_read_pcm16_span_refuses_a_start_past_end_of_file(tmp_path):
    wav = tmp_path / "m.wav"
    _wav(wav, seconds=1.0)
    with pytest.raises(ValueError, match="unsupported wav"):
        w.read_pcm16_span(str(wav), 5.0, 6.0)


class _FakeTensor:
    """Just enough tensor for `_batch`: `from_numpy(...).unsqueeze(0)` in, and
    `.squeeze().detach().cpu().numpy()` out of the encoder."""
    def __init__(self, arr): self.arr = arr
    def unsqueeze(self, _dim): return self
    def squeeze(self): return self
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.array([float(self.arr.mean()), 1.0], dtype=np.float32)


def test_speechbrain_batch_reads_through_the_shared_reader(tmp_path, monkeypatch):
    """Task 9 (31827): the SpeechBrain Stop pass must not need torchaudio.

    It used to call `torchaudio.load`, which torchaudio >= 2.9 routes through
    the separate `torchcodec` package; without it the call raised and the pass
    returned NO SEGMENTS while the live path carried on -- the bake-off's
    first baseline run scored DER 1.000 for exactly this. Here the engine
    reads a real WAV through `diarizer_worker.read_pcm16_span` with a fake
    encoder, and only the requested span reaches it.
    """
    import contextlib
    import inspect
    import types

    from tldw_chatbook.Audio import diarizer_engine_speechbrain as sb
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer
    from tldw_chatbook.Local_Ingestion import diarization_service as ds

    class _FakeService:
        def __init__(self, config=None): self.config = config
        def _cluster_speakers(self, embeddings, num_speakers): return [0] * len(embeddings)

    monkeypatch.setattr(ds, "DiarizationService", _FakeService)
    seen = []
    encoder = types.SimpleNamespace(encode_batch=lambda t: (seen.append(t.arr.shape[0]), t)[1])
    fake_torch = types.SimpleNamespace(no_grad=contextlib.nullcontext, from_numpy=_FakeTensor)

    wav = tmp_path / "mixed.wav"
    _wav(wav, seconds=6.0)
    segments, centroids = sb._batch(encoder, fake_torch, np, OnlineClusterer(), str(wav), 1.0, 3.0, 8)

    assert sum(seen) == 32000            # the 2 s span alone, not the 6 s file
    # 1.5 s windows over that span, with the times offset back to absolute.
    assert [(s["start_s"], s["end_s"]) for s in segments] == [(1.0, 2.5), (2.5, 3.0)]
    assert {s["speaker"] for s in segments} == set(centroids) == {"S1"}
    source = inspect.getsource(sb)          # prose about it is fine; a load path is not
    assert "_lazy_import_torchaudio" not in source and "import torchaudio" not in source
