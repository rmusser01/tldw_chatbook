import math
import struct
import sys
import types
import wave

import numpy as np
import pytest


class _Stream:
    def __init__(self, vec): self.vec = vec; self.chunks = []
    def accept_waveform(self, sample_rate, waveform): self.chunks.append((sample_rate, np.asarray(waveform)))
    def input_finished(self): pass

class _Extractor:
    """Embeds by mean sample sign so two tones separate deterministically."""
    dim = 2
    def __init__(self, config): self.config = config
    def create_stream(self): return _Stream(None)
    def is_ready(self, stream): return True
    def compute(self, stream):
        x = np.concatenate([c[1] for c in stream.chunks]); m = float(x.mean())
        return [1.0, 0.0] if m >= 0 else [0.0, 1.0]

class _Seg:
    def __init__(self, s, e, k): self.start, self.end, self.speaker = s, e, k
class _Result:
    def __init__(self, segs): self._segs = segs
    def sort_by_start_time(self): return sorted(self._segs, key=lambda r: r.start)
class _Diarizer:
    sample_rate = 16000
    last_config = None
    def __init__(self, config): _Diarizer.last_config = config
    def process(self, audio, callback=None):
        return _Result([_Seg(0.0, 2.0, 0), _Seg(1.5, 3.0, 1), _Seg(3.0, 4.0, 0)])   # overlap 1.5-2.0


@pytest.fixture
def fake_sherpa(monkeypatch):
    mod = types.SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=lambda **kw: kw,
        SpeakerEmbeddingExtractor=_Extractor,
        OfflineSpeakerDiarizationConfig=lambda **kw: kw,
        OfflineSpeakerSegmentationModelConfig=lambda **kw: kw,
        OfflineSpeakerSegmentationPyannoteModelConfig=lambda **kw: kw,
        FastClusteringConfig=lambda **kw: kw,
        OfflineSpeakerDiarization=_Diarizer,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", mod)
    return mod


def _wav(path, seconds=4.0, sr=16000):
    frames = int(seconds * sr)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        # Controller ruling 2: positive DC EXCEPT 2.0-3.0 s negative. Speaker 0
        # ([0,2)U[3,4)) is then all-positive on both its segments -> embeds
        # [1,0] and reconciles onto the live S1 ([1,0]); speaker 1's segment
        # (1.5,3.0) mixes 0.5 s positive + 1.0 s negative -> net negative ->
        # embeds [0,1] and is minted fresh.
        neg_lo, neg_hi = int(2.0 * sr), int(3.0 * sr)
        f.writeframes(b"".join(
            struct.pack("<h", -8000 if neg_lo <= i < neg_hi else 8000)
            for i in range(frames)
        ))


def test_engine_module_imports_without_sherpa(monkeypatch):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    import importlib; mod = importlib.import_module("tldw_chatbook.Audio.diarizer_engine_onnx")
    assert mod.model_id_for_embedder("titanet_small").startswith("sherpa-onnx/nemo_en_titanet_small.onnx@")


def test_embed_returns_unit_vector_at_the_given_rate(fake_sherpa, tmp_path):
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer
    (tmp_path / "pyannote-segmentation-3-0.onnx").write_bytes(b"x"); (tmp_path / "nemo_en_titanet_small.onnx").write_bytes(b"x")
    loaded = eng.load(OnlineClusterer(), 8, models_dir_override=tmp_path, verify_hashes=False)
    vec = loaded.embed(b"\x10\x27" * 1600, 16000)
    assert vec == pytest.approx([1.0, 0.0])


def test_batch_reconciles_onto_live_ids_and_keeps_overlap(fake_sherpa, tmp_path):
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer
    (tmp_path / "pyannote-segmentation-3-0.onnx").write_bytes(b"x"); (tmp_path / "nemo_en_titanet_small.onnx").write_bytes(b"x")
    live = OnlineClusterer(); live.assign([1.0, 0.0], seconds=3.0)   # S1 == speaker A
    wav = tmp_path / "mixed.wav"; _wav(wav)
    loaded = eng.load(live, 8, models_dir_override=tmp_path, verify_hashes=False)
    segs, centroids = loaded.batch(str(wav), 0.0, 4.0)
    assert [s["speaker"] for s in segs] == ["S1", "S2", "S1"]          # A reconciled onto S1, B minted
    assert segs[1]["start_s"] == 1.5 and segs[0]["end_s"] == 2.0       # overlap preserved
    assert set(centroids) == {"S1", "S2"} and centroids["S2"] == pytest.approx([0.0, 1.0])
    cfg = _Diarizer.last_config
    assert cfg["min_duration_on"] == 0.3 and cfg["min_duration_off"] == 0.5


def test_batch_folds_clusters_past_max_speakers(fake_sherpa, tmp_path):
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer
    (tmp_path / "pyannote-segmentation-3-0.onnx").write_bytes(b"x"); (tmp_path / "nemo_en_titanet_small.onnx").write_bytes(b"x")
    wav = tmp_path / "mixed.wav"; _wav(wav)
    loaded = eng.load(OnlineClusterer(max_speakers=1), 1, models_dir_override=tmp_path, verify_hashes=False)
    segs, centroids = loaded.batch(str(wav), 0.0, 4.0)
    assert len(centroids) == 1 and len({s["speaker"] for s in segs}) == 1
