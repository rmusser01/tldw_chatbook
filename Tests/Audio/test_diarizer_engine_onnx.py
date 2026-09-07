import dataclasses
import hashlib
import http.server
import io
import math
import struct
import sys
import tarfile
import threading
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


def test_batch_offsets_segment_times_by_the_span_start(fake_sherpa, tmp_path):
    # Review I1: every other batch() test calls start_s=0.0, which passes even
    # if the "start_s + r.start" offset in _batch is dropped entirely. Here
    # start_s=1.0 so the fake's fixed span-relative segments (0.0-2.0,
    # 1.5-3.0, 3.0-4.0) must come back shifted to absolute file time.
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer
    (tmp_path / "pyannote-segmentation-3-0.onnx").write_bytes(b"x"); (tmp_path / "nemo_en_titanet_small.onnx").write_bytes(b"x")
    wav = tmp_path / "mixed.wav"; _wav(wav)
    loaded = eng.load(OnlineClusterer(), 8, models_dir_override=tmp_path, verify_hashes=False)
    segs, _ = loaded.batch(str(wav), 1.0, 4.0)
    assert (segs[0]["start_s"], segs[0]["end_s"]) == (1.0, 3.0)  # was 0.0-2.0
    assert (segs[1]["start_s"], segs[1]["end_s"]) == (2.5, 4.0)  # was 1.5-3.0
    assert (segs[2]["start_s"], segs[2]["end_s"]) == (4.0, 5.0)  # was 3.0-4.0


def test_load_verifies_hashes_by_default_and_can_skip_the_check(fake_sherpa, tmp_path):
    # Review I2: nothing exercised verify_hashes=True (the default) or the
    # "model hash mismatch" message before this test.
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer
    (tmp_path / "pyannote-segmentation-3-0.onnx").write_bytes(b"junk-seg")
    (tmp_path / "nemo_en_titanet_small.onnx").write_bytes(b"junk-emb")

    with pytest.raises(ValueError, match="model hash mismatch"):
        eng.load(OnlineClusterer(), 8, models_dir_override=tmp_path)  # verify_hashes defaults True

    loaded = eng.load(OnlineClusterer(), 8, models_dir_override=tmp_path, verify_hashes=False)
    assert loaded.model_id.startswith("sherpa-onnx/")


# --- Task 3: ensure_models (downloader) -------------------------------------

class _StubHandler(http.server.BaseHTTPRequestHandler):
    """Behaviour is entirely scripted through `self.server.routes[path]`:
    `{"status": int, "location": str, "body": bytes, "bodies": [bytes, ...],
    "sleep_per_chunk": float}`. `"bodies"` serves one entry per request to
    that path (clamped to the last once exhausted) -- how the bad-hash-then-
    good-hash retry test scripts "corrupted first, good second" without
    reaching into the handler class from the test body. `self.server.counts[path]`
    is bumped on every request, script or not."""

    def log_message(self, *a, **k):  # silence stub noise in test output
        pass

    def do_GET(self):
        n = self.server.counts[self.path] = self.server.counts.get(self.path, 0) + 1
        route = self.server.routes.get(self.path)
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        status = route.get("status", 200)
        self.send_response(status)
        if "location" in route:
            self.send_header("Location", route["location"])
        self.end_headers()
        if status in (301, 302, 303, 307, 308):
            return
        if "bodies" in route:
            seq = route["bodies"]
            body = seq[min(n, len(seq)) - 1]
        else:
            body = route.get("body", b"")
        sleep = route.get("sleep_per_chunk")
        if not sleep:
            self.wfile.write(body)
            return
        chunk = 512
        for i in range(0, len(body), chunk):
            try:
                self.wfile.write(body[i:i + chunk])
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            threading.Event().wait(sleep)


@pytest.fixture
def stub_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    server.routes = {}
    server.counts = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _make_tar_bz2(member_name: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        info = tarfile.TarInfo(name=f"sherpa-onnx-fixture/{member_name}")
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _engine_with_manifest_pointing_at(
    stub_server, monkeypatch, *,
    embedder_key="titanet_small",
    seg_body=b"S" * 300_000,
    emb_body=b"E" * 1_900_000,
    seg_route="/seg.tar.bz2",
    emb_route="/emb.onnx",
):
    """Patch the ONNX engine's manifest so `titanet_small` + the segmentation
    asset resolve to `stub_server`, with hashes/sizes matching the given
    fixture bytes -- the URL seam (`_url_for`) plus `ALLOWED_HOSTS` is all
    `ensure_models` needs to treat the stub as a legitimate release host."""
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng

    host, port = stub_server.server_address[0], stub_server.server_address[1]
    base = f"http://{host}:{port}"

    stub_server.routes[seg_route] = {"body": _make_tar_bz2("model.onnx", seg_body)}
    stub_server.routes[emb_route] = {"body": emb_body}

    seg_asset = dataclasses.replace(
        eng.SEGMENTATION, url=f"{base}{seg_route}",
        sha256=hashlib.sha256(seg_body).hexdigest(), size=len(seg_body),
    )
    emb_asset = dataclasses.replace(
        eng.EMBEDDERS[embedder_key], url=f"{base}{emb_route}",
        sha256=hashlib.sha256(emb_body).hexdigest(), size=len(emb_body),
    )
    monkeypatch.setattr(eng, "SEGMENTATION", seg_asset)
    monkeypatch.setattr(eng, "EMBEDDERS", {**eng.EMBEDDERS, embedder_key: emb_asset})
    monkeypatch.setattr(eng, "ALLOWED_HOSTS", eng.ALLOWED_HOSTS + (host,))
    return eng


class _PoisonClient:
    """`ensure_models(..., client=...)` seam: any request means a test bug."""

    def get(self, *a, **k):
        raise AssertionError("network touched in air-gapped/no-op path")

    def stream(self, *a, **k):
        raise AssertionError("network touched in air-gapped/no-op path")


@pytest.mark.loopback_network
def test_ensure_models_downloads_verifies_and_renames_atomically(tmp_path, stub_server, monkeypatch):
    eng = _engine_with_manifest_pointing_at(stub_server, monkeypatch)
    seen = []
    seg, emb = eng.ensure_models("titanet_small", models_dir_override=tmp_path, progress=seen.append)
    assert seg.exists() and emb.exists() and not list(tmp_path.glob("*.tmp*"))
    assert seen and seen[0].startswith("downloading ") and "MB" in seen[0]
    assert seg.read_bytes() == b"S" * 300_000
    assert emb.read_bytes() == b"E" * 1_900_000


@pytest.mark.loopback_network
def test_ensure_models_refuses_a_bad_hash_and_refetches_once(tmp_path, stub_server, monkeypatch):
    good = b"E" * 1_900_000
    eng = _engine_with_manifest_pointing_at(stub_server, monkeypatch, emb_body=good)
    # First response is corrupted; the retry (2nd request) serves the real bytes.
    stub_server.routes["/emb.onnx"] = {"bodies": [b"\x00" * len(good), good]}

    _seg, emb = eng.ensure_models("titanet_small", models_dir_override=tmp_path)
    assert emb.read_bytes() == good
    assert stub_server.counts["/emb.onnx"] == 2
    assert not list(tmp_path.glob("*.tmp*"))


@pytest.mark.loopback_network
def test_ensure_models_refuses_redirects_off_the_allowlist(tmp_path, stub_server, monkeypatch):
    eng = _engine_with_manifest_pointing_at(stub_server, monkeypatch)
    stub_server.routes["/emb.onnx"] = {"status": 302, "location": "http://evil.invalid/payload"}
    models_dir = tmp_path / "models"

    from tldw_chatbook.Audio.diarizer_engine_onnx import ModelsUnavailable

    with pytest.raises(ModelsUnavailable, match="download failed"):
        eng.ensure_models("titanet_small", models_dir_override=models_dir)
    # Nothing written -- not even the segmentation asset, which lands first
    # in the plan and would otherwise have succeeded before the embedder's
    # redirect got refused.
    assert not models_dir.exists() or not any(models_dir.iterdir())


def test_ensure_models_respects_air_gapped_dir_without_network(tmp_path, monkeypatch):
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_engine_onnx import ModelsUnavailable

    good_dir = tmp_path / "good"
    good_dir.mkdir()
    seg_body, emb_body = b"S" * 100, b"E" * 200
    seg_asset = dataclasses.replace(eng.SEGMENTATION, sha256=hashlib.sha256(seg_body).hexdigest(), size=len(seg_body))
    emb_asset = dataclasses.replace(eng.EMBEDDERS["titanet_small"], sha256=hashlib.sha256(emb_body).hexdigest(), size=len(emb_body))
    monkeypatch.setattr(eng, "SEGMENTATION", seg_asset)
    monkeypatch.setattr(eng, "EMBEDDERS", {**eng.EMBEDDERS, "titanet_small": emb_asset})
    (good_dir / seg_asset.file_name).write_bytes(seg_body)
    (good_dir / emb_asset.file_name).write_bytes(emb_body)

    seg, emb = eng.ensure_models("titanet_small", models_dir_override=good_dir, client=_PoisonClient())
    assert seg == good_dir / seg_asset.file_name and emb == good_dir / emb_asset.file_name

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / seg_asset.file_name).write_bytes(seg_body)
    (bad_dir / emb_asset.file_name).write_bytes(b"not-the-right-bytes")  # wrong hash

    with pytest.raises(ModelsUnavailable, match="air-gapped file invalid"):
        eng.ensure_models("titanet_small", models_dir_override=bad_dir, client=_PoisonClient())


@pytest.mark.loopback_network
def test_ensure_models_gives_up_after_the_budget(tmp_path, stub_server, monkeypatch):
    eng = _engine_with_manifest_pointing_at(stub_server, monkeypatch, emb_body=b"E" * 40_000)
    stub_server.routes["/emb.onnx"] = {"body": b"E" * 40_000, "sleep_per_chunk": 0.1}

    from tldw_chatbook.Audio.diarizer_engine_onnx import ModelsUnavailable

    with pytest.raises(ModelsUnavailable, match="budget exceeded"):
        eng.ensure_models("titanet_small", models_dir_override=tmp_path, budget_s=0.2)
    assert not list(tmp_path.glob("*.tmp*"))


def test_ensure_models_is_a_noop_when_files_are_present_and_valid(tmp_path, monkeypatch):
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng

    monkeypatch.setattr("tldw_chatbook.config.get_user_data_dir", lambda: tmp_path)
    models_dir = tmp_path / "models" / "diarization" / "onnx"
    models_dir.mkdir(parents=True)
    # Sizes match the real manifest but the content is garbage: ruling 8 says
    # ensure_models must trust size alone here and never hash a pre-existing
    # file in the DEFAULT (non-air-gapped) path.
    (models_dir / eng.SEGMENTATION.file_name).write_bytes(b"\x00" * eng.SEGMENTATION.size)
    (models_dir / eng.EMBEDDERS["titanet_small"].file_name).write_bytes(b"\x00" * eng.EMBEDDERS["titanet_small"].size)

    seg, emb = eng.ensure_models("titanet_small", client=_PoisonClient())
    assert seg == models_dir / eng.SEGMENTATION.file_name
    assert emb == models_dir / eng.EMBEDDERS["titanet_small"].file_name
