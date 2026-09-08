import contextlib
import dataclasses
import hashlib
import http.server
import io
import math
import struct
import sys
import tarfile
import threading
import time
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


def test_load_reads_embedder_and_models_dir_from_env_when_kwargs_are_none(fake_sherpa, tmp_path, monkeypatch):
    """Task 4 (31827) ruling 2: the worker learns which files the app fetched
    through env vars, not argv -- `load()` must fall back to them only when
    its own kwargs are omitted."""
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer

    (tmp_path / eng.SEGMENTATION.file_name).write_bytes(b"x")
    (tmp_path / eng.EMBEDDERS["wespeaker_resnet34"].file_name).write_bytes(b"x")
    monkeypatch.setenv("TLDW_DIARIZER_EMBEDDER", "wespeaker_resnet34")
    monkeypatch.setenv("TLDW_DIARIZER_MODELS_DIR", str(tmp_path))

    loaded = eng.load(OnlineClusterer(), 8, verify_hashes=False)
    assert loaded.model_id == eng.model_id_for_embedder("wespeaker_resnet34")


def test_load_prefers_explicit_kwargs_over_env(fake_sherpa, tmp_path, monkeypatch):
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer

    (tmp_path / eng.SEGMENTATION.file_name).write_bytes(b"x")
    (tmp_path / eng.EMBEDDERS["titanet_small"].file_name).write_bytes(b"x")
    monkeypatch.setenv("TLDW_DIARIZER_EMBEDDER", "wespeaker_resnet34")
    monkeypatch.setenv("TLDW_DIARIZER_MODELS_DIR", "/nonexistent/path")

    loaded = eng.load(OnlineClusterer(), 8, embedder="titanet_small", models_dir_override=tmp_path, verify_hashes=False)
    assert loaded.model_id == eng.model_id_for_embedder("titanet_small")


# --- Task 8: per-engine live threshold + env sweep overrides ----------------

def _place(eng, tmp_path, key="titanet_small"):
    (tmp_path / eng.SEGMENTATION.file_name).write_bytes(b"x")
    (tmp_path / eng.EMBEDDERS[key].file_name).write_bytes(b"x")


def test_thresholds_are_the_bakeoff_measured_values(fake_sherpa, tmp_path):
    """Task 9 (31827): both threshold dicts carry the MEASURED optima.

    Pinned per key, not merely "covers every embedder" (task 8's version of
    this test asserted the flat 0.25 default): these are the whole product of
    a 188-minute bake-off, and a silent revert to 0.5/0.25 costs live
    labelling -- 0.25 on titanet_small minted S1..S8 inside 30 s. See
    `Docs/STT_Evaluation/task-31827/report.md` and spec §10.
    """
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng

    assert eng.CLUSTER_THRESHOLD == {
        "titanet_small": 0.95, "eres2net_en": 0.90, "wespeaker_resnet34": 0.60, "campplus_en": 0.80,
    }
    assert eng.LIVE_THRESHOLD == {
        "titanet_small": 0.45, "eres2net_en": 0.45, "wespeaker_resnet34": 0.10, "campplus_en": 0.15,
    }
    # A new manifest entry must bring its own measured pair, not inherit one.
    assert set(eng.LIVE_THRESHOLD) == set(eng.CLUSTER_THRESHOLD) == set(eng.EMBEDDERS)
    # The bake-off kept titanet_small: best DER/purity/RTF/latency of the four.
    assert eng.DEFAULT_EMBEDDER == "titanet_small"


def test_load_reports_the_embedders_live_threshold(fake_sherpa, tmp_path):
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer

    _place(eng, tmp_path)
    loaded = eng.load(OnlineClusterer(), 8, models_dir_override=tmp_path, verify_hashes=False)
    assert loaded.live_threshold == eng.LIVE_THRESHOLD["titanet_small"]


def test_env_overrides_both_thresholds_for_the_bakeoff(fake_sherpa, tmp_path, monkeypatch):
    """The bake-off sweeps live x Stop thresholds across four embedders; without
    these the sweep would have to edit the manifest between every cell."""
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer

    _place(eng, tmp_path)
    monkeypatch.setenv("TLDW_DIARIZER_LIVE_THRESHOLD", "0.42")
    monkeypatch.setenv("TLDW_DIARIZER_CLUSTER_THRESHOLD", "0.7")
    wav = tmp_path / "mixed.wav"; _wav(wav)

    loaded = eng.load(OnlineClusterer(), 8, models_dir_override=tmp_path, verify_hashes=False)
    assert loaded.live_threshold == 0.42
    loaded.batch(str(wav), 0.0, 4.0)
    assert _Diarizer.last_config["clustering"]["threshold"] == 0.7


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "5.0", "2.0", "0", "-0.3"])
def test_out_of_range_threshold_env_values_are_ignored(bad, fake_sherpa, tmp_path, monkeypatch):
    """A float that parses is not automatically a usable threshold.

    These are cosine DISTANCES: `nan` makes `(1 - sim) <= nan` false for every
    window, so the live clusterer mints a new speaker per window until the cap;
    `inf` -- or 2.0 itself, the widest possible cosine distance, which the
    guard used to accept against its own docstring (final review Minor 2) --
    collapses every voice into S1. Both are silent, and both destroy live
    labelling for the whole meeting -- so an unusable value falls back to the
    manifest default exactly like an unparseable one (review Minor 3).
    """
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer

    _place(eng, tmp_path)
    monkeypatch.setenv("TLDW_DIARIZER_LIVE_THRESHOLD", bad)
    monkeypatch.setenv("TLDW_DIARIZER_CLUSTER_THRESHOLD", bad)
    wav = tmp_path / "mixed.wav"; _wav(wav)

    loaded = eng.load(OnlineClusterer(), 8, models_dir_override=tmp_path, verify_hashes=False)
    assert loaded.live_threshold == eng.LIVE_THRESHOLD["titanet_small"]
    loaded.batch(str(wav), 0.0, 4.0)
    assert _Diarizer.last_config["clustering"]["threshold"] == eng.CLUSTER_THRESHOLD["titanet_small"]


def test_speechbrain_engine_ignores_the_bakeoff_threshold_env_vars(monkeypatch):
    """The sweep knobs are the ONNX engine's alone (review Minor 5).

    Pinned rather than left to grep: the SpeechBrain engine builds its
    `LoadedEngine` without a `live_threshold`, so it inherits the 0.25 it has
    always run at no matter what the environment says.
    """
    import inspect

    from tldw_chatbook.Audio import diarizer_engine_speechbrain as sb
    from tldw_chatbook.Audio.diarizer_worker import LoadedEngine

    monkeypatch.setenv("TLDW_DIARIZER_LIVE_THRESHOLD", "0.9")
    monkeypatch.setenv("TLDW_DIARIZER_CLUSTER_THRESHOLD", "0.9")
    source = inspect.getsource(sb)
    assert "TLDW_DIARIZER_LIVE_THRESHOLD" not in source
    assert "TLDW_DIARIZER_CLUSTER_THRESHOLD" not in source
    assert "live_threshold" not in source.split("return LoadedEngine(")[1]
    assert LoadedEngine(lambda p, s: [1.0], lambda *a: ([], {}), "x").live_threshold == 0.25


def test_invalid_threshold_env_values_are_ignored(fake_sherpa, tmp_path, monkeypatch):
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer

    _place(eng, tmp_path)
    monkeypatch.setenv("TLDW_DIARIZER_LIVE_THRESHOLD", "not-a-float")
    monkeypatch.setenv("TLDW_DIARIZER_CLUSTER_THRESHOLD", "")
    wav = tmp_path / "mixed.wav"; _wav(wav)

    loaded = eng.load(OnlineClusterer(), 8, models_dir_override=tmp_path, verify_hashes=False)
    assert loaded.live_threshold == eng.LIVE_THRESHOLD["titanet_small"]
    loaded.batch(str(wav), 0.0, 4.0)
    assert _Diarizer.last_config["clustering"]["threshold"] == eng.CLUSTER_THRESHOLD["titanet_small"]


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


def _make_hostile_tar_bz2(real_content: bytes) -> bytes:
    """A tarball where every unsafe way to end up with a member basename of
    `model.onnx` is present alongside the one safe copy (review I3): an
    absolute path, a symlink, a hardlink -- each sized to match
    `real_content` so the C2 exact-size filter alone isn't what rejects
    them -- plus a decoy with the wrong basename entirely."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        decoy = tarfile.TarInfo(name="../evil.onnx")  # wrong basename + traversal
        decoy.size = 4
        tf.addfile(decoy, io.BytesIO(b"evil"))

        absolute = tarfile.TarInfo(name="/abs/model.onnx")  # right basename, absolute path
        absolute.size = len(real_content)
        tf.addfile(absolute, io.BytesIO(b"A" * len(real_content)))

        symlink = tarfile.TarInfo(name="dirB/model.onnx")  # right basename, symlink
        symlink.type = tarfile.SYMTYPE
        symlink.linkname = "/etc/passwd"
        # Matches `real_content`'s size (re-review Minor 1): `size = 0` (a
        # link's natural size) let the C2 exact-size check reject this
        # candidate before the isfile()/issym()/islnk() leg was ever
        # reached, so removing *just* that leg still passed. `addfile` with
        # no `fileobj` writes no data blocks regardless of the declared
        # size, and tarfile does not expect any for a link type on read, so
        # the archive stays internally consistent.
        symlink.size = len(real_content)
        tf.addfile(symlink)

        hardlink = tarfile.TarInfo(name="dirC/model.onnx")  # right basename, hardlink
        hardlink.type = tarfile.LNKTYPE
        hardlink.linkname = "readme.txt"
        hardlink.size = len(real_content)  # see symlink's comment above
        tf.addfile(hardlink)

        readme = tarfile.TarInfo(name="readme.txt")  # unrelated decoy file
        readme.size = 5
        tf.addfile(readme, io.BytesIO(b"hello"))

        real = tarfile.TarInfo(name="real/model.onnx")  # the one safe member
        real.size = len(real_content)
        tf.addfile(real, io.BytesIO(real_content))
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


class _RecordingClient:
    """Wraps a real `httpx.Client` but records every requested URL first --
    proves a forbidden URL was never connected to (review I6), rather than
    merely relying on it failing DNS through the repo's network guard."""

    def __init__(self, inner):
        self._inner = inner
        self.requested: list[str] = []

    def stream(self, method, url, **kw):
        self.requested.append(str(url))
        return self._inner.stream(method, url, **kw)


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
def test_ensure_models_refuses_redirects_off_the_allowlist_but_keeps_a_sibling_that_already_landed(
    tmp_path, stub_server, monkeypatch,
):
    eng = _engine_with_manifest_pointing_at(stub_server, monkeypatch)
    stub_server.routes["/emb.onnx"] = {"status": 302, "location": "http://evil.invalid/payload"}
    models_dir = tmp_path / "models"

    import httpx

    from tldw_chatbook.Audio.diarizer_engine_onnx import ModelsUnavailable

    real_client = httpx.Client(follow_redirects=False, trust_env=True, timeout=5.0)
    recorder = _RecordingClient(real_client)
    try:
        with pytest.raises(ModelsUnavailable, match="download failed"):
            eng.ensure_models("titanet_small", models_dir_override=models_dir, client=recorder)
    finally:
        real_client.close()

    # I6: refused BY THE CODE's own allowlist check, not by evil.invalid
    # failing DNS -- the forbidden URL was never even requested.
    assert not any("evil.invalid" in u for u in recorder.requested)
    # I5: the segmentation asset lands first in the plan and is fully
    # verified before the embedder's redirect gets refused; a whole-call
    # rollback would delete it, contradicting spec §3's "concurrent fetches
    # are harmless" and §8's "retried next Start" (which needs it kept).
    assert (models_dir / eng.SEGMENTATION.file_name).exists()
    assert not (models_dir / eng.EMBEDDERS["titanet_small"].file_name).exists()


@pytest.mark.loopback_network
def test_ensure_models_refuses_a_redirect_that_changes_scheme(tmp_path, stub_server, monkeypatch):
    # Controller ruling (C1): every hop must stay https, with the loopback
    # stub as the sole http exception -- a redirect may never change scheme
    # even onto an otherwise-allowed host. The stub only speaks http, so the
    # reachable direction to pin here is http -> https; the same equality
    # check in `_stream_to_file` (`next_url.scheme != cur_url.scheme`) is
    # what would also refuse the literal https -> http downgrade in
    # production, where the initial hop is always https.
    eng = _engine_with_manifest_pointing_at(stub_server, monkeypatch)
    stub_server.routes["/emb.onnx"] = {"status": 302, "location": "https://github.com/anything"}
    models_dir = tmp_path / "models"

    import httpx

    from tldw_chatbook.Audio.diarizer_engine_onnx import ModelsUnavailable

    real_client = httpx.Client(follow_redirects=False, trust_env=True, timeout=5.0)
    recorder = _RecordingClient(real_client)
    try:
        with pytest.raises(ModelsUnavailable, match="download failed"):
            eng.ensure_models("titanet_small", models_dir_override=models_dir, client=recorder)
    finally:
        real_client.close()

    # The scheme check must refuse this BEFORE ever connecting to the
    # redirect target -- not merely because a real github.com connection
    # would be blocked by this repo's own network guard (the same
    # false-attribution trap review I6 flagged for a different test).
    assert not any("github.com" in u for u in recorder.requested)
    assert not (models_dir / eng.EMBEDDERS["titanet_small"].file_name).exists()


@pytest.mark.loopback_network
def test_ensure_models_follows_an_allowed_redirect_hop(tmp_path, stub_server, monkeypatch):
    # I4: the production happy path (a GitHub release 302ing to
    # objects.githubusercontent.com) had no coverage at all -- a mutant that
    # refuses every 3xx passed 12/12. Route the embedder through one 302.
    # I4 (partial, re-review): the `Location` is relative -- production
    # redirects are absolute today, but pinning the relative-resolution leg
    # (`cur_url.join`, not a bare `httpx.URL(location)`) end-to-end here
    # costs nothing and a mutant swapping that call used to pass 23/23.
    eng = _engine_with_manifest_pointing_at(stub_server, monkeypatch)
    host, port = stub_server.server_address
    stub_server.routes["/redir"] = {"status": 302, "location": "/emb.onnx"}
    emb_asset = dataclasses.replace(eng.EMBEDDERS["titanet_small"], url=f"http://{host}:{port}/redir")
    monkeypatch.setattr(eng, "EMBEDDERS", {**eng.EMBEDDERS, "titanet_small": emb_asset})

    seg, emb = eng.ensure_models("titanet_small", models_dir_override=tmp_path)
    assert seg.exists() and emb.exists()
    assert stub_server.counts["/redir"] == 1 and stub_server.counts["/emb.onnx"] == 1


@pytest.mark.loopback_network
def test_ensure_models_refuses_more_than_five_redirect_hops(tmp_path, stub_server, monkeypatch):
    eng = _engine_with_manifest_pointing_at(stub_server, monkeypatch)
    host, port = stub_server.server_address
    # Six hops -- one past `_MAX_REDIRECTS` -- must be refused.
    for i in range(6):
        stub_server.routes[f"/chain{i}"] = {"status": 302, "location": f"http://{host}:{port}/chain{i + 1}"}
    emb_asset = dataclasses.replace(eng.EMBEDDERS["titanet_small"], url=f"http://{host}:{port}/chain0")
    monkeypatch.setattr(eng, "EMBEDDERS", {**eng.EMBEDDERS, "titanet_small": emb_asset})

    from tldw_chatbook.Audio.diarizer_engine_onnx import ModelsUnavailable

    with pytest.raises(ModelsUnavailable, match="download failed"):
        eng.ensure_models("titanet_small", models_dir_override=tmp_path / "models")
    for i in range(6):
        assert stub_server.counts[f"/chain{i}"] == 1


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


def test_ensure_models_rejects_a_non_positive_budget_up_front(tmp_path, monkeypatch):
    # Minor 1: this must not touch the network at all when there is
    # nothing to fetch OR when budget_s is already exhausted.
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng
    from tldw_chatbook.Audio.diarizer_engine_onnx import ModelsUnavailable

    with pytest.raises(ModelsUnavailable, match="budget exceeded"):
        eng.ensure_models("titanet_small", models_dir_override=tmp_path, budget_s=0.0, client=_PoisonClient())


@pytest.mark.loopback_network
def test_ensure_models_refuses_a_response_bigger_than_the_manifest_size_plus_slack(tmp_path, stub_server, monkeypatch):
    # C2: `_stream_to_file`'s own byte-counted cap, independent of the tar
    # member's own size check below -- a misbehaving/hostile server filling
    # the disk with a plain (non-tarball) asset must be cut off well before
    # any hash check could run.
    eng = _engine_with_manifest_pointing_at(stub_server, monkeypatch, emb_body=b"E" * 1_900_000)
    stub_server.routes["/emb.onnx"] = {"body": b"E" * (1_900_000 + 2 * 1024 * 1024)}
    models_dir = tmp_path / "models"

    from tldw_chatbook.Audio.diarizer_engine_onnx import ModelsUnavailable

    with pytest.raises(ModelsUnavailable, match="download failed"):
        eng.ensure_models("titanet_small", models_dir_override=models_dir)
    assert not (models_dir / eng.EMBEDDERS["titanet_small"].file_name).exists()
    assert not any(models_dir.glob("*.tmp*"))


class _ZeroReader(io.RawIOBase):
    """A `size`-byte stream of zero bytes generated on read, never
    materialised as one giant buffer -- lets the fixture below produce a
    genuinely `size`-byte (and so internally-consistent, bz2-tiny) tar
    member without allocating `size` bytes of Python memory to build it."""

    def __init__(self, size: int) -> None:
        self.remaining = size

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        n = min(len(b), self.remaining)
        b[:n] = bytes(n)
        self.remaining -= n
        return n


def test_extract_tar_member_rejects_a_member_that_lies_about_its_size(tmp_path):
    # C2: a header declaring a huge size must be refused from the header
    # alone -- before `extractfile`/`copyfileobj` ever reads a byte. The
    # reviewer's probe measured this exact shape ballooning a compressed
    # tarball into ~840 MB of RSS with the old (post-read) check; the real
    # (if uninteresting) zero-filled data blocks here keep the archive
    # internally consistent while staying tiny once bz2-compressed.
    from tldw_chatbook.Audio.diarizer_engine_onnx import ModelsUnavailable, _extract_tar_member

    declared_size = 20 * 1024 * 1024  # a multiple of the real ~6 MB manifest size is enough to prove the point
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        huge = tarfile.TarInfo(name="model.onnx")
        huge.size = declared_size
        tf.addfile(huge, _ZeroReader(declared_size))
    tar_path = tmp_path / "huge.tar.bz2"
    tar_path.write_bytes(buf.getvalue())
    dest = tmp_path / "out.onnx"

    with pytest.raises(ModelsUnavailable, match="download failed"):
        _extract_tar_member(tar_path, "model.onnx", dest, expected_size=5_992_913)
    assert not dest.exists()


def test_extract_tar_member_rejects_every_unsafe_candidate_and_picks_the_real_one(tmp_path):
    # I3: mutation coverage for the safe-member filter -- deleting the
    # basename/path/link rejection lines used to leave `12 passed`. Every
    # hostile candidate here shares the "model.onnx" basename (and, for the
    # absolute-path one, the exact expected size too) with the one real,
    # safe member, so only the path/link checks can be what picks correctly.
    from tldw_chatbook.Audio.diarizer_engine_onnx import _extract_tar_member

    real_content = b"R" * 1_000
    tar_path = tmp_path / "fixture.tar.bz2"
    tar_path.write_bytes(_make_hostile_tar_bz2(real_content))
    dest = tmp_path / "out.onnx"

    _extract_tar_member(tar_path, "model.onnx", dest, expected_size=len(real_content))
    assert dest.read_bytes() == real_content


def _make_link_only_tar_bz2(link_type, real_content: bytes) -> bytes:
    """A tarball with exactly one hostile `model.onnx` candidate of
    `link_type` (`tarfile.SYMTYPE` or `tarfile.LNKTYPE`), sized to match
    `real_content`, plus the one real, safe member -- isolates the
    isfile()/issym()/islnk() leg of the filter on its own (re-review item
    3: the combined fixture let a filter that drops only ONE of the two
    link types pass, since ordering/other-member effects could mask it)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        hostile = tarfile.TarInfo(name="model.onnx")
        hostile.type = link_type
        hostile.linkname = "nowhere"  # must not resolve to the real member
        hostile.size = len(real_content)
        tf.addfile(hostile)

        real = tarfile.TarInfo(name="real/model.onnx")
        real.size = len(real_content)
        tf.addfile(real, io.BytesIO(real_content))
    return buf.getvalue()


def test_extract_tar_member_rejects_a_symlink_candidate_on_its_own(tmp_path):
    from tldw_chatbook.Audio.diarizer_engine_onnx import _extract_tar_member

    real_content = b"R" * 1_000
    tar_path = tmp_path / "fixture.tar.bz2"
    tar_path.write_bytes(_make_link_only_tar_bz2(tarfile.SYMTYPE, real_content))
    dest = tmp_path / "out.onnx"

    _extract_tar_member(tar_path, "model.onnx", dest, expected_size=len(real_content))
    assert dest.read_bytes() == real_content


def test_extract_tar_member_rejects_a_hardlink_candidate_on_its_own(tmp_path):
    from tldw_chatbook.Audio.diarizer_engine_onnx import _extract_tar_member

    real_content = b"R" * 1_000
    tar_path = tmp_path / "fixture.tar.bz2"
    tar_path.write_bytes(_make_link_only_tar_bz2(tarfile.LNKTYPE, real_content))
    dest = tmp_path / "out.onnx"

    _extract_tar_member(tar_path, "model.onnx", dest, expected_size=len(real_content))
    assert dest.read_bytes() == real_content


def test_ensure_models_constructs_its_client_with_trust_env_and_manual_redirects(monkeypatch):
    # Minor 8: spec §9 lists "proxy passthrough" as a downloader test; that
    # depends entirely on `trust_env=True` reaching the real `httpx.Client`.
    import httpx

    from tldw_chatbook.Audio import diarizer_engine_onnx as eng

    captured = {}
    real_client_cls = httpx.Client

    def _spy(*a, **kw):
        captured.update(kw)
        return real_client_cls(*a, **kw)

    monkeypatch.setattr(httpx, "Client", _spy)
    client = eng._new_http_client()
    client.close()

    assert captured == {"follow_redirects": False, "trust_env": True, "timeout": 30.0}


class _ScriptedClient:
    """No sockets: maps an exact URL string to one canned response and
    records every URL requested (KeyError on an unscripted one, so a
    forbidden request fails loudly instead of silently proceeding) --
    drives `_stream_to_file` directly against the REAL production
    allowlist (`github.com` / `*.githubusercontent.com`), unlike the
    loopback stub, which is itself an allowed host under the C1 loopback
    exception and so can never reach the host-allowlist checks at all
    (re-review N-I1)."""

    def __init__(self, routes):
        self.routes = routes
        self.requested: list[str] = []

    def stream(self, method, url, **kw):
        self.requested.append(url)
        return contextlib.nullcontext(self.routes[url])


def _fake_response(status_code, *, location=None, body=b""):
    headers = {"location": location} if location else {}
    return types.SimpleNamespace(status_code=status_code, headers=headers, iter_bytes=lambda: iter([body]))


def test_stream_to_file_enforces_the_real_host_allowlist_and_relative_redirects(tmp_path):
    # N-I1: the C1 scheme guard refuses every off-allowlist redirect
    # reachable from the (http-only) loopback stub on SCHEME alone, so no
    # stub-based test could ever reach the host-allowlist checks -- deleting
    # either one left the whole suite green. This drives `_stream_to_file`
    # directly against the real `https://github.com` allowlist instead.
    from tldw_chatbook.Audio.diarizer_engine_onnx import ModelsUnavailable, _stream_to_file

    deadline = time.monotonic() + 30.0

    # (a) initial URL's host is off the allowlist -> refused, nothing requested.
    client = _ScriptedClient({})
    with pytest.raises(ModelsUnavailable, match="download failed"):
        _stream_to_file(client, "https://evil.example/x", tmp_path / "a", deadline, lambda n: None, 999)
    assert client.requested == []

    # (b) redirect target's host is off the allowlist (scheme unchanged) --
    # the HOST check refuses it; evil.invalid is never requested.
    client = _ScriptedClient({
        "https://github.com/x": _fake_response(302, location="https://evil.invalid/y"),
    })
    with pytest.raises(ModelsUnavailable, match="download failed"):
        _stream_to_file(client, "https://github.com/x", tmp_path / "b", deadline, lambda n: None, 999)
    assert client.requested == ["https://github.com/x"]

    # (c) redirect downgrades scheme on an otherwise-allowed host -- the
    # real production downgrade direction (controller ruling C1).
    client = _ScriptedClient({
        "https://github.com/x": _fake_response(302, location="http://github.com/y"),
    })
    with pytest.raises(ModelsUnavailable, match="download failed"):
        _stream_to_file(client, "https://github.com/x", tmp_path / "c", deadline, lambda n: None, 999)
    assert client.requested == ["https://github.com/x"]

    # (d) redirect to an allowed *.githubusercontent.com host, same scheme
    # -> followed, body written.
    dest_d = tmp_path / "d"
    client = _ScriptedClient({
        "https://github.com/x": _fake_response(302, location="https://objects.githubusercontent.com/y"),
        "https://objects.githubusercontent.com/y": _fake_response(200, body=b"hello"),
    })
    _stream_to_file(client, "https://github.com/x", dest_d, deadline, lambda n: None, 999)
    assert dest_d.read_bytes() == b"hello"
    assert client.requested == ["https://github.com/x", "https://objects.githubusercontent.com/y"]

    # (e) a relative Location resolves against the CURRENT url
    # (`cur_url.join`, never a bare `httpx.URL(location)`).
    dest_e = tmp_path / "e"
    client = _ScriptedClient({
        "https://github.com/dir/x": _fake_response(302, location="y"),
        "https://github.com/dir/y": _fake_response(200, body=b"world"),
    })
    _stream_to_file(client, "https://github.com/dir/x", dest_e, deadline, lambda n: None, 999)
    assert dest_e.read_bytes() == b"world"


def test_fetching_the_int8_segmentation_asset_extracts_its_own_member(tmp_path):
    """Final review Minor 1: `_fetch_asset` hard-coded `"model.onnx"` for
    every `kind == "segmentation"` asset, so fetching `SEGMENTATION_INT8` --
    the same tarball, a different member -- would have extracted the FLOAT
    member, failed the size filter and raised "download failed". The member
    comes from the asset now, so the manifest entry is actually fetchable.

    The tarball carries BOTH members at their real relative sizes, so picking
    the wrong one cannot accidentally pass.
    """
    from tldw_chatbook.Audio import diarizer_engine_onnx as eng

    int8_body = b"I" * 1_500
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        for name, content in (("model.onnx", b"F" * 9_000), ("model.int8.onnx", int8_body)):
            info = tarfile.TarInfo(name=f"sherpa-onnx-fixture/{name}")
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    tar = buf.getvalue()

    url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/x/seg.tar.bz2"
    asset = dataclasses.replace(
        eng.SEGMENTATION_INT8, url=url, sha256=hashlib.sha256(int8_body).hexdigest(),
        size=len(int8_body), download_size=len(tar),
    )
    dest = tmp_path / asset.file_name
    client = _ScriptedClient({url: _fake_response(200, body=tar)})

    eng._fetch_asset(client, asset, dest, time.monotonic() + 30.0, lambda n: None)

    assert dest.read_bytes() == int8_body
