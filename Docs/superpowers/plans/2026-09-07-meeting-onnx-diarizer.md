# Meetings ONNX Diarizer Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the base install live speaker labels and a Stop pass without torch, by adding a sherpa-onnx engine to the existing diarizer worker behind the same `Diarizer` seam.

**Architecture:** The worker subprocess keeps one command loop (`serve()`) and gains an `--engine` flag; each engine module returns the `embed`/`batch` callables the loop already consumes. The app-side backend (`LocalDiarizer`) adds the flag, a pre-spawn model download on its warm-up thread, and a per-engine model id. The owner resolves the engine from config and installed packages (by `find_spec`), stamps it on the meeting, and the screen shows it. A bake-off task decides whether ONNX becomes the default.

**Tech Stack:** Python ≥3.11, sherpa-onnx (Apache-2.0, CPU wheels), numpy, httpx, stdlib `wave`/`tarfile`/`hashlib`, pydantic, Textual 8, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-07-meeting-onnx-diarizer-design.md` (binding). Research: `Docs/Design/2026-09-07-self-hosted-diarization-options.md`.

## Global Constraints

- `serve()`, the wire protocol and stdout/stderr framing stay byte-identical; only `embed` now receives `sr`.
- Engine modules import their packages **only inside their loader**; `diarizer_worker.py`, `diarizer_local.py`, `meeting_owner.py`, `meeting_session.py` and the screen never import torch, sherpa-onnx or numpy at module scope (the owner never imports an engine at all; `find_spec` only).
- No network at import, in `prepare()`, or on screen open. Downloads: httpx, initial URL on `github.com`, redirects only to `*.githubusercontent.com`, temp-then-rename, SHA-256 verified on completion and on load, budget 600 s.
- No paths, names, transcript text, vectors, PCM or audio in logs, notifications, worker descriptions or stderr; log `type(exc).__name__` and static reason strings. Any new `logger.*` → `./scripts/preflight.sh` (read the inventory rows, then `--write`).
- Class-keyed TCSS only; UI-ready module census ≤ 972; CSS bare-type-subject ratchet ≤ 274.
- `AUTO_ORDER` starts as `("speechbrain", "onnx")`; it flips to `("onnx", "speechbrain")` only in Task 9 and only if Task 8's gates all pass.
- Never `git stash`. Argument-list subprocesses only. Commit per task with the messages given; trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Run tests with `.venv/bin/python -m pytest <files> -q -p no:cacheprovider` from the worktree.

---

## File map

| File | Responsibility |
|---|---|
| `tldw_chatbook/Audio/diarizer_worker.py` (modify) | command loop, `_framing`, `_map_final_clusters`, `--engine` parsing, `main()` |
| `tldw_chatbook/Audio/diarizer_engine_speechbrain.py` (create) | today's ECAPA loader/embed/batch, moved verbatim; `_cluster_windows` |
| `tldw_chatbook/Audio/diarizer_engine_onnx.py` (create) | manifest, model dir, downloader, sherpa embed/batch loader |
| `tldw_chatbook/Audio/diarizer_local.py` (modify) | `LocalDiarizer(engine)`, `model_id_for`, warm-up fetch, `warmup_status` |
| `tldw_chatbook/Audio/meeting_owner.py` (modify) | config values, `resolve_engine`, `AUTO_ORDER`, `PrepareResult` fields, stamping, `diarizer_status` |
| `tldw_chatbook/Audio/meeting_session.py` (modify) | `MeetingMeta.diarizer_engine/diarizer_model_id`, largest-overlap rule |
| `tldw_chatbook/UI/Screens/meetings_screen.py` (modify) | rail copy per engine, download progress, switch sentence |
| `tldw_chatbook/Utils/optional_deps.py`, `config.py`, `pyproject.toml` (modify) | feature entry, template keys, base deps |
| `Tests/Audio/bakeoff/` (create), `Docs/STT_Evaluation/task-31827/` (create) | harness, DER scorer, report |

---

### Task 1: Split the worker into a loop plus a SpeechBrain engine module

**Files:**
- Modify: `tldw_chatbook/Audio/diarizer_worker.py` (`_reconcile_windows` ~L187-288, `_load_encoder`/`_embed`/`_batch` ~L142-350, `serve()` embed call ~L403, `main()` ~L504-535)
- Create: `tldw_chatbook/Audio/diarizer_engine_speechbrain.py`
- Test: `Tests/Audio/test_diarizer_local.py` (existing worker-loop tests must stay green), `Tests/Audio/test_diarizer_engines.py` (new)

**Interfaces:**
- Produces (worker): `LoadedEngine` (NamedTuple: `embed: Callable[[bytes, int], list[float]]`, `batch: Callable[[str, float, float], tuple[list[dict], dict[str, list[float]]]]`, `model_id: str`); `ENGINES = {"speechbrain": "tldw_chatbook.Audio.diarizer_engine_speechbrain", "onnx": "tldw_chatbook.Audio.diarizer_engine_onnx"}`; `_parse_args(argv) -> tuple[int, str]` (start_id, engine; default engine `"speechbrain"`); `_map_final_clusters(final_clusters, live_centroids, threshold=0.25, start_id=0, out_centroids=None) -> dict[str, str]` where `final_clusters: list[tuple[str, np.ndarray, float]]` = (final label, centroid, seconds).
- Produces (engine module contract): `load(live, max_speakers: int) -> LoadedEngine`; `MODEL_ID: str` constant.
- `serve()` calls `embed(pcm, sr)` (was `embed(pcm)`).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Audio/test_diarizer_engines.py
import numpy as np
import pytest
from tldw_chatbook.Audio import diarizer_worker as w


def test_parse_args_defaults_to_speechbrain_and_reads_engine():
    assert w._parse_args([]) == (0, "speechbrain")
    assert w._parse_args(["--start-id", "7", "--engine", "onnx"]) == (7, "onnx")
    with pytest.raises(ValueError):
        w._parse_args(["--engine", "bogus"])


def test_map_final_clusters_reuses_matches_and_mints_past_start_id():
    live = {"S1": np.array([1.0, 0.0]), "S2": np.array([0.0, 1.0])}
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Audio/test_diarizer_engines.py -q -p no:cacheprovider`
Expected: FAIL (`_parse_args` / `_map_final_clusters` / module not found).

- [ ] **Step 3: Implement**

In `diarizer_worker.py`:
- Add `LoadedEngine`, `ENGINES`, `_parse_args` (replaces `_parse_start_id`; unknown engine → `ValueError`).
- Split `_reconcile_windows`: keep its signature for the SpeechBrain path but have it build `final_clusters` (label, mean centroid, seconds) and call the new `_map_final_clusters`, which holds the existing map / mint / seconds-weighted fold code verbatim (the block starting at `mapping = reconcile(...)` through `out_centroids[lid] = ...`).
- Change the assign branch to `live.assign(embed(pcm, sr), seconds=seconds)`.
- `main()`: `start_id, engine = _parse_args(sys.argv[1:])`; import the module named in `ENGINES[engine]` inside the existing `try` and call `engine_mod.load(live, max_speakers)`; on any exception keep the `ERROR load <Type>` line; then `serve(stdin, stdout, live, loaded.embed, loaded.batch)`. `MODEL_ID` in the worker becomes an import-free alias of the SpeechBrain id (`MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb@unpinned"`) so nothing that still reads it breaks before Task 5.
- Move `_load_encoder`, `_embed`, `_batch` and the window-clustering half of `_reconcile_windows` (as `_cluster_windows`) into `diarizer_engine_speechbrain.py` verbatim, with `load(live, max_speakers)` that imports torch/numpy via the existing lazy loaders and returns `LoadedEngine(lambda pcm, sr: _embed(encoder, torch, np, pcm), lambda wav, s, e: _batch(encoder, torch, np, live, wav, s, e, max_speakers), MODEL_ID)`. `_batch` calls back into `diarizer_worker._map_final_clusters`.

- [ ] **Step 4: Run the worker suites**

Run: `.venv/bin/python -m pytest Tests/Audio/test_diarizer_engines.py Tests/Audio/test_diarizer_local.py Tests/Audio/test_diarizer_cluster.py -q -p no:cacheprovider`
Expected: PASS, all existing worker-loop tests untouched.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/diarizer_worker.py tldw_chatbook/Audio/diarizer_engine_speechbrain.py Tests/Audio/test_diarizer_engines.py
git commit -m "refactor(meetings): diarizer worker takes an --engine flag; SpeechBrain engine moves to its own module (31827)"
```

---

### Task 2: The ONNX engine module (manifest, embed, batch)

**Files:**
- Create: `tldw_chatbook/Audio/diarizer_engine_onnx.py`
- Test: `Tests/Audio/test_diarizer_engine_onnx.py`

**Interfaces:**
- Consumes: `diarizer_worker.LoadedEngine`, `diarizer_worker._map_final_clusters`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class ModelAsset: key: str; kind: str  # "segmentation" | "embedder"
                    file_name: str; url: str; sha256: str; size: int; licence: str
  SEGMENTATION = ModelAsset("segmentation", "segmentation", "pyannote-segmentation-3-0.onnx",
      "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
      sha256="<fill: sha256 of the EXTRACTED model.onnx>", size=<fill>, licence="MIT (pyannote/segmentation-3.0)")
  EMBEDDERS: dict[str, ModelAsset] = {
      "titanet_small": ModelAsset(..., "nemo_en_titanet_small.onnx", ".../speaker-recongition-models/nemo_en_titanet_small.onnx", sha256=<fill>, size=40257283, licence="CC-BY-4.0 (NVIDIA)"),
      "wespeaker_resnet34": ModelAsset(..., "wespeaker_en_voxceleb_resnet34.onnx", ..., size=26534365, licence="Apache-2.0 (WeSpeaker)"),
      "eres2net_en": ModelAsset(..., "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx", ..., size=26485263, licence="Apache-2.0 (3D-Speaker; verify ModelScope terms)"),
      "campplus_en": ModelAsset(..., "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx", ..., size=29596978, licence="Apache-2.0 (3D-Speaker; verify ModelScope terms)"),
  }
  DEFAULT_EMBEDDER = "titanet_small"        # Task 9 may change it
  CLUSTER_THRESHOLD: dict[str, float]         # per embedder; start 0.5 for all, bake-off tunes
  MIN_DURATION_ON, MIN_DURATION_OFF = 0.3, 0.5
  LIVE_THREADS, BATCH_THREADS = 2, 4
  CENTROID_SECONDS = 30.0
  def model_id_for_embedder(key: str) -> str   # f"sherpa-onnx/{file_name}@{sha256[:12]}"
  def models_dir(override: Path | None = None) -> Path   # override or get_user_data_dir()/models/diarization/onnx
  def model_paths(embedder: str, override=None) -> tuple[Path, Path]   # (segmentation, embedder)
  def models_ready(embedder: str, override=None) -> bool               # presence + byte size only
  def load(live, max_speakers: int, *, embedder: str | None = None, models_dir_override: Path | None = None) -> LoadedEngine
  MODEL_ID = model_id_for_embedder(DEFAULT_EMBEDDER)
  ```
- The `<fill>` hashes are computed in Step 3 by downloading each asset once (`shutil`-free: `hashlib.sha256` over the file) and pasted as literals; the commit message records them. `size` values above come from the release listing.

- [ ] **Step 1: Write the failing tests** (a fake `sherpa_onnx` injected through `sys.modules`)

```python
# Tests/Audio/test_diarizer_engine_onnx.py
import sys, types, wave, struct, math
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
        # positive DC for 0-2 s (speaker A), negative for 2-4 s (speaker B)
        f.writeframes(b"".join(struct.pack("<h", 8000 if i < frames // 2 else -8000) for i in range(frames)))


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
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest Tests/Audio/test_diarizer_engine_onnx.py -q -p no:cacheprovider` → FAIL (module missing).

- [ ] **Step 3: Implement** `diarizer_engine_onnx.py`

- `load(...)`: `import sherpa_onnx, numpy` inside; resolve paths via `model_paths`; when `verify_hashes` (default True) compare SHA-256 of each file with the manifest and raise `ValueError("model hash mismatch")` on mismatch (the worker frames it as `ERROR load ValueError`); build `SpeakerEmbeddingExtractor(SpeakerEmbeddingExtractorConfig(model=..., num_threads=LIVE_THREADS, provider="cpu"))`.
- `embed(pcm, sr)`: `np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0`; stream → `accept_waveform(sr, samples)` → `input_finished()` → `compute` → unit-normalise (zero vector → return the zero vector; the worker's `_unit` refuses it downstream).
- `batch(wav, start_s, end_s)`: read the span with `wave` (`setpos`/`readframes` on the 16 kHz mono PCM16 file the meeting wrote; raise on any other format), float32; build `OfflineSpeakerDiarization` with the segmentation + embedder config, `FastClusteringConfig(num_clusters=-1, threshold=CLUSTER_THRESHOLD[embedder])`, `min_duration_on/off`; `result = sd.process(samples).sort_by_start_time()`; group segments per `r.speaker`; per cluster embed its longest segments separately until `CENTROID_SECONDS` is covered and average the unit vectors; if clusters > `max_speakers`, fold the smallest (by seconds) into the nearest remaining centroid until within the cap; build `final_clusters = [(f"F{k}", centroid, seconds)]`; `mapping = _map_final_clusters(final_clusters, live.centroids(), threshold=live.threshold, start_id=live.max_id, out_centroids=out)`; return `([{"start_s": start_s + r.start, "end_s": start_s + r.end, "speaker": mapping[f"F{r.speaker}"]} ...], out)`.
- Compute and paste the manifest hashes: download each asset once into a scratch dir with `httpx`, for the segmentation tarball extract only the member ending in `/model.onnx` with `tarfile` (reject members with `..` or absolute names), hash the resulting `.onnx` files, paste the hex literals. Keep the tarball's `model.int8.onnx` as a second `segmentation_int8` asset for the bake-off.

- [ ] **Step 4: Run** — the new test file, then `Tests/Audio/test_diarizer_local.py` (unchanged) → PASS. Also `python -c "import sys; import tldw_chatbook.Audio.diarizer_engine_onnx; print('sherpa_onnx' in sys.modules, 'numpy' in sys.modules)"` → `False False`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/diarizer_engine_onnx.py Tests/Audio/test_diarizer_engine_onnx.py
git commit -m "feat(meetings): sherpa-onnx diarizer engine — manifest, live embed, offline Stop pass reconciled onto live ids (31827)"
```

---

### Task 3: Model downloader and placement

**Files:**
- Modify: `tldw_chatbook/Audio/diarizer_engine_onnx.py`
- Test: `Tests/Audio/test_diarizer_engine_onnx.py`

**Interfaces:**
- Produces: `ensure_models(embedder: str, *, models_dir_override: Path | None = None, progress: Callable[[str], None] | None = None, budget_s: float = 600.0, client: Any | None = None) -> tuple[Path, Path]`; raises `ModelsUnavailable(RuntimeError)` (static message: `"download failed"`, `"hash mismatch"`, `"budget exceeded"`, `"air-gapped file invalid"`). `progress` receives static strings of the form `"downloading 12 / 35 MB"` (integers only).
- Constants: `ALLOWED_HOSTS = ("github.com",)`, `ALLOWED_REDIRECT_SUFFIX = ".githubusercontent.com"`, `DOWNLOAD_BUDGET_S = 600.0`.

- [ ] **Step 1: Write the failing tests** (a `threading` `http.server` stub on 127.0.0.1 serving the fixture bytes; the manifest URL host is monkeypatched to the stub via a `_url_for(asset)` seam)

```python
def test_ensure_models_downloads_verifies_and_renames_atomically(tmp_path, stub_server, monkeypatch):
    eng = _engine_with_manifest_pointing_at(stub_server, monkeypatch)   # helper: patches ALLOWED_HOSTS + urls + sha of the fixture bytes
    seen = []
    seg, emb = eng.ensure_models("titanet_small", models_dir_override=tmp_path, progress=seen.append)
    assert seg.exists() and emb.exists() and not list(tmp_path.glob("*.tmp*"))
    assert seen and seen[0].startswith("downloading ") and "MB" in seen[0]

def test_ensure_models_refuses_a_bad_hash_and_refetches_once(tmp_path, stub_server, monkeypatch):
    ...  # first response corrupted -> second good -> success and exactly two requests for that asset

def test_ensure_models_refuses_redirects_off_the_allowlist(tmp_path, stub_server, monkeypatch):
    ...  # stub answers 302 to http://evil.invalid -> ModelsUnavailable("download failed"), nothing written

def test_ensure_models_respects_air_gapped_dir_without_network(tmp_path, monkeypatch):
    ...  # pre-placed good files + client that raises on any request -> returns paths, client never called
    ...  # pre-placed BAD file -> ModelsUnavailable("air-gapped file invalid"), client never called

def test_ensure_models_gives_up_after_the_budget(tmp_path, stub_server, monkeypatch):
    ...  # stub sleeps per chunk; budget_s=0.2 -> ModelsUnavailable("budget exceeded")

def test_ensure_models_is_a_noop_when_files_are_present_and_valid(tmp_path, ...):
    ...  # second call performs zero requests
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement** — `ensure_models` for each of (segmentation, embedder): if `models_ready` and hash ok → skip; if `models_dir_override` is set (air-gapped) never touch the network; else `httpx.Client(follow_redirects=False, trust_env=True, timeout=30)` GET, follow at most 5 redirects manually checking the host suffix, stream to `<file>.tmp-<pid>-<token>`, `os.replace`, hash; the tarball path extracts `model.onnx`/`model.int8.onnx` with the safe member filter; one re-fetch on mismatch; wall-clock budget checked between chunks. Progress every ~1 MB.

- [ ] **Step 4: Run** the file → PASS. **Step 5: Commit** — `git commit -m "feat(meetings): download and verify the ONNX diarizer models into the user data dir (31827)"`.

---

### Task 4: `LocalDiarizer` — engine flag, warm-up fetch, model id, status

**Files:**
- Modify: `tldw_chatbook/Audio/diarizer_local.py` (`__init__` ~L100-160, `_command` ~L200, `_start` ~L232, `_watch_stderr` ~L262; constants ~L69-90)
- Test: `Tests/Audio/test_diarizer_local.py`

**Interfaces:**
- Produces: `class LocalDiarizer(engine: str = "speechbrain", max_speakers=8, *, spawn=..., assign_budget_s=..., voiceprint=None, match_threshold=0.2, match_min_seconds=4.0, embedder: str | None = None, models_dir_override: Path | None = None, ensure_models: Callable[..., Any] | None = None)`; `SpeechBrainDiarizer = functools.partial(LocalDiarizer, engine="speechbrain")` kept as a class alias (subclass with the default) so `isinstance`/imports/tests keep working; `def model_id_for(engine: str, embedder: str | None = None) -> str`; property `warmup_status -> str` in `{"downloading <a> / <b> MB", "warming up", "ready", "unavailable"}`; `COARSE_MODELS_UNAVAILABLE = "models unavailable"`; `MODELS_DOWNLOAD_BUDGET_S = 600.0`.
- `_command()` appends `["--engine", engine]`.

- [ ] **Step 1: Write the failing tests** (existing `FakeProc` idiom)

```python
def test_local_diarizer_onnx_downloads_then_spawns_and_reports_status():
    calls = []
    gate = threading.Event()
    def ensure(embedder, *, models_dir_override, progress, budget_s):
        progress("downloading 1 / 35 MB"); calls.append(embedder); gate.wait(2.0); return (Path("s"), Path("e"))
    proc = FakeProc(['{"id": "S1", "seq": 0, "self": false}\n'])
    spawned = []
    d = LocalDiarizer(engine="onnx", spawn=lambda cmd, **k: (spawned.append(cmd), proc)[1], ensure_models=ensure)
    assert d.warmup_status.startswith("downloading")      # constructor returned immediately
    assert spawned == []                                    # not spawned until the fetch finishes
    gate.set()
    assert d.wait_ready(2.0) is True and d.warmup_status == "ready"
    assert "--engine" in spawned[0] and spawned[0][spawned[0].index("--engine") + 1] == "onnx"

def test_local_diarizer_onnx_download_failure_marks_coarse_without_spawning():
    from tldw_chatbook.Audio.diarizer_engine_onnx import ModelsUnavailable
    def ensure(*a, **k): raise ModelsUnavailable("download failed")
    spawned = []
    d = LocalDiarizer(engine="onnx", spawn=lambda cmd, **k: spawned.append(cmd), ensure_models=ensure)
    assert d.wait_ready(2.0) is False and spawned == []
    assert d.coarse_reason == COARSE_MODELS_UNAVAILABLE and d.warmup_status == "unavailable"
    assert d.assign(_PCM, 16000, 0) is None

def test_speechbrain_engine_path_is_unchanged():
    proc = FakeProc(['{"id": "S1", "seq": 0, "self": false}\n'])
    d = _ready(SpeechBrainDiarizer(spawn=lambda *a, **k: proc))
    assert d.assign(_PCM, 16000, 0) == "S1"

def test_model_id_for_each_engine():
    assert model_id_for("speechbrain") == "speechbrain/spkrec-ecapa-voxceleb@unpinned"
    assert model_id_for("onnx").startswith("sherpa-onnx/nemo_en_titanet_small.onnx@")
    assert model_id_for("onnx", "wespeaker_resnet34").startswith("sherpa-onnx/wespeaker_en_voxceleb_resnet34.onnx@")
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement** — for `engine == "onnx"` the constructor starts a daemon thread `_warmup` that calls `ensure_models` (injected or the real one, imported lazily from `diarizer_engine_onnx`) with `progress` writing `self._status`, then calls `_start()`; failure → `_mark_coarse(COARSE_MODELS_UNAVAILABLE)`, `_degraded = True`, `self._ready.set()` (so `wait_ready` returns False). The READY timeout starts when the spawn happens (the fetch has its own budget). `_watch_stderr` sets `_status = "ready"` after `ready.set()`; `_status = "warming up"` between spawn and READY. For `speechbrain` the flow is exactly today's. `model_id_for` reads `diarizer_engine_onnx.model_id_for_embedder` lazily (the module is import-cheap) and returns the SpeechBrain constant otherwise. The worker's env gains `TLDW_DIARIZER_EMBEDDER` and `TLDW_DIARIZER_MODELS_DIR` so the ONNX `load` finds the same files the app fetched.

- [ ] **Step 4: Run** `Tests/Audio/test_diarizer_local.py Tests/Audio/test_diarizer_engine_onnx.py` → PASS (existing tests unchanged). **Step 5: Commit** — `git commit -m "feat(meetings): LocalDiarizer selects the worker engine, fetches ONNX models on its warm-up thread, reports status (31827)"`.

---

### Task 5: Owner — config, engine resolution, prepare fields, stamping, model ids

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_owner.py` (`MeetingSettings` ~L111-170, `from_config` ~L169-230, `DIARIZATION_MODULES` L44, `PrepareResult` ~L296-325, `diarization_requirements` L327, `build_diarizer` L349, `_load_voiceprint` L644, `_voice_match_off_for` L699, `prepare` ~L805-826, `_offer_for` L1138, the three `from .diarizer_worker import MODEL_ID` sites ~L671/1226/1433, `start()` stamping)
- Modify: `tldw_chatbook/config.py` (`[meetings]` template ~L5218-5232), `tldw_chatbook/Utils/optional_deps.py` (~L272), `pyproject.toml` (`dependencies` L38-83)
- Test: `Tests/Audio/test_meeting_owner.py`, `Tests/Audio/test_meeting_import_safety.py`

**Interfaces:**
- Produces: `ENGINE_MODULES = {"onnx": ("sherpa_onnx", "numpy"), "speechbrain": ("torch", "torchaudio", "speechbrain", "sklearn")}`; `AUTO_ORDER: tuple[str, ...] = ("speechbrain", "onnx")`; `def resolve_engine(settings, find_spec=importlib.util.find_spec) -> tuple[str | None, tuple[str, ...]]` (engine, missing packages of the explicit choice or of the last candidate); `MeetingSettings.diarizer_backend: str = "auto"` (validator: `{"auto","onnx","speechbrain","server","local"}`, `local` → `auto`), `onnx_embedder: str = "titanet_small"` (validator: manifest keys), `onnx_models_dir: Path | None = None`; `PrepareResult.diarizer_engine: str | None = None`, `.diarizer_missing: tuple[str, ...] = ()`, `.diarizer_models_ready: bool = False`; `VoiceMatchState.detail: str | None = None`; `MeetingSessionOwner.diarizer_status() -> str | None` (the live diarizer's `warmup_status` or None); `MeetingMeta.diarizer_engine`/`diarizer_model_id` stamped at `start()` (fields added in Task 6 — this task writes them via `setattr`-free dataclass fields, so **Task 6 must land before this task is merged**; order the execution 6 then 5 if working strictly sequentially, or do Task 6's meta fields first inside this task).
- `build_diarizer(settings, voiceprint)` → `LocalDiarizer(engine=resolved, embedder=settings.onnx_embedder, models_dir_override=settings.onnx_models_dir, ...)`; `server` → `NotImplementedError` as today.
- `optional_deps`: new feature `"diarization_onnx"` (`("sherpa_onnx", "numpy")`, area media, label "Speaker diarization (ONNX)").
- `pyproject.toml`: add `"sherpa-onnx>=1.13"` and `"numpy"` to `dependencies`.

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_engine_walks_auto_order_by_find_spec(tmp_path):
    s = _settings(tmp_path, live_diarization=True, diarizer_backend="auto")
    have = {"sherpa_onnx", "numpy"}
    fs = lambda name, *a: object() if name in have else None
    assert mo.resolve_engine(s, find_spec=fs) == ("onnx", ())          # torch absent -> onnx
    have |= {"torch", "torchaudio", "speechbrain", "sklearn"}
    assert mo.resolve_engine(s, find_spec=fs)[0] == mo.AUTO_ORDER[0]  # both present -> order decides
    assert mo.resolve_engine(s, find_spec=lambda *a: None) == (None, mo.ENGINE_MODULES[mo.AUTO_ORDER[-1]])

def test_resolve_engine_explicit_choice_reports_its_missing_packages(tmp_path):
    s = _settings(tmp_path, live_diarization=True, diarizer_backend="speechbrain")
    assert mo.resolve_engine(s, find_spec=lambda *a: None) == (None, ("torch", "torchaudio", "speechbrain", "sklearn"))

def test_legacy_local_backend_value_reads_as_auto(tmp_path):
    assert _settings(tmp_path, diarizer_backend="local").diarizer_backend == "auto"
    with pytest.raises(ValueError):
        _settings(tmp_path, diarizer_backend="bogus")

def test_prepare_reports_engine_missing_and_models_ready(tmp_path, monkeypatch):
    ...  # explicit onnx, find_spec ok, models absent -> diarizer_engine == "onnx", diarizer_models_ready False, live_diarization_active True

def test_build_diarizer_passes_the_resolved_engine(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("tldw_chatbook.Audio.diarizer_local.LocalDiarizer", lambda **kw: seen.update(kw) or object())
    ...  # -> seen["engine"] == "onnx", seen["embedder"] == "titanet_small"

def test_start_stamps_engine_and_model_id_onto_meta(tmp_path, monkeypatch):
    ...  # meta.diarizer_engine == "onnx"; meta.diarizer_model_id == model_id_for("onnx")

def test_voiceprint_load_uses_the_resolved_engines_model_id(tmp_path, monkeypatch):
    ...  # fake store records expected_model_id; with onnx resolved it equals model_id_for("onnx")

def test_no_learning_offer_when_the_stored_voiceprint_is_from_another_engine(tmp_path, monkeypatch):
    ...  # store.load returns reason "needs_reenrollment" -> learning_offer(result) is None even for a plain-call you.wav meeting

def test_switch_copy_names_the_engines(tmp_path, monkeypatch):
    ...  # stored id speechbrain, resolved onnx -> voice_match.detail == "Voiceprint was recorded with SpeechBrain; the active engine is ONNX."

def test_owner_never_imports_an_engine_package(tmp_path):
    src = Path(mo.__file__).read_text()
    assert "import sherpa_onnx" not in src and "import torch" not in src
```

Extend `Tests/Audio/test_meeting_import_safety.py`: the owner/screen probes also assert `"sherpa_onnx" not in sys.modules` and `"numpy" not in sys.modules`.

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement** per the interfaces. `live_diarization_active = settings.live_diarization and engine is not None`. `_voice_match_off_for` reads `prepared.live_diarization_active` (unchanged) and no longer mentions `"local"`. Replace the three `diarizer_worker.MODEL_ID` imports with `model_id_for(self._resolved_engine, self.settings.onnx_embedder)` (`_resolved_engine` cached by `prepare()` and refreshed at `start()`). `_load_voiceprint` sets `detail` when `result.reason == "needs_reenrollment"` and both ids map to known engines (`speechbrain/` → SpeechBrain, `sherpa-onnx/` → ONNX). `diarizer_status()` returns `getattr(self.session._diarizer, "warmup_status", None)` when a session is live, else the retained/enrollment diarizer's, else None. Config template comments for the three keys next to `live_diarization`; `optional_deps` entry; `pyproject` base deps; run `./scripts/preflight.sh` (inventory rows if any logger line changed).

- [ ] **Step 4: Run** `Tests/Audio/test_meeting_owner.py Tests/Audio/test_meeting_import_safety.py Tests/Audio/test_meeting_diarization_session.py Tests/Performance/test_ui_ready_module_census.py` → PASS; `./scripts/preflight.sh` green.

- [ ] **Step 5: Commit** — `git commit -m "feat(meetings): resolve the diarizer engine from config and installed packages; stamp it; per-engine voiceprint model ids (31827)"`.

---

### Task 6: Session — meta fields and the largest-overlap rule

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_session.py` (`MeetingMeta` ~L30-62; Stop overlay ~L731-745)
- Test: `Tests/Audio/test_meeting_diarization_session.py`, `Tests/Audio/test_meeting_session.py`

**Interfaces:**
- Produces: `MeetingMeta.diarizer_engine: str | None = None`, `MeetingMeta.diarizer_model_id: str | None = None` (persisted; back-filled as None on read); `_speaker_for_segment(meeting_segment, speaker_segments) -> str | None` implementing largest overlap with midpoint tiebreak.

- [ ] **Step 1: Write the failing tests**

```python
def test_stop_overlay_picks_the_largest_overlap_not_the_first_midpoint_hit(tmp_path, meeting_session_with_fake_capture):
    # transcript segment 1.0-3.0; batch: S1 0.0-2.2 (overlap 1.2), S2 1.9-4.0 (overlap 1.1) -> S1 by overlap;
    # midpoint 2.0 is inside BOTH; the old rule returned the first in list order.
    class Batch(FakeDiarizer):
        def diarize(self, *a): return [SpeakerSegment(0.0, 2.2, "S1"), SpeakerSegment(1.9, 4.0, "S2")]
    ...
    assert seg.speaker_id == "S1"
    # and the mirror case: batch [S1 0.0-2.05, S2 1.5-4.0] -> S2 (overlap 1.5 vs 1.05)

def test_stop_overlay_falls_back_to_midpoint_when_overlaps_tie(...):
    ...

def test_meta_round_trips_engine_fields_and_backfills_old_folders(tmp_path):
    ...  # to_json carries both; read_meeting_json on a payload without them -> None, None
```

- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement** the helper and use it in the overlay loop (keep the `transitions` bookkeeping identical). **Step 4: Run** the two session files → PASS. **Step 5: Commit** — `git commit -m "feat(meetings): stamp the diarizer engine on meeting.json; Stop overlay picks the largest overlap (31827)"`.

---

### Task 7: Meetings screen — engine copy, download progress, switch sentence

**Files:**
- Modify: `tldw_chatbook/UI/Screens/meetings_screen.py` (`_live_diarization_copy` ~L419-439, `_tick` (timer), `_render_voice_match`)
- Test: `Tests/UI/test_meetings_screen.py` (extend `FakeOwner.prepare` result and add `diarizer_status`)

**Interfaces:**
- Consumes: `PrepareResult.diarizer_engine/diarizer_missing/diarizer_models_ready`, `owner.diarizer_status()`, `VoiceMatchState.detail`.
- Produces: rail copy exactly: `"on (ONNX)"`, `"on (ONNX, ~N MB of models fetched at Start)"` (N = manifest total for the configured embedder, rounded), `"on (SpeechBrain)"`, `"off (not enabled in settings)"`, `"off (install the diarization extra)"` when `auto` found nothing, `"off (missing: <pkgs>)"` for an explicit choice, `"off (unsupported backend: server)"`. While a session is live and `diarizer_status()` starts with `"downloading"`, the live-labels line shows `"Live speaker labels: downloading 12 / 35 MB"`, then `"warming up"`, then the engine copy.

- [ ] **Step 1: Write the failing tests** (pilots at 160×45 AND 100×30; assert the exact strings; a FakeOwner whose `diarizer_status()` returns a scripted sequence and whose `voice_match.detail` is set).
- [ ] **Step 2: Run to verify they fail.** **Step 3: Implement** (no new TCSS; if a class rule is needed keep it class-keyed; run `python tldw_chatbook/css/build_css.py`). **Step 4: Run** `Tests/UI/test_meetings_screen.py Tests/UI/test_meetings_wiring.py Tests/Performance/test_ui_ready_module_census.py` + the CSS ratchet test → PASS, census ≤ 972, ratchet 274. **Step 5: Commit** — `git commit -m "feat(meetings): rail shows the diarizer engine, model download progress and the voiceprint engine switch (31827)"`.

---

### Task 8: The bake-off harness, DER scorer and report

**Files:**
- Create: `Tests/Audio/bakeoff/__init__.py`, `Tests/Audio/bakeoff/der.py`, `Tests/Audio/bakeoff/test_der.py`, `Tests/Audio/bakeoff/corpus.py`, `Tests/Audio/bakeoff/run_bakeoff.py`, `.github/workflows/diarizer-bakeoff.yml`, `Docs/STT_Evaluation/task-31827/README.md`
- Modify: `Tests/Audio/test_diarizer_helper_real.py`, `Tests/Audio/test_voiceprint_real.py` (add an `--engine onnx` run, gated on `TLDW_RUN_DIARIZER_TEST=1` and `sherpa_onnx` importable), `pyproject.toml`/`pytest.ini` (`Tests/Audio/bakeoff/run_bakeoff.py` is not collected: name it without the `test_` prefix and add the directory to `norecollectdirs` only for the runner, keeping `test_der.py` collected)

**Interfaces:**
- `der.py`: `def der(reference: list[tuple[float, float, str]], hypothesis: list[tuple[float, float, str]], collar: float = 0.25) -> float` (missed + false alarm + confusion over total reference speech, optimal speaker mapping by Hungarian assignment over overlap — implement with a small permutation search since speaker counts are ≤ 10).
- `corpus.py`: the fixed list of VoxConverse dev file ids (20) and AMI meeting ids (4), download helpers writing under `Tests/Audio/bakeoff/data/` (git-ignored), RTTM parsing.
- `run_bakeoff.py --engines speechbrain,onnx --embedders titanet_small,wespeaker_resnet34,eres2net_en,campplus_en --out Docs/STT_Evaluation/task-31827/report.md`: per (engine, embedder, segmentation float/int8): Stop-pass DER (drive the real worker through `LocalDiarizer.diarize`), live purity/coverage (feed 3 s windows through `assign` against reference majority speakers), RTF and peak RSS of the worker (`resource`/`psutil` if present), per-window embed latency, self-match separation (enrol from one VoxConverse speaker's turns via `enroll_from_pcm`, score against that speaker's held-out turns and the others); machine info, sherpa-onnx version, model hashes, commit SHA.
- Workflow: `workflow_dispatch` on `ubuntu-latest`, installs the base deps + the diarization extra, runs the harness on the corpus subset, uploads `report.md` as an artifact.

- [ ] **Step 1: Write the failing test** for the scorer:

```python
def test_der_on_a_hand_computed_example():
    ref = [(0.0, 10.0, "A"), (10.0, 20.0, "B")]
    hyp = [(0.0, 12.0, "x"), (12.0, 20.0, "y")]     # 2 s confusion, best map x->A y->B
    assert der(ref, hyp, collar=0.0) == pytest.approx(2.0 / 20.0)
    hyp2 = [(0.0, 20.0, "x")]                        # 10 s confusion
    assert der(ref, hyp2, collar=0.0) == pytest.approx(0.5)
    assert der(ref, [], collar=0.0) == pytest.approx(1.0)   # all missed
```

- [ ] **Step 2: Run to verify it fails.** **Step 3: Implement** `der.py`, then the corpus and runner, then the workflow and the README (how to run, the gates, the corpus list). **Step 4: Run** `Tests/Audio/bakeoff/test_der.py` → PASS; run the harness on the M-series machine and via the workflow; commit `report.md`. **Step 5: Commit** — `git commit -m "test(meetings): diarizer bake-off harness, DER scorer and report for the ONNX default decision (31827)"`.

---

### Task 9: Go/no-go, default flip, docs and invariants

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_owner.py` (`AUTO_ORDER`), `tldw_chatbook/Audio/diarizer_engine_onnx.py` (`DEFAULT_EMBEDDER`, `CLUSTER_THRESHOLD`, licence field), `Docs/superpowers/specs/2026-09-07-meeting-onnx-diarizer-design.md` (§10 outcome), `Docs/User_Guide/meetings.md`, `Tests/Audio/test_meeting_import_safety.py`
- Test: `Tests/Audio/test_meeting_owner.py` (an `AUTO_ORDER` test asserting the recorded outcome), the import-safety file

- [ ] **Step 1:** Read `Docs/STT_Evaluation/task-31827/report.md`. Apply the gates from spec §7 (DER within 2 points of ECAPA; live purity within 3; RTF ≤ 0.15 on both machines; per-window embed ≤ 150 ms M-series / ≤ 300 ms runner; self-match separation within 0.05). Record the outcome in spec §10 with the numbers.
- [ ] **Step 2:** If all pass: `AUTO_ORDER = ("onnx", "speechbrain")`, `DEFAULT_EMBEDDER` = the winner, `CLUSTER_THRESHOLD[winner]` = the tuned value, and the winner's licence line goes into the user guide's attribution. If any fails: leave `AUTO_ORDER` SpeechBrain-first and document the failed gate. Either way update the test that pins `AUTO_ORDER`.
- [ ] **Step 3:** User guide: engines and the three config values, what each needs installed, the first-Start download (size, folder, GitHub source, air-gapped `onnx_models_dir`), that `auto` can change engines when the torch extra is installed or removed (re-enrollment), the scrolling rail note, the attribution line; refresh the "Verified against" stamp.
- [ ] **Step 4:** Extend `Tests/Audio/test_meeting_import_safety.py` with the `--engine onnx` worker probe (fake `sherpa_onnx` in `sys.modules`; assert `"torch" not in sys.modules` after `load`). Run `Tests/Audio` in full, the UI meetings suites, census, CSS ratchet, `./scripts/preflight.sh`.
- [ ] **Step 5: Commit** — `git commit -m "feat(meetings): <ONNX becomes the default live diarizer | ONNX ships as the torch-free fallback> per the bake-off; docs and invariants (31827)"`.

---

## Self-review

**Spec coverage:** §2 engine layout → T1/T2/T4; §3 acquisition → T2 (manifest) + T3 (downloader) + T4 (warm-up); §4 selection → T5 + T7 copy; §5 Stop pass → T2 + T6 overlap rule; §6 voiceprint/model ids → T4 (`model_id_for`) + T5 (load/offer/copy); §7 bake-off → T8 + T9; §8 degradation → T3/T4/T5 reasons; §9 tests → each task + T9 invariants; §10/§11 → T9. Packaging note (frozen app) stays with TASK-31747 as the spec says.
**Placeholders:** the manifest `sha256`/`size` literals are computed in T2 Step 3 (a stated action, not a TBD); T9's outcome is decided by the report.
**Type consistency:** `LoadedEngine.embed(pcm, sr)` (T1) ↔ `serve()` (T1) ↔ ONNX `embed` (T2); `_map_final_clusters(final_clusters, live_centroids, threshold, start_id, out_centroids)` (T1) ↔ T2; `ensure_models(embedder, *, models_dir_override, progress, budget_s, client)` (T3) ↔ T4's injected `ensure`; `LocalDiarizer(engine, embedder, models_dir_override, ensure_models)` (T4) ↔ `build_diarizer` (T5); `model_id_for(engine, embedder)` (T4) ↔ T5; `PrepareResult.diarizer_engine/diarizer_missing/diarizer_models_ready` (T5) ↔ T7; `VoiceMatchState.detail` (T5) ↔ T7; `MeetingMeta.diarizer_engine/diarizer_model_id` (T6) ↔ T5 stamping — **T6 must land before T5's stamping step** (execution order: 1, 2, 3, 4, 6, 5, 7, 8, 9).
