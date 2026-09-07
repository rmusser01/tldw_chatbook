# Meetings Self Voiceprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the local user enroll their own voice once (explicitly, or learned from accepted meetings) so later meetings tag their speech with their display name automatically, with the voiceprint encrypted on-device, exportable under a passphrase, and deletable in one action.

**Architecture:** A pure-Python encrypted store (`Audio/voiceprint.py`) holds one self voiceprint. Matching happens inside the diarizer worker: the backend sends the vector after READY (and after its one restart), the worker flags `self` on `assign`/`diarize` replies, the backend fixes the FIRST flagged cluster, and the session names it with the display name and a marker. Learning merges the batch centroid (or, in plain call mode, `you.wav` embedded via `enroll_from_pcm`) with a per-meeting cap. The Meetings rail hosts the controls (there is no Settings › Meetings section today).

**Tech Stack:** Python ≥3.11, numpy (existing), `Utils/config_encryption.ConfigEncryption` (AES-256-GCM, scrypt), `keyring` (existing dependency), Textual 8.x, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-06-meeting-voiceprint-design.md`. Backlog: TASK-31826.

**Plan deviation (recorded):** the spec says "Settings › Meetings" for Delete/Export/Import/toggles; no such section exists, so those controls live in a "Voice" row on the Meetings rail and the two toggles stay config keys (documented). Building a Settings section is out of scope.

## Global Constraints

- The voiceprint vector NEVER appears in logs, `meeting.json`, transcripts, the meeting folder, or user-facing copy; it crosses only the local pipe to the worker. Logs carry mode names, counts, and exception types only.
- The store's key is its OWN random key (keyring, else an owner-only key file); never the config-encryption password. A key file with permissions broader than owner-only is refused.
- The key is CREATED during enrollment; at meeting Start it is READ on a worker thread with a short timeout — a failed/blocked read never delays Start (matching off with reason "keyring locked").
- No matching in plain call mode (`diarize_mic_channel` off); matching in room mode and in call mode with mic diarization on.
- Stable first match is fixed in the backend (`self_cluster_id`); later flags are counted, never applied.
- Best-effort everywhere: no voiceprint failure may break recording, transcript, ingest, or Start; never block under the session/backend lock.
- No new dependency; `import tldw_chatbook.app` imports no torch/diarizer/voiceprint module; UI-ready census unchanged; new TCSS rules class-keyed only.
- `voiceprint.json` is a new profile-owned path: register it with `scripts/check_profile_owned_path_inventory.py`'s census; any new `logger.*` call → inventory drift review + `--write`; `./scripts/preflight.sh` green before the PR.
- Never `git stash`. Tests from the worktree: `.venv/bin/python -m pytest <files> -q -p no:cacheprovider`. Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

- Create `tldw_chatbook/Audio/voiceprint.py` — `Voiceprint` record, `VoiceprintStore` (load/save/merge_sample/delete/export/import_), key providers (`KeyringKeyProvider`, `KeyfileKeyProvider`), envelope encryption via `ConfigEncryption`. Import-light.
- Modify `tldw_chatbook/Audio/diarizer_cluster.py` — per-cluster accumulated seconds (`assign(embedding, seconds=0.0)`, `seconds(cluster_id)`), `nearest(vector) -> (id, distance)`.
- Modify `tldw_chatbook/Audio/diarizer_worker.py` — `enroll`, `export_centroid`, `enroll_from_pcm` ops; `self` on `assign`/`diarize` replies; `_batch` returns centroids for `export_centroid`.
- Modify `tldw_chatbook/Audio/diarizer_local.py` — `voiceprint=` ctor kwarg, enroll after READY/restart, `self_cluster_id`/`self_candidates_seen`, `export_centroid()`, `enroll_from_pcm()`.
- Modify `tldw_chatbook/Audio/meeting_session.py` — `Diarizer` protocol additions, matched-self application/override/Stop fallback, `MeetingMeta` fields.
- Modify `tldw_chatbook/Audio/meeting_owner.py` — settings/config keys, load-at-Start (off-thread with timeout), learning offer, explicit enrollment, `VoiceMatchState` readout.
- Modify `tldw_chatbook/UI/Screens/meetings_screen.py` — marker, rail line, offer, Enroll flow, Voice row (Delete/Export/Import).
- Modify `tldw_chatbook/config.py` (4 keys), `Docs/User_Guide/meetings.md`, the profile-owned path census.
- Tests: `Tests/Audio/test_voiceprint_store.py`, `Tests/Audio/test_diarizer_cluster.py`, `Tests/Audio/test_diarizer_local.py` (serve-loop + backend), `Tests/Audio/test_meeting_diarization_session.py`, `Tests/Audio/test_meeting_owner.py`, `Tests/UI/test_meetings_screen.py`, `Tests/Audio/test_voiceprint_real.py` (gated).

---

### Task 1: The voiceprint store

**Files:**
- Create: `tldw_chatbook/Audio/voiceprint.py`
- Modify: the profile-owned path census input that `scripts/check_profile_owned_path_inventory.py` reads (open the script to find the census file; add `voiceprint.json` at the user data dir root with owner TASK-31826)
- Test: `Tests/Audio/test_voiceprint_store.py`

**Interfaces:**
- Consumes: `tldw_chatbook.Utils.config_encryption.ConfigEncryption.encrypt_value(plaintext, password) -> str` / `decrypt_value(encrypted, password) -> str`; `tldw_chatbook.config.get_user_data_dir()`.
- Produces:
  ```python
  @dataclass
  class Voiceprint:
      model_id: str
      centroid: list[float]          # unit-normalised
      sample_count: float            # effective samples
      meetings_contributed: int
      created_at: str
      updated_at: str
      threshold_used: float
      last_best_similarity: float | None = None
      format_version: int = 1

  class KeyProvider(Protocol):
      mode: str                       # "keyring" | "keyfile"
      def get_or_create(self) -> str: ...   # may raise; create only from enrollment paths
      def get(self, timeout_s: float) -> str | None: ...  # never raises; None on missing/blocked

  class VoiceprintStore:
      def __init__(self, path: Path, key_provider: KeyProvider, *, clock=None, per_meeting_cap: float = 20.0): ...
      def load(self, expected_model_id: str | None = None, timeout_s: float = 1.5) -> "LoadResult": ...
      def save(self, record: Voiceprint) -> None: ...          # atomic, creates the key if needed
      def merge_sample(self, centroid: Sequence[float], weight: float, model_id: str) -> Voiceprint: ...
      def delete(self) -> bool: ...
      def export(self, dest: Path, passphrase: str) -> None: ...
      def import_(self, src: Path, passphrase: str, *, replace: bool) -> Voiceprint: ...

  @dataclass
  class LoadResult:
      voiceprint: Voiceprint | None
      reason: str | None   # None | "no_voiceprint" | "needs_reenrollment" | "cannot_decrypt" | "keyring_locked"
      mode: str | None
  def default_store(user_data_dir: Path | None = None) -> VoiceprintStore   # keyring → keyfile fallback
  def unit_normalise(v: Sequence[float]) -> list[float]
  ```

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Audio/test_voiceprint_store.py
import json, os, stat
import pytest
from tldw_chatbook.Audio import voiceprint as vp

class FakeKeys:
    mode = "keyring"
    def __init__(self, key="k" * 32, blocked=False): self.key, self.blocked, self.created = key, blocked, 0
    def get_or_create(self): self.created += 1; return self.key
    def get(self, timeout_s): return None if self.blocked else self.key

def _rec(cent=(1.0, 0.0), n=1.0, model="ecapa@rev1"):
    return vp.Voiceprint(model_id=model, centroid=vp.unit_normalise(cent), sample_count=n, meetings_contributed=1,
                         created_at="2026-09-06T00:00:00", updated_at="2026-09-06T00:00:00", threshold_used=0.2)

def test_round_trip_is_encrypted_at_rest(tmp_path):
    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", FakeKeys())
    store.save(_rec())
    raw = (tmp_path / "voiceprint.json").read_text()
    assert "1.0" not in raw and '"mode": "keyring"' in raw          # payload ciphertext, envelope mode visible
    assert store.load().voiceprint.centroid == [1.0, 0.0]

def test_blocked_key_reports_keyring_locked_without_raising(tmp_path):
    keys = FakeKeys(); store = vp.VoiceprintStore(tmp_path / "voiceprint.json", keys); store.save(_rec())
    keys.blocked = True
    res = store.load(timeout_s=0.1)
    assert res.voiceprint is None and res.reason == "keyring_locked"

def test_model_mismatch_needs_reenrollment(tmp_path):
    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", FakeKeys()); store.save(_rec(model="ecapa@rev1"))
    assert store.load(expected_model_id="ecapa@rev2").reason == "needs_reenrollment"

def test_atomic_save_keeps_old_file_when_write_fails(tmp_path, monkeypatch):
    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", FakeKeys()); store.save(_rec((1.0, 0.0)))
    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        store.save(_rec((0.0, 1.0)))
    assert store.load().voiceprint.centroid == [1.0, 0.0]

def test_merge_is_capped_normalised_weighted_mean(tmp_path):
    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", FakeKeys(), per_meeting_cap=2.0); store.save(_rec((1.0, 0.0), n=2.0))
    out = store.merge_sample((0.0, 10.0), weight=100.0, model_id="ecapa@rev1")   # weight capped to 2.0
    assert out.sample_count == pytest.approx(4.0) and out.meetings_contributed == 2
    assert out.centroid == pytest.approx(vp.unit_normalise((0.5, 0.5)))

def test_export_import_with_passphrase_and_model_gate(tmp_path):
    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", FakeKeys()); store.save(_rec())
    with pytest.raises(ValueError):
        store.export(tmp_path / "out.json", "")
    store.export(tmp_path / "out.json", "correct horse")
    other = vp.VoiceprintStore(tmp_path / "b" / "voiceprint.json", FakeKeys(key="j" * 32))
    other.save(_rec((0.0, 1.0), model="ecapa@rev9"))
    with pytest.raises(vp.ModelMismatch):
        other.import_(tmp_path / "out.json", "correct horse", replace=False)
    assert other.import_(tmp_path / "out.json", "correct horse", replace=True).model_id == "ecapa@rev1"

def test_keyfile_provider_refuses_broad_permissions(tmp_path):
    p = tmp_path / "voiceprint.key"; p.write_text("k" * 32); p.chmod(0o644)
    assert vp.KeyfileKeyProvider(p).get(timeout_s=0.1) is None

def test_delete_removes_file_only(tmp_path):
    keys = FakeKeys(); store = vp.VoiceprintStore(tmp_path / "voiceprint.json", keys); store.save(_rec())
    assert store.delete() is True and not (tmp_path / "voiceprint.json").exists() and keys.key
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest Tests/Audio/test_voiceprint_store.py -q -p no:cacheprovider` → FAIL (module missing).
- [ ] **Step 3: Implement** `voiceprint.py` per the interface: envelope `{"format_version": 1, "mode": ..., "payload": <ConfigEncryption.encrypt_value(json_record, key)>}`; `save` = write to `path.with_suffix(".tmp")` then `os.replace`; `load` = read key via `key_provider.get(timeout_s)` → `keyring_locked` on None, `cannot_decrypt` on any decrypt/JSON error, `needs_reenrollment` on model mismatch; `merge_sample` = `w = min(weight, cap)`, `new = unit_normalise((c*n + s*w)/(n+w))`, `sample_count += w`, `meetings_contributed += 1`; `export` = record re-encrypted with mode `passphrase` (empty → `ValueError`); `import_` reads only mode `passphrase` (`ModelMismatch` when models differ and `replace=False`, else merge with `weight=min(sample_count, cap)`); `KeyringKeyProvider` (service `"tldw_chatbook"`, username `"meeting-voiceprint"`, `secrets.token_urlsafe(32)` on create; `get` runs the keyring read on a thread joined with `timeout_s`); `KeyfileKeyProvider` (0o600 check via `stat`); `default_store` picks keyring when a backend is available (`keyring.get_keyring()` not the fail backend) else keyfile. Register the path in the census.
- [ ] **Step 4: Run to verify they pass**; run `./scripts/preflight.sh` (profile-owned path census must be green).
- [ ] **Step 5: Commit** — `git add tldw_chatbook/Audio/voiceprint.py Tests/Audio/test_voiceprint_store.py <census file>` / `git commit -m "feat(meetings): encrypted self voiceprint store (31826)"`.

---

### Task 2: Worker matching ops (torch-free via `serve()`)

**Files:**
- Modify: `tldw_chatbook/Audio/diarizer_cluster.py` (`assign(embedding, seconds=0.0)`, `seconds(cid) -> float`, `nearest(vector) -> tuple[str, float] | None`)
- Modify: `tldw_chatbook/Audio/diarizer_worker.py` (`serve()` ops; `_batch` centroids)
- Test: `Tests/Audio/test_diarizer_cluster.py`, `Tests/Audio/test_diarizer_local.py` (serve-loop tests live there already)

**Interfaces:**
- Consumes: `serve(stdin, stdout, live, embed, batch)` (exists); `OnlineClusterer` (exists).
- Produces (wire protocol): `{"cmd": "enroll", "vector": [...], "threshold": 0.2, "min_seconds": 4.0}` (no reply); `assign` reply `{"id", "seq", "self": true|false}`; `{"cmd": "diarize", ...}` reply `{"segments": [...], "self": "<live id>" | null}`; `{"cmd": "export_centroid", "id": "S1"}` → `{"centroid": [...], "seconds": 12.3}` or `{"centroid": null}`; `{"cmd": "enroll_from_pcm", "sr": 16000, "n": <bytes>}` + PCM → `{"centroid": [...], "seconds": 30.0}`. The worker also embeds each `assign`'s PCM length into `seconds = n / (2*sr)` for the clusterer.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Audio/test_diarizer_cluster.py (add)
def test_assign_accumulates_seconds_and_nearest_reports_distance():
    c = OnlineClusterer()
    a = c.assign(_v(1, 0, 0), seconds=1.5); c.assign(_v(0.99, 0.01, 0), seconds=2.0)
    assert c.seconds(a) == pytest.approx(3.5)
    cid, dist = c.nearest(_v(1, 0, 0))
    assert cid == a and dist == pytest.approx(0.0, abs=1e-3)

# Tests/Audio/test_diarizer_local.py (add, next to the existing serve-loop test)
def _serve_lines(lines, embed, batch=None):
    import io, json
    from tldw_chatbook.Audio.diarizer_worker import serve
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer
    stdin = io.BytesIO(b"".join(lines)); stdout = io.BytesIO()
    serve(stdin, stdout, OnlineClusterer(max_speakers=8), embed, batch or (lambda *a, **k: ([], {})))
    return [json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()]

def test_self_flag_requires_threshold_and_min_seconds():
    pcm = b"\x00\x00" * 16000  # 1 s
    def ctl(d): return (json.dumps(d) + "\n").encode()
    lines = [ctl({"cmd": "enroll", "vector": [1, 0, 0], "threshold": 0.2, "min_seconds": 2.0})]
    for seq in range(3):
        lines += [ctl({"cmd": "assign", "sr": 16000, "seq": seq, "n": len(pcm)}), pcm]
    out = _serve_lines(lines, embed=lambda pcm: [0.99, 0.01, 0.0])
    assert [o["self"] for o in out] == [False, True, True]   # 1 s < 2 s, then 2 s, 3 s

def test_enroll_from_pcm_returns_unit_centroid_and_export_centroid_roundtrip():
    pcm = b"\x00\x00" * 16000 * 3
    def ctl(d): return (json.dumps(d) + "\n").encode()
    out = _serve_lines([ctl({"cmd": "enroll_from_pcm", "sr": 16000, "n": len(pcm)}), pcm,
                        ctl({"cmd": "assign", "sr": 16000, "seq": 0, "n": len(pcm)}), pcm,
                        ctl({"cmd": "export_centroid", "id": "S1"})],
                       embed=lambda pcm: [3.0, 4.0, 0.0])
    assert out[0]["centroid"] == pytest.approx([0.6, 0.8, 0.0]) and out[0]["seconds"] == pytest.approx(3.0)
    assert out[2]["centroid"] == pytest.approx([0.6, 0.8, 0.0]) and out[2]["seconds"] == pytest.approx(3.0)
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** — clusterer: `self._seconds: dict[str, float]`, add `seconds` on every assign/fold; `nearest` = argmin cosine distance over centroids. Worker `serve`: hold `enrolled = None` state; `enroll` sets `(vec, threshold, min_seconds)`; on `assign` after clustering, `is_self = enrolled is not None and cos_dist(live.centroids()[sid], vec) <= threshold and live.seconds(sid) >= min_seconds`; `diarize` reply adds `"self"` = the reconciled live id of the batch cluster nearest the vector within the threshold (the batch centroids come back from `_batch`, which now returns `(segments, final_centroids_by_live_id)`; adapt the existing `batch` callable and `main()`); `export_centroid` returns the batch centroid for the id if a `diarize` ran, else the live centroid, plus `live.seconds(id)`; `enroll_from_pcm` embeds the PCM (fake embed in tests) and returns `unit_normalise(embedding)` + seconds. Never write PCM to disk.
- [ ] **Step 4: Run** both files + `Tests/Audio/test_diarizer_worker*.py` if present → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(meetings): worker voiceprint matching, export and enrollment ops (31826)"`.

---

### Task 3: Backend enrolls after READY and fixes the first self match

**Files:**
- Modify: `tldw_chatbook/Audio/diarizer_local.py`
- Test: `Tests/Audio/test_diarizer_local.py`

**Interfaces:**
- Produces: `SpeechBrainDiarizer(..., voiceprint: Sequence[float] | None = None, match_threshold: float = 0.2, match_min_seconds: float = 4.0)`; attributes `self_cluster_id: str | None` (first flagged), `self_candidates_seen: int`; methods `export_centroid(cluster_id) -> tuple[list[float], float] | None` (bounded wait, best-effort) and `enroll_from_pcm(pcm: bytes, sample_rate: int) -> tuple[list[float], float] | None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_enroll_is_sent_after_ready_and_after_restart():
    made = []
    def spawn(cmd, *a, **k):
        p = FakeProc(['{"id": "S1", "seq": 0, "self": false}\n'] if len(made) == 0 else []); made.append(p); return p
    d = SpeechBrainDiarizer(spawn=spawn, voiceprint=[1.0, 0.0], match_threshold=0.2)
    made[0].emit_ready()
    assert any(b'"cmd": "enroll"' in c for c in made[0].stdin.chunks)
    d.assign(b"\x00\x00" * 1600, 16000, 0); made[0].die(); d.assign(b"\x00\x00" * 1600, 16000, 1)  # crash -> restart
    made[1].emit_ready()
    assert any(b'"cmd": "enroll"' in c for c in made[1].stdin.chunks)

def test_first_self_flag_is_fixed_and_later_ones_counted():
    proc = FakeProc(['{"id": "S1", "seq": 0, "self": true}\n', '{"id": "S2", "seq": 1, "self": true}\n'])
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc, voiceprint=[1.0, 0.0]); proc.emit_ready()
    d.assign(b"\x00\x00" * 1600, 16000, 0); d.assign(b"\x00\x00" * 1600, 16000, 1)
    assert d.self_cluster_id == "S1" and d.self_candidates_seen == 2

def test_vector_never_reaches_logs(captured_lines):
    proc = FakeProc([]); d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc, voiceprint=[0.123456, 0.654321]); proc.emit_ready(); proc.die()
    d.assign(b"\x00\x00" * 1600, 16000, 0)
    assert "0.123456" not in "\n".join(captured_lines)
```

(Adapt `emit_ready`/`die` to the file's real `FakeProc` idiom.)

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** — store the vector privately; in the READY watcher (after `_ready_ok`), send `enroll` (inside the lock, non-blocking); the restart path reuses it; parse `"self"` from assign replies: first `True` sets `self_cluster_id`, every `True` increments `self_candidates_seen`; `export_centroid`/`enroll_from_pcm` send their ops and wait bounded (reuse `_await_reply`-style plumbing, budget ≤ 10 s) returning `None` on any failure; never log the vector (log only lengths/types).
- [ ] **Step 4: Run** `Tests/Audio/test_diarizer_local.py` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(meetings): backend enrolls the voiceprint and fixes the first self match (31826)"`.

---

### Task 4: Session and owner — apply the match, learn, enroll

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_session.py` (`Diarizer` protocol: `self_cluster_id` attr optional via `getattr`; `MeetingMeta.matched_self: str | None = None`, `matched_self_overridden: bool = False`, `self_candidates_seen: int = 0`; application in `_on_final` after `assign`; override in the rename path used by the screen; Stop-pass `self` fallback)
- Modify: `tldw_chatbook/Audio/meeting_owner.py` (`MeetingSettings.voice_match: bool = True`, `voice_match_threshold: float = 0.2`, `voice_match_min_seconds: float = 4.0`, `voice_learn_offer: bool = True`; `from_config`; `VoiceMatchState` readout on `PrepareResult`; load at Start on a thread with timeout via `VoiceprintStore.load(expected_model_id=..., timeout_s=1.5)`; build the diarizer with the vector when matching applies; `learning_offer(result) -> LearningOffer | None` and `accept_learning(offer) -> bool`; `enroll_from_mic(seconds=30, progress=None) -> EnrollResult`)
- Modify: `tldw_chatbook/config.py` (four keys under `[meetings]` with comments)
- Test: `Tests/Audio/test_meeting_diarization_session.py`, `Tests/Audio/test_meeting_owner.py`

**Interfaces:**
- Consumes: Task 1 store, Task 3 backend attributes/methods, `render_label` (unchanged), `meeting_user_display_name()` (exists).
- Produces: `PrepareResult.voice_match: VoiceMatchState` with `.state in {"on","off"}` and `.reason in {None,"no_voiceprint","needs_reenrollment","cannot_decrypt","keyring_locked","plain_call_mode","disabled"}`; `MeetingResult` exposes `matched_self`; owner methods above. Model id source: the worker's `MODEL` constant plus revision (`diarizer_worker.MODEL_ID` — add it; `"speechbrain/spkrec-ecapa-voxceleb@unpinned"` unless the loader exposes a revision).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Audio/test_meeting_diarization_session.py (add)
def test_first_self_cluster_is_named_and_persisted(meeting_session_with_fake_capture):
    fake = FakeDiarizer(["S1", "S2"]); fake.self_cluster_id = None
    session = meeting_session_with_fake_capture(diarizer=fake, mode="room", user_display_name="Me")
    session.start(); fake.self_cluster_id = "S1"; session._on_final_for_test("hi", label=None)
    assert session.meta.speaker_names["S1"] == "Me" and session.meta.matched_self == "S1"
    assert read_meeting_json(session.meta.folder)["matched_self"] == "S1"

def test_rename_overrides_match_but_display_name_does_not(meeting_session_with_fake_capture):
    ... apply match as above; session.rename_speaker("S1", "Bob") -> matched_self_overridden True
    ... fresh session; session.rename_speaker("S1", "Me") -> matched_self_overridden False

def test_stop_pass_self_applies_only_without_live_match(meeting_session_with_fake_capture):
    class StopSelf(FakeDiarizer):
        def diarize(self, *a): self.stop_self = "S3"; return [SpeakerSegment(0.0, 1e6, "S3")]
    ... no live match -> after stop, meta.matched_self == "S3"; with a live match "S1" -> stays "S1"

# Tests/Audio/test_meeting_owner.py (add)
def test_plain_call_mode_never_matches(tmp_path, monkeypatch):
    ... owner with a stored voiceprint, tap_kind="native_macos", diarize_mic_channel=False
    assert owner.prepare().voice_match.reason == "plain_call_mode"
    session = owner.start(); assert getattr(session._diarizer, "voiceprint", None) is None

def test_locked_keyring_never_delays_start(tmp_path, monkeypatch):
    ... store whose key provider sleeps 5 s; owner.start() returns in < 2 s; prepare().voice_match.reason == "keyring_locked"

def test_learning_offer_once_and_merge_on_accept(tmp_path, monkeypatch):
    ... result with matched_self "S1" not overridden -> learning_offer(result) is not None; second call -> None
    ... accept_learning(offer) uses diarizer.export_centroid -> store.load().voiceprint.meetings_contributed == 2

def test_learning_in_plain_call_mode_embeds_you_wav(tmp_path, monkeypatch):
    ... call mode, you.wav present, offer.kind == "mic_channel"; accept -> diarizer.enroll_from_pcm called with you.wav's PCM

def test_enroll_from_mic_refused_while_meeting_active(tmp_path, monkeypatch):
    ... owner.start(); res = owner.enroll_from_mic(seconds=1); assert res.ok is False and res.reason == "capture_busy"
```

Fill the elided parts with the files' existing fakes/fixtures; the assertions are the contract.

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** per the interfaces: matching applies only when `voice_match` and the load succeeded and `(mode == "room") or (mode == "call" and settings.diarize_mic_channel)`; at Start the load runs on a `threading.Thread` joined with the timeout so Start never blocks; the session, after each `assign`, reads `getattr(self._diarizer, "self_cluster_id", None)` and applies once (name + `matched_self` + re-emit) unless overridden; the screen's rename path calls `session.rename_speaker(cluster_id, name)` (add it; it updates the map, pins, and sets `matched_self_overridden` unless `name == meta.user_display_name`); Stop: apply `diarize`'s `self` id only when `matched_self is None`; `learning_offer` (kinds `"matched_cluster"` / `"mic_channel"`), once per result; `accept_learning` → `export_centroid` or `enroll_from_pcm(you.wav)` → `store.merge_sample(centroid, weight=seconds, model_id=MODEL_ID)`; `enroll_from_mic` refuses when `is_active` or another capture is running (reuse the Console guard predicate), records via `mic_recorder_factory` into memory, spawns a `SpeechBrainDiarizer` if none, `enroll_from_pcm`, `store.save(...)` (key created here). Config keys + comments.
- [ ] **Step 4: Run** the two files + `Tests/Audio` fully → PASS; if any `logger.*` was added, inventory review + `--write`; preflight green.
- [ ] **Step 5: Commit** — `git commit -m "feat(meetings): voiceprint matching, learning offer and enrollment in the session/owner (31826)"`.

---

### Task 5: Meetings screen — marker, rail line, offer, Enroll, Voice row

**Files:**
- Modify: `tldw_chatbook/UI/Screens/meetings_screen.py`
- Test: `Tests/UI/test_meetings_screen.py`

**Interfaces:**
- Consumes: `PrepareResult.voice_match`, `MeetingResult.matched_self`, owner `learning_offer`/`accept_learning`/`enroll_from_mic`, `VoiceprintStore.delete/export/import_`.
- Produces: rail `Static(id="meetings-voice-match-status")` ("Voice match: on" / "off (<reason copy>)"); a `·` marker suffix on the matched row and legend label; a post-Stop offer rendered as a rail prompt with Accept / Not now / Don't ask again (never a modal that blocks; lapses on unmount); "Enroll my voice" button (`#meetings-enroll`) with a countdown Static and Cancel, running `enroll_from_mic` on a worker thread; a "Voice" row with Delete (confirm), Export (passphrase Input + path Input), Import (path + passphrase, replace/merge choice); all callbacks `is_mounted`-guarded; class-keyed CSS only.

- [ ] **Step 1: Write the failing tests** — rail line per reason; matched row shows the marker and the display name; rename clears the marker; offer buttons call the owner and the "don't ask again" flips `voice_learn_offer` via the config save seam; Enroll refused copy when busy; countdown visible while enrolling; Voice row Delete/Export/Import call the store with the typed passphrase (fake store); an unmounted screen never touches widgets.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** following the screen's existing patterns (rail Statics, `_apply_rename`, `is_mounted` guards, `run_worker(thread=True, exit_on_error=False)` + `call_from_thread`).
- [ ] **Step 4: Run** `Tests/UI/test_meetings_screen.py Tests/UI/test_meetings_wiring.py` + the CSS ratchet + census → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(meetings): voice match marker, rail status, learning offer, enrollment and voice controls (31826)"`.

---

### Task 6: Docs, invariants, gated real test

**Files:**
- Modify: `Docs/User_Guide/meetings.md` (enrollment, matching, learning offer, export/import, privacy, threshold caveat)
- Modify: `Tests/Audio/test_meeting_import_safety.py` (boot pulls in no `tldw_chatbook.Audio.voiceprint` either — it is imported lazily by the owner)
- Create: `Tests/Audio/test_voiceprint_real.py` (gated)
- Test: the two above + `Tests/Performance/test_ui_ready_module_census.py`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Audio/test_meeting_import_safety.py (extend the subprocess check)
assert "tldw_chatbook.Audio.voiceprint" not in imported_modules

# Tests/Audio/test_voiceprint_real.py
import os, shutil, sys, pytest
pytestmark = [pytest.mark.real_audio_device, pytest.mark.integration]
@pytest.mark.skipif(
    os.environ.get("TLDW_RUN_VOICEPRINT_TEST") != "1" or shutil.which("say") is None or sys.platform != "darwin",
    reason="opt-in: TLDW_RUN_VOICEPRINT_TEST=1 on macOS with the diarization extra and the `say` TTS",
)
def test_enrolled_voice_matches_itself_and_not_another(tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("speechbrain")
    # `say -v Samantha` and `say -v Daniel` → two 16 kHz mono WAVs of a paragraph; enroll from voice A via enroll_from_pcm;
    # run a fake-capture session over A then B with the real SpeechBrainDiarizer; assert matched_self is A's cluster and
    # B's cluster is not named with the display name; print the best similarity for calibration.
```

- [ ] **Step 2: Run** — the import-safety test fails until the owner's import is lazy; the gated test skips by default.
- [ ] **Step 3: Implement** — lazy import in the owner; docs section "Remember my voice" (what it stores, where, encryption mode, export/import, how to delete, that the threshold is a starting value, that plain call mode never matches); refresh the stamp honestly.
- [ ] **Step 4: Run** import safety + census + `Tests/Audio` → PASS; `./scripts/preflight.sh` green.
- [ ] **Step 5: Commit** — `git commit -m "docs(meetings): voiceprint enrollment guide, boot invariant, gated real test (31826)"`.

---

## Self-Review

**Coverage:** §3.1 store → T1 (incl. census registration, key modes, Keychain-timing via `get(timeout)` + create-at-enroll in T4); §3.2 worker ops → T2; §3.3 backend → T3; §3.4 session/owner (mode gate, stable first, override, Stop fallback, learning incl. `you.wav`, enrollment refused when busy) → T4; §3.5 screens → T5 (plan deviation: Meetings rail, not Settings — recorded above); §5 config → T4; §6 degradation → T1/T3/T4 (reasons, best-effort); §7 tests → each task + T6 gated TTS test + invariants.
**Placeholders:** elided test bodies name the real fixtures/fakes and state the assertions; no TBDs.
**Type consistency:** `self_cluster_id`/`self_candidates_seen` (T3→T4), `export_centroid`/`enroll_from_pcm` return `(centroid, seconds)` (T2 wire → T3 → T4), `LoadResult.reason` values (T1→T4→T5), `matched_self`/`matched_self_overridden` (T4→T5), `MODEL_ID` (T2/T4), `VoiceMatchState` (T4→T5).
