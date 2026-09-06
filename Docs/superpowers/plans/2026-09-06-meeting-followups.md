# Meetings Diarization Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close six bounded follow-ups from the phase-2 diarization program: log redaction (31748), live-rename pin forwarding (31744), post-crash name safety (31749), the mic-channel display name (31746), hybrid-room mic diarization (31743), and after-the-fact rename in the live media reader (31745).

**Architecture:** Small, contained changes to the shipped diarization stack: the `Diarizer` protocol gains `pin`; the subprocess backend records where it crashed and restarts its worker with a fresh id namespace; the session runs the Stop pass only over the post-crash span; one shared helper renders the mic-channel name everywhere; a config flag routes mic-channel segments through the diarizer; the rename functions move to a `Library/` module and the tested legend is mounted in `LibraryMediaViewer`.

**Tech Stack:** Python ≥3.11, Textual 8.x, numpy (existing), pytest with `--import-mode=importlib`.

**Spec:** `Docs/superpowers/specs/2026-09-05-meeting-diarization-design.md` (phase 2; these tasks close its §10 follow-ups and the review deferrals). Backlog: 31748, 31744, 31749, 31746, 31743, 31745.

## Global Constraints

- Best-effort: no change may make a diarizer failure break recording, the transcript, or Library ingest; never hold the session lock or the backend lock across blocking work.
- No transcript text, speaker names, PCM, or user paths in persistent logs or user-facing failure copy (exception type / static copy / `redact_user_paths` only).
- No new dependency; `import tldw_chatbook.app` imports no torch and no diarizer module (`Tests/Audio/test_meeting_import_safety.py`); the UI-ready module census must not rise.
- New TCSS rules must be class-keyed (`Static.some-class`), never ancestor-scoped bare types (`#x Static`) — the Perf Guard CSS ratchet (PR #2465 lesson).
- Every change to a `logger.*` call requires reading the diagnostic-inventory drift rows, then `scripts/check_persistent_diagnostic_inventory.py --write`; `./scripts/preflight.sh` must be green before the PR.
- Default behaviour is unchanged for every existing user: new flags default off; the mic channel still renders "You" for an unset display name.
- Run tests from the worktree with `.venv/bin/python -m pytest <files> -q -p no:cacheprovider`. Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

- Modify `tldw_chatbook/Audio/meeting_session.py` — redaction (T1); `Diarizer.pin` in the protocol (T2); post-crash Stop-pass span (T3); shared display-name use in `render_markdown` and the `you`-branch bypass (T4/T5).
- Modify `tldw_chatbook/Audio/meeting_owner.py` — redaction (T1); `diarize_mic_channel` setting + `meeting_user_display_name()` helper (T4/T5).
- Modify `tldw_chatbook/Audio/diarizer_local.py` — `pin` command (T2); `crashed_at_seq` + restart with `start_id` (T3).
- Modify `tldw_chatbook/Audio/diarizer_worker.py` — `pin` op (T2); `--start-id` (T3).
- Modify `tldw_chatbook/Audio/diarizer_cluster.py` — `OnlineClusterer(start_id=)` (T3).
- Modify `tldw_chatbook/UI/Screens/meetings_screen.py` — use the shared display-name helper (T4).
- Create `tldw_chatbook/Library/meeting_speaker_rename.py` — the rename functions moved out of the canvas (T6).
- Modify `tldw_chatbook/Widgets/Library/library_media_canvas.py` — re-export from the new module (T6).
- Modify `tldw_chatbook/Library/library_media_viewer_state.py`, `tldw_chatbook/Widgets/Library/library_media_viewer.py`, `tldw_chatbook/UI/Screens/library_screen.py` — legend in the live reader (T6).
- Modify `tldw_chatbook/config.py` — `diarize_mic_channel` key (T5).
- Modify `Docs/User_Guide/meetings.md` — mic-channel flag, display name, reader rename (T5/T6).
- Tests under `Tests/Audio/` and `Tests/UI/`.

---

### Task 1: Redact raw exceptions in the phase-1 meeting logs (31748)

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_session.py` (≈361 sink-failed, ≈433 stop_dictation-failed, ≈441 capture-stop-failed), `tldw_chatbook/Audio/meeting_owner.py` (≈439 device-enumeration)
- Test: `Tests/Audio/test_meeting_log_privacy.py` (new)

**Interfaces:** none new. Uses the existing `captured_lines` loguru-sink fixture pattern from `Tests/Audio/test_meeting_diarization_session.py`.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Audio/test_meeting_log_privacy.py
import pytest
from loguru import logger

@pytest.fixture
def captured_lines():
    lines: list[str] = []
    handle = logger.add(lambda m: lines.append(str(m)), level="DEBUG")
    yield lines
    logger.remove(handle)

def test_sink_failure_log_carries_no_path(meeting_session_with_fake_capture, captured_lines):
    class BoomSink:
        def on_started(self, meta): raise OSError("/Users/alice/secret/meeting.jsonl: denied")
        def on_partial(self, *a): ...
        def on_segment(self, *a): ...
        def on_stopped(self, *a): ...
    session = meeting_session_with_fake_capture(sinks=[BoomSink()], mode="room")
    session.start()
    joined = "\n".join(captured_lines)
    assert "/Users/alice" not in joined and "secret" not in joined

def test_device_enumeration_log_carries_no_path(tmp_path, monkeypatch, captured_lines):
    from tldw_chatbook.Audio import meeting_owner as mo
    class Rec:
        def __init__(self, **k): ...
        def get_audio_devices(self): raise RuntimeError("/Users/alice/dev: boom")
    owner = mo.MeetingSessionOwner(settings=mo.MeetingSettings(recordings_dir=tmp_path), call_from_thread=lambda f, *a, **k: f(*a, **k),
                                   submit_ingest=lambda **k: None, job_state=lambda j: None, mic_recorder_factory=Rec,
                                   facade_factory=lambda: object(), dictation_factory=lambda c, f, g: object(),
                                   tap_probe=lambda **k: __import__("tldw_chatbook.Audio.system_audio_tap", fromlist=["TapMode"]).TapMode("unavailable", "r"),
                                   tap_builder=lambda m, **k: None, vad_factory=object)
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: None)
    owner.prepare()
    assert "/Users/alice" not in "\n".join(captured_lines)
```

Adapt the owner-construction kwargs to `_owner(...)` in `Tests/Audio/test_meeting_owner.py` if that helper fits better — the assertion is what matters. Also add a regression test that `_apply_rename`'s persist-failure log (meetings_screen) and `rename_meeting_speaker`'s failure path do not carry a path (force `update_meeting_json` to raise `OSError("/Users/alice/x")`).

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/python -m pytest Tests/Audio/test_meeting_log_privacy.py -q -p no:cacheprovider` → FAIL (path present).
- [ ] **Step 3: Implement** — each named site becomes `logger.<level>("... ({})", type(exc).__name__)` or, where the message is useful, `redact_user_paths(str(exc))` (already imported in both modules). Do not change log levels.
- [ ] **Step 4: Run to verify it passes**; then `Tests/Audio` fully; then read the inventory drift rows and `--write`; `./scripts/preflight.sh` green.
- [ ] **Step 5: Commit** — `git add tldw_chatbook/Audio/meeting_session.py tldw_chatbook/Audio/meeting_owner.py Tests/Audio/test_meeting_log_privacy.py Docs/security/production-diagnostic-inventory.json` / `git commit -m "fix(meetings): redact raw exceptions in phase-1 meeting logs (31748)"`.

---

### Task 2: Forward `pin()` to the worker clusterer (31744)

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_session.py` (`Diarizer` protocol), `tldw_chatbook/Audio/diarizer_local.py` (`SpeechBrainDiarizer.pin`), `tldw_chatbook/Audio/diarizer_worker.py` (`"pin"` op)
- Test: `Tests/Audio/test_diarizer_local.py`, `Tests/UI/test_meetings_screen.py` (existing pin test keeps passing)

**Interfaces:**
- Produces: `Diarizer.pin(cluster_id: str) -> None` (best-effort, never raises). Wire: `{"cmd": "pin", "id": "<cluster_id>"}` on stdin; worker calls `live.pin(id)`; no reply.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Audio/test_diarizer_local.py (add)
def test_pin_sends_a_pin_command_to_the_worker():
    proc = FakeProc(['{"id": "S1", "seq": 0}\n'])
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc)
    proc.ready()  # whatever the existing fake uses to emit READY
    assert d.assign(b"\x00\x00" * 1600, 16000, 0) == "S1"
    d.pin("S1")
    assert any(b'"cmd": "pin"' in chunk and b'"S1"' in chunk for chunk in proc.stdin.writes)

def test_pin_is_a_noop_when_coarse_only():
    proc = FakeProc([])
    d = SpeechBrainDiarizer(spawn=lambda *a, **k: proc)
    d._mark_coarse("backend crashed")
    d.pin("S1")  # must not raise, must not write
    assert not any(b'"cmd": "pin"' in c for c in proc.stdin.writes)
```

- [ ] **Step 2: Run to verify it fails** (`AttributeError: pin`).
- [ ] **Step 3: Implement** — add `def pin(self, cluster_id: str) -> None: ...` to the `Diarizer` Protocol; in `SpeechBrainDiarizer.pin`: return if not ready / coarse-only / degraded; else under `self._lock` `self._send(proc, {"cmd": "pin", "id": cluster_id})` inside try/except (log type only). Worker: `elif op == "pin": live.pin(str(cmd.get("id", "")))`. The Meetings screen already calls `pin` via `hasattr`; the FakeDiarizer there already has it.
- [ ] **Step 4: Run** the two test files → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(meetings): forward live-rename pin to the worker clusterer (31744)"`.

---

### Task 3: Post-crash name safety (31749)

**Files:**
- Modify: `tldw_chatbook/Audio/diarizer_cluster.py` (`OnlineClusterer(start_id=)`), `tldw_chatbook/Audio/diarizer_worker.py` (`--start-id`), `tldw_chatbook/Audio/diarizer_local.py` (`crashed_at_seq`, restart passes start id), `tldw_chatbook/Audio/meeting_session.py` (Stop pass over the post-crash span only)
- Test: `Tests/Audio/test_diarizer_cluster.py`, `Tests/Audio/test_diarizer_local.py`, `Tests/Audio/test_meeting_diarization_session.py`

**Interfaces:**
- Produces: `OnlineClusterer(threshold=0.25, max_speakers=8, start_id: int = 0)` — first minted id is `S{start_id+1}`; `SpeechBrainDiarizer.crashed_at_seq: int | None` (the `seq` of the assign in flight when the worker died; `None` if never crashed) and `max_id_seen: int` (highest `S<n>` returned so far); the restarted worker is spawned with `--start-id <max_id_seen>`.
- Session rule: on stop, if `getattr(diarizer, "crashed_at_seq", None) is not None`, the batch pass runs over `[t_start_of_that_seq, duration]` and the overlay touches only segments with `seq >= crashed_at_seq`; earlier segments keep their near-live ids and names.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Audio/test_diarizer_cluster.py (add)
def test_start_id_mints_past_the_pre_crash_ids():
    c = OnlineClusterer(start_id=3)
    assert c.assign(_v(1, 0, 0)) == "S4"

# Tests/Audio/test_diarizer_local.py (add)
def test_restart_spawns_with_start_id_past_max_seen_and_records_crash_seq():
    made = []
    def spawn(cmd, *a, **k):
        made.append(cmd); return FakeProc(['{"id": "S2", "seq": 0}\n']) if len(made) == 1 else FakeProc([])
    d = SpeechBrainDiarizer(spawn=spawn); ... assign seq 0 -> "S2"; kill the proc; assign seq 1 -> None
    assert d.crashed_at_seq == 1 and d.max_id_seen == 2
    assert "--start-id" in made[1] and made[1][made[1].index("--start-id") + 1] == "2"

# Tests/Audio/test_meeting_diarization_session.py (add)
def test_stop_pass_after_a_crash_leaves_pre_crash_segments_and_names_alone(meeting_session_with_fake_capture):
    class Crashy(FakeDiarizer):
        crashed_at_seq = 2
        def diarize(self, wav_path, start_s, end_s):
            self.seen = (start_s, end_s); return [SpeakerSegment(start_s, end_s, "S9")]
    # segments 0,1 assigned S1 (named "Alice"); segments 2,3 coarse after the crash
    ... start(); four finals; stop()
    assert [s.speaker_id for s in session.segments[:2]] == ["S1", "S1"]
    assert session.meta.speaker_names["S1"] == "Alice"
    assert all(s.speaker_id == "S9" for s in session.segments[2:])
    assert fake.seen[0] == session.segments[2].t_audio_start
```

Fill the elided parts with the file's existing fixture/fake idioms; the assertions are the contract.

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** — `OnlineClusterer.__init__(..., start_id=0)`: `self._n = start_id`. Worker `main()`: parse `--start-id N` (argv, alongside how `max_speakers` arrives) → `OnlineClusterer(max_speakers=..., start_id=N)`. Backend: track `max_id_seen` from every non-None assign reply; in `_fail`, set `crashed_at_seq` to the seq in flight (pass it into `_fail` from `assign`); `_command()` appends `["--start-id", str(self.max_id_seen)]` on restart. Session `stop()`: compute `crash_seq = getattr(self._diarizer, "crashed_at_seq", None)`; if not None and `< len(self.segments)`, `start_s = self.segments[crash_seq].t_audio_start` and overlay only `seq >= crash_seq`; otherwise unchanged.
- [ ] **Step 4: Run** the three files + `Tests/Audio` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "fix(meetings): post-crash Stop pass keeps pre-crash ids and names (31749)"`.

---

### Task 4: One shared display name for the mic channel (31746)

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_owner.py` (helper), `tldw_chatbook/UI/Screens/meetings_screen.py` (`_user_display_name` → helper), `tldw_chatbook/Audio/meeting_session.py` (`render_markdown` and `LocalMeetingSink` use the helper's value), `tldw_chatbook/Library/meeting_speaker_rename.py` after T6 (or the canvas render now) uses it too
- Test: `Tests/Audio/test_meeting_owner.py`, `Tests/UI/test_meetings_screen.py`, `Tests/Audio/test_meeting_session.py`

**Interfaces:**
- Produces: `meeting_user_display_name(get_setting=...) -> str` in `meeting_owner.py`: returns `chat_defaults.user_display_name` when it is set AND differs from the factory default `"User"` (compare against `config`'s shipped default constant, not a literal), else `"You"`. `MeetingMeta` gains `user_display_name: str = "You"` so after-the-fact renders agree with the live session; `LocalMeetingSink.on_stopped`/`render_markdown` use `meta.user_display_name`.

- [ ] **Step 1: Write the failing tests** — helper: unset → "You"; factory "User" → "You"; "Alice" → "Alice". Session: `render_markdown` renders `**Alice:**` for a `you` segment when `meta.user_display_name == "Alice"`. Screen: the legend/transcript shows "Alice:" for the mic row.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** — the helper; owner stamps `meta.user_display_name` at `start()`; `render_markdown` replaces the hardcoded `"You"`; the screen's `_user_display_name` delegates to the helper (keep its comment, updated); `read_meeting_json` back-fills `user_display_name="You"`.
- [ ] **Step 4: Run** the named files + `Tests/Audio` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(meetings): mic channel renders the configured display name everywhere (31746)"`.

---

### Task 5: Hybrid-room mic diarization behind a flag (31743)

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_owner.py` (`MeetingSettings.diarize_mic_channel: bool = False`, `from_config`), `tldw_chatbook/config.py` (`[meetings] diarize_mic_channel = false` with a comment), `tldw_chatbook/Audio/meeting_session.py` (`MeetingMeta.diarize_mic_channel`; `_on_final` also assigns `you`/`both` segments from the `"you"`/`"mixed"` PCM when on; Stop pass uses `mixed.wav` when on; `render_label` yields to `speaker_id` for a `you` segment when the meta flag is on), `Docs/User_Guide/meetings.md`
- Test: `Tests/Audio/test_meeting_diarization_session.py`, `Tests/Audio/test_meeting_owner.py`, `Tests/Audio/test_meeting_speaker_model.py`

**Interfaces:**
- Consumes: T4's `meta.user_display_name`.
- Produces: `render_label(segment, names, user_display_name, diarize_mic=False)` — when `diarize_mic` is True and the segment has a `speaker_id`, the mic branch is skipped. Callers pass `meta.diarize_mic_channel`.

- [ ] **Step 1: Write the failing tests** — flag off: a `you` segment is never sent to `assign` (existing behaviour); flag on in call mode: a `you` segment is assigned from `pcm_window("you", ...)` and gets a `speaker_id`; `render_label(seg_you_with_S2, {"S2": "Bob"}, "You", diarize_mic=True) == "Bob"` and `== "You"` when `diarize_mic=False`; Stop pass path is `mixed.wav` when the flag is on in call mode; settings/config round-trip; old `meeting.json` back-fills `False`.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** per the interface; keep default off; document the flag (and that "You" pre-naming no longer applies with it on) in `meetings.md`.
- [ ] **Step 4: Run** the named files + `Tests/Audio` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(meetings): optional mic-channel diarization for hybrid rooms (31743)"`.

---

### Task 6: After-the-fact rename in the live media reader (31745)

**Files:**
- Create: `tldw_chatbook/Library/meeting_speaker_rename.py` (move `can_rename_meeting_speakers`, `rename_meeting_speaker`, `SpeakerRenameResult`, `RENAME_REFUSED_*`, `_read_meeting_transcript_segments`, `_render_meeting_transcript`, `_write_meeting_transcript_row`, `_meeting_speaker_legend_rows` out of the canvas unchanged)
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` (import + re-export the moved names so `Tests/UI/test_library_media_speaker_rename.py` stays green), `tldw_chatbook/Library/library_media_viewer_state.py` (`can_rename_speakers: bool = False`, `speaker_legend_rows: tuple[tuple[str, str], ...] = ()`), `tldw_chatbook/UI/Screens/library_screen.py` (populate both when building the viewer state at ≈34143, using the existing memoized `_library_media_can_rename_speakers` and `_meeting_speaker_legend_rows`), `tldw_chatbook/Widgets/Library/library_media_viewer.py` (a `_compose_speaker_legend()` section after the content: one row per speaker via `render_label`, an `Input`, `@on(Input.Submitted)` → `rename_meeting_speaker` on a worker thread + `call_from_thread` refresh; on a refused result show the static explanation "This transcript came from ingest; rename the live transcript in Meetings." via the app's notify pattern), `Docs/User_Guide/meetings.md` (rename-after is now reachable in the reader)
- Test: `Tests/UI/test_library_media_viewer_speaker_rename.py` (new)

**Interfaces:**
- Consumes: T4's display-name helper for the mic row label.
- CSS: class-keyed only (`Static.library-media-speaker-label`, `Input.library-media-speaker-input`, reuse the existing classes/rules — do not add `#… Static` rules).

- [ ] **Step 1: Write the failing tests** — viewer state carries `can_rename_speakers=True` + rows for a meeting item and `False` for a plain document; mounting `LibraryMediaViewer` with rows renders the legend; submitting a rename calls `rename_meeting_speaker` and refreshes the shown content; a refused result surfaces the static explanation and leaves content untouched; the ratchet test `Tests/Performance/test_textual_css_fastpath.py::test_ancestor_scoped_bare_type_rule_count_is_a_ratchet` still passes.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** — move first (pure relocation commit), then the viewer legend; keep the hidden canvas legend as-is; regenerate the CSS bundle if `BUNDLED_CSS` changes (`tldw_chatbook/css/build_css.py`).
- [ ] **Step 4: Run** the new test, `Tests/UI/test_library_media_speaker_rename.py`, the ratchet test, `Tests/Performance/test_ui_ready_module_census.py`, and `./scripts/preflight.sh` → all green.
- [ ] **Step 5: Commit** in two commits — `refactor(library): move meeting speaker rename into Library/meeting_speaker_rename.py` then `feat(library): rename meeting speakers from the live media reader (31745)`.

---

## Self-Review

**Coverage:** 31748→T1, 31744→T2, 31749→T3, 31746→T4, 31743→T5, 31745→T6; the review corrections (post-crash span-limited Stop pass, one shared display name incl. `render_markdown`, `you`-branch bypass, the fifth redaction site, the module move) are all in their tasks.
**Placeholders:** the elided test bodies name the file's existing fixtures/fakes to reuse and state the assertions; no TBDs.
**Type consistency:** `crashed_at_seq`/`max_id_seen`/`start_id` (T3), `meeting_user_display_name`/`meta.user_display_name` (T4→T5/T6), `render_label(..., diarize_mic=)` (T5), `can_rename_speakers`/`speaker_legend_rows` (T6) are used with the same names throughout.
