# Meeting Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record a Zoom call or in-person meeting from inside the TUI with a live You/Others transcript, then land it in the Library as an audio media item with a diarized post-meeting transcript.

**Architecture:** One existing dictation pipeline (`LazyLiveDictationService`) runs over a mixed mic+system-audio stream produced by a new `MeetingCapture` that duck-types the recorder. System audio comes from a new `SystemAudioTap` (Swift Core Audio helper on macOS, `parec` on Linux, WASAPI loopback device on Windows, any input device as fallback). A Textual-free `MeetingSession` turns dictation callbacks into labelled segments and fans out to sinks; an app-owned `MeetingSessionOwner` keeps the session alive across tab switches and hands the recording to the Library ingest registry on stop. A new Meetings screen drives it.

**Tech Stack:** Python 3.11+, Textual 8.x, numpy, webrtcvad-wheels, sounddevice/pyaudio (already in the speech extras), stdlib `wave`/`subprocess`/`queue`, Swift 6 (macOS helper only), pytest + Hypothesis.

**Spec:** `Docs/superpowers/specs/2026-09-04-meeting-transcription-design.md` (read it first; every task below cites the section it implements).

## Global Constraints

- Python ≥ 3.11; Textual ≥ 8.0.0,<9. No new runtime dependencies (spec §3.7).
- No schema change to any DB (spec §3.7).
- Every new Audio module is Textual-free; only `UI/Screens/meetings_screen.py` imports Textual (spec §3).
- PCM everywhere is 16 kHz, mono, int16; one frame = 20 ms = 320 samples = 640 bytes (spec §3.1/§3.2).
- Meetings never use the deferred STT executor: the transcription facade is built with `local_stt_dispatcher=None` (spec §4).
- The Library ingest registry is UI-thread-only: every `submit(...)` goes through `app.call_from_thread` (spec §3.4).
- Subprocesses are argument lists, never `shell=True`; PulseAudio sink names must match `^[A-Za-z0-9._-]+$` (spec §3.1).
- Tests never open real audio hardware; inject fakes through the factory kwargs (spec §8).
- Config keys live in a flat `[meetings]` section read as `get_cli_setting("meetings", key, default)` (spec §6).
- Config values that name paths go through `tldw_chatbook.Utils.path_validation.validate_path_simple` (spec §6).
- Commit after every task with a message that names the task; use the repo's trailer (`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`).
- Work happens in the worktree `.claude/worktrees/meeting-spec` on branch `docs/meeting-transcription-spec` (rename to `feat/meeting-transcription` in Task 0). Backlog task per CLAUDE.md §Backlog.

## File structure

| Path | Responsibility |
|---|---|
| `tldw_chatbook/Audio/wav_writer.py` (new) | Placeholder-header WAV writer, header patching, unfinished-file detection (spec §4 file layout, §7 crash safety) |
| `tldw_chatbook/Audio/meeting_capture.py` (new) | `EnergyRing` (attribution), `SpeechRun`, `MeetingCapture` (mixing, alignment, VAD runs, writers, fault) (spec §3.2, §4 attribution) |
| `tldw_chatbook/Audio/system_audio_tap.py` (new) | Platform resolvers, `SubprocessTap`, `DeviceTap`, `probe()`, `build_tap()`, macOS helper lookup/compile (spec §3.1, §3.6) |
| `tldw_chatbook/Audio/audiotap/main.swift` (new) | The macOS Core Audio process-tap helper source (spec §3.6; shipped inside the package so the dev fallback can compile it) |
| `tldw_chatbook/Audio/meeting_session.py` (new) | `MeetingSegment`/`MeetingMeta`/`MeetingResult`, `MeetingSink`/`Diarizer` protocols, `MeetingSession`, `LocalMeetingSink`, markdown render (spec §3.3, §5) |
| `tldw_chatbook/Audio/meeting_owner.py` (new) | `MeetingSessionOwner`: prepare, start/stop, watchdog, shutdown, recovery scan, raw-track cleanup (spec §3.4, §7) |
| `tldw_chatbook/UI/Screens/meetings_screen.py` (new) | `MeetingsScreen` rail + canvas (spec §3.5) |
| `tldw_chatbook/Audio/recording_service.py` (modify) | `retain_audio` kwarg (spec §3.7) |
| `tldw_chatbook/Audio/dictation_service_lazy.py` (modify) | `recorder_factory` kwarg (spec §3.7) |
| `tldw_chatbook/UI/Console_Modules/dictation.py`, `hands_free.py` (modify) | refuse while a meeting is active (spec §3.4) |
| `tldw_chatbook/Constants.py`, `UI/Navigation/shell_destinations.py`, `UI/Navigation/screen_registry.py`, `app.py`, `config.py` (modify) | tab, destination, route, owner + shutdown, `[meetings]` defaults (spec §3.5, §3.7, §6) |
| `Packaging/macos/build_app.py`, `Packaging/macos/Info.plist.template` (modify) | compile helper into the bundle, usage strings (spec §3.6) |
| `Tests/conftest.py` (modify) | recorder guard (spec §3.7) |
| `Docs/User_Guide/meetings.md` (new), `Docs/User_Guide/index.md` (modify) | user guide (spec §3.7) |

Deviation from the spec, decided here: the Swift source lives at `tldw_chatbook/Audio/audiotap/main.swift` instead of `Packaging/macos/audiotap/main.swift`, because the runtime dev-fallback compile (spec §3.6) needs the source inside the installed package. `build_app.py` compiles from that path.

---

### Task 0: Branch, backlog task, venv, baseline

**Files:**
- Modify: none (git + backlog + venv only)

- [ ] **Step 1: Rename the worktree branch and confirm the base**

```bash
git branch -m docs/meeting-transcription-spec feat/meeting-transcription
git log --oneline -3
```
Expected: top commits are `d5f3354de4 docs: revise meeting transcription spec…` and `bce5ce5f02 docs: design meeting transcription…` over `91757b61e9`.

- [ ] **Step 2: Create the venv in this worktree and install dev deps**

```bash
uv venv .venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -e ".[dev,speech_recording]"
.venv/bin/python -c "import numpy, webrtcvad, hypothesis; print('ok')"
```
Expected: `ok`. If `speech_recording` is not an extra name, run `grep -n 'speech_recording\|^audio = \|^\[project.optional' pyproject.toml` and use the extra that lists `pyaudio`/`sounddevice`/`webrtcvad-wheels` (pyproject lines ~270-280).

- [ ] **Step 3: Baseline the Audio and UI navigation suites**

```bash
.venv/bin/python -m pytest Tests/Audio Tests/UI/test_shell_destinations.py Tests/UI/test_command_palette_shell_routes.py -q -p no:cacheprovider
```
Expected: all pass (record the counts in the backlog task). Any failure here is pre-existing; note it, do not fix it.

- [ ] **Step 4: File the backlog task**

```bash
backlog task create "Meeting transcription phase 1: live You/Others transcript with native system audio and Library handoff" \
  -d "Record Zoom or in-person meetings from the TUI with a live labelled transcript, persist crash-safe audio, and hand the recording to Library ingest with diarization. Spec: Docs/superpowers/specs/2026-09-04-meeting-transcription-design.md" \
  --ac "Meetings screen records mic plus system audio on macOS and Linux and shows a live transcript labelled You/Others" \
  --ac "Stopping a meeting produces mixed.wav plus transcript.jsonl and meeting.json in the meetings folder and queues a Library audio ingest with diarization" \
  --ac "A meeting survives tab switches and app quit without losing recorded audio (headers patched, recovery offered on next visit)" \
  --ac "Console dictation and hands-free refuse to start while a meeting is active" \
  --ac "All new logic is covered by hardware-free tests and the suite is green" \
  -l audio,meetings --priority high -s "In Progress"
```
Then `backlog task edit <id> --plan "Follow Docs/superpowers/plans/2026-09-04-meeting-transcription.md tasks 1-12 in order"`. Record the id; later commits reference it.

- [ ] **Step 5: Commit the branch state**

Nothing to commit yet (venv is ignored). Confirm with `git status --short` → empty.

---

### Task 1: Recorder `retain_audio` flag and the hardware guard

Spec §3.7 (recording_service change, conftest guard).

**Files:**
- Modify: `tldw_chatbook/Audio/recording_service.py` (`__init__` at ~line 144, `_handle_audio_chunk` at ~line 535)
- Modify: `Tests/conftest.py` (`_no_real_audio_device` at ~line 584)
- Test: `Tests/Audio/test_recording_retain_audio.py` (new)

**Interfaces:**
- Produces: `AudioRecordingService(..., retain_audio: bool = True)`; when `False`, `_handle_audio_chunk` still invokes `self.callback(chunk)` but never appends to `audio_buffer` or `audio_queue`.
- Produces: the autouse guard replaces `AudioRecordingService._recording_loop` with a stub that sets `is_recording = False`, unless the test carries `@pytest.mark.real_audio_device`.

- [ ] **Step 1: Write the failing tests**

`Tests/Audio/test_recording_retain_audio.py`:
```python
"""Task 1: `retain_audio=False` keeps the recorder from accumulating PCM."""
from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def _make_service(**kwargs):
    from tldw_chatbook.Audio.recording_service import AudioRecordingService

    with patch("tldw_chatbook.Audio.recording_service.PYAUDIO_AVAILABLE", True):
        with patch("tldw_chatbook.Audio.recording_service.pyaudio"):
            return AudioRecordingService(use_vad=False, **kwargs)


def test_retain_audio_false_skips_buffer_and_queue_but_calls_back():
    service = _make_service(retain_audio=False)
    seen: list[bytes] = []
    service.callback = seen.append

    service._handle_audio_chunk(b"\x01\x00" * 320)

    assert seen == [b"\x01\x00" * 320]
    assert service.audio_buffer == []
    assert service._audio_buffer_bytes == 0
    assert service.audio_queue.empty()


def test_retain_audio_default_keeps_old_behaviour():
    service = _make_service()
    service._handle_audio_chunk(b"\x01\x00" * 320)

    assert len(service.audio_buffer) == 1
    assert service._audio_buffer_bytes == 640
    assert not service.audio_queue.empty()


def test_autouse_guard_replaces_the_recording_loop():
    from tldw_chatbook.Audio.recording_service import AudioRecordingService

    service = _make_service()
    service.is_recording = True
    AudioRecordingService._recording_loop(service)

    assert service.is_recording is False
    assert AudioRecordingService._recording_loop.__name__ == "_guarded_recording_loop"
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest Tests/Audio/test_recording_retain_audio.py -q -p no:cacheprovider
```
Expected: 3 failures (`TypeError: unexpected keyword 'retain_audio'`, buffer not empty, `__name__ != "_guarded_recording_loop"`).

- [ ] **Step 3: Add the kwarg and the branch**

In `recording_service.py` `__init__` signature, after `on_buffer_limit`:
```python
        retain_audio: bool = True,
```
Docstring line to add under Args:
```
            retain_audio: When False, delivered chunks go only to the
                callback -- nothing is appended to ``audio_buffer`` or
                ``audio_queue``. Long captures (meetings) that stream to
                disk themselves set this; the default keeps every existing
                caller's behaviour.
```
In the body next to `self.on_buffer_limit = on_buffer_limit`:
```python
        self.retain_audio = bool(retain_audio)
```
In `_handle_audio_chunk`, replace the two retention statements:
```python
        if retained and self.retain_audio:
            self.audio_buffer.append(retained)
            self._audio_buffer_bytes += len(retained)
        if retained and self.retain_audio:
            self.audio_queue.put(retained)
```

- [ ] **Step 4: Extend the conftest guard**

In `Tests/conftest.py`, at the end of `_no_real_audio_device` (after the `streaming_sink` monkeypatch):
```python
    # Meeting transcription (2026-09-04): the same backstop for the INPUT
    # side. `AudioRecordingService.start_recording` only spawns a thread
    # running `_recording_loop`, which is where the backend opens the
    # device -- so stubbing the loop is the single chokepoint. Tests that
    # exercise `_pyaudio_recording_loop` / `_sounddevice_recording_loop`
    # directly with mocked backends are unaffected.
    import tldw_chatbook.Audio.recording_service as recording_service

    def _guarded_recording_loop(self) -> None:
        self.is_recording = False

    monkeypatch.setattr(
        recording_service.AudioRecordingService,
        "_recording_loop",
        _guarded_recording_loop,
    )
```
Also add one sentence to the fixture docstring: "Since the meeting-transcription work it also stubs `AudioRecordingService._recording_loop` so no test opens a microphone."

- [ ] **Step 5: Run the new tests and the existing recorder suite**

```bash
.venv/bin/python -m pytest Tests/Audio/test_recording_retain_audio.py Tests/Audio/test_recording_service.py Tests/Audio/test_recording_vad_preroll.py -q -p no:cacheprovider
```
Expected: all pass. If an existing test asserts on `_recording_loop` being the real dispatcher, mark that test `@pytest.mark.real_audio_device` only if it mocks the backend module itself (it then opens no hardware).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Audio/recording_service.py Tests/conftest.py Tests/Audio/test_recording_retain_audio.py
git commit -m "feat(audio): retain_audio recorder flag + input-side hardware guard (meeting transcription task 1)"
```

---

### Task 2: Placeholder-header WAV writer and recovery patching

Spec §4 file layout, §7 crash safety.

**Files:**
- Create: `tldw_chatbook/Audio/wav_writer.py`
- Test: `Tests/Audio/test_wav_writer.py`

**Interfaces:**
- Produces:
  - `wav_header(data_bytes: int, *, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes` (exactly 44 bytes)
  - `class PlaceholderWavWriter` with `__init__(self, path: Path, *, sample_rate=16000, channels=1, sample_width=2)`, `write(pcm: bytes) -> None`, `close() -> None`, attribute `bytes_written: int`, property `audio_position_s: float`, property `closed: bool`
  - `wav_needs_patch(path: Path) -> bool`
  - `patch_wav_header(path: Path, *, sample_rate=16000, channels=1, sample_width=2) -> int` (returns data bytes)

- [ ] **Step 1: Write the failing tests**

`Tests/Audio/test_wav_writer.py`:
```python
"""Task 2: crash-safe WAV files (spec §4, §7)."""
from __future__ import annotations

import wave

import pytest

from tldw_chatbook.Audio.wav_writer import (
    PlaceholderWavWriter,
    patch_wav_header,
    wav_header,
    wav_needs_patch,
)

pytestmark = pytest.mark.unit

FRAME = b"\x10\x00" * 320  # 20 ms of a constant sample


def test_header_is_44_bytes_and_encodes_sizes():
    header = wav_header(640)
    assert len(header) == 44
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    assert int.from_bytes(header[40:44], "little") == 640
    assert int.from_bytes(header[4:8], "little") == 36 + 640


def test_writer_streams_and_patches_on_close(tmp_path):
    path = tmp_path / "mixed.wav"
    writer = PlaceholderWavWriter(path)
    writer.write(FRAME)
    writer.write(FRAME)
    assert writer.bytes_written == 1280
    assert writer.audio_position_s == pytest.approx(0.04)
    writer.close()
    assert writer.closed

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        assert handle.getnframes() == 640


def test_unclosed_file_is_detected_and_patched(tmp_path):
    path = tmp_path / "you.wav"
    writer = PlaceholderWavWriter(path)
    writer.write(FRAME)
    writer._handle.flush()  # simulate a crash: never close()

    assert wav_needs_patch(path)
    assert patch_wav_header(path) == 640
    assert not wav_needs_patch(path)
    with wave.open(str(path), "rb") as handle:
        assert handle.getnframes() == 320


def test_write_after_close_raises(tmp_path):
    writer = PlaceholderWavWriter(tmp_path / "x.wav")
    writer.close()
    with pytest.raises(ValueError):
        writer.write(FRAME)


def test_needs_patch_false_for_missing_or_tiny_file(tmp_path):
    assert not wav_needs_patch(tmp_path / "absent.wav")
    (tmp_path / "tiny.wav").write_bytes(b"RIFF")
    assert not wav_needs_patch(tmp_path / "tiny.wav")
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest Tests/Audio/test_wav_writer.py -q -p no:cacheprovider
```
Expected: `ModuleNotFoundError: tldw_chatbook.Audio.wav_writer`.

- [ ] **Step 3: Implement**

`tldw_chatbook/Audio/wav_writer.py`:
```python
"""Crash-safe WAV writing for meeting recordings.

The stdlib ``wave`` module writes its header only on ``close()``, so a crash
mid-meeting would leave an unreadable file. This writer puts a 44-byte
header with a zero data length first, appends raw PCM as it arrives, and
patches the header on close. A header still reading zero length marks an
unfinished file; ``patch_wav_header`` repairs it from the file size.

No Textual, no numpy: stdlib only.
"""
from __future__ import annotations

import struct
from pathlib import Path

HEADER_BYTES = 44


def wav_header(
    data_bytes: int,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Return a canonical 44-byte PCM WAV header for ``data_bytes`` of audio."""
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8,
        b"data",
        data_bytes,
    )


class PlaceholderWavWriter:
    """Append-only WAV writer whose header is finalised on ``close()``."""

    def __init__(
        self,
        path: Path,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.bytes_written = 0
        self._closed = False
        self._handle = open(self.path, "wb")  # noqa: SIM115 - long-lived stream
        self._handle.write(
            wav_header(0, sample_rate=sample_rate, channels=channels, sample_width=sample_width)
        )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def audio_position_s(self) -> float:
        return self.bytes_written / float(self.sample_rate * self.channels * self.sample_width)

    def write(self, pcm: bytes) -> None:
        if self._closed:
            raise ValueError(f"{self.path.name} is closed")
        self._handle.write(pcm)
        self.bytes_written += len(pcm)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._handle.seek(0)
        self._handle.write(
            wav_header(
                self.bytes_written,
                sample_rate=self.sample_rate,
                channels=self.channels,
                sample_width=self.sample_width,
            )
        )
        self._handle.close()


def wav_needs_patch(path: Path) -> bool:
    """True when ``path`` has a placeholder header but audio bytes after it."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= HEADER_BYTES:
        return False
    with open(path, "rb") as handle:
        head = handle.read(HEADER_BYTES)
    if len(head) < HEADER_BYTES or head[:4] != b"RIFF" or head[36:40] != b"data":
        return False
    return int.from_bytes(head[40:44], "little") == 0


def patch_wav_header(
    path: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> int:
    """Rewrite the header from the file size; return the data byte count."""
    path = Path(path)
    data_bytes = max(0, path.stat().st_size - HEADER_BYTES)
    with open(path, "r+b") as handle:
        handle.seek(0)
        handle.write(
            wav_header(
                data_bytes, sample_rate=sample_rate, channels=channels, sample_width=sample_width
            )
        )
    return data_bytes
```

- [ ] **Step 4: Run to verify they pass**

```bash
.venv/bin/python -m pytest Tests/Audio/test_wav_writer.py -q -p no:cacheprovider
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/wav_writer.py Tests/Audio/test_wav_writer.py
git commit -m "feat(audio): placeholder-header WAV writer with crash recovery (meeting transcription task 2)"
```

---
### Task 3: `recorder_factory` kwarg on the lazy dictation service

Spec §3.7. Mirrors `RealtimeMicTap.recorder_factory`.

**Files:**
- Modify: `tldw_chatbook/Audio/dictation_service_lazy.py` (`__init__` ~line 208, `audio_service` property ~line 480)
- Test: `Tests/Audio/test_dictation_recorder_factory.py` (new)

**Interfaces:**
- Produces: `LazyLiveDictationService(..., recorder_factory: Optional[Callable[..., Any]] = None)`. When set, the `audio_service` property calls `recorder_factory(backend=..., use_vad=True, vad_aggressiveness=..., vad_preroll_ms=..., chunk_size=..., max_buffer_bytes=..., on_buffer_limit=...)` instead of `AudioRecordingService(...)`. Meetings pass `lambda **_: capture`.

- [ ] **Step 1: Write the failing test**

`Tests/Audio/test_dictation_recorder_factory.py`:
```python
"""Task 3: the lazy dictation service accepts an injected recorder factory."""
from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _stub_settings(monkeypatch) -> None:
    from tldw_chatbook.Audio import dictation_service_lazy

    values = {
        "dictation.buffer_duration_ms": 10,
        "dictation.privacy.save_history": False,
        "dictation.privacy.encrypt_history": True,
        "dictation.privacy.local_only": False,
        "dictation.privacy.auto_clear_buffer": True,
    }

    def _get(section: str, key: Any = None, default: Any = None) -> Any:
        if key is not None and not isinstance(key, str):
            key, default = None, key
        path = section if key is None else f"{section}.{key}"
        return values.get(path, default)

    monkeypatch.setattr(dictation_service_lazy, "get_cli_setting", _get)


class _Recorder:
    sample_rate = 16000
    channels = 1


def test_recorder_factory_is_used_and_receives_recorder_kwargs(monkeypatch):
    _stub_settings(monkeypatch)
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    recorder = _Recorder()
    seen: dict[str, Any] = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return recorder

    service = LazyLiveDictationService(
        transcription_provider="faster-whisper",
        enable_commands=False,
        recorder_factory=factory,
    )

    assert service.audio_service is recorder
    assert seen["use_vad"] is True
    assert "chunk_size" in seen and "vad_preroll_ms" in seen
    assert service._audio_service is recorder  # cached, not rebuilt


def test_default_factory_is_the_real_recorder_class(monkeypatch):
    _stub_settings(monkeypatch)
    from tldw_chatbook.Audio import dictation_service_lazy
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    service = LazyLiveDictationService(enable_commands=False)
    assert service._recorder_factory is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/bin/python -m pytest Tests/Audio/test_dictation_recorder_factory.py -q -p no:cacheprovider
```
Expected: `TypeError: __init__() got an unexpected keyword argument 'recorder_factory'`.

- [ ] **Step 3: Implement**

In `__init__` signature, after `transcription_service_factory`:
```python
        recorder_factory: Optional[Callable[..., Any]] = None,
```
Docstring Args entry:
```
            recorder_factory: Callable used to construct the recorder,
                receiving the same kwargs `AudioRecordingService` would.
                Defaults to `AudioRecordingService`. Exists so a caller can
                substitute its own capture (meetings hand in a mixed
                mic+system stream) and so tests can inject a fake.
```
Body, next to `self._transcription_service_factory = ...`:
```python
        self._recorder_factory = recorder_factory
```
In the `audio_service` property replace the construction:
```python
                from .recording_service import AudioRecordingService

                factory = self._recorder_factory or AudioRecordingService
                self._audio_service = factory(
                    backend=self.audio_backend_preference,
                    use_vad=True,
                    vad_aggressiveness=self.vad_aggressiveness,
                    vad_preroll_ms=self.vad_preroll_ms,
                    chunk_size=int(self.buffer_duration_ms * 16),
                    max_buffer_bytes=self.max_buffer_bytes,
                    on_buffer_limit=self.on_buffer_limit,
                )
```
(keep the existing comment lines and the `logger.info` that follow).

- [ ] **Step 4: Run to verify it passes, plus the lazy-service suite**

```bash
.venv/bin/python -m pytest Tests/Audio/test_dictation_recorder_factory.py Tests/Audio/test_dictation_lazy_transcription.py Tests/Audio/test_dictation_service.py -q -p no:cacheprovider
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/dictation_service_lazy.py Tests/Audio/test_dictation_recorder_factory.py
git commit -m "feat(audio): recorder_factory injection on LazyLiveDictationService (meeting transcription task 3)"
```

---

### Task 4: `EnergyRing` attribution

Spec §3.2 step 2 and §4 "Attribution (call mode)".

**Files:**
- Create: `tldw_chatbook/Audio/meeting_capture.py` (this task adds `EnergyRing`, `SpeechRun`, `rms_int16`; Task 5 adds `MeetingCapture` to the same file)
- Test: `Tests/Audio/test_meeting_capture.py` (new; Task 5 appends to it)

**Interfaces:**
- Produces:
  - `rms_int16(pcm: bytes) -> float` (0.0 for empty)
  - `ABS_MIN_RMS: float` (−60 dBFS ≈ 32.77), `SHARE_YOU = 0.7`, `SHARE_OTHERS = 0.3`
  - `@dataclass SpeechRun(start_s: float, end_s: float | None = None)`
  - `class EnergyRing(bucket_s: float = 0.1, horizon_s: float = 600.0, floor_window_s: float = 30.0)` with `add(position_s: float, mic_rms: float, sys_rms: float) -> None`, `dominant_source(start_s: float, end_s: float) -> str` returning `"you" | "others" | "both"`, `floor(source: str, end_s: float) -> float` (`source` is `"mic"` or `"sys"`)

- [ ] **Step 1: Write the failing tests**

`Tests/Audio/test_meeting_capture.py` (initial content):
```python
"""Meeting capture: energy attribution (Task 4) and the mixer (Task 5)."""
from __future__ import annotations

import math

import pytest

from tldw_chatbook.Audio.meeting_capture import (
    ABS_MIN_RMS,
    EnergyRing,
    SpeechRun,
    rms_int16,
)

pytestmark = pytest.mark.unit


def _fill(ring: EnergyRing, start_s: float, end_s: float, mic: float, sys_: float) -> None:
    t = start_s
    while t < end_s:
        ring.add(t, mic, sys_)
        t += 0.1


def test_rms_of_constant_and_empty():
    assert rms_int16(b"") == 0.0
    assert rms_int16(b"\x00\x10" * 10) == pytest.approx(4096.0)


def test_abs_min_is_minus_60_dbfs():
    assert ABS_MIN_RMS == pytest.approx(32768 * 10 ** (-60 / 20), rel=1e-6)


def test_mic_dominant_window_is_you():
    ring = EnergyRing()
    _fill(ring, 0.0, 5.0, mic=2000.0, sys_=0.0)
    assert ring.dominant_source(0.0, 5.0) == "you"


def test_system_dominant_window_is_others():
    ring = EnergyRing()
    _fill(ring, 0.0, 5.0, mic=0.0, sys_=2000.0)
    assert ring.dominant_source(0.0, 5.0) == "others"


def test_balanced_window_is_both():
    ring = EnergyRing()
    _fill(ring, 0.0, 5.0, mic=2000.0, sys_=2000.0)
    assert ring.dominant_source(0.0, 5.0) == "both"


def test_room_noise_below_adaptive_floor_does_not_flip_to_both():
    ring = EnergyRing()
    # 30 s of steady room noise on the mic (p10 == 200 -> floor 600),
    # then the remote party talks while the mic keeps its noise.
    _fill(ring, 0.0, 30.0, mic=200.0, sys_=0.0)
    _fill(ring, 30.0, 35.0, mic=200.0, sys_=3000.0)
    assert ring.floor("mic", 35.0) == pytest.approx(600.0)
    assert ring.dominant_source(30.0, 35.0) == "others"


def test_digital_silence_uses_absolute_minimum_floor():
    ring = EnergyRing()
    _fill(ring, 0.0, 30.0, mic=0.0, sys_=0.0)
    assert ring.floor("sys", 30.0) == pytest.approx(ABS_MIN_RMS)


def test_no_active_buckets_falls_back_to_higher_raw_sum():
    ring = EnergyRing()
    _fill(ring, 0.0, 30.0, mic=0.0, sys_=0.0)
    _fill(ring, 30.0, 31.0, mic=5.0, sys_=20.0)  # both under ABS_MIN
    assert ring.dominant_source(30.0, 31.0) == "others"


def test_ring_forgets_beyond_horizon():
    ring = EnergyRing(horizon_s=1.0)
    _fill(ring, 0.0, 3.0, mic=1000.0, sys_=0.0)
    assert ring.dominant_source(0.0, 0.5) == "others"  # evicted: nothing active, sums tie -> not "you"


def test_speech_run_defaults_open():
    run = SpeechRun(1.5)
    assert run.end_s is None and math.isclose(run.start_s, 1.5)
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest Tests/Audio/test_meeting_capture.py -q -p no:cacheprovider
```
Expected: `ModuleNotFoundError: tldw_chatbook.Audio.meeting_capture`.

- [ ] **Step 3: Implement**

`tldw_chatbook/Audio/meeting_capture.py` (first version; Task 5 appends `MeetingCapture`):
```python
"""Meeting capture: mic + system audio mixed into one dictation stream.

Textual-free. numpy is required (the recorder already requires it).

`EnergyRing` answers "who was talking in this window" from per-source RMS
history (spec §4). `MeetingCapture` (Task 5) duck-types the recorder
surface `LazyLiveDictationService` uses.
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np

#: -60 dBFS expressed as int16 RMS. Digital silence must not yield a zero
#: adaptive floor that any dither would exceed.
ABS_MIN_RMS: float = 32768 * 10 ** (-60 / 20)
SHARE_YOU: float = 0.7
SHARE_OTHERS: float = 0.3
FLOOR_MULTIPLIER: float = 3.0


def rms_int16(pcm: bytes) -> float:
    """RMS of little-endian int16 PCM; 0.0 for an empty buffer."""
    if len(pcm) < 2:
        return 0.0
    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


@dataclass
class SpeechRun:
    """A span of VAD-detected speech in audio seconds; ``end_s`` None while open."""

    start_s: float
    end_s: float | None = None


class EnergyRing:
    """Per-source RMS history in fixed buckets, bounded by a time horizon."""

    def __init__(
        self,
        bucket_s: float = 0.1,
        horizon_s: float = 600.0,
        floor_window_s: float = 30.0,
    ) -> None:
        self.bucket_s = bucket_s
        self.horizon_s = horizon_s
        self.floor_window_s = floor_window_s
        self._times: Deque[float] = deque()
        self._mic: Deque[float] = deque()
        self._sys: Deque[float] = deque()

    def add(self, position_s: float, mic_rms: float, sys_rms: float) -> None:
        bucket = math.floor(position_s / self.bucket_s) * self.bucket_s
        if self._times and bucket <= self._times[-1]:
            # Same bucket: keep the louder reading for each source.
            self._mic[-1] = max(self._mic[-1], mic_rms)
            self._sys[-1] = max(self._sys[-1], sys_rms)
        else:
            self._times.append(bucket)
            self._mic.append(mic_rms)
            self._sys.append(sys_rms)
        while self._times and self._times[0] < bucket - self.horizon_s:
            self._times.popleft()
            self._mic.popleft()
            self._sys.popleft()

    def _slice(self, start_s: float, end_s: float) -> Tuple[list, list]:
        times = list(self._times)
        lo = bisect.bisect_left(times, math.floor(start_s / self.bucket_s) * self.bucket_s)
        hi = bisect.bisect_right(times, end_s)
        return list(self._mic)[lo:hi], list(self._sys)[lo:hi]

    def floor(self, source: str, end_s: float) -> float:
        """Adaptive noise floor: 3x the 10th percentile of the last 30 s, never below ABS_MIN_RMS."""
        mic, sys_ = self._slice(end_s - self.floor_window_s, end_s)
        values = mic if source == "mic" else sys_
        if not values:
            return ABS_MIN_RMS
        p10 = float(np.percentile(np.asarray(values, dtype=np.float64), 10))
        return max(FLOOR_MULTIPLIER * p10, ABS_MIN_RMS)

    def dominant_source(self, start_s: float, end_s: float) -> str:
        """Label a window ``you`` / ``others`` / ``both`` (spec §4)."""
        mic, sys_ = self._slice(start_s, end_s)
        mic_floor = self.floor("mic", end_s)
        sys_floor = self.floor("sys", end_s)
        mic_active = sum(v for v in mic if v > mic_floor)
        sys_active = sum(v for v in sys_ if v > sys_floor)
        total = mic_active + sys_active
        if total <= 0.0:
            # ponytail: energy-share heuristic; the Diarizer seam (meeting_session.py)
            # replaces this whole method in phase 2.
            return "you" if sum(mic) > sum(sys_) else "others"
        share_you = mic_active / total
        if share_you >= SHARE_YOU:
            return "you"
        if share_you <= SHARE_OTHERS:
            return "others"
        return "both"
```

- [ ] **Step 4: Run to verify they pass**

```bash
.venv/bin/python -m pytest Tests/Audio/test_meeting_capture.py -q -p no:cacheprovider
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/meeting_capture.py Tests/Audio/test_meeting_capture.py
git commit -m "feat(audio): EnergyRing speaker attribution for meetings (meeting transcription task 4)"
```

---
### Task 5: `MeetingCapture` — mixing, alignment, VAD runs, writers, fault

Spec §3.2 in full, §4 "Frames" and "Pause".

**Files:**
- Modify: `tldw_chatbook/Audio/meeting_capture.py` (append)
- Test: `Tests/Audio/test_meeting_capture.py` (append)

**Interfaces:**
- Consumes: `PlaceholderWavWriter` (Task 2), `EnergyRing`/`SpeechRun`/`rms_int16` (Task 4).
- Consumes (tap contract, implemented in Task 6): an object with `start(on_frames: Callable[[bytes], None]) -> bool`, `stop() -> None`, attribute `state: str`.
- Produces:
  - `mix_int16(a: bytes, b: bytes) -> bytes` (saturating add; equal lengths)
  - `class MeetingCapture` with `__init__(self, *, mic_recorder_factory: Callable[..., Any], tap: Any | None, writers: Mapping[str, PlaceholderWavWriter], vad_factory: Callable[[], Any] | None = None, silence_threshold_s: float = 2.0, preroll_frames: int = 12)`
  - recorder surface: `start_recording(callback=None, save_to_file=None) -> bool`, `stop_recording() -> None`, `get_audio_level() -> float`, `get_audio_devices() -> list`, `set_device(device_id) -> bool`, `is_available() -> bool`, `sample_rate = 16000`, `channels = 1`
  - meeting surface: `levels() -> tuple[float, float]`, `audio_position_s: float`, `last_speech_position_s: float`, `fault: Exception | None`, `pause()`, `resume()`, `paused: bool`, `closed_runs_after(t: float) -> list[SpeechRun]`, `dominant_source(start_s, end_s) -> str`, `mode: str` (`"call"` if a tap was given else `"room"`)
  - `writers` keys: `"mixed"` required; `"you"` and `"others"` present in call mode.

- [ ] **Step 1: Append the failing tests**

Append to `Tests/Audio/test_meeting_capture.py`:
```python
# ---------------------------------------------------------------- Task 5
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_chatbook.Audio.meeting_capture import MeetingCapture, mix_int16
from tldw_chatbook.Audio.wav_writer import PlaceholderWavWriter

FRAME_BYTES = 640
SILENT = b"\x00\x00" * 320
LOUD = b"\x00\x20" * 320  # 8192 amplitude
QUIET = b"\x10\x00" * 320  # 16 amplitude


class FakeRecorder:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.callback = None
        self.stopped = 0

    def start_recording(self, callback=None, save_to_file=None):
        self.callback = callback
        return True

    def stop_recording(self):
        self.stopped += 1
        return None

    def get_audio_devices(self):
        return [{"id": 0, "name": "fake"}]

    def set_device(self, device_id):
        return True


class FakeTap:
    def __init__(self):
        self.on_frames = None
        self.state = "stopped"
        self.stops = 0

    def start(self, on_frames):
        self.on_frames = on_frames
        self.state = "running"
        return True

    def stop(self):
        self.stops += 1
        self.state = "stopped"

    def push(self, frame: bytes):
        self.on_frames(frame)


class EnergyVad:
    """Stand-in for webrtcvad: speech == RMS above 100."""

    def is_speech(self, frame: bytes, rate: int) -> bool:
        return rms_int16(frame) > 100.0


def _capture(tmp_path, *, call_mode=True, silence=2.0, preroll=12):
    writers = {"mixed": PlaceholderWavWriter(tmp_path / "mixed.wav")}
    tap = None
    if call_mode:
        writers["you"] = PlaceholderWavWriter(tmp_path / "you.wav")
        writers["others"] = PlaceholderWavWriter(tmp_path / "others.wav")
        tap = FakeTap()
    recorders: list[FakeRecorder] = []

    def factory(**kwargs):
        recorders.append(FakeRecorder(**kwargs))
        return recorders[-1]

    cap = MeetingCapture(
        mic_recorder_factory=factory,
        tap=tap,
        writers=writers,
        vad_factory=EnergyVad,
        silence_threshold_s=silence,
        preroll_frames=preroll,
    )
    return cap, recorders, tap, writers


def test_mix_saturates_and_keeps_length():
    a = np.full(320, 30000, dtype=np.int16).tobytes()
    b = np.full(320, 30000, dtype=np.int16).tobytes()
    mixed = np.frombuffer(mix_int16(a, b), dtype=np.int16)
    assert mixed.size == 320 and int(mixed[0]) == 32767


@settings(max_examples=100, deadline=None)
@given(
    st.lists(st.integers(-32768, 32767), min_size=1, max_size=64),
    st.lists(st.integers(-32768, 32767), min_size=1, max_size=64),
)
def test_mix_equals_clipped_sum(a_vals, b_vals):
    n = min(len(a_vals), len(b_vals))
    a = np.asarray(a_vals[:n], dtype=np.int16)
    b = np.asarray(b_vals[:n], dtype=np.int16)
    expected = np.clip(a.astype(np.int32) + b.astype(np.int32), -32768, 32767)
    got = np.frombuffer(mix_int16(a.tobytes(), b.tobytes()), dtype=np.int16)
    assert np.array_equal(got.astype(np.int32), expected)


def test_start_builds_mic_with_retain_off_and_starts_tap(tmp_path):
    cap, recorders, tap, _ = _capture(tmp_path)
    assert cap.mode == "call"
    assert cap.start_recording(callback=lambda b: None) is True
    assert recorders[0].init_kwargs["retain_audio"] is False
    assert recorders[0].init_kwargs["use_vad"] is False
    assert recorders[0].init_kwargs["chunk_size"] == 320
    assert tap.state == "running"


def test_mic_frame_pulls_one_tap_frame_and_zero_fills(tmp_path):
    cap, recorders, tap, writers = _capture(tmp_path)
    cap.start_recording(callback=lambda b: None)
    tap.push(QUIET)
    recorders[0].callback(QUIET)   # pairs with the pushed frame
    recorders[0].callback(QUIET)   # nothing queued -> zeros
    assert writers["you"].bytes_written == 2 * FRAME_BYTES
    assert writers["others"].bytes_written == 2 * FRAME_BYTES
    assert writers["mixed"].bytes_written == 2 * FRAME_BYTES
    assert cap.audio_position_s == pytest.approx(0.04)
    mixed_second = np.frombuffer(QUIET, dtype=np.int16)
    assert cap.levels()[1] == 0.0 or cap.levels()[1] < cap.levels()[0]


def test_backlog_over_200ms_drops_one_extra_frame_per_tick(tmp_path):
    cap, recorders, tap, _ = _capture(tmp_path)
    cap.start_recording(callback=lambda b: None)
    for _ in range(20):          # 400 ms queued
        tap.push(QUIET)
    recorders[0].callback(QUIET)  # takes one, drops one extra
    assert cap._tap_backlog_bytes() == 18 * FRAME_BYTES
    for _ in range(8):
        recorders[0].callback(QUIET)
    assert cap._tap_backlog_bytes() <= 10 * FRAME_BYTES


def test_tap_buffer_is_bounded_to_one_second(tmp_path):
    cap, _, tap, _ = _capture(tmp_path)
    cap.start_recording(callback=lambda b: None)
    for _ in range(80):
        tap.push(QUIET)
    assert cap._tap_backlog_bytes() == 50 * FRAME_BYTES


def test_vad_runs_open_with_preroll_close_after_silence(tmp_path):
    cap, recorders, tap, _ = _capture(tmp_path, silence=0.1, preroll=2)
    got: list[bytes] = []
    cap.start_recording(callback=got.append)
    mic = recorders[0].callback
    for _ in range(5):
        mic(SILENT)              # 0.00-0.10 s, pre-roll keeps last 2
    mic(LOUD)                    # 0.10-0.12 s speech
    mic(LOUD)                    # 0.12-0.14 s
    assert cap.closed_runs_after(0.0) == []
    assert cap.last_speech_position_s == pytest.approx(0.14)
    for _ in range(6):
        mic(SILENT)              # gap of 0.12 s >= 0.1 -> run closes
    runs = cap.closed_runs_after(0.0)
    assert len(runs) == 1
    assert runs[0].start_s == pytest.approx(0.06)   # 0.10 - 2 pre-roll frames
    assert runs[0].end_s == pytest.approx(0.14)
    assert b"".join(got) == SILENT + SILENT + LOUD + LOUD


def test_closed_runs_after_filters_by_end(tmp_path):
    cap, recorders, _, _ = _capture(tmp_path, silence=0.02, preroll=0)
    cap.start_recording(callback=lambda b: None)
    mic = recorders[0].callback
    mic(LOUD); mic(SILENT); mic(SILENT)   # run 1 ends at 0.02
    mic(LOUD); mic(SILENT); mic(SILENT)   # run 2 ends at 0.08
    assert [r.end_s for r in cap.closed_runs_after(0.05)] == [pytest.approx(0.08)]


def test_pause_discards_tap_frames_and_writes_nothing(tmp_path):
    cap, recorders, tap, writers = _capture(tmp_path)
    cap.start_recording(callback=lambda b: None)
    cap.pause()
    assert cap.paused
    tap.push(QUIET)
    recorders[0].callback(QUIET)
    assert writers["mixed"].bytes_written == 0
    assert cap._tap_backlog_bytes() == 0
    cap.resume()
    recorders[0].callback(QUIET)
    assert writers["mixed"].bytes_written == FRAME_BYTES


def test_writer_error_is_recorded_as_fault_not_raised(tmp_path):
    cap, recorders, _, writers = _capture(tmp_path)
    cap.start_recording(callback=lambda b: None)
    writers["mixed"].close()   # next write raises ValueError
    recorders[0].callback(QUIET)
    assert isinstance(cap.fault, ValueError)


def test_stop_closes_writers_open_run_and_tap(tmp_path):
    cap, recorders, tap, writers = _capture(tmp_path, silence=5.0, preroll=0)
    cap.start_recording(callback=lambda b: None)
    recorders[0].callback(LOUD)
    cap.stop_recording()
    assert all(w.closed for w in writers.values())
    assert recorders[0].stopped == 1 and tap.stops == 1
    assert cap.closed_runs_after(0.0)[0].end_s == pytest.approx(0.02)


def test_room_mode_has_no_tap_and_only_mixed_writer(tmp_path):
    cap, recorders, tap, writers = _capture(tmp_path, call_mode=False)
    assert cap.mode == "room" and tap is None
    cap.start_recording(callback=lambda b: None)
    recorders[0].callback(QUIET)
    assert set(writers) == {"mixed"} and writers["mixed"].bytes_written == FRAME_BYTES
    assert cap.dominant_source(0.0, 0.02) in {"you", "others", "both"}


def test_recorder_surface_forwards_to_mic(tmp_path):
    cap, recorders, _, _ = _capture(tmp_path)
    cap.start_recording()
    assert cap.get_audio_devices() == [{"id": 0, "name": "fake"}]
    assert cap.set_device(0) is True
    assert cap.is_available() is True
    assert cap.sample_rate == 16000 and cap.channels == 1
    assert 0.0 <= cap.get_audio_level() <= 1.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest Tests/Audio/test_meeting_capture.py -q -p no:cacheprovider
```
Expected: `ImportError: cannot import name 'MeetingCapture'`.

- [ ] **Step 3: Implement**

Append to `tldw_chatbook/Audio/meeting_capture.py`:
```python
import threading
from typing import Any, Callable, Mapping, Optional

from loguru import logger

from .wav_writer import PlaceholderWavWriter

FRAME_BYTES = 640            # 20 ms at 16 kHz mono int16
FRAME_S = 0.02
BACKLOG_BYTES = 10 * FRAME_BYTES   # 200 ms
TAP_BUFFER_MAX = 50 * FRAME_BYTES  # 1 s
BYTES_PER_S = 32000


def mix_int16(a: bytes, b: bytes) -> bytes:
    """Saturating sum of two equal-length int16 buffers."""
    x = np.frombuffer(a, dtype=np.int16).astype(np.int32)
    y = np.frombuffer(b, dtype=np.int16).astype(np.int32)
    return np.clip(x + y, -32768, 32767).astype(np.int16).tobytes()


def _default_vad_factory():
    import webrtcvad

    vad = webrtcvad.Vad()
    vad.set_mode(2)
    return vad


class MeetingCapture:
    """Recorder-shaped source for one meeting: mic + system tap -> one mixed stream.

    The mic callback is the clock (spec §3.2): every mic chunk pulls the
    same number of system bytes from the tap buffer (zero-filled when
    short), writes all tracks, updates the energy ring, gates the mix
    through VAD and hands speech to the dictation callback.
    """

    sample_rate = 16000
    channels = 1

    def __init__(
        self,
        *,
        mic_recorder_factory: Callable[..., Any],
        tap: Any | None,
        writers: Mapping[str, PlaceholderWavWriter],
        vad_factory: Callable[[], Any] | None = None,
        silence_threshold_s: float = 2.0,
        preroll_frames: int = 12,
    ) -> None:
        if "mixed" not in writers:
            raise ValueError("writers must include 'mixed'")
        self._mic_factory = mic_recorder_factory
        self._tap = tap
        self._writers = dict(writers)
        self._vad_factory = vad_factory or _default_vad_factory
        self._silence_threshold_s = silence_threshold_s
        self._preroll: deque = deque(maxlen=max(0, preroll_frames))
        self.mode = "call" if tap is not None else "room"
        self._mic: Any | None = None
        self._vad: Any | None = None
        self._callback: Optional[Callable[[bytes], None]] = None
        self._running = False
        self._paused = False
        self._tap_lock = threading.Lock()
        self._tap_buf = bytearray()
        self._ring = EnergyRing()
        self._runs: list[SpeechRun] = []
        self._open_run: SpeechRun | None = None
        self._levels = (0.0, 0.0)
        self.last_speech_position_s = 0.0
        self.fault: Exception | None = None

    # ---- recorder surface -------------------------------------------------
    def start_recording(self, callback=None, save_to_file=None) -> bool:
        self._callback = callback
        try:
            self._vad = self._vad_factory()
        except Exception as exc:  # noqa: BLE001 - VAD optional; gate everything through
            logger.warning("Meeting VAD unavailable, passing all audio: {}", exc)
            self._vad = None
        self._mic = self._mic_factory(use_vad=False, retain_audio=False, chunk_size=320)
        self._running = True
        if self._tap is not None and not self._tap.start(self._on_tap_frame):
            logger.warning("System audio tap failed to start; continuing in room mode")
            self._tap = None
            self.mode = "room"
        return bool(self._mic.start_recording(callback=self._on_mic_frame))

    def stop_recording(self) -> None:
        self._running = False
        if self._mic is not None:
            try:
                self._mic.stop_recording()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Mic stop failed: {}", exc)
        if self._tap is not None:
            self._tap.stop()
        self._close_open_run()
        for writer in self._writers.values():
            writer.close()
        return None

    def get_audio_level(self) -> float:
        return self._levels[0]

    def get_audio_devices(self) -> list:
        return list(self._mic.get_audio_devices()) if self._mic is not None else []

    def set_device(self, device_id) -> bool:
        return bool(self._mic.set_device(device_id)) if self._mic is not None else False

    def is_available(self) -> bool:
        return True

    # ---- meeting surface --------------------------------------------------
    def levels(self) -> tuple[float, float]:
        return self._levels

    @property
    def audio_position_s(self) -> float:
        return self._writers["mixed"].audio_position_s

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def closed_runs_after(self, t: float) -> list[SpeechRun]:
        return [r for r in self._runs if r.end_s is not None and r.end_s > t]

    def dominant_source(self, start_s: float, end_s: float) -> str:
        return self._ring.dominant_source(start_s, end_s)

    def _tap_backlog_bytes(self) -> int:
        with self._tap_lock:
            return len(self._tap_buf)

    # ---- frame handling ---------------------------------------------------
    def _on_tap_frame(self, frame: bytes) -> None:
        with self._tap_lock:
            self._tap_buf.extend(frame)
            overflow = len(self._tap_buf) - TAP_BUFFER_MAX
            if overflow > 0:
                del self._tap_buf[:overflow]

    def _take_tap_bytes(self, n: int) -> bytes:
        with self._tap_lock:
            part = bytes(self._tap_buf[:n])
            del self._tap_buf[:n]
            if len(self._tap_buf) > BACKLOG_BYTES:
                del self._tap_buf[:FRAME_BYTES]   # one extra frame per tick
        if len(part) < n:
            part += b"\x00" * (n - len(part))
        return part

    def _on_mic_frame(self, chunk: bytes) -> None:
        if not self._running:
            return
        try:
            if self._paused:
                with self._tap_lock:
                    self._tap_buf.clear()
                return
            n = len(chunk)
            sys_part = self._take_tap_bytes(n) if self._tap is not None else b"\x00" * n
            mixed = mix_int16(chunk, sys_part) if self._tap is not None else chunk
            start_pos = self.audio_position_s
            self._writers["mixed"].write(mixed)
            if "you" in self._writers:
                self._writers["you"].write(chunk)
            if "others" in self._writers:
                self._writers["others"].write(sys_part)
            mic_rms, sys_rms = rms_int16(chunk), rms_int16(sys_part)
            self._levels = (min(1.0, mic_rms / 32768.0), min(1.0, sys_rms / 32768.0))
            self._ring.add(start_pos, mic_rms, sys_rms)
            speech = self._gate(mixed, start_pos)
            if speech and self._callback is not None:
                self._callback(speech)
        except Exception as exc:  # noqa: BLE001 - recorder would swallow it (spec §3.2)
            if self.fault is None:
                self.fault = exc
                logger.error("Meeting capture fault: {}", exc)

    def _gate(self, mixed: bytes, start_pos: float) -> bytes:
        out = bytearray()
        for i in range(0, len(mixed) - FRAME_BYTES + 1, FRAME_BYTES):
            frame = bytes(mixed[i : i + FRAME_BYTES])
            frame_pos = start_pos + (i // FRAME_BYTES) * FRAME_S
            is_speech = True if self._vad is None else bool(
                self._vad.is_speech(frame, self.sample_rate)
            )
            if is_speech:
                if self._open_run is None:
                    self._open_run = SpeechRun(max(0.0, frame_pos - len(self._preroll) * FRAME_S))
                    for buffered in self._preroll:
                        out.extend(buffered)
                    self._preroll.clear()
                out.extend(frame)
                self.last_speech_position_s = frame_pos + FRAME_S
            else:
                self._preroll.append(frame)
                if (
                    self._open_run is not None
                    and frame_pos + FRAME_S - self.last_speech_position_s
                    >= self._silence_threshold_s
                ):
                    self._close_open_run()
        return bytes(out)

    def _close_open_run(self) -> None:
        if self._open_run is not None:
            self._open_run.end_s = self.last_speech_position_s
            self._runs.append(self._open_run)
            self._open_run = None
```

- [ ] **Step 4: Run to verify they pass**

```bash
.venv/bin/python -m pytest Tests/Audio/test_meeting_capture.py -q -p no:cacheprovider
```
Expected: 24 passed. If `test_vad_runs_open_with_preroll_close_after_silence` disagrees on `start_s`, check the arithmetic: the first LOUD frame starts at 0.10 s (five silent frames × 0.02), the pre-roll holds 2 frames, so the run opens at 0.06.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/meeting_capture.py Tests/Audio/test_meeting_capture.py
git commit -m "feat(audio): MeetingCapture mixes mic and system audio into one dictation stream (meeting transcription task 5)"
```

---
### Task 6: `SystemAudioTap` — resolvers, subprocess reader, device fallback, probe

Spec §3.1 in full, §3.6 (helper lookup and dev-fallback compile only; the Swift source itself is Task 12).

**Files:**
- Create: `tldw_chatbook/Audio/system_audio_tap.py`
- Create: `Tests/Audio/fake_audiotap.py` (stand-in helper script, not a test module)
- Test: `Tests/Audio/test_system_audio_tap.py`

**Interfaces:**
- Consumes: `AudioRecordingService(retain_audio=..., use_vad=..., chunk_size=...)` (Task 1), `get_user_data_dir()` from `tldw_chatbook.config`.
- Produces (all in `system_audio_tap.py`):
  - `@dataclass(frozen=True) TapMode(kind: str, reason: str, command: tuple[str, ...] | None = None, device_name: str | None = None, device_index: int | None = None)`; `kind` ∈ `native_macos | native_parec | native_wasapi | virtual_device | unavailable`
  - `SINK_NAME_RE`, `validate_sink_name(name) -> str`, `parse_default_sink(output) -> str`, `linux_capture_command(tool, sink) -> tuple[str, ...]`, `resolve_wasapi_loopback(devices, default_output_name) -> int | None`, `macos_version_ok(mac_ver) -> bool`
  - `helper_source_path() -> Path`, `bundled_helper_path(executable=sys.executable) -> Path | None`, `dev_helper_path(data_dir) -> Path`, `ensure_helper(data_dir, *, run=subprocess.run, which=shutil.which, executable=sys.executable) -> Path | None`
  - `probe(*, system_source="auto", platform=sys.platform, mac_ver=None, which=shutil.which, run=subprocess.run, data_dir=None, query_devices=None, default_output_name=None) -> TapMode`
  - `class SubprocessTap(command, *, frame_bytes=640, restart_delay_s=2.0, spawn=subprocess.Popen, sleep=time.sleep)` with `start(on_frames) -> bool`, `stop() -> None`, `state: str` (`stopped|running|lost`), `restarts: int`, `exit_code: int | None`, `last_stderr: str`
  - `class DeviceTap(device_name, *, recorder_factory=None)` with the same `start/stop/state`
  - `build_tap(mode: TapMode, *, recorder_factory=None) -> SubprocessTap | DeviceTap | None`

- [ ] **Step 1: Write the stand-in helper script**

`Tests/Audio/fake_audiotap.py`:
```python
"""Stand-in for the macOS `tldw-audiotap` helper and for `parec`.

Emits `--frames` frames of 640 bytes on stdout, then either holds until
stdin closes (`--hold`) or exits with `--exit-code`.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()

    sys.stderr.write("READY\n")
    sys.stderr.flush()
    frame = bytes([1, 0]) * 320
    out = sys.stdout.buffer
    for _ in range(args.frames):
        out.write(frame)
    out.flush()
    if args.hold:
        sys.stdin.buffer.read()  # block until the parent closes stdin
        return 0
    return args.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write the failing tests**

`Tests/Audio/test_system_audio_tap.py`:
```python
"""Task 6: system-audio tap resolvers, subprocess reader, probe."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tldw_chatbook.Audio import system_audio_tap as sat

pytestmark = pytest.mark.unit

FAKE = Path(__file__).with_name("fake_audiotap.py")


def _fake_cmd(*extra: str) -> tuple[str, ...]:
    return (sys.executable, str(FAKE), *extra)


# ---- pure resolvers -------------------------------------------------------

def test_sink_name_validation():
    assert sat.validate_sink_name("alsa_output.pci-0000_00_1f.3.analog-stereo") == (
        "alsa_output.pci-0000_00_1f.3.analog-stereo"
    )
    for bad in ("", "sink;rm -rf /", "a b", "x$(y)", "über"):
        with pytest.raises(ValueError):
            sat.validate_sink_name(bad)


def test_parse_default_sink_takes_first_line():
    assert sat.parse_default_sink("my_sink\n") == "my_sink"
    with pytest.raises(ValueError):
        sat.parse_default_sink("")


def test_linux_capture_commands():
    assert sat.linux_capture_command("parec", "s1") == (
        "parec", "--device=s1.monitor", "--format=s16le", "--rate=16000",
        "--channels=1", "--latency-msec=20",
    )
    assert sat.linux_capture_command("pw-record", "s1") == (
        "pw-record", "--target", "s1.monitor", "--rate", "16000", "--channels", "1",
        "--format", "s16", "-",
    )
    with pytest.raises(ValueError):
        sat.linux_capture_command("arecord", "s1")


def test_resolve_wasapi_loopback_matches_default_output():
    devices = [
        {"name": "Speakers (Realtek)", "max_input_channels": 0},
        {"name": "Speakers (Realtek) [Loopback]", "max_input_channels": 2},
        {"name": "Headphones [Loopback]", "max_input_channels": 2},
    ]
    assert sat.resolve_wasapi_loopback(devices, "Speakers (Realtek)") == 1
    assert sat.resolve_wasapi_loopback(devices, "Monitor") is None


def test_macos_version_gate():
    assert sat.macos_version_ok("14.2") and sat.macos_version_ok("26.5.2")
    assert not sat.macos_version_ok("14.1.1") and not sat.macos_version_ok("")


# ---- helper lookup --------------------------------------------------------

def test_helper_source_ships_in_package():
    assert sat.helper_source_path().name == "main.swift"


def test_ensure_helper_prefers_bundled_then_dev_then_compiles(tmp_path, monkeypatch):
    bundled_dir = tmp_path / "Contents" / "MacOS"
    bundled_dir.mkdir(parents=True)
    exe = bundled_dir / "python"
    exe.write_text("")
    helper = bundled_dir / "tldw-audiotap"
    helper.write_text("")
    assert sat.ensure_helper(tmp_path / "data", executable=str(exe)) == helper

    data_dir = tmp_path / "data"
    dev = sat.dev_helper_path(data_dir)
    dev.parent.mkdir(parents=True)
    dev.write_text("")
    assert sat.ensure_helper(data_dir, executable=str(tmp_path / "nowhere")) == dev
    dev.unlink()

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        Path(args[args.index("-o") + 1]).write_text("")
        return subprocess.CompletedProcess(args, 0)

    got = sat.ensure_helper(
        data_dir, run=fake_run, which=lambda name: "/usr/bin/swiftc",
        executable=str(tmp_path / "nowhere"),
    )
    assert got == dev and calls[0][0] == "/usr/bin/swiftc" and "-O" in calls[0]
    assert sat.ensure_helper(
        tmp_path / "other", which=lambda name: None, executable=str(tmp_path / "nowhere")
    ) is None


# ---- probe ----------------------------------------------------------------

def test_probe_virtual_device_wins_over_platform():
    mode = sat.probe(system_source="BlackHole 2ch", platform="darwin")
    assert mode.kind == "virtual_device" and mode.device_name == "BlackHole 2ch"


def test_probe_macos_old_version_unavailable():
    mode = sat.probe(platform="darwin", mac_ver="13.6", data_dir=Path("/tmp/x"))
    assert mode.kind == "unavailable" and "14.2" in mode.reason


def test_probe_macos_uses_helper(tmp_path, monkeypatch):
    monkeypatch.setattr(sat, "ensure_helper", lambda data_dir, **kw: tmp_path / "tap")
    mode = sat.probe(platform="darwin", mac_ver="15.0", data_dir=tmp_path)
    assert mode.kind == "native_macos" and mode.command == (str(tmp_path / "tap"),)


def test_probe_linux_parec_then_pw_record_then_unavailable():
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="sink_a\n")

    mode = sat.probe(platform="linux", which=lambda n: "/usr/bin/parec" if n == "parec" else None, run=run)
    assert mode.kind == "native_parec" and mode.command[0] == "parec"
    mode = sat.probe(platform="linux", which=lambda n: "/usr/bin/pw-record" if n == "pw-record" else None, run=run)
    assert mode.kind == "native_parec" and mode.command[0] == "pw-record"
    mode = sat.probe(platform="linux", which=lambda n: None, run=run)
    assert mode.kind == "unavailable"


def test_probe_linux_rejects_hostile_sink_name():
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="bad;name\n")

    mode = sat.probe(platform="linux", which=lambda n: "/usr/bin/parec", run=run)
    assert mode.kind == "unavailable" and "sink" in mode.reason.lower()


def test_probe_windows_loopback():
    devices = [{"name": "Speakers [Loopback]", "max_input_channels": 2}]
    mode = sat.probe(platform="win32", query_devices=lambda: devices, default_output_name="Speakers")
    assert mode.kind == "native_wasapi" and mode.device_index == 0
    mode = sat.probe(platform="win32", query_devices=lambda: [], default_output_name="Speakers")
    assert mode.kind == "unavailable"


# ---- SubprocessTap --------------------------------------------------------

def _collect(tap: sat.SubprocessTap, expected: int, timeout: float = 5.0) -> list[bytes]:
    frames: list[bytes] = []
    done = threading.Event()

    def on_frames(frame: bytes) -> None:
        frames.append(frame)
        if len(frames) >= expected:
            done.set()

    assert tap.start(on_frames) is True
    done.wait(timeout)
    return frames


def test_subprocess_tap_delivers_frames_and_stops_cleanly():
    tap = sat.SubprocessTap(_fake_cmd("--frames", "5", "--hold"))
    frames = _collect(tap, 5)
    assert len(frames) == 5 and all(len(f) == 640 for f in frames)
    assert tap.state == "running"
    tap.stop()
    assert tap.state == "stopped" and tap.exit_code == 0 and tap.restarts == 0


def test_subprocess_tap_restarts_once_then_reports_lost():
    tap = sat.SubprocessTap(
        _fake_cmd("--frames", "3", "--exit-code", "1"),
        restart_delay_s=0.01, sleep=lambda s: None,
    )
    frames = _collect(tap, 6)
    deadline = time.monotonic() + 5
    while tap.state != "lost" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(frames) == 6 and tap.restarts == 1 and tap.state == "lost"
    assert tap.exit_code == 1
    tap.stop()


def test_subprocess_tap_start_false_when_spawn_fails():
    def boom(*args, **kwargs):
        raise FileNotFoundError("no helper")

    tap = sat.SubprocessTap(("missing",), spawn=boom)
    assert tap.start(lambda f: None) is False and tap.state == "lost"


# ---- DeviceTap + build_tap ------------------------------------------------

class _Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = None
        self.callback = None
        self.stopped = 0

    def get_audio_devices(self):
        return [{"id": 3, "name": "BlackHole 2ch"}, {"id": 4, "name": "MacBook Pro Microphone"}]

    def set_device(self, device_id):
        self.device = device_id
        return True

    def start_recording(self, callback=None, save_to_file=None):
        self.callback = callback
        return True

    def stop_recording(self):
        self.stopped += 1


def test_device_tap_selects_device_by_name_and_forwards_frames():
    made: list[_Recorder] = []

    def factory(**kwargs):
        made.append(_Recorder(**kwargs))
        return made[-1]

    tap = sat.DeviceTap("BlackHole 2ch", recorder_factory=factory)
    got: list[bytes] = []
    assert tap.start(got.append) is True
    assert made[0].kwargs == {"use_vad": False, "retain_audio": False, "chunk_size": 320}
    assert made[0].device == 3 and tap.state == "running"
    made[0].callback(b"\x00" * 640)
    assert got == [b"\x00" * 640]
    tap.stop()
    assert made[0].stopped == 1 and tap.state == "stopped"


def test_build_tap_by_kind():
    assert sat.build_tap(sat.TapMode("unavailable", "x")) is None
    assert isinstance(sat.build_tap(sat.TapMode("native_parec", "x", command=("parec",))), sat.SubprocessTap)
    assert isinstance(sat.build_tap(sat.TapMode("native_macos", "x", command=("/h",))), sat.SubprocessTap)
    assert isinstance(
        sat.build_tap(sat.TapMode("virtual_device", "x", device_name="BlackHole"), recorder_factory=_Recorder),
        sat.DeviceTap,
    )
    assert isinstance(
        sat.build_tap(sat.TapMode("native_wasapi", "x", device_index=1), recorder_factory=_Recorder),
        sat.DeviceTap,
    )
```

- [ ] **Step 3: Run to verify they fail**

```bash
.venv/bin/python -m pytest Tests/Audio/test_system_audio_tap.py -q -p no:cacheprovider
```
Expected: `ModuleNotFoundError: tldw_chatbook.Audio.system_audio_tap`.

- [ ] **Step 4: Implement**

`tldw_chatbook/Audio/system_audio_tap.py`:
```python
"""System-audio capture for meetings (spec §3.1, §3.6).

Delivers 20 ms PCM16 mono 16 kHz frames of what the computer is playing.
Native routes are a spawned subprocess writing PCM to stdout (the Swift
helper on macOS, ``parec``/``pw-record`` on Linux) or a WASAPI loopback
input device on Windows; the fallback on any OS is a user-chosen input
device (BlackHole, VB-Cable). Textual-free; sounddevice/pyaudio are only
touched through ``AudioRecordingService``.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

FRAME_BYTES = 640
SINK_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
MACOS_MIN = (14, 2)
HELPER_NAME = "tldw-audiotap"


@dataclass(frozen=True)
class TapMode:
    kind: str
    reason: str
    command: tuple[str, ...] | None = None
    device_name: str | None = None
    device_index: int | None = None


# ---- pure resolvers -------------------------------------------------------

def validate_sink_name(name: str) -> str:
    if not name or not SINK_NAME_RE.match(name):
        raise ValueError(f"unsafe PulseAudio sink name: {name!r}")
    return name


def parse_default_sink(output: str) -> str:
    first = (output or "").strip().splitlines()[:1]
    if not first:
        raise ValueError("pactl returned no default sink")
    return validate_sink_name(first[0].strip())


def linux_capture_command(tool: str, sink: str) -> tuple[str, ...]:
    sink = validate_sink_name(sink)
    if tool == "parec":
        return (
            "parec", f"--device={sink}.monitor", "--format=s16le", "--rate=16000",
            "--channels=1", "--latency-msec=20",
        )
    if tool == "pw-record":
        return (
            "pw-record", "--target", f"{sink}.monitor", "--rate", "16000",
            "--channels", "1", "--format", "s16", "-",
        )
    raise ValueError(f"unsupported capture tool: {tool}")


def resolve_wasapi_loopback(devices: list[dict], default_output_name: str) -> int | None:
    for index, device in enumerate(devices):
        name = str(device.get("name", ""))
        if (
            name.endswith("[Loopback]")
            and name.startswith(default_output_name)
            and int(device.get("max_input_channels", 0) or 0) > 0
        ):
            return index
    return None


def macos_version_ok(mac_ver: str) -> bool:
    try:
        parts = tuple(int(p) for p in (mac_ver or "").split(".")[:2])
    except ValueError:
        return False
    return len(parts) >= 1 and (parts + (0,))[:2] >= MACOS_MIN


# ---- macOS helper lookup --------------------------------------------------

def helper_source_path() -> Path:
    return Path(__file__).with_name("audiotap") / "main.swift"


def bundled_helper_path(executable: str = sys.executable) -> Path | None:
    candidate = Path(executable).resolve().parent / HELPER_NAME
    return candidate if candidate.exists() else None


def dev_helper_path(data_dir: Path) -> Path:
    digest = hashlib.sha256(helper_source_path().read_bytes()).hexdigest()[:12]
    return Path(data_dir) / "bin" / f"{HELPER_NAME}-{digest}"


def ensure_helper(
    data_dir: Path,
    *,
    run: Callable[..., Any] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    executable: str = sys.executable,
) -> Path | None:
    """Return a runnable helper binary, compiling with swiftc if needed."""
    bundled = bundled_helper_path(executable)
    if bundled is not None:
        return bundled
    target = dev_helper_path(data_dir)
    if target.exists():
        return target
    swiftc = which("swiftc")
    if swiftc is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    result = run(
        [swiftc, "-O", "-o", str(target), str(helper_source_path()),
         "-framework", "CoreAudio", "-framework", "AVFoundation"],
        capture_output=True, text=True,
    )
    if getattr(result, "returncode", 1) != 0 or not target.exists():
        logger.warning("audiotap helper compile failed: {}", getattr(result, "stderr", ""))
        return None
    return target


# ---- probe ----------------------------------------------------------------

def probe(
    *,
    system_source: str = "auto",
    platform: str = sys.platform,
    mac_ver: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., Any] = subprocess.run,
    data_dir: Path | None = None,
    query_devices: Callable[[], list] | None = None,
    default_output_name: str | None = None,
) -> TapMode:
    source = (system_source or "auto").strip()
    if source and source.lower() != "auto":
        return TapMode("virtual_device", f"Virtual device: {source}", device_name=source)
    if platform == "darwin":
        if mac_ver is None:
            import platform as _platform

            mac_ver = _platform.mac_ver()[0]
        if not macos_version_ok(mac_ver):
            return TapMode("unavailable", "Native system audio needs macOS 14.2 or newer; pick a virtual device such as BlackHole")
        if data_dir is None:
            from tldw_chatbook.config import get_user_data_dir

            data_dir = get_user_data_dir()
        helper = ensure_helper(data_dir, run=run, which=which)
        if helper is None:
            return TapMode("unavailable", "System audio helper unavailable (no bundled binary and no swiftc); pick a virtual device")
        return TapMode("native_macos", "Native (macOS tap)", command=(str(helper),))
    if platform.startswith("linux"):
        tool = "parec" if which("parec") else ("pw-record" if which("pw-record") else None)
        if tool is None:
            return TapMode("unavailable", "Neither parec nor pw-record found; install pulseaudio-utils or pipewire-pulse")
        try:
            result = run(["pactl", "get-default-sink"], capture_output=True, text=True, timeout=5)
            sink = parse_default_sink(getattr(result, "stdout", ""))
        except Exception as exc:  # noqa: BLE001 - reason goes to the rail
            return TapMode("unavailable", f"Could not resolve the default sink: {exc}")
        return TapMode("native_parec", f"Native ({tool})", command=linux_capture_command(tool, sink))
    if platform == "win32":
        if query_devices is None or default_output_name is None:
            try:
                import sounddevice as sd

                query_devices = query_devices or (lambda: [dict(d) for d in sd.query_devices()])
                default_output_name = default_output_name or str(sd.query_devices(kind="output")["name"])
            except Exception as exc:  # noqa: BLE001
                return TapMode("unavailable", f"sounddevice unavailable: {exc}")
        index = resolve_wasapi_loopback(list(query_devices()), default_output_name)
        if index is None:
            return TapMode("unavailable", "No WASAPI [Loopback] device for the default output; pick a virtual device")
        return TapMode("native_wasapi", "Native (WASAPI loopback)", device_index=index)
    return TapMode("unavailable", f"No native system-audio capture on {platform}")


# ---- taps -----------------------------------------------------------------

class SubprocessTap:
    """Reads fixed-size PCM frames from a helper process's stdout."""

    def __init__(
        self,
        command: tuple[str, ...],
        *,
        frame_bytes: int = FRAME_BYTES,
        restart_delay_s: float = 2.0,
        spawn: Callable[..., Any] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._command = tuple(command)
        self._frame_bytes = frame_bytes
        self._restart_delay_s = restart_delay_s
        self._spawn = spawn
        self._sleep = sleep
        self._proc: Any | None = None
        self._thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stopping = False
        self._on_frames: Optional[Callable[[bytes], None]] = None
        self._stderr_lines: deque = deque(maxlen=5)
        self.state = "stopped"
        self.restarts = 0
        self.exit_code: int | None = None

    @property
    def last_stderr(self) -> str:
        return "\n".join(self._stderr_lines)

    def _launch(self) -> bool:
        try:
            self._proc = self._spawn(
                list(self._command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("system audio helper failed to start: {}", exc)
            self._stderr_lines.append(str(exc))
            self.state = "lost"
            return False
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True, name="audiotap-stderr")
        self._stderr_thread.start()
        return True

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            self._stderr_lines.append(raw.decode("utf-8", "replace").rstrip())

    def start(self, on_frames: Callable[[bytes], None]) -> bool:
        self._on_frames = on_frames
        self._stopping = False
        if not self._launch():
            return False
        self.state = "running"
        self._thread = threading.Thread(target=self._reader, daemon=True, name="audiotap-reader")
        self._thread.start()
        return True

    def _reader(self) -> None:
        while True:
            proc = self._proc
            stdout = proc.stdout if proc is not None else None
            while stdout is not None:
                data = stdout.read(self._frame_bytes)
                if not data:
                    break
                if len(data) == self._frame_bytes and self._on_frames is not None:
                    self._on_frames(data)
            if proc is not None:
                self.exit_code = proc.wait()
            if self._stopping:
                self.state = "stopped"
                return
            if self.restarts == 0:
                self.restarts = 1
                logger.warning("system audio helper exited ({}); restarting once", self.exit_code)
                self._sleep(self._restart_delay_s)
                if not self._stopping and self._launch():
                    continue
            self.state = "lost"
            logger.error("system audio source lost (exit {}): {}", self.exit_code, self.last_stderr)
            return

    def stop(self) -> None:
        self._stopping = True
        proc = self._proc
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=1.0)
            except Exception as exc:  # noqa: BLE001
                logger.debug("audiotap stop: {}", exc)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.state != "lost":
            self.state = "stopped"


class DeviceTap:
    """System audio through an ordinary input device (loopback or virtual cable)."""

    def __init__(self, device_name: str | None, *, device_index: int | None = None, recorder_factory=None) -> None:
        self._device_name = device_name
        self._device_index = device_index
        self._factory = recorder_factory
        self._recorder: Any | None = None
        self.state = "stopped"

    def start(self, on_frames: Callable[[bytes], None]) -> bool:
        factory = self._factory
        if factory is None:
            from .recording_service import AudioRecordingService

            factory = AudioRecordingService
        try:
            self._recorder = factory(use_vad=False, retain_audio=False, chunk_size=320)
            device_id = self._device_index
            if device_id is None and self._device_name:
                for device in self._recorder.get_audio_devices():
                    if str(device.get("name", "")) == self._device_name:
                        device_id = device.get("id", device.get("index"))
                        break
            if device_id is not None:
                self._recorder.set_device(device_id)
            ok = bool(self._recorder.start_recording(callback=on_frames))
        except Exception as exc:  # noqa: BLE001
            logger.error("device tap failed: {}", exc)
            ok = False
        self.state = "running" if ok else "lost"
        return ok

    def stop(self) -> None:
        if self._recorder is not None:
            try:
                self._recorder.stop_recording()
            except Exception as exc:  # noqa: BLE001
                logger.debug("device tap stop: {}", exc)
        self.state = "stopped"


def build_tap(mode: TapMode, *, recorder_factory=None):
    if mode.kind in ("native_macos", "native_parec") and mode.command:
        return SubprocessTap(mode.command)
    if mode.kind == "native_wasapi":
        return DeviceTap(None, device_index=mode.device_index, recorder_factory=recorder_factory)
    if mode.kind == "virtual_device":
        return DeviceTap(mode.device_name, recorder_factory=recorder_factory)
    return None
```
Create the placeholder for the Swift source so `helper_source_path()` and `dev_helper_path()` work before Task 12: `mkdir -p tldw_chatbook/Audio/audiotap` and write `tldw_chatbook/Audio/audiotap/main.swift` containing one line `// Task 12 replaces this file with the Core Audio tap helper.` Also add `tldw_chatbook/Audio/audiotap/__init__.py` (empty) so the directory ships as package data.

- [ ] **Step 5: Run to verify they pass**

```bash
.venv/bin/python -m pytest Tests/Audio/test_system_audio_tap.py -q -p no:cacheprovider
```
Expected: 19 passed. The two subprocess tests spawn `python fake_audiotap.py`; if they hang, confirm `--hold` blocks on `sys.stdin.buffer.read()` and that `stop()` closes stdin.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Audio/system_audio_tap.py tldw_chatbook/Audio/audiotap Tests/Audio/fake_audiotap.py Tests/Audio/test_system_audio_tap.py
git commit -m "feat(audio): SystemAudioTap with native probes, subprocess reader and device fallback (meeting transcription task 6)"
```

---
### Task 7: `MeetingSession`, sink and diarizer seams, `LocalMeetingSink`

Spec §3.3 in full, §4 "Stop", §5 Library handoff (the sink builds the submit kwargs; the owner in Task 8 supplies the callable that marshals onto the UI thread).

**Files:**
- Create: `tldw_chatbook/Audio/meeting_session.py`
- Modify: `tldw_chatbook/Audio/meeting_capture.py` (`stop_recording` becomes idempotent, see Step 4)
- Test: `Tests/Audio/test_meeting_session.py`

**Interfaces:**
- Consumes: `MeetingCapture` surface (Task 5): `mode`, `audio_position_s`, `last_speech_position_s`, `closed_runs_after(t)`, `dominant_source(a, b)`, `stop_recording()`, `pause()`, `resume()`, `fault`.
- Consumes: `LazyLiveDictationService` (Task 3 signature) via `dictation_factory(capture) -> service` where `service.start_dictation(**callbacks) -> bool`, `service.stop_dictation() -> DictationResult` (has `.transcription_complete`), `service.privacy_settings: dict`, class attr `MAX_NON_STREAMING_SEGMENT_SECONDS`.
- Produces:
  - `@dataclass MeetingMeta(folder: Path, mode: str, started_at: str, mic_device: str, system_source: str, provider: str, model: str)` with `to_json() -> dict`
  - `@dataclass MeetingSegment(seq, t_audio_start, t_audio_end, t_wall_start, t_wall_end, label: str | None, text: str)` with `to_json()`
  - `@dataclass MeetingResult(meta, ended_at: str, duration_s: float, segment_count: int, transcription_complete: bool, failed_segments: int, stop_reason: str, recovered: bool = False)` with `to_json()`
  - `@dataclass SpeakerSegment(start_s: float, end_s: float, speaker: str, text: str = "")`
  - `class MeetingSink(Protocol)`: `on_started(meta)`, `on_partial(text, label)`, `on_segment(segment)`, `on_stopped(result)`
  - `class Diarizer(Protocol)`: `diarize(wav_path: Path, start_s: float, end_s: float) -> list[SpeakerSegment]`
  - `write_meeting_json(folder, payload: dict) -> None`, `read_meeting_json(folder) -> dict`, `update_meeting_json(folder, **fields) -> dict`
  - `MEETING_JSON = "meeting.json"`, `TRANSCRIPT_JSONL = "transcript.jsonl"`, `MEETING_SEGMENT_CAP_S = 10.0`
  - `class MeetingSession(*, meta, capture, dictation_factory, sinks, clock=time.time)` with `state: str` (`idle|starting|recording|paused|stopping|stopped|error`), `segments: list[MeetingSegment]`, `failed_segments: int`, `subscribe(listener: Callable[[str, Any], None])`, `unsubscribe(listener)`, `start() -> bool`, `pause()`, `resume()`, `stop(reason: str = "user") -> MeetingResult`, `service` (the built dictation service)
  - listener events: `("state", str)`, `("partial", (text, label))`, `("segment", MeetingSegment)`, `("transcribing", bool)`, `("error", str)`
  - `class LocalMeetingSink(folder: Path, *, submit: Callable[..., str | None], post_transcribe: bool = True, post_diarize: bool = True)` with `last_submit_error: str | None`, `job_id: str | None`
  - `render_markdown(result: MeetingResult, segments: list[MeetingSegment]) -> str`
  - `format_clock(seconds: float) -> str` (`hh:mm:ss`)

- [ ] **Step 1: Write the failing tests**

`Tests/Audio/test_meeting_session.py`:
```python
"""Task 7: MeetingSession segment windows, sinks, Library submit kwargs."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Audio.meeting_capture import SpeechRun
from tldw_chatbook.Audio.meeting_session import (
    LocalMeetingSink,
    MeetingMeta,
    MeetingSession,
    format_clock,
    read_meeting_json,
    render_markdown,
)

pytestmark = pytest.mark.unit


class FakeCapture:
    def __init__(self, mode="call"):
        self.mode = mode
        self.audio_position_s = 0.0
        self.last_speech_position_s = 0.0
        self.runs: list[SpeechRun] = []
        self.labels: dict[tuple[float, float], str] = {}
        self.default_label = "you"
        self.stops = 0
        self.paused = False
        self.fault = None

    def closed_runs_after(self, t):
        return [r for r in self.runs if r.end_s is not None and r.end_s > t]

    def dominant_source(self, a, b):
        return self.labels.get((round(a, 2), round(b, 2)), self.default_label)

    def stop_recording(self):
        self.stops += 1

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class FakeDictation:
    MAX_NON_STREAMING_SEGMENT_SECONDS = 30.0

    def __init__(self, capture):
        self.capture = capture
        self.privacy_settings = {"auto_clear_buffer": False, "local_only": True}
        self.callbacks: dict[str, Any] = {}
        self.stopped = 0
        self.complete = True

    def start_dictation(self, **callbacks):
        self.callbacks = callbacks
        return True

    def stop_dictation(self):
        self.stopped += 1
        return SimpleNamespace(transcription_complete=self.complete)


class RecordingSink:
    def __init__(self):
        self.calls: list[tuple[str, Any]] = []

    def on_started(self, meta):
        self.calls.append(("started", meta))

    def on_partial(self, text, label):
        self.calls.append(("partial", (text, label)))

    def on_segment(self, segment):
        self.calls.append(("segment", segment))

    def on_stopped(self, result):
        self.calls.append(("stopped", result))


def _meta(tmp_path, mode="call") -> MeetingMeta:
    return MeetingMeta(
        folder=tmp_path, mode=mode, started_at="2026-09-04T14:30:00",
        mic_device="MacBook Pro Microphone", system_source="Native (macOS tap)",
        provider="faster-whisper", model="base.en",
    )


def _session(tmp_path, mode="call", sinks=None):
    capture = FakeCapture(mode)
    built: list[FakeDictation] = []

    def factory(cap):
        built.append(FakeDictation(cap))
        return built[-1]

    ticks = iter(range(1000, 2000))
    session = MeetingSession(
        meta=_meta(tmp_path, mode), capture=capture, dictation_factory=factory,
        sinks=sinks or [], clock=lambda: float(next(ticks)),
    )
    return session, capture, built


def test_start_configures_service_and_writes_meeting_json(tmp_path):
    sink = RecordingSink()
    session, capture, built = _session(tmp_path, sinks=[sink])
    assert session.start() is True
    service = built[0]
    assert service.capture is capture
    assert service.privacy_settings["auto_clear_buffer"] is True
    assert service.MAX_NON_STREAMING_SEGMENT_SECONDS == 10.0
    assert "on_command" not in service.callbacks
    assert set(service.callbacks) == {
        "on_partial_transcript", "on_final_transcript", "on_state_change", "on_error",
        "on_segment_transcribing", "on_speech_resumed", "on_segment_no_final",
    }
    assert session.state == "recording"
    assert sink.calls[0][0] == "started"
    payload = read_meeting_json(tmp_path)
    assert payload["mode"] == "call" and payload["started_at"] == "2026-09-04T14:30:00"
    assert payload["schema"] == 1


def test_final_uses_contiguous_window_and_closed_run_end(tmp_path):
    sink = RecordingSink()
    session, capture, built = _session(tmp_path, sinks=[sink])
    session.start()
    cb = built[0].callbacks
    capture.runs = [SpeechRun(0.5, 3.0)]
    capture.audio_position_s = 5.0
    capture.last_speech_position_s = 4.8      # next speaker already talking
    capture.labels[(0.0, 3.0)] = "others"
    cb["on_final_transcript"]("hello there")
    seg = session.segments[0]
    assert (seg.t_audio_start, seg.t_audio_end) == (0.0, 3.0)
    assert seg.label == "others" and seg.text == "hello there" and seg.seq == 0
    assert seg.t_wall_end - seg.t_wall_start == pytest.approx(3.0)
    assert sink.calls[-1] == ("segment", seg)


def test_final_without_closed_run_uses_last_speech_position(tmp_path):
    session, capture, built = _session(tmp_path)
    session.start()
    cb = built[0].callbacks
    capture.last_speech_position_s = 2.2
    capture.audio_position_s = 2.5
    cb["on_final_transcript"]("first")
    capture.runs = [SpeechRun(2.3, 6.0)]
    capture.last_speech_position_s = 6.0
    cb["on_final_transcript"]("second")
    assert [(s.t_audio_start, s.t_audio_end) for s in session.segments] == [(0.0, 2.2), (2.2, 6.0)]


def test_one_final_spanning_two_runs_and_a_cap_split(tmp_path):
    session, capture, built = _session(tmp_path)
    session.start()
    cb = built[0].callbacks
    capture.runs = [SpeechRun(0.0, 1.0), SpeechRun(1.5, 4.0)]
    capture.last_speech_position_s = 4.0
    cb["on_final_transcript"]("spans two runs")
    assert session.segments[0].t_audio_end == 4.0
    # 10 s cap split: a second final arrives while the run is still open
    capture.last_speech_position_s = 12.0
    cb["on_final_transcript"]("cap split part")
    assert (session.segments[1].t_audio_start, session.segments[1].t_audio_end) == (4.0, 12.0)


def test_room_mode_has_no_labels(tmp_path):
    session, capture, built = _session(tmp_path, mode="room")
    session.start()
    capture.last_speech_position_s = 1.0
    built[0].callbacks["on_final_transcript"]("hi")
    built[0].callbacks["on_partial_transcript"]("h")
    assert session.segments[0].label is None


def test_partial_and_transcribing_and_error_events_reach_listeners(tmp_path):
    session, capture, built = _session(tmp_path)
    events: list[tuple[str, Any]] = []
    session.subscribe(lambda kind, payload: events.append((kind, payload)))
    session.start()
    cb = built[0].callbacks
    capture.audio_position_s = 3.0
    capture.labels[(2.0, 3.0)] = "others"
    cb["on_partial_transcript"]("par")
    cb["on_segment_transcribing"](False)
    cb["on_segment_transcribing"](True)
    cb["on_error"](RuntimeError("boom"))
    assert ("partial", ("par", "others")) in events
    assert ("transcribing", True) in events and ("transcribing", False) in events
    assert ("error", "boom") in events and session.failed_segments == 1


def test_blank_final_is_ignored(tmp_path):
    session, capture, built = _session(tmp_path)
    session.start()
    built[0].callbacks["on_final_transcript"]("   ")
    assert session.segments == []


def test_pause_resume_forward_and_change_state(tmp_path):
    session, capture, _ = _session(tmp_path)
    session.start()
    session.pause()
    assert capture.paused and session.state == "paused"
    session.resume()
    assert not capture.paused and session.state == "recording"


def test_stop_returns_result_and_finalises_files(tmp_path):
    sink = RecordingSink()
    session, capture, built = _session(tmp_path, sinks=[sink])
    session.start()
    capture.last_speech_position_s = 2.0
    built[0].callbacks["on_final_transcript"]("one")
    capture.audio_position_s = 7.5
    built[0].complete = False
    result = session.stop(reason="user")
    assert built[0].stopped == 1 and capture.stops == 1
    assert result.segment_count == 1 and result.duration_s == 7.5
    assert result.transcription_complete is False and result.stop_reason == "user"
    assert session.state == "stopped" and sink.calls[-1][0] == "stopped"
    payload = read_meeting_json(tmp_path)
    assert payload["ended_at"] and payload["segment_count"] == 1 and payload["stop_reason"] == "user"


def test_stop_twice_is_a_no_op(tmp_path):
    session, capture, built = _session(tmp_path)
    session.start()
    first = session.stop()
    second = session.stop()
    assert first is second and capture.stops == 1


def test_start_failure_sets_error_state(tmp_path):
    session, capture, built = _session(tmp_path)
    FakeDictation.start_dictation = lambda self, **cb: False  # type: ignore[assignment]
    try:
        assert session.start() is False and session.state == "error"
    finally:
        del FakeDictation.start_dictation


# ---- LocalMeetingSink ----------------------------------------------------

def _run_meeting(tmp_path, sink, mode="call"):
    session, capture, built = _session(tmp_path, mode=mode, sinks=[sink])
    session.start()
    capture.runs = [SpeechRun(0.0, 2.0)]
    capture.last_speech_position_s = 2.0
    capture.labels[(0.0, 2.0)] = "you"
    built[0].callbacks["on_final_transcript"]("hello")
    capture.runs.append(SpeechRun(2.5, 4.0))
    capture.last_speech_position_s = 4.0
    capture.labels[(2.0, 4.0)] = "others"
    built[0].callbacks["on_final_transcript"]("hi back")
    capture.audio_position_s = 4.0
    return session.stop()


def test_local_sink_writes_jsonl_and_submits_audio_with_diarization(tmp_path):
    calls: list[dict] = []

    def submit(**kwargs):
        calls.append(kwargs)
        return "ingest-job-7"

    sink = LocalMeetingSink(tmp_path, submit=submit, post_transcribe=True, post_diarize=True)
    _run_meeting(tmp_path, sink)
    lines = [json.loads(l) for l in (tmp_path / "transcript.jsonl").read_text().splitlines()]
    assert [l["label"] for l in lines] == ["you", "others"]
    assert lines[0] == {
        "seq": 0, "t_audio_start": 0.0, "t_audio_end": 2.0,
        "t_wall_start": lines[0]["t_wall_start"], "t_wall_end": lines[0]["t_wall_end"],
        "label": "you", "text": "hello",
    }
    assert calls == [{
        "source_path": str(tmp_path / "mixed.wav"),
        "title": "Meeting 2026-09-04 14:30",
        "keywords": ("meeting",),
        "detected_type": "audio",
        "ingest_options": {"diarization": True},
    }]
    assert sink.job_id == "ingest-job-7"
    assert read_meeting_json(tmp_path)["ingest_job_id"] == "ingest-job-7"


def test_local_sink_without_post_transcribe_submits_markdown(tmp_path):
    calls: list[dict] = []
    sink = LocalMeetingSink(tmp_path, submit=lambda **kw: calls.append(kw) or "j1", post_transcribe=False)
    _run_meeting(tmp_path, sink)
    md = (tmp_path / "transcript.md").read_text()
    assert "# Meeting 2026-09-04 14:30" in md and "mixed.wav" in md
    assert "[00:00:00] **You:** hello" in md and "[00:00:02] **Others:** hi back" in md
    assert calls[0]["source_path"] == str(tmp_path / "transcript.md")
    assert calls[0]["detected_type"] == "document" and calls[0]["ingest_options"] == {}


def test_local_sink_records_submit_failure(tmp_path):
    def submit(**kwargs):
        raise RuntimeError("registry refused")

    sink = LocalMeetingSink(tmp_path, submit=submit)
    _run_meeting(tmp_path, sink)
    assert sink.job_id is None and "registry refused" in sink.last_submit_error
    assert read_meeting_json(tmp_path)["ingest_error"] == "registry refused"


def test_render_markdown_room_mode_omits_labels(tmp_path):
    sink = LocalMeetingSink(tmp_path, submit=lambda **kw: None, post_transcribe=False)
    _run_meeting(tmp_path, sink, mode="room")
    md = (tmp_path / "transcript.md").read_text()
    assert "[00:00:00] hello" in md and "**You:**" not in md


def test_format_clock():
    assert format_clock(0) == "00:00:00" and format_clock(3725.9) == "01:02:05"
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest Tests/Audio/test_meeting_session.py -q -p no:cacheprovider
```
Expected: `ModuleNotFoundError: tldw_chatbook.Audio.meeting_session`.

- [ ] **Step 3: Implement**

`tldw_chatbook/Audio/meeting_session.py`:
```python
"""One meeting: dictation callbacks -> labelled segments -> sinks (spec §3.3).

Textual-free. The session owns no devices (the capture does) and no app
objects (the owner does); it knows the capture surface, the dictation
service surface, and a list of sinks.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from loguru import logger

MEETING_JSON = "meeting.json"
TRANSCRIPT_JSONL = "transcript.jsonl"
MEETING_SEGMENT_CAP_S = 10.0
PARTIAL_LABEL_WINDOW_S = 1.0


@dataclass
class MeetingMeta:
    folder: Path
    mode: str
    started_at: str
    mic_device: str
    system_source: str
    provider: str
    model: str

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["folder"] = str(self.folder)
        return payload


@dataclass
class MeetingSegment:
    seq: int
    t_audio_start: float
    t_audio_end: float
    t_wall_start: float
    t_wall_end: float
    label: str | None
    text: str

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class MeetingResult:
    meta: MeetingMeta
    ended_at: str
    duration_s: float
    segment_count: int
    transcription_complete: bool
    failed_segments: int
    stop_reason: str
    recovered: bool = False

    def to_json(self) -> dict:
        payload = self.meta.to_json()
        payload.update(
            ended_at=self.ended_at, duration_s=self.duration_s, segment_count=self.segment_count,
            transcription_complete=self.transcription_complete, failed_segments=self.failed_segments,
            stop_reason=self.stop_reason, recovered=self.recovered,
        )
        return payload


@dataclass
class SpeakerSegment:
    start_s: float
    end_s: float
    speaker: str
    text: str = ""


class MeetingSink(Protocol):
    def on_started(self, meta: MeetingMeta) -> None: ...
    def on_partial(self, text: str, label: str | None) -> None: ...
    def on_segment(self, segment: MeetingSegment) -> None: ...
    def on_stopped(self, result: MeetingResult) -> None: ...


class Diarizer(Protocol):
    """Phase-2 seam: MOSS or the server plugs in here (spec §3.3)."""

    def diarize(self, wav_path: Path, start_s: float, end_s: float) -> list[SpeakerSegment]: ...


def write_meeting_json(folder: Path, payload: dict) -> None:
    (Path(folder) / MEETING_JSON).write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_meeting_json(folder: Path) -> dict:
    path = Path(folder) / MEETING_JSON
    return json.loads(path.read_text()) if path.exists() else {}


def update_meeting_json(folder: Path, **fields: Any) -> dict:
    payload = read_meeting_json(folder)
    payload.update(fields)
    write_meeting_json(folder, payload)
    return payload


def format_clock(seconds: float) -> str:
    total = int(max(0.0, seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


class MeetingSession:
    """Turns one dictation service's callbacks into labelled segments."""

    def __init__(
        self,
        *,
        meta: MeetingMeta,
        capture: Any,
        dictation_factory: Callable[[Any], Any],
        sinks: Sequence[MeetingSink],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.meta = meta
        self.capture = capture
        self._dictation_factory = dictation_factory
        self._sinks = list(sinks)
        self._clock = clock
        self.service: Any | None = None
        self.state = "idle"
        self.segments: list[MeetingSegment] = []
        self.failed_segments = 0
        self._listeners: list[Callable[[str, Any], None]] = []
        self._lock = threading.RLock()
        self._last_end_s = 0.0
        self._result: MeetingResult | None = None

    # ---- listeners --------------------------------------------------------
    def subscribe(self, listener: Callable[[str, Any], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[str, Any], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _emit(self, kind: str, payload: Any) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(kind, payload)
            except Exception as exc:  # noqa: BLE001
                logger.error("meeting listener error: {}", exc)

    def _set_state(self, state: str) -> None:
        self.state = state
        self._emit("state", state)

    def _each_sink(self, method: str, *args: Any) -> None:
        with self._lock:
            for sink in self._sinks:
                try:
                    getattr(sink, method)(*args)
                except Exception as exc:  # noqa: BLE001
                    logger.error("meeting sink {} failed: {}", method, exc)

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> bool:
        self._set_state("starting")
        Path(self.meta.folder).mkdir(parents=True, exist_ok=True)
        payload = self.meta.to_json()
        payload.update(schema=1, ended_at=None, segment_count=0, recovered=False)
        write_meeting_json(self.meta.folder, payload)
        service = self._dictation_factory(self.capture)
        service.privacy_settings["auto_clear_buffer"] = True
        service.MAX_NON_STREAMING_SEGMENT_SECONDS = MEETING_SEGMENT_CAP_S
        self.service = service
        ok = bool(
            service.start_dictation(
                on_partial_transcript=self._on_partial,
                on_final_transcript=self._on_final,
                on_state_change=self._on_service_state,
                on_error=self._on_error,
                on_segment_transcribing=self._on_transcribing,
                on_speech_resumed=self._on_speech_resumed,
                on_segment_no_final=self._on_no_final,
            )
        )
        if not ok:
            self._set_state("error")
            return False
        self._set_state("recording")
        self._each_sink("on_started", self.meta)
        return True

    def pause(self) -> None:
        self.capture.pause()
        self._set_state("paused")

    def resume(self) -> None:
        self.capture.resume()
        self._set_state("recording")

    def stop(self, reason: str = "user") -> MeetingResult:
        with self._lock:
            if self._result is not None:
                return self._result
            self._set_state("stopping")
        complete = True
        if self.service is not None:
            try:
                outcome = self.service.stop_dictation()
                complete = bool(getattr(outcome, "transcription_complete", True))
            except Exception as exc:  # noqa: BLE001
                logger.error("stop_dictation failed: {}", exc)
                complete = False
        try:
            self.capture.stop_recording()
        except Exception as exc:  # noqa: BLE001
            logger.error("capture stop failed: {}", exc)
        result = MeetingResult(
            meta=self.meta,
            ended_at=datetime.now().isoformat(timespec="seconds"),
            duration_s=float(self.capture.audio_position_s),
            segment_count=len(self.segments),
            transcription_complete=complete,
            failed_segments=self.failed_segments,
            stop_reason=reason,
        )
        payload = read_meeting_json(self.meta.folder)
        payload.update(result.to_json())
        payload.setdefault("schema", 1)
        write_meeting_json(self.meta.folder, payload)
        with self._lock:
            self._result = result
        self._each_sink("on_stopped", result)
        self._set_state("stopped")
        return result

    # ---- dictation callbacks (capture / processing threads) ---------------
    def _label(self, start_s: float, end_s: float) -> str | None:
        if self.capture.mode != "call":
            return None
        return self.capture.dominant_source(start_s, end_s)

    def _on_partial(self, text: str) -> None:
        end = float(self.capture.audio_position_s)
        label = self._label(max(0.0, end - PARTIAL_LABEL_WINDOW_S), end)
        self._emit("partial", (text, label))
        self._each_sink("on_partial", text, label)

    def _on_final(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            start = self._last_end_s
            closed = self.capture.closed_runs_after(start)
            end = closed[-1].end_s if closed else float(self.capture.last_speech_position_s)
            if end <= start:
                end = float(self.capture.audio_position_s)
            wall_end = float(self._clock())
            segment = MeetingSegment(
                seq=len(self.segments),
                t_audio_start=start,
                t_audio_end=end,
                t_wall_start=wall_end - (end - start),
                t_wall_end=wall_end,
                label=self._label(start, end),
                text=text,
            )
            self.segments.append(segment)
            self._last_end_s = end
        self._emit("segment", segment)
        self._each_sink("on_segment", segment)

    def _on_service_state(self, state: str) -> None:
        self._emit("service_state", state)

    def _on_error(self, exc: Exception) -> None:
        self.failed_segments += 1
        self._emit("error", str(exc))

    def _on_transcribing(self, done: bool) -> None:
        self._emit("transcribing", not done)

    def _on_speech_resumed(self) -> None:
        self._emit("speech", True)

    def _on_no_final(self) -> None:
        self._emit("transcribing", False)


def render_markdown(result: MeetingResult, segments: list[MeetingSegment]) -> str:
    meta = result.meta
    started = datetime.fromisoformat(meta.started_at)
    lines = [
        f"# Meeting {started:%Y-%m-%d %H:%M}",
        "",
        f"- Audio: `{Path(meta.folder) / 'mixed.wav'}`",
        f"- Mode: {meta.mode}",
        f"- Duration: {format_clock(result.duration_s)}",
        f"- Transcriber: {meta.provider} {meta.model}".rstrip(),
        "",
    ]
    names = {"you": "You", "others": "Others", "both": "You + Others"}
    for segment in segments:
        stamp = f"[{format_clock(segment.t_audio_start)}]"
        if segment.label:
            lines.append(f"{stamp} **{names.get(segment.label, segment.label)}:** {segment.text}")
        else:
            lines.append(f"{stamp} {segment.text}")
    return "\n".join(lines) + "\n"


class LocalMeetingSink:
    """JSONL transcript + Library ingest submit on stop (spec §5)."""

    def __init__(
        self,
        folder: Path,
        *,
        submit: Callable[..., Optional[str]],
        post_transcribe: bool = True,
        post_diarize: bool = True,
    ) -> None:
        self.folder = Path(folder)
        self._submit = submit
        self.post_transcribe = post_transcribe
        self.post_diarize = post_diarize
        self._handle = None
        self._segments: list[MeetingSegment] = []
        self.job_id: str | None = None
        self.last_submit_error: str | None = None

    def on_started(self, meta: MeetingMeta) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.folder / TRANSCRIPT_JSONL, "a", encoding="utf-8")  # noqa: SIM115

    def on_partial(self, text: str, label: str | None) -> None:
        return None

    def on_segment(self, segment: MeetingSegment) -> None:
        self._segments.append(segment)
        if self._handle is not None:
            self._handle.write(json.dumps(segment.to_json()) + "\n")
            self._handle.flush()

    def on_stopped(self, result: MeetingResult) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        started = datetime.fromisoformat(result.meta.started_at)
        title = f"Meeting {started:%Y-%m-%d %H:%M}"
        if self.post_transcribe:
            kwargs = dict(
                source_path=str(self.folder / "mixed.wav"), title=title, keywords=("meeting",),
                detected_type="audio", ingest_options={"diarization": bool(self.post_diarize)},
            )
        else:
            md_path = self.folder / "transcript.md"
            md_path.write_text(render_markdown(result, self._segments), encoding="utf-8")
            kwargs = dict(
                source_path=str(md_path), title=title, keywords=("meeting",),
                detected_type="document", ingest_options={},
            )
        try:
            self.job_id = self._submit(**kwargs)
            update_meeting_json(self.folder, ingest_job_id=self.job_id)
        except Exception as exc:  # noqa: BLE001 - the footer reports it (spec §7)
            self.last_submit_error = str(exc)
            update_meeting_json(self.folder, ingest_error=str(exc))
            logger.error("meeting ingest submit failed: {}", exc)
```

- [ ] **Step 4: Make `MeetingCapture.stop_recording` idempotent**

`LazyLiveDictationService.stop_dictation` stops the recorder it holds (the capture) and the session stops it again; the second call must be a no-op. In `meeting_capture.py` add `self._stopped = False` to `__init__` and at the top of `stop_recording`:
```python
        if self._stopped:
            return None
        self._stopped = True
```

- [ ] **Step 5: Run to verify they pass**

```bash
.venv/bin/python -m pytest Tests/Audio/test_meeting_session.py Tests/Audio/test_meeting_capture.py -q -p no:cacheprovider
```
Expected: all pass (17 new).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Audio/meeting_session.py tldw_chatbook/Audio/meeting_capture.py Tests/Audio/test_meeting_session.py
git commit -m "feat(audio): MeetingSession with contiguous segment windows, sink/diarizer seams, local sink (meeting transcription task 7)"
```

---
### Task 8: `MeetingSessionOwner` — prepare, start/stop, watchdog, shutdown, recovery

Spec §3.4 in full, §7 (watchdog, crash recovery, raw-track cleanup), §5 (`keep_raw_tracks`).

**Files:**
- Create: `tldw_chatbook/Audio/meeting_owner.py`
- Test: `Tests/Audio/test_meeting_owner.py`

**Interfaces:**
- Consumes: `MeetingCapture`, `PlaceholderWavWriter`, `wav_needs_patch`, `patch_wav_header`, `probe`/`build_tap`, `MeetingSession`, `LocalMeetingSink`, `MeetingMeta`, `read_meeting_json`/`update_meeting_json`; `LazyLiveDictationService(recorder_factory=..., transcription_service_factory=..., enable_commands=False, ...)`; `TranscriptionService(local_stt_dispatcher=None)`; `console_voice_input.resolve() -> EffectiveConfig | None` (fields `provider`, `model`, `language`).
- Produces (all in `meeting_owner.py`):
  - `@dataclass MeetingSettings(provider: str = "auto", model: str = "", system_source: str = "auto", mic_device: str = "", recordings_dir: Path | None = None, keep_raw_tracks: bool = True, post_transcribe: bool = True, post_diarize: bool = True)` and `MeetingSettings.from_config(get_setting: Callable[[str, str, Any], Any], data_dir: Path) -> MeetingSettings`
  - `@dataclass PrepareResult(tap_mode: TapMode, provider: str, model: str, diarization_available: bool, diarization_missing: tuple[str, ...], recoverable: tuple[Path, ...])`
  - `diarization_requirements(find_spec=importlib.util.find_spec) -> tuple[str, ...]` (names of missing modules among `torch`, `torchaudio`, `speechbrain`, `sklearn`)
  - `scan_recoverable(meetings_dir: Path) -> list[Path]`, `recover_folder(folder: Path) -> dict`
  - `class MeetingSessionOwner(*, settings: MeetingSettings, call_from_thread: Callable[..., Any], submit_ingest: Callable[..., str | None], job_state: Callable[[str], str | None] = lambda job_id: None, facade_factory=None, dictation_factory=None, tap_probe=probe, tap_builder=build_tap, mic_recorder_factory=None, vad_factory=None, clock=time.monotonic, watchdog_interval_s: float = 1.0, stall_after_s: float = 3.0, sleep=time.sleep)` with `prepare() -> PrepareResult`, `prepared: PrepareResult | None`, `start() -> MeetingSession`, `pause()`, `resume()`, `stop(reason="user") -> MeetingResult | None`, `is_active: bool`, `session: MeetingSession | None`, `last_result: MeetingResult | None`, `local_sink: LocalMeetingSink | None`, `shutdown() -> None`, `cleanup_raw_tracks_if_done() -> bool`
  - `MEETINGS_DIRNAME = "meetings"`

- [ ] **Step 1: Write the failing tests**

`Tests/Audio/test_meeting_owner.py`:
```python
"""Task 8: app-owned session owner (watchdog, shutdown, recovery, cleanup)."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Audio import meeting_owner as mo
from tldw_chatbook.Audio.system_audio_tap import TapMode
from tldw_chatbook.Audio.wav_writer import PlaceholderWavWriter, wav_needs_patch

pytestmark = pytest.mark.unit


class FakeRecorder:
    def __init__(self, **kwargs):
        self.callback = None

    def start_recording(self, callback=None, save_to_file=None):
        self.callback = callback
        return True

    def stop_recording(self):
        return None

    def get_audio_devices(self):
        return []

    def set_device(self, device_id):
        return True


class FakeDictation:
    MAX_NON_STREAMING_SEGMENT_SECONDS = 30.0

    def __init__(self, capture):
        self.capture = capture
        self.privacy_settings = {"auto_clear_buffer": False}
        self.callbacks = {}

    def start_dictation(self, **callbacks):
        self.callbacks = callbacks
        return True

    def stop_dictation(self):
        return SimpleNamespace(transcription_complete=True)


class EnergyVad:
    def is_speech(self, frame, rate):
        return False


def _settings(tmp_path, **over) -> mo.MeetingSettings:
    base = dict(recordings_dir=tmp_path / "meetings", system_source="auto")
    base.update(over)
    return mo.MeetingSettings(**base)


def _owner(tmp_path, *, tap_kind="unavailable", job_state=None, **over):
    marshalled: list[tuple] = []
    submitted: list[dict] = []

    def call_from_thread(fn, *args, **kwargs):
        marshalled.append((fn, args, kwargs))
        return fn(*args, **kwargs)

    def submit_ingest(**kwargs):
        submitted.append(kwargs)
        return "ingest-job-1"

    owner = mo.MeetingSessionOwner(
        settings=_settings(tmp_path, **over),
        call_from_thread=call_from_thread,
        submit_ingest=submit_ingest,
        job_state=job_state or (lambda job_id: None),
        facade_factory=lambda: SimpleNamespace(name="facade"),
        dictation_factory=lambda capture, facade, cfg: FakeDictation(capture),
        tap_probe=lambda **kw: TapMode(tap_kind, "reason", command=("x",)),
        tap_builder=lambda mode, **kw: None,
        mic_recorder_factory=FakeRecorder,
        vad_factory=EnergyVad,
        watchdog_interval_s=0.01,
        stall_after_s=0.05,
    )
    return owner, marshalled, submitted


def test_settings_from_config_reads_flat_meetings_section(tmp_path):
    values = {"provider": "parakeet-mlx", "keep_raw_tracks": False, "recordings_dir": str(tmp_path / "rec")}

    def get(section, key, default):
        assert section == "meetings"
        return values.get(key, default)

    settings = mo.MeetingSettings.from_config(get, data_dir=tmp_path)
    assert settings.provider == "parakeet-mlx" and settings.keep_raw_tracks is False
    assert settings.recordings_dir == (tmp_path / "rec").resolve()
    default = mo.MeetingSettings.from_config(lambda s, k, d: d, data_dir=tmp_path)
    assert default.recordings_dir == (tmp_path / "meetings").resolve()


def test_diarization_requirements_uses_find_spec_not_imports():
    missing = mo.diarization_requirements(find_spec=lambda name: None if name in ("torch", "speechbrain") else object())
    assert missing == ("torch", "speechbrain")


def test_prepare_reports_tap_provider_and_diarization(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="faster-whisper", model="base.en", language="en"))
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ("torch",))
    owner, _, _ = _owner(tmp_path, tap_kind="native_macos")
    prepared = owner.prepare()
    assert prepared.tap_mode.kind == "native_macos"
    assert prepared.provider == "faster-whisper" and prepared.model == "base.en"
    assert prepared.diarization_available is False and prepared.diarization_missing == ("torch",)
    assert owner.prepared is prepared and owner._facade.name == "facade"


def test_start_creates_folder_writers_and_session_in_room_mode_when_tap_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, _, _ = _owner(tmp_path)
    owner.prepare()
    session = owner.start()
    assert owner.is_active and session.state == "recording"
    folder = session.meta.folder
    assert folder.parent == (tmp_path / "meetings").resolve()
    assert (folder / "mixed.wav").exists() and not (folder / "you.wav").exists()
    assert session.meta.mode == "room" and session.meta.provider == "p"
    owner.stop()
    assert not owner.is_active


def test_start_call_mode_has_three_writers(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))

    class Tap:
        state = "stopped"

        def start(self, on_frames):
            self.state = "running"
            return True

        def stop(self):
            self.state = "stopped"

    owner, _, _ = _owner(tmp_path, tap_kind="native_macos")
    owner._tap_builder = lambda mode, **kw: Tap()
    owner.prepare()
    session = owner.start()
    folder = session.meta.folder
    assert {p.name for p in folder.glob("*.wav")} == {"mixed.wav", "you.wav", "others.wav"}
    assert session.meta.mode == "call"
    owner.stop()


def test_stop_submits_through_call_from_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, marshalled, submitted = _owner(tmp_path)
    owner.prepare()
    owner.start()
    result = owner.stop()
    assert result is owner.last_result and result.stop_reason == "user"
    assert submitted[0]["detected_type"] == "audio" and submitted[0]["ingest_options"] == {"diarization": True}
    assert marshalled and marshalled[0][0] is owner._submit_ingest
    assert owner.local_sink.job_id == "ingest-job-1"


def test_watchdog_stops_on_fault(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, _, _ = _owner(tmp_path)
    owner.prepare()
    session = owner.start()
    session.capture.fault = OSError("disk full")
    deadline = time.monotonic() + 2
    while owner.is_active and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not owner.is_active and owner.last_result.stop_reason == "disk_error"


def test_watchdog_stops_on_stalled_clock_but_not_while_paused(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, _, _ = _owner(tmp_path)
    owner.prepare()
    session = owner.start()
    session.pause()
    time.sleep(0.15)
    assert owner.is_active            # paused: no stall verdict
    session.resume()
    deadline = time.monotonic() + 2   # no mic frames ever arrive -> stall
    while owner.is_active and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not owner.is_active and owner.last_result.stop_reason == "mic_lost"


def test_shutdown_finalises_files_without_submitting(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, _, submitted = _owner(tmp_path)
    owner.prepare()
    session = owner.start()
    owner.shutdown()
    assert not owner.is_active and submitted == []
    assert not wav_needs_patch(session.meta.folder / "mixed.wav")
    payload = json.loads((session.meta.folder / "meeting.json").read_text())
    assert payload["stop_reason"] == "shutdown" and payload["ended_at"]


def test_scan_and_recover_unfinished_folder(tmp_path):
    folder = tmp_path / "meetings" / "2026-09-04_1000"
    folder.mkdir(parents=True)
    writer = PlaceholderWavWriter(folder / "mixed.wav")
    writer.write(b"\x00\x00" * 320 * 50)   # 1 s
    writer._handle.flush()                  # crash: never closed
    (folder / "meeting.json").write_text(json.dumps({"schema": 1, "started_at": "2026-09-04T10:00:00", "ended_at": None, "mode": "room"}))
    assert mo.scan_recoverable(tmp_path / "meetings") == [folder]
    payload = mo.recover_folder(folder)
    assert payload["recovered"] is True and payload["duration_s"] == pytest.approx(1.0)
    assert payload["ended_at"] and not wav_needs_patch(folder / "mixed.wav")
    assert mo.scan_recoverable(tmp_path / "meetings") == []


def test_cleanup_raw_tracks_only_when_job_done(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    states = {"ingest-job-1": "parsing"}

    class Tap:
        state = "stopped"

        def start(self, on_frames):
            return True

        def stop(self):
            return None

    owner, _, _ = _owner(tmp_path, tap_kind="native_macos", job_state=lambda j: states.get(j), keep_raw_tracks=False)
    owner._tap_builder = lambda mode, **kw: Tap()
    owner.prepare()
    session = owner.start()
    folder = session.meta.folder
    owner.stop()
    assert owner.cleanup_raw_tracks_if_done() is False and (folder / "you.wav").exists()
    states["ingest-job-1"] = "done"
    assert owner.cleanup_raw_tracks_if_done() is True
    assert not (folder / "you.wav").exists() and not (folder / "others.wav").exists()
    assert (folder / "mixed.wav").exists()
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest Tests/Audio/test_meeting_owner.py -q -p no:cacheprovider
```
Expected: `ModuleNotFoundError: tldw_chatbook.Audio.meeting_owner`.

- [ ] **Step 3: Implement**

`tldw_chatbook/Audio/meeting_owner.py`:
```python
"""App-owned meeting session lifecycle (spec §3.4, §7).

Screens are never cached across tab switches, so the running session
lives here. Textual-free: the app hands in `call_from_thread` and the
ingest submit callable; everything else is injectable for tests.
"""
from __future__ import annotations

import importlib.util
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from .meeting_capture import MeetingCapture
from .meeting_session import (
    LocalMeetingSink,
    MeetingMeta,
    MeetingResult,
    MeetingSession,
    read_meeting_json,
    update_meeting_json,
)
from .system_audio_tap import TapMode, build_tap, probe
from .wav_writer import PlaceholderWavWriter, patch_wav_header, wav_needs_patch

MEETINGS_DIRNAME = "meetings"
DIARIZATION_MODULES = ("torch", "torchaudio", "speechbrain", "sklearn")


def resolve_effective_config():
    """Late import: `console_voice_input` pulls config; keep this module light."""
    from tldw_chatbook.Chat.console_voice_input import resolve

    return resolve()


@dataclass
class MeetingSettings:
    provider: str = "auto"
    model: str = ""
    system_source: str = "auto"
    mic_device: str = ""
    recordings_dir: Path | None = None
    keep_raw_tracks: bool = True
    post_transcribe: bool = True
    post_diarize: bool = True

    @classmethod
    def from_config(cls, get_setting: Callable[[str, str, Any], Any], data_dir: Path) -> "MeetingSettings":
        from tldw_chatbook.Utils.path_validation import validate_path_simple

        raw_dir = get_setting("meetings", "recordings_dir", "") or ""
        recordings_dir = validate_path_simple(raw_dir) if raw_dir else Path(data_dir) / MEETINGS_DIRNAME
        return cls(
            provider=str(get_setting("meetings", "provider", "auto") or "auto"),
            model=str(get_setting("meetings", "model", "") or ""),
            system_source=str(get_setting("meetings", "system_source", "auto") or "auto"),
            mic_device=str(get_setting("meetings", "mic_device", "") or ""),
            recordings_dir=Path(recordings_dir).resolve(),
            keep_raw_tracks=bool(get_setting("meetings", "keep_raw_tracks", True)),
            post_transcribe=bool(get_setting("meetings", "post_transcribe", True)),
            post_diarize=bool(get_setting("meetings", "post_diarize", True)),
        )


@dataclass
class PrepareResult:
    tap_mode: TapMode
    provider: str
    model: str
    diarization_available: bool
    diarization_missing: tuple[str, ...]
    recoverable: tuple[Path, ...]


def diarization_requirements(find_spec=importlib.util.find_spec) -> tuple[str, ...]:
    """Missing diarization modules, checked WITHOUT importing them (spec §3.5)."""
    missing = []
    for name in DIARIZATION_MODULES:
        try:
            present = find_spec(name) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            missing.append(name)
    return tuple(missing)


def scan_recoverable(meetings_dir: Path) -> list[Path]:
    meetings_dir = Path(meetings_dir)
    if not meetings_dir.exists():
        return []
    found = []
    for folder in sorted(p for p in meetings_dir.iterdir() if p.is_dir()):
        if any(wav_needs_patch(folder / name) for name in ("mixed.wav", "you.wav", "others.wav")):
            found.append(folder)
    return found


def recover_folder(folder: Path) -> dict:
    folder = Path(folder)
    data_bytes = 0
    for name in ("mixed.wav", "you.wav", "others.wav"):
        path = folder / name
        if wav_needs_patch(path):
            patched = patch_wav_header(path)
            if name == "mixed.wav":
                data_bytes = patched
    duration_s = data_bytes / 32000.0
    payload = read_meeting_json(folder)
    if not payload.get("ended_at"):
        payload["ended_at"] = datetime.fromtimestamp((folder / "mixed.wav").stat().st_mtime).isoformat(timespec="seconds")
    payload.update(recovered=True, duration_s=duration_s, stop_reason=payload.get("stop_reason") or "crash")
    return update_meeting_json(folder, **payload)


def _default_facade_factory():
    from tldw_chatbook.Local_Ingestion.transcription_service import TranscriptionService

    return TranscriptionService(local_stt_dispatcher=None)


def _default_dictation_factory(capture: MeetingCapture, facade: Any, cfg: Any):
    from .dictation_service_lazy import LazyLiveDictationService

    return LazyLiveDictationService(
        transcription_provider=cfg.provider,
        transcription_model=cfg.model,
        language=getattr(cfg, "language", "en"),
        enable_commands=False,
        recorder_factory=lambda **_: capture,
        transcription_service_factory=lambda: facade,
    )


def _default_mic_factory(**kwargs):
    from .recording_service import AudioRecordingService

    return AudioRecordingService(**kwargs)


class MeetingSessionOwner:
    def __init__(
        self,
        *,
        settings: MeetingSettings,
        call_from_thread: Callable[..., Any],
        submit_ingest: Callable[..., Optional[str]],
        job_state: Callable[[str], Optional[str]] = lambda job_id: None,
        facade_factory: Callable[[], Any] | None = None,
        dictation_factory: Callable[[MeetingCapture, Any, Any], Any] | None = None,
        tap_probe: Callable[..., TapMode] = probe,
        tap_builder: Callable[..., Any] = build_tap,
        mic_recorder_factory: Callable[..., Any] | None = None,
        vad_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        watchdog_interval_s: float = 1.0,
        stall_after_s: float = 3.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._call_from_thread = call_from_thread
        self._submit_ingest = submit_ingest
        self._job_state = job_state
        self._facade_factory = facade_factory or _default_facade_factory
        self._dictation_factory = dictation_factory or _default_dictation_factory
        self._tap_probe = tap_probe
        self._tap_builder = tap_builder
        self._mic_factory = mic_recorder_factory or _default_mic_factory
        self._vad_factory = vad_factory
        self._clock = clock
        self._watchdog_interval_s = watchdog_interval_s
        self._stall_after_s = stall_after_s
        self._sleep = sleep
        self.prepared: PrepareResult | None = None
        self._facade: Any | None = None
        self._cfg: Any | None = None
        self.session: MeetingSession | None = None
        self.local_sink: LocalMeetingSink | None = None
        self.last_result: MeetingResult | None = None
        self._watchdog: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._lock = threading.RLock()

    # ---- prepare ----------------------------------------------------------
    def prepare(self) -> PrepareResult:
        cfg = resolve_effective_config()
        provider = self.settings.provider if self.settings.provider != "auto" else getattr(cfg, "provider", "auto")
        model = self.settings.model or (getattr(cfg, "model", "") or "")
        self._cfg = type("Cfg", (), {"provider": provider, "model": model or None, "language": getattr(cfg, "language", "en")})()
        if self._facade is None:
            self._facade = self._facade_factory()
        tap_mode = self._tap_probe(system_source=self.settings.system_source)
        missing = diarization_requirements()
        recoverable = tuple(scan_recoverable(self.settings.recordings_dir))
        self.prepared = PrepareResult(
            tap_mode=tap_mode, provider=provider, model=model or "",
            diarization_available=not missing, diarization_missing=missing, recoverable=recoverable,
        )
        return self.prepared

    # ---- lifecycle --------------------------------------------------------
    @property
    def is_active(self) -> bool:
        session = self.session
        return session is not None and session.state in ("starting", "recording", "paused")

    def _submit_on_ui_thread(self, **kwargs) -> Optional[str]:
        return self._call_from_thread(self._submit_ingest, **kwargs)

    def start(self) -> MeetingSession:
        if self.prepared is None:
            self.prepare()
        with self._lock:
            if self.is_active:
                raise RuntimeError("a meeting is already running")
            folder = Path(self.settings.recordings_dir) / datetime.now().strftime("%Y-%m-%d_%H%M")
            suffix = 1
            while folder.exists():
                suffix += 1
                folder = folder.with_name(f"{folder.name.split('-')[0]}-{suffix}") if False else Path(self.settings.recordings_dir) / f"{datetime.now():%Y-%m-%d_%H%M}-{suffix}"
            folder.mkdir(parents=True, exist_ok=True)
            tap = self._tap_builder(self.prepared.tap_mode, recorder_factory=self._mic_factory)
            writers = {"mixed": PlaceholderWavWriter(folder / "mixed.wav")}
            if tap is not None:
                writers["you"] = PlaceholderWavWriter(folder / "you.wav")
                writers["others"] = PlaceholderWavWriter(folder / "others.wav")
            capture = MeetingCapture(
                mic_recorder_factory=self._mic_factory, tap=tap, writers=writers,
                vad_factory=self._vad_factory,
            )
            meta = MeetingMeta(
                folder=folder, mode=capture.mode,
                started_at=datetime.now().isoformat(timespec="seconds"),
                mic_device=self.settings.mic_device or "default",
                system_source=self.prepared.tap_mode.reason,
                provider=self.prepared.provider, model=self.prepared.model,
            )
            self.local_sink = LocalMeetingSink(
                folder, submit=self._submit_on_ui_thread,
                post_transcribe=self.settings.post_transcribe, post_diarize=self.settings.post_diarize,
            )
            facade, cfg = self._facade, self._cfg
            session = MeetingSession(
                meta=meta, capture=capture,
                dictation_factory=lambda cap: self._dictation_factory(cap, facade, cfg),
                sinks=[self.local_sink],
            )
            self.session = session
            if not session.start():
                self.session = None
                raise RuntimeError("meeting failed to start (see log)")
            self._start_watchdog()
            return session

    def pause(self) -> None:
        if self.session is not None:
            self.session.pause()

    def resume(self) -> None:
        if self.session is not None:
            self.session.resume()

    def stop(self, reason: str = "user") -> MeetingResult | None:
        with self._lock:
            session = self.session
            if session is None:
                return None
            self._watchdog_stop.set()
            result = session.stop(reason=reason)
            self.last_result = result
            return result

    def shutdown(self) -> None:
        """App quit: finalise files, skip the ingest submit (spec §3.4)."""
        session = self.session
        if session is None or not self.is_active:
            return
        sink = self.local_sink
        if sink is not None:
            sink._submit = lambda **kwargs: None
        self.stop(reason="shutdown")

    # ---- watchdog ---------------------------------------------------------
    def _start_watchdog(self) -> None:
        self._watchdog_stop.clear()
        self._watchdog = threading.Thread(target=self._watch, daemon=True, name="meeting-watchdog")
        self._watchdog.start()

    def _watch(self) -> None:
        last_pos = -1.0
        last_change = self._clock()
        while not self._watchdog_stop.wait(self._watchdog_interval_s):
            session = self.session
            if session is None or not self.is_active:
                return
            capture = session.capture
            if capture.fault is not None:
                logger.error("meeting watchdog: capture fault {}", capture.fault)
                self.stop(reason="disk_error")
                return
            pos = float(capture.audio_position_s)
            now = self._clock()
            if pos != last_pos or session.state == "paused":
                last_pos, last_change = pos, now
                continue
            if now - last_change >= self._stall_after_s:
                logger.error("meeting watchdog: audio clock stalled for {:.1f}s", now - last_change)
                self.stop(reason="mic_lost")
                return

    # ---- cleanup ----------------------------------------------------------
    def cleanup_raw_tracks_if_done(self) -> bool:
        """Delete you/others once the ingest job is done (best effort, spec §5)."""
        if self.settings.keep_raw_tracks or self.last_result is None or self.local_sink is None:
            return False
        job_id = self.local_sink.job_id
        if not job_id or self._job_state(job_id) != "done":
            return False
        folder = Path(self.last_result.meta.folder)
        for name in ("you.wav", "others.wav"):
            path = folder / name
            if path.exists():
                path.unlink()
        return True
```
Simplify the folder-collision loop before committing: replace the `while folder.exists()` block with
```python
            base = datetime.now().strftime("%Y-%m-%d_%H%M")
            folder = Path(self.settings.recordings_dir) / base
            suffix = 1
            while folder.exists():
                suffix += 1
                folder = Path(self.settings.recordings_dir) / f"{base}-{suffix}"
```

- [ ] **Step 4: Run to verify they pass**

```bash
.venv/bin/python -m pytest Tests/Audio/test_meeting_owner.py -q -p no:cacheprovider
```
Expected: 11 passed. The two watchdog tests rely on the fake mic never delivering frames (so the clock never advances) and on `watchdog_interval_s=0.01`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Audio/meeting_owner.py Tests/Audio/test_meeting_owner.py
git commit -m "feat(audio): MeetingSessionOwner with watchdog, shutdown finalisation, recovery and raw-track cleanup (meeting transcription task 8)"
```

---
### Task 9: Console dictation and hands-free refuse while a meeting is active

Spec §3.4 last paragraph, §7 "Contention".

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/dictation.py` (`ConsoleDictationController._request_console_dictation_start`, ~line 1950)
- Modify: `tldw_chatbook/UI/Console_Modules/hands_free.py` (`ConsoleHandsFreeController.action_toggle_console_hands_free`, ~line 563)
- Test: `Tests/UI/test_console_meeting_guard.py` (new)

**Interfaces:**
- Consumes: `app_instance.meeting_session_owner.is_active` (Task 8; attribute wired in Task 10). Both guards use `getattr` so an app without the owner behaves as before.

- [ ] **Step 1: Write the failing tests**

`Tests/UI/test_console_meeting_guard.py`:
```python
"""Task 9: Console voice entry points refuse while a meeting is active."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Console_Modules.dictation import ConsoleDictationController
from tldw_chatbook.UI.Console_Modules.hands_free import ConsoleHandsFreeController

pytestmark = pytest.mark.unit


class _Bare:
    """Deliberately has ONLY the attributes the guard may touch: if the
    method proceeds past the guard it raises AttributeError."""

    def __init__(self, active: bool):
        self.notices: list[tuple[str, str]] = []
        self.app_instance = SimpleNamespace(
            meeting_session_owner=SimpleNamespace(is_active=active),
            notify=lambda message, severity="information": self.notices.append((message, severity)),
        )
        self._console_dictation_state = "idle"
        self._console_hands_free = None
        self._console_realtime = None


def test_dictation_start_refuses_during_meeting():
    host = _Bare(active=True)
    ConsoleDictationController._request_console_dictation_start(host)
    assert host.notices == [("Meeting in progress: stop it in Meetings before using Console dictation.", "warning")]
    assert host._console_dictation_state == "idle"


def test_dictation_start_proceeds_without_meeting():
    host = _Bare(active=False)
    with pytest.raises(AttributeError):   # past the guard: reaches real state handling
        ConsoleDictationController._request_console_dictation_start(host)
    assert host.notices == []


def test_hands_free_toggle_refuses_during_meeting():
    host = _Bare(active=True)
    ConsoleHandsFreeController.action_toggle_console_hands_free(host)
    assert host.notices == [("Meeting in progress: stop it in Meetings before using hands-free.", "warning")]


def test_hands_free_toggle_proceeds_without_meeting():
    host = _Bare(active=False)
    with pytest.raises(AttributeError):
        ConsoleHandsFreeController.action_toggle_console_hands_free(host)
    assert host.notices == []


def test_guards_tolerate_apps_without_an_owner():
    host = _Bare(active=False)
    del host.app_instance.meeting_session_owner
    with pytest.raises(AttributeError):
        ConsoleDictationController._request_console_dictation_start(host)
    assert host.notices == []
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest Tests/UI/test_console_meeting_guard.py -q -p no:cacheprovider
```
Expected: the two "refuses" tests fail with `AttributeError` (guard absent), the others pass.

- [ ] **Step 3: Implement the guards**

`dictation.py`, in `_request_console_dictation_start` directly after
```python
        if self._console_dictation_state != "idle":
            return
```
add
```python
        owner = getattr(self.app_instance, "meeting_session_owner", None)
        if owner is not None and getattr(owner, "is_active", False):
            # Meetings hold the mic in-process, so the executor's "local STT
            # busy" signal never fires for them (meeting spec §3.4).
            self.app_instance.notify(
                "Meeting in progress: stop it in Meetings before using Console dictation.",
                severity="warning",
            )
            return
```
`hands_free.py`, in `action_toggle_console_hands_free` directly before
```python
        self._enter_console_hands_free_loop(
```
add
```python
        owner = getattr(self.app_instance, "meeting_session_owner", None)
        if owner is not None and getattr(owner, "is_active", False):
            self.app_instance.notify(
                "Meeting in progress: stop it in Meetings before using hands-free.",
                severity="warning",
            )
            return
```

- [ ] **Step 4: Run to verify they pass, plus the Console voice suites**

```bash
.venv/bin/python -m pytest Tests/UI/test_console_meeting_guard.py Tests/Audio/test_console_dictation.py -q -p no:cacheprovider
.venv/bin/python -m pytest Tests/UI -q -p no:cacheprovider -k "dictation or hands_free" -x
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Console_Modules/dictation.py tldw_chatbook/UI/Console_Modules/hands_free.py Tests/UI/test_console_meeting_guard.py
git commit -m "feat(console): refuse dictation and hands-free while a meeting is active (meeting transcription task 9)"
```

---

### Task 10: Tab, destination, route, config section, app-owned owner

Spec §3.5 (tab/route), §3.7 (files), §6 (config), §3.4 (owner + shutdown hook).

**Files:**
- Modify: `tldw_chatbook/Constants.py` (`TAB_SETTINGS` line ~43; `TAB_DISPLAY_LABELS` ~line 100)
- Modify: `tldw_chatbook/UI/Navigation/shell_destinations.py` (`SHELL_DESTINATION_ORDER`, after the `"workflows"` entry ~line 128)
- Modify: `tldw_chatbook/UI/Navigation/screen_registry.py` (route table, after `"workflows"` ~line 89)
- Modify: `tldw_chatbook/app.py` (imports ~line 171 and ~546; `TAB_HELP_TEXT` ~line 1249; owner creation ~line 7689; `_shutdown_app_owned_lifecycles` ~line 16979)
- Modify: `tldw_chatbook/config.py` (`COMPREHENSIVE_CONFIG_RAW`, insert before `[transcription]` ~line 4871)
- Modify: `tldw_chatbook/Audio/meeting_owner.py` (add `build_meeting_session_owner(app)`)
- Test: `Tests/UI/test_meetings_wiring.py` (new)

**Interfaces:**
- Produces: `TAB_MEETINGS = "meetings"`; destination id `"meetings"`, primary route `"meetings"`; `ScreenRoute("meetings", "meetings", "tldw_chatbook.UI.Screens.meetings_screen", "MeetingsScreen")` (module created in Task 11); `app.meeting_session_owner: MeetingSessionOwner`; `build_meeting_session_owner(app) -> MeetingSessionOwner`; `[meetings]` template block.

- [ ] **Step 1: Write the failing tests**

`Tests/UI/test_meetings_wiring.py`:
```python
"""Task 10: Meetings tab/destination/route/config/owner wiring."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_tab_constant_and_label():
    from tldw_chatbook.Constants import TAB_DISPLAY_LABELS, TAB_MEETINGS

    assert TAB_MEETINGS == "meetings" and TAB_DISPLAY_LABELS[TAB_MEETINGS] == "Meetings"


def test_shell_destination_registered_after_workflows():
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER, get_shell_destination

    ids = [d.destination_id for d in SHELL_DESTINATION_ORDER]
    assert ids.index("meetings") == ids.index("workflows") + 1
    dest = get_shell_destination("meetings")
    assert dest.primary_route == "meetings" and dest.label == "Meetings"


def test_screen_route_points_at_meetings_screen():
    from tldw_chatbook.UI.Navigation.screen_registry import registered_screen_routes

    route = next(r for r in registered_screen_routes() if r.screen_name == "meetings")
    assert (route.module_path, route.class_name) == (
        "tldw_chatbook.UI.Screens.meetings_screen", "MeetingsScreen",
    )


def test_app_help_text_covers_meetings():
    from tldw_chatbook.Constants import TAB_MEETINGS
    from tldw_chatbook.app import TldwCli

    assert "Meetings" in TldwCli.TAB_HELP_TEXT[TAB_MEETINGS]
    assert TAB_MEETINGS in TldwCli.NAVIGATION_TABS


def test_config_template_has_meetings_section():
    from tldw_chatbook.config import COMPREHENSIVE_CONFIG_RAW

    block = COMPREHENSIVE_CONFIG_RAW.split("[meetings]", 1)[1].split("\n[", 1)[0]
    for key in ("provider", "model", "system_source", "mic_device", "recordings_dir",
                "keep_raw_tracks", "post_transcribe", "post_diarize"):
        assert f"\n{key} = " in block, key


def test_build_owner_marshals_submit_and_reads_job_state(tmp_path, monkeypatch):
    from tldw_chatbook.Audio import meeting_owner as mo

    monkeypatch.setattr(mo, "_config_accessors", lambda: (lambda s, k, d: d, lambda: tmp_path))
    jobs = {}

    class Registry:
        def submit(self, **kwargs):
            jobs["kw"] = kwargs
            return SimpleNamespace(job_id="ingest-job-9", state=SimpleNamespace(value="queued"))

        def get_job(self, job_id):
            return SimpleNamespace(job_id=job_id, state=SimpleNamespace(value="done")) if job_id == "ingest-job-9" else None

    marshalled = []

    class App:
        library_ingest_jobs = Registry()
        _thread_id = threading.get_ident() + 1   # pretend the UI thread is another thread

        def call_from_thread(self, fn, *args, **kwargs):
            marshalled.append(fn)
            return fn(*args, **kwargs)

    owner = mo.build_meeting_session_owner(App())
    assert owner.settings.recordings_dir == (tmp_path / "meetings").resolve()
    assert owner._submit_on_ui_thread(source_path="x") == "ingest-job-9" and marshalled
    assert owner._job_state("ingest-job-9") == "done" and owner._job_state("nope") is None


def test_build_owner_calls_directly_when_already_on_ui_thread(tmp_path, monkeypatch):
    from tldw_chatbook.Audio import meeting_owner as mo

    monkeypatch.setattr(mo, "_config_accessors", lambda: (lambda s, k, d: d, lambda: tmp_path))

    class App:
        library_ingest_jobs = SimpleNamespace(submit=lambda **kw: SimpleNamespace(job_id="j"), get_job=lambda j: None)
        _thread_id = threading.get_ident()

        def call_from_thread(self, fn, *args, **kwargs):
            raise RuntimeError("must not marshal from the UI thread")

    owner = mo.build_meeting_session_owner(App())
    assert owner._submit_on_ui_thread(source_path="x") == "j"


@pytest.mark.asyncio
async def test_real_app_owns_a_meeting_owner_and_shuts_it_down():
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Audio.meeting_owner import MeetingSessionOwner

    app = _build_test_app()
    assert isinstance(app.meeting_session_owner, MeetingSessionOwner)
    calls = []
    app.meeting_session_owner.shutdown = lambda: calls.append("shutdown")
    await app._shutdown_app_owned_lifecycles()
    assert calls == ["shutdown"]
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest Tests/UI/test_meetings_wiring.py -q -p no:cacheprovider
```
Expected: `ImportError: cannot import name 'TAB_MEETINGS'` and friends.

- [ ] **Step 3: Constants**

`Constants.py` after `TAB_SETTINGS = "settings"`:
```python
TAB_MEETINGS = "meetings"
```
and inside `TAB_DISPLAY_LABELS` after `TAB_SETTINGS: "Settings",`:
```python
    TAB_MEETINGS: "Meetings",
```

- [ ] **Step 4: Shell destination and screen route**

`shell_destinations.py`, immediately after the `"workflows"` `ShellDestination(...)` entry:
```python
    ShellDestination(
        "meetings",
        "Meetings",
        "meetings",
        "Record a call or a room with a live labelled transcript, then file it in the Library.",
        "Record and transcribe a meeting.",
        palette_aliases=("meeting", "record", "transcribe"),
        navigation_priority=75,
    ),
```
`screen_registry.py`, immediately after the `"workflows"` route:
```python
    "meetings": ScreenRoute(
        "meetings", "meetings", "tldw_chatbook.UI.Screens.meetings_screen", "MeetingsScreen"
    ),
```

- [ ] **Step 5: app.py help text, owner, shutdown**

Add `TAB_MEETINGS,` to the `from .Constants import (...)` list (near line 171). In `TAB_HELP_TEXT` after the `TAB_SETTINGS` line:
```python
        TAB_MEETINGS: "Open Meetings to record a call or a room with a live transcript",
```
Import near line 546:
```python
from .Audio.meeting_owner import build_meeting_session_owner
```
After `self.audio_cpp_model_install_owner = AudioCppModelInstallOwner()` (~line 7689):
```python
        self.meeting_session_owner = build_meeting_session_owner(self)
```
In `_shutdown_app_owned_lifecycles`, after `await self.audio_cpp_model_install_owner.shutdown()`:
```python
        await asyncio.to_thread(self.meeting_session_owner.shutdown)
```
(`asyncio` is already imported in app.py.)

- [ ] **Step 6: Owner factory in `meeting_owner.py`**

Append:
```python
def _config_accessors():
    """Late import seam (tests monkeypatch this)."""
    from tldw_chatbook.config import get_cli_setting, get_user_data_dir

    return get_cli_setting, get_user_data_dir


def build_meeting_session_owner(app: Any) -> "MeetingSessionOwner":
    """Wire the owner to a `TldwCli`: config, ingest registry, UI-thread marshalling."""
    get_setting, get_data_dir = _config_accessors()
    settings = MeetingSettings.from_config(get_setting, get_data_dir())

    def marshal(fn, *args, **kwargs):
        # Textual's call_from_thread raises when already on the app thread.
        if threading.get_ident() == getattr(app, "_thread_id", None):
            return fn(*args, **kwargs)
        return app.call_from_thread(fn, *args, **kwargs)

    def submit_ingest(**kwargs):
        job = app.library_ingest_jobs.submit(**kwargs)
        return getattr(job, "job_id", None)

    def job_state(job_id: str):
        job = app.library_ingest_jobs.get_job(job_id)
        state = getattr(job, "state", None)
        return getattr(state, "value", state)

    return MeetingSessionOwner(
        settings=settings, call_from_thread=marshal, submit_ingest=submit_ingest, job_state=job_state,
    )
```

- [ ] **Step 7: Config template block**

In `config.py`, immediately before the `[transcription]` line of `COMPREHENSIVE_CONFIG_RAW`:
```toml
[meetings]
# Meetings screen: record a call (mic + system audio) or a room (mic only).
# STT provider for the live transcript: "auto" uses the Console dictation
# resolution (privacy local-only mode honoured). Never the shared executor.
provider = "auto"
model = ""
# "auto" = native system audio (macOS 14.2+ tap, Linux parec/pw-record,
# Windows WASAPI loopback). Or name an input device such as "BlackHole 2ch".
system_source = "auto"
# Input device name for the mic; empty = system default.
mic_device = ""
# Where meeting folders go; empty = <data_dir>/meetings.
recordings_dir = ""
# Keep you.wav / others.wav after the Library ingest finishes (mixed.wav is always kept).
keep_raw_tracks = true
# Re-transcribe mixed.wav offline after the meeting (needed for speaker labels).
post_transcribe = true
# Ask that offline pass for speaker diarization (needs torch + speechbrain).
post_diarize = true

```

- [ ] **Step 8: Run the wiring tests and the navigation suites**

```bash
.venv/bin/python -m pytest Tests/UI/test_meetings_wiring.py Tests/UI/test_shell_destinations.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_shell_chrome_contract.py Tests/UI/test_chrome_ux_fixes.py -q -p no:cacheprovider
```
Expected: all pass except `test_screen_route_points_at_meetings_screen` may pass already (metadata only). If a navigation test pins the destination count or a fixed strip width, update its expectation to include Meetings and say so in the commit body.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Constants.py tldw_chatbook/UI/Navigation/shell_destinations.py tldw_chatbook/UI/Navigation/screen_registry.py tldw_chatbook/app.py tldw_chatbook/config.py tldw_chatbook/Audio/meeting_owner.py Tests/UI/test_meetings_wiring.py
git commit -m "feat(app): Meetings tab, destination, route, [meetings] config and app-owned session owner (meeting transcription task 10)"
```

---
### Task 11: `MeetingsScreen`

Spec §3.5 in full; §7 footer copy; §3.4 attach/detach.

**Files:**
- Create: `tldw_chatbook/UI/Screens/meetings_screen.py`
- Modify: `tldw_chatbook/Audio/meeting_owner.py` (`PrepareResult.input_devices`, device enumeration in `prepare()`, `apply_device_choice()`)
- Test: `Tests/UI/test_meetings_screen.py` (new)

**Interfaces:**
- Consumes: `MeetingSessionOwner` (Task 8/10): `prepare() -> PrepareResult`, `start() -> MeetingSession`, `pause()`, `resume()`, `stop(reason) -> MeetingResult`, `is_active`, `session`, `local_sink.job_id`, `local_sink.last_submit_error`, `settings`; `MeetingSession.subscribe/unsubscribe`, `segments`, `state`, `capture.levels()`, `capture.audio_position_s`, `meta.folder`, `meta.mode`; `recover_folder(folder)`; `format_clock`; `NavigateToScreen` from `UI/Navigation/main_navigation.py`; `TAB_LIBRARY`, `LIBRARY_NAV_CONTEXT_INGEST` from `Constants`.
- Produces: `class MeetingsScreen(BaseAppScreen)` with ids `#meetings-title`, `#meetings-mic-select`, `#meetings-system-select`, `#meetings-system-status`, `#meetings-provider-status`, `#meetings-diarization-status`, `#meetings-consent`, `#meetings-start`, `#meetings-pause`, `#meetings-stop`, `#meetings-timer`, `#meetings-level-mic`, `#meetings-level-sys`, `#meetings-recovery`, `#meetings-recover`, `#meetings-transcript` (RichLog), `#meetings-partial`, `#meetings-footer`, `#meetings-open-library`; attribute `rendered_lines: list[str]` (what the log shows, for replay and tests).
- Produces on the owner: `PrepareResult.input_devices: tuple[str, ...]`, `MeetingSessionOwner.apply_device_choice(kind: str, value: str) -> None` (`kind` ∈ `mic|system`; persists via `save_setting_to_cli_config("meetings", key, value)` and updates `settings`).

- [ ] **Step 1: Extend the owner for device enumeration and choice**

In `meeting_owner.py`:
- add `input_devices: tuple[str, ...] = ()` as the last field of `PrepareResult`;
- in `prepare()`, before building `PrepareResult`:
```python
        devices: tuple[str, ...] = ()
        try:
            probe_recorder = self._mic_factory(use_vad=False, retain_audio=False, chunk_size=320)
            devices = tuple(str(d.get("name", "")) for d in probe_recorder.get_audio_devices() if d.get("name"))
        except Exception as exc:  # noqa: BLE001 - no backend: pickers stay empty
            logger.info("meeting device enumeration unavailable: {}", exc)
```
  and pass `input_devices=devices`;
- add the method:
```python
    def apply_device_choice(self, kind: str, value: str) -> None:
        from tldw_chatbook.config import save_setting_to_cli_config

        key = "mic_device" if kind == "mic" else "system_source"
        value = "" if (kind == "mic" and value == "default") else value
        setattr(self.settings, key, value)
        save_setting_to_cli_config("meetings", key, value)
        self.prepared = None   # next prepare() re-probes with the new source
```
Add to `Tests/Audio/test_meeting_owner.py`:
```python
def test_prepare_enumerates_input_devices_and_choice_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    saved = []
    monkeypatch.setattr("tldw_chatbook.config.save_setting_to_cli_config", lambda s, k, v: saved.append((s, k, v)) or True)

    class Rec(FakeRecorder):
        def get_audio_devices(self):
            return [{"id": 0, "name": "MacBook Pro Microphone"}, {"id": 1, "name": "BlackHole 2ch"}]

    owner, _, _ = _owner(tmp_path)
    owner._mic_factory = Rec
    assert owner.prepare().input_devices == ("MacBook Pro Microphone", "BlackHole 2ch")
    owner.apply_device_choice("system", "BlackHole 2ch")
    assert owner.settings.system_source == "BlackHole 2ch" and owner.prepared is None
    owner.apply_device_choice("mic", "default")
    assert saved == [("meetings", "system_source", "BlackHole 2ch"), ("meetings", "mic_device", "")]
```
Run `.venv/bin/python -m pytest Tests/Audio/test_meeting_owner.py -q -p no:cacheprovider` → 12 passed.

- [ ] **Step 2: Write the failing screen tests**

`Tests/UI/test_meetings_screen.py`:
```python
"""Task 11: Meetings screen pilots with a faked owner (no hardware, no STT)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Audio.meeting_owner import PrepareResult
from tldw_chatbook.Audio.meeting_session import MeetingMeta, MeetingResult, MeetingSegment
from tldw_chatbook.Audio.system_audio_tap import TapMode
from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_INGEST, TAB_LIBRARY
from tldw_chatbook.UI.Screens.meetings_screen import MeetingsScreen

pytestmark = pytest.mark.unit


class FakeSession:
    def __init__(self, folder: Path, mode="call"):
        self.meta = MeetingMeta(folder=folder, mode=mode, started_at="2026-09-04T14:30:00",
                                mic_device="default", system_source="Native (macOS tap)",
                                provider="faster-whisper", model="base.en")
        self.state = "recording"
        self.segments: list[MeetingSegment] = []
        self.failed_segments = 0
        self.listeners: list[Any] = []
        self.capture = SimpleNamespace(levels=lambda: (0.5, 0.25), audio_position_s=65.0, mode=mode)

    def subscribe(self, listener):
        self.listeners.append(listener)

    def unsubscribe(self, listener):
        self.listeners.remove(listener)

    def emit(self, kind, payload):
        for listener in list(self.listeners):
            listener(kind, payload)

    def add_segment(self, text, label):
        seg = MeetingSegment(len(self.segments), 0.0, 2.0, 0.0, 2.0, label, text)
        self.segments.append(seg)
        self.emit("segment", seg)
        return seg


class FakeOwner:
    def __init__(self, tmp_path: Path, *, tap_kind="native_macos", recoverable=(), mode="call"):
        self.tmp_path = tmp_path
        self.mode = mode
        self.session: FakeSession | None = None
        self.local_sink = SimpleNamespace(job_id=None, last_submit_error=None)
        self.settings = SimpleNamespace(post_diarize=True, mic_device="", system_source="auto")
        self.choices: list[tuple[str, str]] = []
        self.prepared = PrepareResult(
            tap_mode=TapMode(tap_kind, "Native (macOS tap)" if tap_kind == "native_macos" else "Unavailable, mic only"),
            provider="faster-whisper", model="base.en", diarization_available=False,
            diarization_missing=("torch",), recoverable=tuple(recoverable),
            input_devices=("MacBook Pro Microphone", "BlackHole 2ch"),
        )
        self.stop_reasons: list[str] = []

    @property
    def is_active(self):
        return self.session is not None and self.session.state in ("recording", "paused")

    def prepare(self):
        return self.prepared

    def start(self):
        self.session = FakeSession(self.tmp_path / "2026-09-04_1430", self.mode)
        return self.session

    def pause(self):
        self.session.state = "paused"
        self.session.emit("state", "paused")

    def resume(self):
        self.session.state = "recording"
        self.session.emit("state", "recording")

    def stop(self, reason="user"):
        self.stop_reasons.append(reason)
        self.session.state = "stopped"
        self.local_sink.job_id = "ingest-job-3"
        return MeetingResult(meta=self.session.meta, ended_at="2026-09-04T15:35:00", duration_s=65.0,
                             segment_count=len(self.session.segments), transcription_complete=False,
                             failed_segments=1, stop_reason=reason)

    def apply_device_choice(self, kind, value):
        self.choices.append((kind, value))

    def cleanup_raw_tracks_if_done(self):
        return False


class Host(ConsolidatedCSSApp):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance
        self.seen: list[tuple[str, dict]] = []

    async def on_mount(self) -> None:
        await self.push_screen(MeetingsScreen(self.app_instance))

    def on_navigate_to_screen(self, message) -> None:
        self.seen.append((message.screen_name, dict(message.screen_context)))


def _text(widget) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


async def _boot(tmp_path, **owner_kwargs):
    app = _build_test_app()
    owner = FakeOwner(tmp_path, **owner_kwargs)
    app.meeting_session_owner = owner
    host = Host(app)
    return host, owner


@pytest.mark.asyncio
async def test_mount_shows_probe_results(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        assert "Native (macOS tap)" in _text(screen.query_one("#meetings-system-status", Static))
        assert "faster-whisper" in _text(screen.query_one("#meetings-provider-status", Static))
        assert "torch" in _text(screen.query_one("#meetings-diarization-status", Static))
        assert "consent" in _text(screen.query_one("#meetings-consent", Static)).lower()
        assert screen.query_one("#meetings-start", Button).disabled is False
        assert screen.query_one("#meetings-stop", Button).disabled is True


@pytest.mark.asyncio
async def test_start_pause_stop_flow_renders_transcript_and_footer(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        await pilot.click("#meetings-start")
        await pilot.pause(0.3)
        assert owner.is_active and screen.query_one("#meetings-stop", Button).disabled is False
        owner.session.emit("partial", ("hel", "others"))
        await pilot.pause(0.1)
        assert "Others" in _text(screen.query_one("#meetings-partial", Static))
        owner.session.add_segment("hello there", "others")
        owner.session.add_segment("hi", "you")
        await pilot.pause(0.1)
        assert screen.rendered_lines == ["[00:00:00] Others: hello there", "[00:00:00] You: hi"]
        assert _text(screen.query_one("#meetings-partial", Static)) == ""
        assert _text(screen.query_one("#meetings-timer", Static)) == "00:01:05"
        await pilot.click("#meetings-pause")
        await pilot.pause(0.1)
        assert owner.session.state == "paused"
        assert str(screen.query_one("#meetings-pause", Button).label) == "Resume"
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        footer = _text(screen.query_one("#meetings-footer", Static))
        assert "2 segments" in footer and "00:01:05" in footer
        assert "last segment was dropped" in footer and "1 failed" in footer
        assert "ingest-job-3" in footer and str(tmp_path) in footer
        assert screen.query_one("#meetings-open-library", Button).disabled is False
        assert owner.stop_reasons == ["user"]


@pytest.mark.asyncio
async def test_open_in_library_navigates_with_ingest_context(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        await pilot.click("#meetings-open-library")
        await pilot.pause(0.1)
        assert host.seen == [(TAB_LIBRARY, {LIBRARY_NAV_CONTEXT_INGEST: True})]


@pytest.mark.asyncio
async def test_attach_on_mount_replays_running_session(tmp_path):
    app = _build_test_app()
    owner = FakeOwner(tmp_path)
    owner.start()
    owner.session.add_segment("already said", "you")
    app.meeting_session_owner = owner
    host = Host(app)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        assert screen.rendered_lines == ["[00:00:00] You: already said"]
        assert screen.query_one("#meetings-stop", Button).disabled is False
        assert owner.session.listeners  # subscribed
    assert owner.session.listeners == []  # unsubscribed on unmount


@pytest.mark.asyncio
async def test_room_mode_omits_labels_and_submit_error_shows_saved_locally(tmp_path):
    host, owner = await _boot(tmp_path, tap_kind="unavailable", mode="room")
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        assert "mic only" in _text(screen.query_one("#meetings-system-status", Static)).lower()
        await pilot.click("#meetings-start")
        await pilot.pause(0.2)
        owner.session.add_segment("solo", None)
        await pilot.pause(0.1)
        assert screen.rendered_lines == ["[00:00:00] solo"]
        owner.local_sink.last_submit_error = "registry refused"
        real_stop = owner.stop

        def stop(reason="user"):
            result = real_stop(reason)
            owner.local_sink.job_id = None
            return result

        owner.stop = stop
        await pilot.click("#meetings-stop")
        await pilot.pause(0.3)
        footer = _text(screen.query_one("#meetings-footer", Static))
        assert "saved locally, not queued" in footer and "registry refused" in footer
        assert screen.query_one("#meetings-open-library", Button).disabled is True


@pytest.mark.asyncio
async def test_recoverable_folder_offers_recover_and_submits(tmp_path, monkeypatch):
    folder = tmp_path / "2026-09-04_1000"
    folder.mkdir()
    host, owner = await _boot(tmp_path, recoverable=(folder,))
    submitted = []
    owner._submit_on_ui_thread = lambda **kw: submitted.append(kw) or "ingest-job-8"
    monkeypatch.setattr("tldw_chatbook.UI.Screens.meetings_screen.recover_folder",
                        lambda f: {"started_at": "2026-09-04T10:00:00", "duration_s": 12.0})
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        assert folder.name in _text(screen.query_one("#meetings-recovery", Static))
        await pilot.click("#meetings-recover")
        await pilot.pause(0.3)
        assert submitted[0]["source_path"] == str(folder / "mixed.wav")
        assert submitted[0]["detected_type"] == "audio"
        assert "ingest-job-8" in _text(screen.query_one("#meetings-footer", Static))


@pytest.mark.asyncio
async def test_device_selects_apply_choice(tmp_path):
    host, owner = await _boot(tmp_path)
    async with host.run_test(size=(160, 45)) as pilot:
        await pilot.pause(0.3)
        screen = host.screen_stack[-1]
        screen.query_one("#meetings-system-select").value = "BlackHole 2ch"
        await pilot.pause(0.1)
        assert owner.choices == [("system", "BlackHole 2ch")]
```

- [ ] **Step 3: Run to verify they fail**

```bash
.venv/bin/python -m pytest Tests/UI/test_meetings_screen.py -q -p no:cacheprovider
```
Expected: `ModuleNotFoundError: tldw_chatbook.UI.Screens.meetings_screen`.

- [ ] **Step 4: Implement the screen**

`tldw_chatbook/UI/Screens/meetings_screen.py`:
```python
"""Meetings destination: record a call or a room with a live transcript.

The running session is app-owned (`app.meeting_session_owner`, spec §3.4);
this screen attaches on mount and detaches on unmount. Session callbacks
arrive on capture threads and cross to the loop with `call_from_thread`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, ProgressBar, RichLog, Select, Static

from ...Audio.meeting_owner import PrepareResult, recover_folder
from ...Audio.meeting_session import MeetingResult, MeetingSegment, format_clock
from ...Constants import LIBRARY_NAV_CONTEXT_INGEST, TAB_LIBRARY
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen

LABELS = {"you": "You", "others": "Others", "both": "You + Others"}
STOP_REASON_COPY = {
    "mic_lost": "Microphone stopped delivering audio; the meeting was ended.",
    "disk_error": "Recording stopped: the disk write failed.",
}


class MeetingsScreen(BaseAppScreen):
    """Record a call (mic + system audio) or a room (mic only)."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "meetings", **kwargs)
        self._owner = getattr(app_instance, "meeting_session_owner", None)
        self._attached: Any | None = None
        self._level_timer = None
        self._transcribing = False
        self.rendered_lines: list[str] = []

    # ---- compose ----------------------------------------------------------
    def compose_content(self) -> ComposeResult:
        with Vertical(id="meetings-shell"):
            yield Static(
                "Meetings | Record a call or a room | Live transcript | Library handoff",
                id="meetings-title",
                classes="ds-destination-header",
            )
            with Horizontal(id="meetings-workbench", classes="ds-panel destination-workbench"):
                with Vertical(id="meetings-rail", classes="destination-workbench-pane"):
                    yield Static("Sources", classes="destination-section")
                    yield Select([("System default", "default")], value="default", id="meetings-mic-select", allow_blank=False)
                    yield Select([("Native (auto)", "auto")], value="auto", id="meetings-system-select", allow_blank=False)
                    yield Static("System audio: probing…", id="meetings-system-status")
                    yield Static("Transcriber: probing…", id="meetings-provider-status")
                    yield Static("Speaker labels after the meeting: probing…", id="meetings-diarization-status")
                    yield Static("Recording other people may require their consent.", id="meetings-consent", classes="destination-note")
                    with Horizontal(id="meetings-controls"):
                        yield Button("Start", id="meetings-start", variant="success", disabled=True)
                        yield Button("Pause", id="meetings-pause", disabled=True)
                        yield Button("Stop", id="meetings-stop", variant="error", disabled=True)
                    yield Static("00:00:00", id="meetings-timer")
                    yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="meetings-level-mic")
                    yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="meetings-level-sys")
                    yield Static("", id="meetings-recovery")
                    yield Button("Recover", id="meetings-recover", disabled=True)
                with Vertical(id="meetings-canvas", classes="destination-workbench-pane"):
                    yield RichLog(id="meetings-transcript", wrap=True, highlight=False, markup=False)
                    yield Static("", id="meetings-partial")
                    yield Static("", id="meetings-footer")
                    yield Button("Open in Library", id="meetings-open-library", disabled=True)

    # ---- lifecycle --------------------------------------------------------
    def on_mount(self) -> None:
        self._attach_if_running()
        self._run_prepare()
        self._level_timer = self.set_interval(0.2, self._tick)

    def on_unmount(self) -> None:
        self._detach()
        if self._level_timer is not None:
            self._level_timer.stop()
        super().on_unmount()

    def _attach_if_running(self) -> None:
        owner = self._owner
        if owner is None or not owner.is_active or owner.session is None:
            return
        self._attached = owner.session
        self._attached.subscribe(self._on_session_event)
        for segment in list(self._attached.segments):
            self._render_segment(segment)
        self._set_buttons(self._attached.state)

    def _detach(self) -> None:
        if self._attached is not None:
            try:
                self._attached.unsubscribe(self._on_session_event)
            except Exception as exc:  # noqa: BLE001
                logger.debug("meetings detach: {}", exc)
            self._attached = None

    # ---- prepare (worker) -------------------------------------------------
    @work(exclusive=True, group="meetings-prepare", thread=True)
    def _run_prepare(self) -> None:
        if self._owner is None:
            self.app.call_from_thread(self._show_prepare_error, "Meetings are unavailable in this build.")
            return
        try:
            prepared = self._owner.prepare()
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self._show_prepare_error, str(exc))
            return
        self.app.call_from_thread(self._apply_prepared, prepared)

    def _show_prepare_error(self, reason: str) -> None:
        self.query_one("#meetings-provider-status", Static).update(f"Transcriber: {reason}")

    def _apply_prepared(self, prepared: PrepareResult) -> None:
        if not self.is_mounted:
            return
        mode = prepared.tap_mode
        system_copy = mode.reason if mode.kind != "unavailable" else f"Unavailable, mic only ({mode.reason})"
        self.query_one("#meetings-system-status", Static).update(f"System audio: {system_copy}")
        self.query_one("#meetings-provider-status", Static).update(
            f"Transcriber: {prepared.provider} {prepared.model}".rstrip() + " (finalises per segment)"
        )
        if prepared.diarization_available:
            diar = "Speaker labels after the meeting: on"
        else:
            diar = f"Speaker labels after the meeting: off ({', '.join(prepared.diarization_missing)} missing)"
        self.query_one("#meetings-diarization-status", Static).update(diar)
        devices = list(prepared.input_devices)
        mic = self.query_one("#meetings-mic-select", Select)
        mic.set_options([("System default", "default")] + [(d, d) for d in devices])
        settings = getattr(self._owner, "settings", None)
        mic.value = getattr(settings, "mic_device", "") or "default"
        system = self.query_one("#meetings-system-select", Select)
        system.set_options([("Native (auto)", "auto")] + [(d, d) for d in devices])
        system.value = getattr(settings, "system_source", "auto") or "auto"
        recovery = self.query_one("#meetings-recovery", Static)
        recover = self.query_one("#meetings-recover", Button)
        if prepared.recoverable:
            recovery.update("Unfinished meeting found: " + ", ".join(p.name for p in prepared.recoverable))
            recover.disabled = False
        else:
            recovery.update("")
            recover.disabled = True
        if not (self._owner is not None and self._owner.is_active):
            self.query_one("#meetings-start", Button).disabled = False

    # ---- device pickers ---------------------------------------------------
    @on(Select.Changed, "#meetings-mic-select")
    def _mic_changed(self, event: Select.Changed) -> None:
        if self._owner is not None and event.value not in (None, Select.BLANK):
            self._owner.apply_device_choice("mic", str(event.value))

    @on(Select.Changed, "#meetings-system-select")
    def _system_changed(self, event: Select.Changed) -> None:
        if self._owner is not None and event.value not in (None, Select.BLANK):
            self._owner.apply_device_choice("system", str(event.value))
            self._run_prepare()

    # ---- start / pause / stop ---------------------------------------------
    @on(Button.Pressed, "#meetings-start")
    def _start_pressed(self) -> None:
        self.query_one("#meetings-start", Button).disabled = True
        self.rendered_lines.clear()
        self.query_one("#meetings-transcript", RichLog).clear()
        self.query_one("#meetings-footer", Static).update("")
        self.query_one("#meetings-open-library", Button).disabled = True
        self._start_worker()

    @work(exclusive=True, group="meetings-start", thread=True)
    def _start_worker(self) -> None:
        try:
            session = self._owner.start()
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self._start_failed, str(exc))
            return
        self.app.call_from_thread(self._on_started, session)

    def _start_failed(self, reason: str) -> None:
        self.app_instance.notify(f"Meeting failed to start: {reason}", severity="error")
        self.query_one("#meetings-start", Button).disabled = False

    def _on_started(self, session: Any) -> None:
        self._attached = session
        session.subscribe(self._on_session_event)
        self._set_buttons(session.state)

    @on(Button.Pressed, "#meetings-pause")
    def _pause_pressed(self) -> None:
        session = self._attached
        if session is None:
            return
        if session.state == "paused":
            self._owner.resume()
        else:
            self._owner.pause()

    @on(Button.Pressed, "#meetings-stop")
    def _stop_pressed(self) -> None:
        self.query_one("#meetings-stop", Button).disabled = True
        self.query_one("#meetings-pause", Button).disabled = True
        self._stop_worker()

    @work(exclusive=True, group="meetings-stop", thread=True)
    def _stop_worker(self) -> None:
        result = self._owner.stop(reason="user")
        self.app.call_from_thread(self._on_stopped, result)

    def _on_stopped(self, result: MeetingResult | None) -> None:
        self._detach()
        self._set_buttons("stopped")
        if result is None:
            return
        sink = getattr(self._owner, "local_sink", None)
        job_id = getattr(sink, "job_id", None)
        error = getattr(sink, "last_submit_error", None)
        parts = [f"Saved {result.segment_count} segments, {format_clock(result.duration_s)}."]
        if not result.transcription_complete:
            parts.append("The last segment was dropped (transcriber did not finish in time).")
        if result.failed_segments:
            parts.append(f"{result.failed_segments} failed segment(s).")
        parts.append(f"Folder: {result.meta.folder}.")
        if job_id:
            parts.append(f"Library ingest queued: {job_id}.")
        else:
            parts.append(f"Saved locally, not queued ({error or 'no ingest job'}).")
        if result.stop_reason in STOP_REASON_COPY:
            self.app_instance.notify(STOP_REASON_COPY[result.stop_reason], severity="error")
        self.query_one("#meetings-footer", Static).update(" ".join(parts))
        self.query_one("#meetings-open-library", Button).disabled = not bool(job_id)
        self.query_one("#meetings-partial", Static).update("")

    # ---- session events (capture threads -> loop) -------------------------
    def _on_session_event(self, kind: str, payload: Any) -> None:
        try:
            self.app.call_from_thread(self._apply_event, kind, payload)
        except Exception as exc:  # noqa: BLE001 - screen may be tearing down
            logger.debug("meetings event dropped: {}", exc)

    def _apply_event(self, kind: str, payload: Any) -> None:
        if not self.is_mounted:
            return
        if kind == "segment":
            self._render_segment(payload)
            self._transcribing = False
            self.query_one("#meetings-partial", Static).update("")
        elif kind == "partial":
            text, label = payload
            prefix = f"{LABELS.get(label, label)}: " if label else ""
            self.query_one("#meetings-partial", Static).update(f"{prefix}{text}…")
        elif kind == "transcribing":
            self._transcribing = bool(payload)
            partial = self.query_one("#meetings-partial", Static)
            if self._transcribing and not str(getattr(partial.renderable, "plain", partial.renderable)):
                partial.update("transcribing…")
            elif not self._transcribing and str(getattr(partial.renderable, "plain", partial.renderable)) == "transcribing…":
                partial.update("")
        elif kind == "state":
            self._set_buttons(str(payload))
            if payload == "stopped" and self._attached is not None and self._owner is not None:
                # Ended by the watchdog or shutdown, not by our Stop button.
                self._on_stopped(getattr(self._owner, "last_result", None))

    def _render_segment(self, segment: MeetingSegment) -> None:
        stamp = f"[{format_clock(segment.t_audio_start)}]"
        line = f"{stamp} {LABELS.get(segment.label, segment.label)}: {segment.text}" if segment.label else f"{stamp} {segment.text}"
        self.rendered_lines.append(line)
        self.query_one("#meetings-transcript", RichLog).write(line)

    def _set_buttons(self, state: str) -> None:
        active = state in ("starting", "recording", "paused")
        self.query_one("#meetings-start", Button).disabled = active
        self.query_one("#meetings-stop", Button).disabled = not active
        pause = self.query_one("#meetings-pause", Button)
        pause.disabled = not active
        pause.label = "Resume" if state == "paused" else "Pause"

    def _tick(self) -> None:
        session = self._attached
        if session is None or not self.is_mounted:
            return
        try:
            self.query_one("#meetings-timer", Static).update(format_clock(float(session.capture.audio_position_s)))
            mic, sys_ = session.capture.levels()
            self.query_one("#meetings-level-mic", ProgressBar).progress = int(mic * 100)
            self.query_one("#meetings-level-sys", ProgressBar).progress = int(sys_ * 100)
        except Exception as exc:  # noqa: BLE001
            logger.debug("meetings tick: {}", exc)

    # ---- recovery + Library -----------------------------------------------
    @on(Button.Pressed, "#meetings-recover")
    def _recover_pressed(self) -> None:
        prepared = getattr(self._owner, "prepared", None)
        folders = tuple(getattr(prepared, "recoverable", ()) or ())
        if not folders:
            return
        self.query_one("#meetings-recover", Button).disabled = True
        self._recover_worker(folders[0])

    @work(exclusive=True, group="meetings-recover", thread=True)
    def _recover_worker(self, folder: Path) -> None:
        payload = recover_folder(folder)
        started = str(payload.get("started_at", ""))[:16].replace("T", " ")
        try:
            job_id = self._owner._submit_on_ui_thread(
                source_path=str(Path(folder) / "mixed.wav"), title=f"Meeting {started} (recovered)",
                keywords=("meeting",), detected_type="audio",
                ingest_options={"diarization": bool(getattr(self._owner.settings, "post_diarize", True))},
            )
            copy = f"Recovered {Path(folder).name}: Library ingest queued: {job_id}."
        except Exception as exc:  # noqa: BLE001
            copy = f"Recovered {Path(folder).name}: saved locally, not queued ({exc})."
        self.app.call_from_thread(self._recovered, copy)

    def _recovered(self, copy: str) -> None:
        self.query_one("#meetings-footer", Static).update(copy)
        self.query_one("#meetings-recovery", Static).update("")

    @on(Button.Pressed, "#meetings-open-library")
    def _open_library(self) -> None:
        self.app.post_message(NavigateToScreen(TAB_LIBRARY, {LIBRARY_NAV_CONTEXT_INGEST: True}))
```
Note on `_recover_worker`: it runs on a worker thread, so `_submit_on_ui_thread` marshals correctly. In the screen test the owner is a fake whose `_submit_on_ui_thread` is a plain lambda.

- [ ] **Step 5: Run the screen tests and the wiring test that imports the screen**

```bash
.venv/bin/python -m pytest Tests/UI/test_meetings_screen.py Tests/UI/test_meetings_wiring.py -q -p no:cacheprovider
```
Expected: all pass. Common failures and their fixes: `Select.set_options` resets `value`, so set `value` after `set_options` (already done); `pilot.click` on a disabled button is a no-op, so the assertions on `disabled` must come first; if `RichLog.write` complains about markup, the constructor already passes `markup=False`.

- [ ] **Step 6: Add the destination-mount parametrize row**

In `Tests/UI/test_destination_shells.py`, add `MeetingsScreen` to `SCREEN_BY_ROUTE` (`"meetings": MeetingsScreen,` with the import next to the other screen imports) and add the row `("meetings", "#meetings-title", "record a call")` to `test_primary_destination_wrappers_mount`'s parametrize list. The purpose assertion reads `.destination-purpose`, so also add `yield Static("Record a call or a room and get a live transcript into the Library.", classes="destination-purpose")` directly under the title Static in `compose_content`. Run:
```bash
.venv/bin/python -m pytest Tests/UI/test_destination_shells.py -q -p no:cacheprovider -k "primary_destination_wrappers_mount"
```
Expected: pass for all rows.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/meetings_screen.py tldw_chatbook/Audio/meeting_owner.py Tests/UI/test_meetings_screen.py Tests/Audio/test_meeting_owner.py Tests/UI/test_destination_shells.py
git commit -m "feat(ui): Meetings screen with live transcript, device pickers, recovery and Library handoff (meeting transcription task 11)"
```

---
### Task 12: macOS Core Audio tap helper and packaging

Spec §3.6 in full; §10 last assumption (unsandboxed app).

**Files:**
- Modify: `tldw_chatbook/Audio/audiotap/main.swift` (replaces the Task 6 placeholder)
- Modify: `Packaging/macos/build_app.py` (`create_launcher_script` neighbour; `create_info_plist_additions`)
- Modify: `pyproject.toml` (package data for `Audio/audiotap/*.swift` if the build backend needs it)
- Test: `Tests/Audio/test_audiotap_helper_macos.py` (new, opt-in)

**Interfaces:**
- Consumes: `ensure_helper()`/`SubprocessTap` (Task 6): the helper must write 640-byte PCM16 mono 16 kHz frames to stdout, print `READY` on stderr once the IO proc runs, exit on stdin EOF or SIGTERM, exit code 2 on tap-creation failure, 3 on unsupported OS.
- Produces: `tldw-audiotap` binary in `Contents/MacOS/` of the DMG app; `NSAudioCaptureUsageDescription` and `NSMicrophoneUsageDescription` in the bundle's Info.plist.

- [ ] **Step 1: Write the opt-in test**

`Tests/Audio/test_audiotap_helper_macos.py`:
```python
"""Opt-in: compiles and runs the real macOS tap helper. Never in CI.

Run: pytest Tests/Audio/test_audiotap_helper_macos.py -m real_audio_device -p no:cacheprovider
"""
from __future__ import annotations

import platform
import shutil
import sys
import threading

import pytest

from tldw_chatbook.Audio import system_audio_tap as sat

pytestmark = [pytest.mark.real_audio_device, pytest.mark.integration]


@pytest.mark.skipif(sys.platform != "darwin" or shutil.which("swiftc") is None, reason="macOS + swiftc only")
def test_helper_compiles_and_emits_frames(tmp_path):
    assert sat.macos_version_ok(platform.mac_ver()[0])
    helper = sat.ensure_helper(tmp_path, executable=str(tmp_path / "nowhere"))
    assert helper is not None and helper.exists()
    tap = sat.SubprocessTap((str(helper),))
    frames: list[bytes] = []
    got = threading.Event()

    def on_frames(frame: bytes) -> None:
        frames.append(frame)
        if len(frames) >= 50:
            got.set()

    assert tap.start(on_frames)
    assert got.wait(10.0), f"no frames; stderr: {tap.last_stderr}"
    tap.stop()
    assert tap.state == "stopped" and all(len(f) == 640 for f in frames)
```

- [ ] **Step 2: Run it to see it fail (placeholder source does not compile)**

```bash
.venv/bin/python -m pytest Tests/Audio/test_audiotap_helper_macos.py -m real_audio_device -q -p no:cacheprovider
```
Expected: FAIL — `ensure_helper` returns None because `swiftc` rejects the one-line placeholder.

- [ ] **Step 3: Write the helper**

`tldw_chatbook/Audio/audiotap/main.swift`:
```swift
// tldw-audiotap: macOS 14.2+ system-audio capture helper.
// Emits 20 ms frames of PCM16 mono 16 kHz on stdout; READY on stderr once
// the IO proc runs; exits on stdin EOF / SIGTERM. Exit 2 = tap creation
// failed (usually the System Audio Recording permission), 3 = unsupported OS.
import AVFoundation
import CoreAudio
import Foundation

let frameBytes = 640
let ringSeconds = 2

final class Ring {
    private var buffer = [UInt8](repeating: 0, count: 32_000 * ringSeconds)
    private var head = 0, count = 0
    private let lock = NSLock()
    private(set) var dropped = 0

    func push(_ data: UnsafeRawBufferPointer) {
        lock.lock(); defer { lock.unlock() }
        for byte in data {
            if count == buffer.count { head = (head + 1) % buffer.count; count -= 1; dropped += 1 }
            buffer[(head + count) % buffer.count] = byte
            count += 1
        }
    }

    func pop(_ n: Int) -> [UInt8]? {
        lock.lock(); defer { lock.unlock() }
        guard count >= n else { return nil }
        var out = [UInt8](repeating: 0, count: n)
        for i in 0..<n { out[i] = buffer[(head + i) % buffer.count] }
        head = (head + n) % buffer.count
        count -= n
        return out
    }
}

func stderr(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

guard #available(macOS 14.2, *) else { stderr("unsupported macOS"); exit(3) }

func processObject(for pid: pid_t) -> AudioObjectID? {
    var pidVar = pid
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
        mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var object = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = withUnsafePointer(to: &pidVar) { ptr in
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr,
                                   UInt32(MemoryLayout<pid_t>.size), ptr, &size, &object)
    }
    return status == noErr ? object : nil
}

var exclude: [NSNumber] = []
if let own = processObject(for: ProcessInfo.processInfo.processIdentifier) { exclude.append(NSNumber(value: own)) }
if let parent = processObject(for: getppid()) { exclude.append(NSNumber(value: parent)) }

let description = CATapDescription(stereoGlobalTapButExcludeProcesses: exclude)
description.uuid = UUID()
description.muteBehavior = .unmuted
description.name = "tldw-audiotap"
var tapID = AudioObjectID(kAudioObjectUnknown)
var status = AudioHardwareCreateProcessTap(description, &tapID)
guard status == noErr else { stderr("process tap failed: \(status) (grant System Audio Recording in Privacy & Security)"); exit(2) }

let aggregate: [String: Any] = [
    kAudioAggregateDeviceNameKey: "tldw-audiotap",
    kAudioAggregateDeviceUIDKey: UUID().uuidString,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceTapListKey: [[
        kAudioSubTapUIDKey: description.uuid.uuidString,
        kAudioSubTapDriftCompensationKey: true,
    ]],
]
var aggregateID = AudioObjectID(kAudioObjectUnknown)
status = AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &aggregateID)
guard status == noErr else { stderr("aggregate device failed: \(status)"); exit(2) }

var formatAddr = AudioObjectPropertyAddress(
    mSelector: kAudioTapPropertyFormat, mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain)
var asbd = AudioStreamBasicDescription()
var asbdSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
status = AudioObjectGetPropertyData(tapID, &formatAddr, 0, nil, &asbdSize, &asbd)
guard status == noErr, let inFormat = AVAudioFormat(streamDescription: &asbd) else { stderr("tap format failed: \(status)"); exit(2) }
guard let outFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true),
      let converter = AVAudioConverter(from: inFormat, to: outFormat) else { stderr("converter failed"); exit(2) }

let ring = Ring()
var procID: AudioDeviceIOProcID?
status = AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, nil) { _, inData, _, _, _ in
    let frames = AVAudioFrameCount(inData.pointee.mBuffers.mDataByteSize) / AVAudioFrameCount(max(1, asbd.mBytesPerFrame))
    guard frames > 0, let input = AVAudioPCMBuffer(pcmFormat: inFormat, bufferListNoCopy: inData, deallocator: nil) else { return }
    input.frameLength = frames
    let capacity = AVAudioFrameCount(Double(frames) * 16_000.0 / inFormat.sampleRate) + 16
    guard let output = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
    var consumed = false
    var error: NSError?
    converter.convert(to: output, error: &error) { _, outStatus in
        if consumed { outStatus.pointee = .noDataNow; return nil }
        consumed = true; outStatus.pointee = .haveData; return input
    }
    guard error == nil, let bytes = output.int16ChannelData?.pointee else { return }
    let byteCount = Int(output.frameLength) * 2
    ring.push(UnsafeRawBufferPointer(start: bytes, count: byteCount))
}
guard status == noErr, let ioProc = procID else { stderr("io proc failed: \(status)"); exit(2) }
status = AudioDeviceStart(aggregateID, ioProc)
guard status == noErr else { stderr("device start failed: \(status)"); exit(2) }
stderr("READY")

let writer = Thread {
    let out = FileHandle.standardOutput
    var reported = 0
    while true {
        if let frame = ring.pop(frameBytes) {
            out.write(Data(frame))
        } else {
            usleep(5_000)
        }
        if ring.dropped - reported >= 32_000 { reported = ring.dropped; stderr("dropped \(reported) bytes") }
    }
}
writer.start()

signal(SIGTERM) { _ in exit(0) }
signal(SIGPIPE) { _ in exit(0) }
// Block until the parent closes our stdin.
_ = FileHandle.standardInput.readDataToEndOfFile()
AudioDeviceStop(aggregateID, ioProc)
AudioDeviceDestroyIOProcID(aggregateID, ioProc)
AudioHardwareDestroyAggregateDevice(aggregateID)
AudioHardwareDestroyProcessTap(tapID)
exit(0)
```

- [ ] **Step 4: Compile by hand and fix compile errors until clean**

```bash
swiftc -O -o /tmp/tldw-audiotap tldw_chatbook/Audio/audiotap/main.swift -framework CoreAudio -framework AVFoundation
```
Expected: a binary at `/tmp/tldw-audiotap`. Swift API names in Step 3 follow Apple's macOS 14.2 Core Audio taps headers; if `swiftc` reports a renamed symbol (for example `CATapDescription.init(stereoGlobalTapButExcludeProcesses:)` or `kAudioSubTapUIDKey`), open `/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/CoreAudio.framework/Headers/CATapDescription.h` and `AudioHardwareTapping.h` (or the CommandLineTools SDK path under `/Library/Developer/CommandLineTools/SDKs/`) and use the spelling there. Then run it once from a Terminal that has the permission: `/tmp/tldw-audiotap | head -c 6400 | wc -c` while any audio plays → `6400`; the first run prompts for System Audio Recording.

- [ ] **Step 5: Run the opt-in test**

```bash
.venv/bin/python -m pytest Tests/Audio/test_audiotap_helper_macos.py -m real_audio_device -q -p no:cacheprovider
```
Expected: 1 passed (audio must be playing for frames to be non-silent, but silence still produces frames).

- [ ] **Step 6: Packaging**

In `Packaging/macos/build_app.py` add a method next to `create_launcher_script` and call it from `build()` right after `self.create_launcher_script()`:
```python
    def build_audiotap_helper(self):
        """Compile the Core Audio tap helper into Contents/MacOS (meeting transcription)."""
        source = self.project_root / "tldw_chatbook" / "Audio" / "audiotap" / "main.swift"
        app_path = self.dist_dir / f"{self.app_name}.app"
        target = app_path / "Contents" / "MacOS" / "tldw-audiotap"
        if not source.exists():
            print("audiotap source missing; skipping helper")
            return
        result = subprocess.run(
            ["swiftc", "-O", "-o", str(target), str(source), "-framework", "CoreAudio", "-framework", "AVFoundation"],
            cwd=self.project_root,
        )
        print("audiotap helper built" if result.returncode == 0 else "audiotap helper build FAILED (meetings fall back to a virtual device)")
```
In `create_info_plist_additions`, before the final `plistlib.dump`:
```python
            plist_data['NSAudioCaptureUsageDescription'] = (
                'tldw_chatbook records what your computer plays so meetings can be transcribed.'
            )
            plist_data['NSMicrophoneUsageDescription'] = (
                'tldw_chatbook records your microphone for dictation and meetings.'
            )
```
Check package data: run `.venv/bin/python -c "import tldw_chatbook.Audio.system_audio_tap as s; print(s.helper_source_path().exists())"` → `True` for the editable install. Then `grep -n 'package-data\|include =' pyproject.toml`; if the build backend lists package data explicitly, add `"tldw_chatbook/Audio/audiotap/*.swift"` there so wheels ship the source.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Audio/audiotap/main.swift Packaging/macos/build_app.py pyproject.toml Tests/Audio/test_audiotap_helper_macos.py
git commit -m "feat(macos): Core Audio process-tap helper for meeting system audio + bundle packaging (meeting transcription task 12)"
```

---

### Task 13: Docs, full suite, live verification, backlog close-out

Spec §3.7 (docs), §8 live verification.

**Files:**
- Create: `Docs/User_Guide/meetings.md`
- Modify: `Docs/User_Guide/index.md` (screens table, after the Workflows row)
- Modify: the backlog task file from Task 0

- [ ] **Step 1: Full suite**

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider -x --timeout=600 2>&1 | tail -20
```
Expected: green, or only failures already recorded as pre-existing in Task 0 Step 3. Fix anything new before continuing.

- [ ] **Step 2: Live verification (record evidence, per `backlog/docs/lessons-live-verification.md`)**

```bash
tmux new-session -d -s meetings -x 200 -y 50 '.venv/bin/python -m tldw_chatbook.app'
```
Then, in another terminal, play any video or podcast through the speakers and speak into the mic. In the TUI: open Meetings (command palette → "Meetings"), confirm the rail reads `System audio: Native (macOS tap)` (or `Native (parec)` on Linux), press Start, wait ~30 s, press Stop. Capture `tmux capture-pane -p -t meetings > /tmp/meetings-live.txt` before Stop and after. Check:
- transcript rows carry both `You:` and `Others:` labels;
- the footer names the folder and an `ingest-job-N`;
- `ls <folder>` shows `mixed.wav you.wav others.wav transcript.jsonl meeting.json`;
- `python -c "import wave; print(wave.open('<folder>/mixed.wav').getnframes())"` is > 0;
- Library ▸ Ingest shows the job reaching done, and the media item has a transcript.
Then switch to Console mid-meeting and press the mic button → toast "Meeting in progress…". Quit the app mid-meeting (`q` / ctrl+q), relaunch, open Meetings → "Unfinished meeting found" → Recover → footer shows a job id. Paste the captured pane text and the `ls` output into the backlog task's Implementation Notes.

- [ ] **Step 3: User guide page**

Write `Docs/User_Guide/meetings.md` from `Docs/User_Guide/_template.md`, filling every section from the live session in Step 2 (the template forbids unverified claims). Required content: getting there (palette "Meetings"; no hotkey), the rail's four status lines and what each value means, the consent line, the Start/Pause/Stop flow, the footer, "Open in Library", recovery, the `[meetings]` keys table (copy §6 of the spec), the macOS permission prompt (System Audio Recording), the virtual-device fallback (BlackHole / VB-Cable), the known limits (per-segment finals with up to ~10 s + transcription lag; speaker labels only after the meeting when torch + speechbrain are installed; Windows loopback unverified). End with `*Verified against dev @ <short-sha> — 2026-MM-DD*`.

Add to `Docs/User_Guide/index.md` after the Workflows row:
```
| — | [Meetings](meetings.md) 🚧 | Record a call or a room with a live labelled transcript, then file it in the Library. |
```

- [ ] **Step 4: Commit docs**

```bash
git add Docs/User_Guide/meetings.md Docs/User_Guide/index.md
git commit -m "docs: Meetings user guide (meeting transcription task 13)"
```

- [ ] **Step 5: Backlog close-out and follow-up tasks**

Tick every AC in the task file, add `## Implementation Notes` (approach, files, the live evidence, deviations: Swift source path, `update_privacy_settings` not used because it persists config), then:
```bash
backlog task edit <id> -s Done
backlog task create "Wire process_audio on ParakeetMLXStreamingTranscriber so the dictation streaming regime works" -d "The lazy dictation service requires process_audio; the MLX transcriber only has add_audio, so streaming partials are dead for Console and Meetings (meeting spec §9 item 1)." --ac "Console dictation on Apple Silicon shows word-level partials with parakeet-mlx" -l audio
backlog task create "Verify Windows WASAPI loopback capture for Meetings" -d "Meeting spec §3.1/§10: confirm sounddevice enumerates '[Loopback]' devices and auto_convert works on a real Windows box." --ac "A Zoom test call on Windows yields an Others-labelled transcript" -l audio,windows
backlog task create "Meetings server sink over tldw_server WebSocket live ingest" -d "Meeting spec §9 item 3." --ac "A meeting started with an active server also appears as a server meeting session with live transcript" -l audio,server
backlog task create "Meetings phase 2: live speaker labels via a Diarizer implementation (MOSS candidate)" -d "Meeting spec §9 item 2; needs its own design." --ac "Design approved" -l audio
```
Also update the spec file's §3.6 path sentence to name `tldw_chatbook/Audio/audiotap/main.swift` and commit: `git commit -am "docs: spec path for the audiotap source"`.

- [ ] **Step 6: Lessons**

If any of these bit during implementation, add an entry with the incident to `backlog/docs/lessons-testing-evidence.md`: the deferred-executor 60 s ceiling silently dropping audio, the recorder swallowing callback exceptions, screens never being cached, or `update_privacy_settings` persisting config as a side effect. Skip if nothing new surfaced.

---

## Self-review

**Spec coverage** (section → task): §1 decisions → all; §2 building blocks → Tasks 1, 3, 7, 8, 11; §3.1 tap → Task 6; §3.2 capture → Tasks 4, 5; §3.3 session/sinks/diarizer → Task 7; §3.4 owner + Console refusal → Tasks 8, 9, 10; §3.5 screen → Task 11; §3.6 helper + packaging → Task 12; §3.7 existing-file changes → Tasks 1, 3, 9, 10, 12; §4 data flow/attribution/live output/pause/stop/file layout → Tasks 2, 4, 5, 7, 8; §5 Library handoff incl. `post_transcribe=false` and raw-track cleanup → Tasks 7, 8, 10; §6 config → Tasks 8, 10; §7 errors/watchdog/recovery/contention → Tasks 5, 8, 9, 11; §8 tests → every task; §9 follow-ups → Task 13 Step 5; §10 assumptions → Task 12 Step 6 (package data), Task 11 Step 4 (`LIBRARY_NAV_CONTEXT_INGEST` resolves the jobs-view assumption), Task 13 (Windows follow-up).

**Known deviations from the spec, all deliberate:** Swift source lives in the package (`Audio/audiotap/`); privacy auto-clear is set on the dict rather than via `update_privacy_settings()` because that method persists to config as a side effect; the partial-label window and `MEETING_SEGMENT_CAP_S` are module constants, not config keys.

**Type consistency checked:** `TapMode` fields used identically in Tasks 6, 8, 11; `MeetingSegment`/`MeetingResult` fields identical in Tasks 7, 8, 11; `PrepareResult.input_devices` added in Task 11 with a default so Task 8 tests keep passing; `MeetingCapture` constructor kwargs identical in Tasks 5 and 8; the owner's `_submit_on_ui_thread` name used by both Task 10's test and Task 11's recover worker.
