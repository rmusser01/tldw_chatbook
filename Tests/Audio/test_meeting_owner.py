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


class FakeJobRegistry:
    """Stand-in for `app.library_ingest_jobs`: listeners + one job state."""

    def __init__(self, state: str = "queued"):
        self.state = state
        self.listeners: list = []

    def add_listener(self, callback):
        self.listeners.append(callback)

    def remove_listener(self, callback):
        if callback in self.listeners:
            self.listeners.remove(callback)

    def job_state(self, job_id):
        return self.state

    def fire(self):
        for callback in list(self.listeners):
            callback()


def _owner(tmp_path, *, tap_kind="unavailable", job_state=None, registry=None, **over):
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
        job_state=job_state or (registry.job_state if registry else (lambda job_id: None)),
        subscribe_jobs=registry.add_listener if registry else None,
        unsubscribe_jobs=registry.remove_listener if registry else None,
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


def test_no_diarizer_built_when_live_off(tmp_path, monkeypatch):
    owner, _, _ = _owner(tmp_path, live_diarization=False)
    owner.prepare(); session = owner.start()
    assert session._diarizer is None


def test_diarizer_built_when_live_on_and_deps_present(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    built = {}
    monkeypatch.setattr(mo, "build_diarizer", lambda settings: built.setdefault("d", object()))
    owner, _, _ = _owner(tmp_path, live_diarization=True)
    owner.prepare(); session = owner.start()
    assert session._diarizer is built["d"]


def test_live_on_missing_deps_falls_back_to_coarse(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ("torch",))
    owner, _, _ = _owner(tmp_path, live_diarization=True)
    owner.prepare(); session = owner.start()
    assert session._diarizer is None


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


def _wait_for_result(owner: mo.MeetingSessionOwner, timeout: float = 2.0) -> None:
    """Poll for the stop outcome, not for `is_active` to flip.

    `is_active` can read False for a brief window before `last_result` is
    assigned (MeetingSession.stop() flips state to "stopping" well before it
    finishes computing the result) -- polling on it here raced the watchdog
    thread. Waiting on the actual outcome is deterministic instead.
    """
    deadline = time.monotonic() + timeout
    while owner.last_result is None and time.monotonic() < deadline:
        time.sleep(0.01)


def test_watchdog_stops_on_fault(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, _, _ = _owner(tmp_path)
    owner.prepare()
    session = owner.start()
    session.capture.fault = OSError("disk full")
    _wait_for_result(owner)
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
    _wait_for_result(owner)           # no mic frames ever arrive -> stall
    assert not owner.is_active and owner.last_result.stop_reason == "mic_lost"


def test_shutdown_finalises_files_without_submitting(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, _, submitted = _owner(tmp_path)
    owner.prepare()
    session = owner.start()
    owner.shutdown()
    assert not owner.is_active and submitted == []
    assert not wav_needs_patch(session.meta.folder / "mixed.wav")
    assert owner.local_sink._handle is None          # transcript handle released
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


def test_recover_folder_survives_a_folder_key_in_meeting_json(tmp_path):
    # TASK-31551 live-verification finding: the real writer (MeetingSession /
    # meeting_owner.start()/stop()) always persists a "folder" field in
    # meeting.json (see write_meeting_json call sites) -- the two tests
    # above hand-write a meeting.json WITHOUT that key, which is exactly
    # the field recover_folder's `update_meeting_json(folder, **payload)`
    # collides with, so they can never see the bug. Recovering an actual
    # crashed meeting reproducibly raised
    # `TypeError: update_meeting_json() got multiple values for argument
    # 'folder'`, and Textual's default `exit_on_error=True` on the
    # `@work(thread=True)`-decorated recover worker took the whole app
    # down with it.
    folder = tmp_path / "meetings" / "2026-09-04_1200"
    folder.mkdir(parents=True)
    writer = PlaceholderWavWriter(folder / "mixed.wav")
    writer.write(b"\x00\x00" * 320 * 10)
    writer._handle.flush()  # crash: never closed
    (folder / "meeting.json").write_text(json.dumps({
        "schema": 1,
        "folder": str(folder),
        "started_at": "2026-09-04T12:00:00",
        "ended_at": None,
        "mode": "call",
    }))
    payload = mo.recover_folder(folder)
    assert payload["recovered"] is True
    assert payload["folder"] == str(folder)


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


def _tap_owner(tmp_path, monkeypatch, **over):
    """An owner in call mode (three writers) with a trivial always-up tap."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))

    class Tap:
        state = "stopped"

        def start(self, on_frames):
            return True

        def stop(self):
            return None

    owner, marshalled, submitted = _owner(tmp_path, tap_kind="native_macos", **over)
    owner._tap_builder = lambda mode, **kw: Tap()
    owner.prepare()
    return owner


def test_stop_subscribes_to_the_registry_and_cleans_up_when_the_job_is_done(tmp_path, monkeypatch):
    """Q12: `cleanup_raw_tracks_if_done` had no production caller at all, so
    `keep_raw_tracks = false` never deleted anything. The wait outlives the
    Meetings screen, so the owner (not the screen) holds the listener."""
    registry = FakeJobRegistry("parsing")
    owner = _tap_owner(tmp_path, monkeypatch, registry=registry, keep_raw_tracks=False)
    session = owner.start()
    folder = session.meta.folder
    owner.stop()
    assert len(registry.listeners) == 1

    registry.fire()                       # still parsing: nothing to do yet
    assert (folder / "you.wav").exists() and registry.listeners

    registry.state = "done"
    registry.fire()
    assert not (folder / "you.wav").exists() and not (folder / "others.wav").exists()
    assert (folder / "mixed.wav").exists()
    assert registry.listeners == []       # unsubscribed itself


def test_a_failed_ingest_job_stops_waiting_without_deleting(tmp_path, monkeypatch):
    registry = FakeJobRegistry("failed")
    owner = _tap_owner(tmp_path, monkeypatch, registry=registry, keep_raw_tracks=False)
    session = owner.start()
    folder = session.meta.folder
    owner.stop()
    registry.fire()
    assert (folder / "you.wav").exists() and (folder / "others.wav").exists()
    assert registry.listeners == []


def test_keep_raw_tracks_never_subscribes(tmp_path, monkeypatch):
    registry = FakeJobRegistry("done")
    owner = _tap_owner(tmp_path, monkeypatch, registry=registry, keep_raw_tracks=True)
    session = owner.start()
    owner.stop()
    assert registry.listeners == []
    assert (session.meta.folder / "you.wav").exists()


def test_shutdown_unsubscribes_the_registry_listener(tmp_path, monkeypatch):
    registry = FakeJobRegistry("parsing")
    owner = _tap_owner(tmp_path, monkeypatch, registry=registry, keep_raw_tracks=False)
    owner.start()
    owner.stop()
    assert registry.listeners
    owner.shutdown()
    assert registry.listeners == []


def test_cleanup_tolerates_an_unlink_failure(tmp_path, monkeypatch):
    registry = FakeJobRegistry("done")
    owner = _tap_owner(tmp_path, monkeypatch, registry=registry, keep_raw_tracks=False)
    session = owner.start()
    folder = session.meta.folder
    owner.stop()
    real_unlink = Path.unlink

    def flaky(self, *args, **kwargs):
        if self.name == "you.wav":
            raise PermissionError("Operation not permitted")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky)
    registry.fire()
    monkeypatch.undo()
    assert (folder / "you.wav").exists()          # the failure is tolerated
    assert not (folder / "others.wav").exists()   # ... and the pass continues
    assert registry.listeners == []


def test_capture_receives_the_configured_microphone_name(tmp_path, monkeypatch):
    """Q15: the owner persisted and displayed the picked mic but built the
    capture without it, so meetings recorded from the system default."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, _, _ = _owner(tmp_path, mic_device="Shure MV7")
    owner.prepare()
    session = owner.start()
    assert session.capture._mic_device_name == "Shure MV7"
    owner.stop()

    default_owner, _, _ = _owner(tmp_path)
    default_owner.prepare()
    default_session = default_owner.start()
    assert default_session.capture._mic_device_name is None
    default_owner.stop()


def test_recovery_keeps_the_duration_of_an_already_closed_mixed_track(tmp_path):
    """Q16: duration came only from a mixed.wav that itself needed patching.
    Writers close sequentially, so a crash after mixed.wav closed but before
    others.wav did reported a valid 1 s recording as duration 0."""
    folder = tmp_path / "meetings" / "2026-09-04_1300"
    folder.mkdir(parents=True)
    with PlaceholderWavWriter(folder / "mixed.wav") as writer:
        writer.write(b"\x00\x00" * 320 * 50)   # 1 s, header patched on close
    unfinished = PlaceholderWavWriter(folder / "others.wav")
    unfinished.write(b"\x00\x00" * 320 * 10)
    unfinished._handle.flush()                 # crash before this one closed
    (folder / "meeting.json").write_text(json.dumps({"schema": 1, "started_at": "2026-09-04T13:00:00", "ended_at": None, "mode": "call"}))

    assert mo.scan_recoverable(tmp_path / "meetings") == [folder]
    payload = mo.recover_folder(folder)
    assert payload["duration_s"] == pytest.approx(1.0)
    assert not wav_needs_patch(folder / "others.wav")


def test_settings_reject_an_unusable_config_value(tmp_path):
    """Q5: loosely typed config values are validated at the boundary now."""
    from pydantic import ValidationError

    values = {"keep_raw_tracks": "maybe"}

    with pytest.raises(ValidationError) as excinfo:
        mo.MeetingSettings.from_config(lambda s, k, d: values.get(k, d), data_dir=tmp_path)
    assert "keep_raw_tracks" in str(excinfo.value)

    with pytest.raises(ValidationError):
        mo.MeetingSettings(recordings_dir=tmp_path, provider=object())

    # Assignment is validated too: `apply_device_choice` writes these back.
    settings = mo.MeetingSettings(recordings_dir=tmp_path)
    settings.mic_device = "Shure MV7"
    assert settings.mic_device == "Shure MV7"
    with pytest.raises(ValidationError):
        settings.system_source = 5


def test_settings_recordings_dir_goes_through_the_path_validator(tmp_path, monkeypatch):
    seen: list[str] = []
    import tldw_chatbook.Utils.path_validation as pv

    real = pv.validate_path_simple
    monkeypatch.setattr(pv, "validate_path_simple", lambda p, *a, **kw: seen.append(str(p)) or real(p, *a, **kw))
    settings = mo.MeetingSettings.from_config(
        lambda s, k, d: str(tmp_path / "rec") if k == "recordings_dir" else d, data_dir=tmp_path
    )
    assert seen == [str(tmp_path / "rec")]
    assert settings.recordings_dir == (tmp_path / "rec").resolve()


def test_failed_start_closes_writers_and_removes_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    monkeypatch.setattr(FakeDictation, "start_dictation", lambda self, **callbacks: False)
    owner, _, _ = _owner(tmp_path)
    owner.prepare()
    with pytest.raises(RuntimeError):
        owner.start()
    assert owner.session is None
    assert not owner.is_active
    assert list((tmp_path / "meetings").glob("*")) == []


def test_failed_start_closes_the_transcript_sink(tmp_path, monkeypatch):
    """Q4: the sink's JSONL handle is released on every exit from start(),
    not only on the stop() that never happens after a failed start."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    closed: list[int] = []

    class SpySink(mo.LocalMeetingSink):
        def close(self) -> None:
            closed.append(1)
            super().close()

    monkeypatch.setattr(mo, "LocalMeetingSink", SpySink)
    monkeypatch.setattr(FakeDictation, "start_dictation", lambda self, **callbacks: False)
    owner, _, _ = _owner(tmp_path)
    owner.prepare()
    with pytest.raises(RuntimeError):
        owner.start()
    assert closed


def test_raising_start_cleans_up_and_leaves_no_session(tmp_path, monkeypatch):
    """I3: `self.session = session` is assigned BEFORE `session.start()`, and
    only a `False` return used to run the cleanup path. A raising start (a
    dictation service that blows up building its model, say) therefore left
    the owner holding a session that never started, plus an orphan folder.
    """
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))

    def boom(self, **callbacks):
        raise RuntimeError("model failed to load")

    monkeypatch.setattr(FakeDictation, "start_dictation", boom)
    owner, _, _ = _owner(tmp_path)
    owner.prepare()
    with pytest.raises(RuntimeError, match="model failed to load"):
        owner.start()
    assert owner.session is None
    assert not owner.is_active
    assert list((tmp_path / "meetings").glob("*")) == []


def test_raising_capture_constructor_leaks_no_folder(tmp_path, monkeypatch):
    """The capture constructor resolves numpy AFTER the folder and its WAV
    handles exist; a raise there must not leave them behind."""
    import tldw_chatbook.Audio.meeting_capture as mc

    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))

    def boom(**kwargs):
        raise ImportError("numpy is required")

    monkeypatch.setattr(mc, "MeetingCapture", boom)
    owner, _, _ = _owner(tmp_path)
    owner.prepare()
    with pytest.raises(ImportError, match="numpy"):
        owner.start()
    assert owner.session is None
    assert list((tmp_path / "meetings").glob("*")) == []


def test_prepare_reports_a_missing_recorder_as_capture_error(tmp_path, monkeypatch):
    """C1: a numpy-less / backend-less install must say so on the rail rather
    than offer a Start that can only fail. Only "no usable recorder" errors
    qualify -- an ordinary enumeration hiccup still leaves Start available.
    """
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    from tldw_chatbook.Audio.recording_service import AudioRecordingError

    owner, _, _ = _owner(tmp_path)

    def no_recorder(**kwargs):
        raise AudioRecordingError("Audio recording functionality requires NumPy\nfor efficient processing.")

    owner._mic_factory = no_recorder
    assert owner.prepare().capture_error == "Audio recording functionality requires NumPy"

    def flaky(**kwargs):
        raise ValueError("device list temporarily unavailable")

    owner.prepared = None
    owner._mic_factory = flaky
    prepared = owner.prepare()
    assert prepared.capture_error is None and prepared.input_devices == ()


def test_recover_folder_survives_missing_mixed_wav(tmp_path):
    folder = tmp_path / "meetings" / "2026-09-04_1100"
    folder.mkdir(parents=True)
    writer = PlaceholderWavWriter(folder / "others.wav")
    writer.write(b"\x00\x00" * 320 * 20)
    writer._handle.flush()                  # crash: never closed, mixed.wav absent
    (folder / "meeting.json").write_text(json.dumps({"schema": 1, "started_at": "2026-09-04T11:00:00", "ended_at": None, "mode": "call"}))
    assert mo.scan_recoverable(tmp_path / "meetings") == [folder]
    payload = mo.recover_folder(folder)
    assert payload["recovered"] is True
    assert payload["duration_s"] == 0.0
    assert payload["ended_at"]
    assert not wav_needs_patch(folder / "others.wav")


def test_start_waits_for_an_in_flight_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, _, _ = _owner(tmp_path)
    owner.prepare()
    first = owner.start()
    gate = threading.Event()
    real_stop = first.stop

    def slow_stop(reason="user"):
        gate.wait(2.0)
        return real_stop(reason=reason)

    first.stop = slow_stop
    stopper = threading.Thread(target=owner.stop)
    stopper.start()
    time.sleep(0.05)                      # stop() is now blocked inside session.stop under _stop_lock
    started: list = []
    starter = threading.Thread(target=lambda: started.append(owner.start()))
    starter.start()
    time.sleep(0.1)
    assert started == []                  # start() is waiting on _stop_lock
    gate.set()
    stopper.join(2.0); starter.join(2.0)
    assert len(started) == 1 and started[0] is owner.session and owner.is_active
    owner.stop()


def test_stop_does_not_hold_owner_lock_during_session_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, _, _ = _owner(tmp_path)
    owner.prepare()
    owner.start()

    real_stop = owner.session.stop
    acquired: list[bool] = []

    def wrapper(reason="user"):
        def probe():
            got = owner._lock.acquire(timeout=0.5)
            acquired.append(got)
            if got:
                owner._lock.release()

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()
        return real_stop(reason=reason)

    owner.session.stop = wrapper
    owner.stop()
    assert acquired == [True]
    assert not owner.is_active


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
