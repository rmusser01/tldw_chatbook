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


#: What `resolve_engine` is told is installed unless a test says otherwise
#: (31827 task 5). Every verdict in this file has to be a property of the
#: SETTINGS under test, never of what the host venv happens to have: the
#: `_store` fixture below mints its records with the SpeechBrain model id,
#: and which engine resolves decides which model id the owner asks the store
#: for. Without this pin a machine with sherpa-onnx but no torch resolves
#: "onnx" and every stored record reads as `needs_reenrollment`.
_PINNED_INSTALLED = frozenset(mo.DIARIZATION_MODULES)


@pytest.fixture(autouse=True)
def _pin_installed_engine_packages(monkeypatch):
    """Resolve engines against `_PINNED_INSTALLED`, not the host's site-packages.

    A test that passes its OWN `find_spec` (the `resolve_engine` tests) is
    driving resolution deliberately and runs against the real implementation.
    """
    real = mo.resolve_engine

    def pinned(settings, find_spec=None):
        if find_spec is not None:
            return real(settings, find_spec=find_spec)
        return real(settings, find_spec=lambda name, *a: object() if name in _PINNED_INSTALLED else None)

    monkeypatch.setattr(mo, "resolve_engine", pinned)


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


def _owner(
    tmp_path, *, tap_kind="unavailable", job_state=None, registry=None, voiceprint_store=None,
    watchdog_interval_s=0.01, stall_after_s=0.05, **over,
):
    """The owner under test. The watchdog timings are deliberately tiny so a
    test can observe a `mic_lost` stop -- but `FakeCapture.audio_position_s`
    is frozen, so EVERY meeting looks stalled ~50 ms in. A test whose body
    keeps running past that (final review I2: `accept_learning` spends longer
    than that inside the real `VoiceprintStore.merge_sample`'s AES + key
    derivation) has to pass a `stall_after_s` well past its own work, or the
    watchdog stops the meeting out from under its assertions."""
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
        watchdog_interval_s=watchdog_interval_s,
        stall_after_s=stall_after_s,
        voiceprint_store_factory=(lambda: voiceprint_store) if voiceprint_store is not None else None,
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


def test_settings_diarize_mic_channel_round_trip(tmp_path):
    """task 31743: the flag defaults off and round-trips through config."""
    values = {"diarize_mic_channel": True}

    def get(section, key, default):
        assert section == "meetings"
        return values.get(key, default)

    settings = mo.MeetingSettings.from_config(get, data_dir=tmp_path)
    assert settings.diarize_mic_channel is True
    default = mo.MeetingSettings.from_config(lambda s, k, d: d, data_dir=tmp_path)
    assert default.diarize_mic_channel is False


def test_meeting_user_display_name_defaults_to_you_when_unset():
    """No `chat_defaults.user_display_name` at all: the validated getter
    returns the factory default it was handed, same as the real
    `get_chat_defaults_user_display_name`."""
    assert mo.meeting_user_display_name(get_display_name=lambda default: default) == "You"


def test_meeting_user_display_name_defaults_to_you_for_the_factory_default():
    """A configured value that merely ECHOES the shipped factory default
    (task 31746) must not be mistaken for a deliberate choice -- compared
    against `config.py`'s own shipped constant, not a literal re-typed here."""
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    factory_default = DEFAULT_CONFIG_FROM_TOML["chat_defaults"]["user_display_name"]
    assert mo.meeting_user_display_name(get_display_name=lambda default: factory_default) == "You"


def test_meeting_user_display_name_honours_a_real_override():
    assert mo.meeting_user_display_name(get_display_name=lambda default: "Alice") == "Alice"


def test_meeting_user_display_name_treats_whitespace_only_as_unset(tmp_path, monkeypatch):
    """task 31746 review (Important): a whitespace-only configured name must
    not leak through as a literal "   " display name. Reuses Console's own
    validated getter (`get_chat_defaults_user_display_name`), which
    normalizes it away to the neutral "User" -- exercised here against a
    REAL config file, same pattern as
    `test_config_console_defaults.py::test_blank_chat_display_name_falls_back_to_user`."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("[chat_defaults]\nuser_display_name = '   '\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    assert mo.meeting_user_display_name() == "You"


def test_meeting_user_display_name_rejects_a_control_character_name(tmp_path, monkeypatch):
    """A hostile/control-character name is normalized/rejected the same way
    Console does, not passed through verbatim."""
    config_path = tmp_path / "config.toml"
    invalid_value = "unsafe-secret\u202e"
    config_path.write_text(
        f'[chat_defaults]\nuser_display_name = "{invalid_value}"\n', encoding="utf-8"
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    assert mo.meeting_user_display_name() == "You"


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


def test_build_diarizer_server_backend_returns_none_without_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    settings = _settings(tmp_path, live_diarization=True, diarizer_backend="server")
    assert mo.build_diarizer(settings) is None


def test_build_diarizer_construction_failure_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(diarizer_local, "LocalDiarizer", boom)
    settings = _settings(tmp_path, live_diarization=True, diarizer_backend="local")
    assert mo.build_diarizer(settings) is None


def test_live_diarization_active_false_for_server_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    owner, _, _ = _owner(tmp_path, live_diarization=True, diarizer_backend="server")
    prepared = owner.prepare()
    assert prepared.live_diarization_active is False


def test_live_diarization_active_true_for_local_backend_with_deps(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    owner, _, _ = _owner(tmp_path, live_diarization=True, diarizer_backend="local")
    prepared = owner.prepare()
    assert prepared.live_diarization_active is True


# ---- engine resolution (31827 task 5, spec §4) ---------------------------
def test_auto_order_records_the_bakeoff_outcome(tmp_path):
    """Task 9 (31827, spec §10): the bake-off FAILED its separation gate, so
    `auto` still prefers SpeechBrain on an install that has both.

    Pinned as a value, not read off `AUTO_ORDER[0]` like the walk test below:
    flipping the default engine invalidates every enrolled voiceprint on an
    install that has the torch extra (spec §6), so it must be a deliberate
    edit here and in `Docs/User_Guide/meetings.md`, never a drive-by.
    """
    assert mo.AUTO_ORDER == ("speechbrain", "onnx")


def test_resolve_engine_walks_auto_order_by_find_spec(tmp_path):
    s = _settings(tmp_path, live_diarization=True, diarizer_backend="auto")
    have = {"sherpa_onnx", "numpy"}
    fs = lambda name, *a: object() if name in have else None           # noqa: E731
    assert mo.resolve_engine(s, find_spec=fs) == ("onnx", ())          # torch absent -> onnx
    have |= set(mo.ENGINE_MODULES["speechbrain"])
    assert mo.resolve_engine(s, find_spec=fs)[0] == mo.AUTO_ORDER[0]   # both present -> order decides
    assert mo.resolve_engine(s, find_spec=lambda *a: None) == (None, mo.ENGINE_MODULES[mo.AUTO_ORDER[-1]])


def test_resolve_engine_explicit_choice_reports_its_missing_packages(tmp_path):
    s = _settings(tmp_path, live_diarization=True, diarizer_backend="speechbrain")
    assert mo.resolve_engine(s, find_spec=lambda *a: None) == (None, ("torch", "torchaudio", "speechbrain", "sklearn"))
    # An explicit choice never walks on to the other engine, however complete
    # that one's packages are (spec §8: "explicit choice only").
    onnx_only = lambda name, *a: object() if name in ("sherpa_onnx", "numpy") else None    # noqa: E731
    assert mo.resolve_engine(s, find_spec=onnx_only)[0] is None
    assert mo.resolve_engine(s, find_spec=lambda *a: object()) == ("speechbrain", ())
    # "server" is reserved: no engine, and nothing to install for it either.
    server = _settings(tmp_path, diarizer_backend="server")
    assert mo.resolve_engine(server, find_spec=lambda *a: object()) == (None, ())


def test_resolve_engine_treats_an_unimportable_module_as_absent(tmp_path):
    """`find_spec` raises for a namespace-package shadow or a broken install;
    the owner must read that as "not available", never propagate it."""
    def boom(name, *a):
        raise ValueError(f"{name}.__spec__ is not set")

    s = _settings(tmp_path, live_diarization=True, diarizer_backend="onnx")
    assert mo.resolve_engine(s, find_spec=boom) == (None, ("sherpa_onnx", "numpy"))


def test_legacy_local_backend_value_reads_as_auto(tmp_path):
    assert _settings(tmp_path, diarizer_backend="local").diarizer_backend == "auto"
    assert _settings(tmp_path).diarizer_backend == "auto"              # the new default
    with pytest.raises(ValueError):
        _settings(tmp_path, diarizer_backend="bogus")
    with pytest.raises(ValueError):
        _settings(tmp_path, onnx_embedder="bogus")
    assert _settings(tmp_path).onnx_embedder == "titanet_small"
    assert _settings(tmp_path, onnx_models_dir="").onnx_models_dir is None
    assert _settings(tmp_path, onnx_models_dir=str(tmp_path / "m")).onnx_models_dir == tmp_path / "m"
    # Absolute and expanded: the worker SUBPROCESS is handed this path and
    # does not share the app's cwd (review M4).
    assert _settings(tmp_path, onnx_models_dir="~/models").onnx_models_dir == Path.home() / "models"


def test_settings_from_config_reads_the_engine_keys(tmp_path):
    values = {"diarizer_backend": "onnx", "onnx_embedder": "campplus_en",
              "onnx_models_dir": str(tmp_path / "air-gapped")}
    settings = mo.MeetingSettings.from_config(lambda s, k, d: values.get(k, d), data_dir=tmp_path)
    assert settings.diarizer_backend == "onnx" and settings.onnx_embedder == "campplus_en"
    assert settings.onnx_models_dir == tmp_path / "air-gapped"

    default = mo.MeetingSettings.from_config(lambda s, k, d: d, data_dir=tmp_path)
    assert default.diarizer_backend == "auto" and default.onnx_embedder == "titanet_small"
    assert default.onnx_models_dir is None


def test_prepare_reports_engine_missing_and_models_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    monkeypatch.setattr(mo, "resolve_engine", lambda settings, find_spec=None: ("onnx", ()))
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, diarizer_backend="onnx",
        onnx_models_dir=str(tmp_path / "models"),      # empty: nothing fetched yet
    )
    prepared = owner.prepare()
    assert prepared.diarizer_engine == "onnx" and prepared.diarizer_missing == ()
    assert prepared.diarizer_models_ready is False     # fetched at Start, not now...
    assert prepared.live_diarization_active is True    # ... so they never gate live labels

    monkeypatch.setattr(mo, "resolve_engine", lambda settings, find_spec=None: ("speechbrain", ()))
    owner.prepared = None
    prepared = owner.prepare()
    assert prepared.diarizer_engine == "speechbrain" and prepared.diarizer_models_ready is True

    monkeypatch.setattr(mo, "resolve_engine", lambda settings, find_spec=None: (None, ("sherpa_onnx",)))
    owner.prepared = None
    prepared = owner.prepare()
    assert prepared.diarizer_engine is None and prepared.diarizer_missing == ("sherpa_onnx",)
    assert prepared.live_diarization_active is False and prepared.diarizer_models_ready is False


def test_prepare_reports_onnx_models_ready_once_the_files_are_there(tmp_path, monkeypatch):
    """The positive control for the flag above: `models_ready` is presence +
    byte size under `onnx_models_dir`, so an air-gapped directory reads True."""
    from tldw_chatbook.Audio import diarizer_engine_onnx as onnx

    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    monkeypatch.setattr(mo, "resolve_engine", lambda settings, find_spec=None: ("onnx", ()))
    models = tmp_path / "models"
    models.mkdir()
    (models / onnx.SEGMENTATION.file_name).write_bytes(b"\0" * onnx.SEGMENTATION.size)
    (models / onnx.EMBEDDERS["titanet_small"].file_name).write_bytes(
        b"\0" * onnx.EMBEDDERS["titanet_small"].size
    )
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, diarizer_backend="onnx", onnx_models_dir=str(models),
    )
    assert owner.prepare().diarizer_models_ready is True


def test_build_diarizer_passes_the_resolved_engine(tmp_path, monkeypatch):
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    seen: dict = {}
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda **kw: seen.update(kw) or object())
    monkeypatch.setattr(mo, "resolve_engine", lambda settings, find_spec=None: ("onnx", ()))
    settings = _settings(
        tmp_path, live_diarization=True, diarizer_backend="onnx",
        onnx_models_dir=str(tmp_path / "m"),
    )
    assert mo.build_diarizer(settings) is not None
    assert seen["engine"] == "onnx" and seen["embedder"] == "titanet_small"
    assert seen["models_dir_override"] == tmp_path / "m"


def test_no_diarizer_built_when_live_off(tmp_path, monkeypatch):
    owner, _, _ = _owner(tmp_path, live_diarization=False)
    owner.prepare(); session = owner.start()
    assert session._diarizer is None


def test_diarizer_built_when_live_on_and_deps_present(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    built = {}
    monkeypatch.setattr(mo, "build_diarizer", lambda settings, **kw: built.setdefault("d", object()))
    owner, _, _ = _owner(tmp_path, live_diarization=True)
    owner.prepare(); session = owner.start()
    assert session._diarizer is built["d"]


def test_max_speakers_must_be_at_least_one(tmp_path):
    """Qodo Q7: 0/negative silently disabled the Stop pass instead of failing
    at the settings boundary like every other unusable config value."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        _settings(tmp_path, max_speakers=0)
    with pytest.raises(pydantic.ValidationError):
        _settings(tmp_path, max_speakers=-3)
    assert _settings(tmp_path, max_speakers=1).max_speakers == 1


def test_a_live_diarizer_does_not_force_the_offline_ingest_pass(tmp_path, monkeypatch):
    """Qodo Q12: `post_diarize` only asks Library ingest for a SECOND, offline
    diarization of mixed.wav that knows nothing of the live ids or renames.
    The live backend's own Stop pass is driven by the session's `_diarizer`,
    so building one must not override an explicit `post_diarize = false`."""
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    monkeypatch.setattr(mo, "build_diarizer", lambda settings, **kw: object())
    owner, _, _ = _owner(tmp_path, live_diarization=True, post_diarize=False)
    owner.prepare()
    session = owner.start()
    assert owner.local_sink.post_diarize is False   # the user's setting stands
    assert session._diarizer is not None            # ... the Stop pass still runs
    owner.stop()


def test_live_on_missing_deps_falls_back_to_coarse(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ("torch",))
    # No engine resolves at all (31827): neither stack is installed.
    monkeypatch.setattr(mo, "resolve_engine", lambda settings, find_spec=None: (None, ("torch",)))
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


def test_start_stamps_the_configured_display_name_onto_meta(tmp_path, monkeypatch):
    """task 31746: the owner stamps the SAME name the live session shows onto
    `meta.user_display_name`, so an after-the-fact render (transcript.md, the
    Library rename view) agrees with what the user saw while recording."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    monkeypatch.setattr(mo, "meeting_user_display_name", lambda **kw: "Alice")
    owner, _, _ = _owner(tmp_path)
    owner.prepare()
    session = owner.start()
    assert session.meta.user_display_name == "Alice"
    owner.stop()


def test_start_stamps_diarize_mic_channel_onto_meta(tmp_path, monkeypatch):
    """task 31743: the owner stamps the configured flag onto `meta` so the
    session/render path can read a single source of truth."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, _, _ = _owner(tmp_path, diarize_mic_channel=True)
    owner.prepare()
    session = owner.start()
    assert session.meta.diarize_mic_channel is True
    owner.stop()


def test_start_leaves_diarize_mic_channel_off_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, _, _ = _owner(tmp_path)
    owner.prepare()
    session = owner.start()
    assert session.meta.diarize_mic_channel is False
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
    inside = threading.Event()
    real_stop = first.stop

    def slow_stop(reason="user"):
        inside.set()
        gate.wait(2.0)
        return real_stop(reason=reason)

    first.stop = slow_stop
    stopper = threading.Thread(target=owner.stop)
    stopper.start()
    # An Event, not a sleep (final review Minor 10): this is the ORDERING the
    # test is about -- stop() has to be inside session.stop, holding
    # _stop_lock, before the Start below can mean anything.
    assert inside.wait(2.0)
    started: list = []
    starter = threading.Thread(target=lambda: started.append(owner.start()))
    starter.start()
    # The one wait that stays: proving a thread has NOT proceeded needs a
    # window, not an event. It is only ever a false PASS if it is too short.
    starter.join(0.1)
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


# ---- 31826 task 4: voiceprint match, learning offer, explicit enrollment ----

@pytest.fixture
def captured_lines():
    """Collect every loguru message emitted during the test.

    `caplog` does not see loguru's own sink -- mirrors the fixture of the same
    name in `Tests/Audio/test_meeting_diarization_session.py`.
    """
    from loguru import logger as loguru_logger

    lines: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: lines.append(message.record["message"]),
        level="TRACE", format="{message}", diagnose=False,
    )
    try:
        yield lines
    finally:
        loguru_logger.remove(sink_id)


class FakeKeys:
    """Injected key provider (never touches the real keyring)."""

    mode = "keyring"

    def __init__(self, key="k" * 32, delay=0.0):
        self.key, self.delay, self.reads = key, delay, 0

    def get_or_create(self):
        return self.key

    def get(self, timeout_s):
        self.reads += 1
        if self.delay:
            time.sleep(self.delay)
        return self.key


def _store(tmp_path, *, keys=None, centroid=(0.6, 0.8), enrolled=True, meetings=1, model_id=None):
    """A real `VoiceprintStore` on a temp path with an injected key provider.

    `model_id` defaults to the SpeechBrain engine's -- the engine
    `_pin_installed_engine_packages` resolves -- so a record minted here is
    the ACTIVE engine's unless a test deliberately says otherwise (31827).
    """
    from tldw_chatbook.Audio import voiceprint as vp
    from tldw_chatbook.Audio.diarizer_worker import MODEL_ID

    store = vp.VoiceprintStore(tmp_path / "voiceprint.json", keys or FakeKeys())
    if enrolled:
        store.save(vp.Voiceprint(
            model_id=model_id or MODEL_ID, centroid=vp.unit_normalise(centroid), sample_count=10.0,
            meetings_contributed=meetings, created_at="2026-09-06T00:00:00",
            updated_at="2026-09-06T00:00:00", threshold_used=0.2,
        ))
    return store


class FakeBackend:
    """Stands in for `LocalDiarizer` (its whole task-4 surface)."""

    def __init__(self, *, voiceprint=None, centroid=(0.0, 1.0), seconds=5.0, **kwargs):
        self.voiceprint = list(voiceprint) if voiceprint else None
        self.kwargs = kwargs
        self.centroid = list(centroid)
        self.seconds = seconds
        self.closed = 0
        self.exports: list[str] = []
        self.enrolled: list[tuple[bytes, int]] = []
        self.self_cluster_id = None
        self.stop_self = None
        self.self_candidates_seen = 0

    def assign(self, pcm, sample_rate, seq):
        return None

    def diarize(self, wav_path, start_s, end_s):
        return []

    def pin(self, cluster_id):
        return None

    def centroids(self):
        return {}

    def wait_ready(self, timeout):
        return True

    def export_centroid(self, cluster_id):
        self.exports.append(cluster_id)
        return list(self.centroid), self.seconds

    def enroll_from_pcm(self, pcm, sample_rate):
        self.enrolled.append((pcm, sample_rate))
        return list(self.centroid), self.seconds

    def close(self):
        self.closed += 1


def _backend_spy(monkeypatch, backend=None, *, engine="speechbrain"):
    """Replace `build_diarizer` with a spy; returns (backend, seen kwargs).

    Also declares the diarization stack installed and pins the RESOLVED
    engine (ruling 11). A spied backend means "this run HAS live speaker
    labels", and since final review I1 the voice gate reads
    `prepared.live_diarization_active` -- computed from `resolve_engine()`
    (31827 task 5), i.e. from what happens to be in the test machine's venv.
    `diarization_requirements` stays pinned too: the Library's offline ingest
    flag still reads it.

    `engine` also decides the model id the owner asks the store for, so it
    defaults to the engine `_store` mints its records with; a test about the
    ONNX engine passes `engine="onnx"` (and a matching store).
    """
    seen: dict = {}
    made = backend if backend is not None else FakeBackend()

    def build(settings, **kwargs):
        # What the real `build_diarizer` would forward to `LocalDiarizer`,
        # so a spied test still pins the engine/embedder/models-dir kwargs.
        seen.update(
            engine=engine, embedder=settings.onnx_embedder,
            models_dir_override=settings.onnx_models_dir, **kwargs,
        )
        made.voiceprint = list(kwargs.get("voiceprint") or []) or None
        return made

    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    monkeypatch.setattr(mo, "resolve_engine", lambda settings, find_spec=None: (engine, ()))
    monkeypatch.setattr(mo, "build_diarizer", build)
    return made, seen


def _write_wav(path: Path, pcm: bytes, sample_rate: int = 16000) -> None:
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def test_voice_settings_round_trip_and_bounds(tmp_path):
    values = {
        "voice_match": False, "voice_match_threshold": 0.35,
        "voice_match_min_seconds": 2.0, "voice_learn_offer": False,
    }
    settings = mo.MeetingSettings.from_config(lambda s, k, d: values.get(k, d), data_dir=tmp_path)
    assert settings.voice_match is False and settings.voice_learn_offer is False
    assert settings.voice_match_threshold == 0.35 and settings.voice_match_min_seconds == 2.0

    default = mo.MeetingSettings.from_config(lambda s, k, d: d, data_dir=tmp_path)
    assert default.voice_match is True and default.voice_learn_offer is True
    assert default.voice_match_threshold == 0.2 and default.voice_match_min_seconds == 4.0

    import pydantic

    for bad in ({"voice_match_threshold": 0.0}, {"voice_match_threshold": 1.5}, {"voice_match_min_seconds": -1.0}):
        with pytest.raises(pydantic.ValidationError):
            _settings(tmp_path, **bad)


def test_room_mode_builds_the_diarizer_with_the_stored_vector(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    backend, seen = _backend_spy(monkeypatch)
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path, centroid=(0.6, 0.8)),
    )
    prepared = owner.prepare()
    assert prepared.voice_match.state == "on" and prepared.voice_match.reason is None
    session = owner.start()
    assert session._diarizer is backend and backend.voiceprint == [0.6, 0.8]
    assert seen["voiceprint"] == [0.6, 0.8]
    owner.stop()


def test_build_diarizer_without_a_vector_enrolls_nothing(tmp_path, monkeypatch):
    """Review M5: the "no vector" half of the real path, not just the fake."""
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    seen: dict = {}
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda **kw: seen.update(kw) or object())
    mo.build_diarizer(_settings(tmp_path, live_diarization=True))
    assert seen["voiceprint"] is None


def test_build_diarizer_forwards_the_vector_and_the_match_settings(tmp_path, monkeypatch):
    """The thresholds are the worker's match gate -- they must travel with
    the vector, not stay behind in the settings object."""
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    seen: dict = {}
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda **kw: seen.update(kw) or object())
    settings = _settings(
        tmp_path, live_diarization=True, voice_match_threshold=0.3, voice_match_min_seconds=6.0,
    )
    mo.build_diarizer(settings, voiceprint=[0.6, 0.8])
    assert seen["voiceprint"] == [0.6, 0.8]
    assert seen["match_threshold"] == 0.3 and seen["match_min_seconds"] == 6.0
    assert seen["max_speakers"] == settings.max_speakers


def test_no_stored_voiceprint_reports_no_voiceprint_and_passes_no_vector(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    backend, _ = _backend_spy(monkeypatch)
    owner, _, _ = _owner(tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path, enrolled=False))
    assert owner.prepare().voice_match.reason == "no_voiceprint"
    owner.start()
    assert backend.voiceprint is None
    owner.stop()


def test_voice_match_off_never_reads_the_store(tmp_path, monkeypatch):
    """Matching off (and no learning offer) means the key is never touched --
    no Keychain prompt for a feature the user turned off."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    _backend_spy(monkeypatch)
    keys = FakeKeys()
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, voice_match=False, voice_learn_offer=False,
        voiceprint_store=_store(tmp_path, keys=keys),
    )
    reads_after_save = keys.reads
    assert owner.prepare().voice_match.reason == "disabled"
    owner.start()
    owner.stop()
    assert keys.reads == reads_after_save        # no key read at all


def test_plain_call_mode_never_matches(tmp_path, monkeypatch):
    """Spec §3.4: in plain call mode the mic channel already IS the user, so
    remote clusters are never compared against the voiceprint."""
    _backend_spy(monkeypatch)
    keys = FakeKeys()
    owner = _tap_owner(
        tmp_path, monkeypatch, live_diarization=True, voiceprint_store=_store(tmp_path, keys=keys),
    )
    reads_after_save = keys.reads
    assert owner.prepare().voice_match.reason == "plain_call_mode"
    session = owner.start()
    assert getattr(session._diarizer, "voiceprint", None) is None
    assert keys.reads == reads_after_save        # no Keychain prompt either
    owner.stop()


def test_hybrid_call_mode_does_match(tmp_path, monkeypatch):
    """`diarize_mic_channel` makes the mic one cluster among many again, so
    matching is back on in call mode."""
    backend, _ = _backend_spy(monkeypatch)
    owner = _tap_owner(
        tmp_path, monkeypatch, live_diarization=True, diarize_mic_channel=True,
        voiceprint_store=_store(tmp_path, centroid=(0.6, 0.8)),
    )
    assert owner.prepare().voice_match.state == "on"
    owner.start()
    assert backend.voiceprint == [0.6, 0.8]
    owner.stop()


def test_locked_keyring_never_delays_start(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    _backend_spy(monkeypatch)
    store = _store(tmp_path, keys=FakeKeys(delay=30.0))
    owner, _, _ = _owner(tmp_path, live_diarization=True, voiceprint_store=store)
    started = time.monotonic()
    owner.start()
    elapsed = time.monotonic() - started
    # Real headroom (review M7): the bound is 2x the join, so a slow host
    # cannot make this flake -- only a broken join would blow it.
    assert elapsed < 2 * mo.VOICEPRINT_LOAD_TIMEOUT_S
    assert owner.voice_match.reason == "keyring_locked"
    assert owner.prepare().voice_match.reason == "keyring_locked"   # sticky for the rail
    owner.stop()


def test_a_locked_keyring_is_retried_by_the_next_start(tmp_path, monkeypatch):
    """Review I5: the design's own scenario is "the prompt appeared, the join
    timed out, the user then approved it" -- the next meeting must try again
    instead of staying off until an app restart."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    backend, _ = _backend_spy(monkeypatch)
    keys = FakeKeys(delay=30.0)
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path, keys=keys, centroid=(0.6, 0.8)),
    )
    owner.start()
    assert owner.voice_match.reason == "keyring_locked"
    owner.stop()

    keys.delay = 0.0                       # the user approved the prompt
    owner.start()
    assert owner.voice_match.state == "on" and backend.voiceprint == [0.6, 0.8]
    owner.stop()


def test_invalidate_voiceprint_never_waits_for_an_in_flight_load(tmp_path, monkeypatch):
    """Final review Minor 7: `_voice_load` is read and written from five
    threads. It is guarded now -- but only for the POINTER SWAP: hold the lock
    across the (bounded, 1.5 s) read and a Delete or Import on the UI thread
    stalls behind a meeting Start."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    keys = FakeKeys(delay=5.0)
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path, keys=keys),
    )
    reads_after_save = keys.reads
    loader = threading.Thread(target=owner._load_voiceprint, daemon=True)
    loader.start()
    while keys.reads == reads_after_save:          # the store read is in flight
        time.sleep(0.005)

    started = time.monotonic()
    owner.invalidate_voiceprint()
    # Generous: the read this is racing is pinned at 1.5 s by the join above.
    assert time.monotonic() - started < 0.5
    loader.join(5.0)


def test_a_terminal_verdict_is_not_retried_every_start(tmp_path, monkeypatch):
    """... but a record that simply is not there stays cached: no repeated
    reads, and `invalidate_voiceprint` is the escape."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    _backend_spy(monkeypatch)
    keys = FakeKeys()
    store = _store(tmp_path, keys=keys, enrolled=False)
    owner, _, _ = _owner(tmp_path, live_diarization=True, voiceprint_store=store)
    owner.start()
    owner.stop()
    assert owner.voice_match.reason == "no_voiceprint"
    owner.start()
    owner.stop()
    assert owner.voice_match.reason == "no_voiceprint"


def test_prepare_never_reads_the_key(tmp_path, monkeypatch):
    """Controller ruling / review I2: `prepare()` runs at screen mount and
    after every device change, so it may only STAT the store -- the decrypt
    (and any Keychain prompt) belongs to Start, on its bounded thread."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    _backend_spy(monkeypatch)
    keys = FakeKeys()
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path, keys=keys),
    )
    reads_after_save = keys.reads

    prepared = owner.prepare()
    assert keys.reads == reads_after_save                 # zero key reads
    assert prepared.voice_match.state == "on"             # a record exists (unverified)

    owner.prepared = None
    owner.prepare()
    assert keys.reads == reads_after_save                 # still zero, every time

    owner.start()                                          # ... and Start does the read
    assert keys.reads == reads_after_save + 1
    assert owner.voice_match.state == "on" and owner.voice_match.reason is None
    owner.stop()


def test_prepare_reports_no_voiceprint_from_a_stat_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    keys = FakeKeys()
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path, keys=keys, enrolled=False),
    )
    assert owner.prepare().voice_match.reason == "no_voiceprint"
    assert keys.reads == 0


def test_voice_match_is_off_without_live_speaker_labels(tmp_path, monkeypatch):
    """Final review I1: `live_diarization` is off by DEFAULT, and with it off
    `build_diarizer` returns None -- no clusters, so nothing can ever be
    matched. The rail used to say "Voice match: on" for that config."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    keys = FakeKeys()
    owner, _, _ = _owner(
        tmp_path, live_diarization=False,
        voiceprint_store=_store(tmp_path, keys=keys, centroid=(0.6, 0.8)),
    )
    reads_after_save = keys.reads

    prepared = owner.prepare()                                  # room mode (no tap)
    assert prepared.voice_match == mo.VoiceMatchState("off", "live_labels_off")
    # The gate has to agree with the thing it models: same settings, no backend.
    assert mo.build_diarizer(owner.settings) is None
    assert prepared.live_diarization_active is False

    owner.start()
    assert owner.voice_match == mo.VoiceMatchState("off", "live_labels_off")
    assert keys.reads == reads_after_save     # gated before the store: no Keychain prompt
    owner.stop()


def test_voice_match_is_on_again_once_live_labels_are_enabled(tmp_path, monkeypatch):
    """The negative control for the gate above: the ONLY thing changing here
    is `live_diarization`, and the same store then reports "on"."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    _backend_spy(monkeypatch)
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path, centroid=(0.6, 0.8)),
    )
    assert owner.prepare().voice_match == mo.VoiceMatchState("on", None)


# ---- per-engine model ids (31827 task 5, spec §6) ------------------------
class _RecordingStore:
    """A voiceprint store that records the model id it is asked for."""

    def __init__(self, result):
        self.result = result
        self.expected: list[str | None] = []

    def exists(self):
        return True

    def load(self, expected_model_id=None, timeout_s=1.5):
        self.expected.append(expected_model_id)
        # Honours `expected_model_id` like the real store, so it cannot lie
        # about the very thing these tests pin: a record from another engine
        # reads as `needs_reenrollment`, never as a usable vector.
        record = getattr(self.result, "voiceprint", None)
        if record is not None and expected_model_id not in (None, record.model_id):
            from tldw_chatbook.Audio import voiceprint as vp

            return vp.LoadResult(
                voiceprint=None, reason="needs_reenrollment", mode="keyring",
                stored_model_id=record.model_id,
            )
        return self.result


def _reenrollment_store(stored_model_id):
    """A store whose record was written by ANOTHER engine."""
    from tldw_chatbook.Audio import voiceprint as vp

    return _RecordingStore(vp.LoadResult(
        voiceprint=None, reason="needs_reenrollment", mode="keyring",
        stored_model_id=stored_model_id,
    ))


def test_start_stamps_engine_and_model_id_onto_meta(tmp_path, monkeypatch):
    from tldw_chatbook.Audio.diarizer_local import model_id_for

    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    _backend_spy(monkeypatch, engine="onnx")
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, diarizer_backend="onnx",
        voiceprint_store=_store(tmp_path, enrolled=False),
    )
    owner.prepare()
    session = owner.start()
    assert session.meta.diarizer_engine == "onnx"
    assert session.meta.diarizer_model_id == model_id_for("onnx")
    folder = Path(session.meta.folder)
    owner.stop()
    payload = json.loads((folder / "meeting.json").read_text())
    assert payload["diarizer_engine"] == "onnx"
    assert payload["diarizer_model_id"] == model_id_for("onnx")


def test_start_stamps_no_engine_when_none_resolves(tmp_path, monkeypatch):
    """The negative control: a meeting recorded with coarse labels says so
    rather than claiming an engine that never ran (spec §4 "Stamping")."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    monkeypatch.setattr(mo, "resolve_engine", lambda settings, find_spec=None: (None, ("torch",)))
    owner, _, _ = _owner(tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path, enrolled=False))
    owner.prepare()
    session = owner.start()
    assert session.meta.diarizer_engine is None and session.meta.diarizer_model_id is None
    owner.stop()


def test_voiceprint_load_uses_the_resolved_engines_model_id(tmp_path, monkeypatch):
    from tldw_chatbook.Audio import voiceprint as vp
    from tldw_chatbook.Audio.diarizer_local import model_id_for

    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    _backend_spy(monkeypatch, engine="onnx")
    record = vp.Voiceprint(
        model_id=model_id_for("onnx"), centroid=vp.unit_normalise((0.6, 0.8)), sample_count=10.0,
        meetings_contributed=1, created_at="2026-09-06T00:00:00",
        updated_at="2026-09-06T00:00:00", threshold_used=0.2,
    )
    store = _RecordingStore(vp.LoadResult(voiceprint=record, reason=None, mode="keyring"))
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, diarizer_backend="onnx", voiceprint_store=store,
    )
    owner.prepare()
    owner.start()
    assert store.expected == [model_id_for("onnx")]     # never the SpeechBrain id
    assert owner.voice_match == mo.VoiceMatchState("on", None)
    owner.stop()


def test_switch_copy_names_the_engines(tmp_path, monkeypatch):
    from tldw_chatbook.Audio.diarizer_worker import MODEL_ID

    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    _backend_spy(monkeypatch, engine="onnx")
    store = _reenrollment_store(MODEL_ID)               # recorded with SpeechBrain
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, diarizer_backend="onnx", voiceprint_store=store,
    )
    owner.prepare()
    owner.start()
    assert owner.voice_match.state == "off" and owner.voice_match.reason == "needs_reenrollment"
    assert owner.voice_match.detail == "Voiceprint was recorded with SpeechBrain; the active engine is ONNX."
    owner.stop()


def test_no_switch_copy_for_an_unrecognisable_stored_model_id(tmp_path, monkeypatch):
    """Only the two known engines get named; anything else leaves the reason
    to speak for itself rather than putting a raw model id on screen."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    _backend_spy(monkeypatch, engine="onnx")
    store = _reenrollment_store("some-other-vendor/model@abc")
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, diarizer_backend="onnx", voiceprint_store=store,
    )
    owner.prepare()
    owner.start()
    assert owner.voice_match.reason == "needs_reenrollment" and owner.voice_match.detail is None
    owner.stop()


def test_a_changed_engine_drops_the_cached_voiceprint_load(tmp_path, monkeypatch):
    """Review M6: the cached load was verified against the PREVIOUS engine's
    model id. A run where the resolved engine changes (the user installed or
    removed the torch extra) must re-read under the new id -- otherwise the
    stale "on" vector, from a different vector space, is handed to the new
    backend and every comparison against it is noise."""
    from tldw_chatbook.Audio import voiceprint as vp
    from tldw_chatbook.Audio.diarizer_local import model_id_for

    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())
    engine = ["speechbrain"]
    monkeypatch.setattr(mo, "resolve_engine", lambda settings, find_spec=None: (engine[0], ()))
    monkeypatch.setattr(mo, "build_diarizer", lambda settings, **kw: FakeBackend())
    record = vp.Voiceprint(
        model_id=model_id_for("speechbrain"), centroid=vp.unit_normalise((0.6, 0.8)),
        sample_count=10.0, meetings_contributed=1, created_at="2026-09-06T00:00:00",
        updated_at="2026-09-06T00:00:00", threshold_used=0.2,
    )
    store = _RecordingStore(vp.LoadResult(voiceprint=record, reason=None, mode="keyring"))
    owner, _, _ = _owner(tmp_path, live_diarization=True, voiceprint_store=store)
    owner.prepare()
    owner.start()
    assert store.expected == [model_id_for("speechbrain")]
    assert owner.voice_match == mo.VoiceMatchState("on", None)
    owner.stop()

    engine[0] = "onnx"                       # the packages changed under this run
    owner.start()
    assert store.expected == [model_id_for("speechbrain"), model_id_for("onnx")]
    assert owner.voice_match.reason == "needs_reenrollment"     # not the stale "on"
    owner.stop()


def test_no_learning_offer_when_the_stored_voiceprint_is_from_another_engine(tmp_path, monkeypatch):
    """Spec §6: an offer merges into the stored print, and a print from
    another engine is in a different vector space -- Enroll replaces it."""
    from tldw_chatbook.Audio.diarizer_worker import MODEL_ID

    _backend_spy(monkeypatch, engine="onnx")
    store = _reenrollment_store(MODEL_ID)
    owner = _tap_owner(
        tmp_path, monkeypatch, live_diarization=True, diarizer_backend="onnx", voiceprint_store=store,
    )
    session = owner.start()
    folder = Path(session.meta.folder)
    result = owner.stop()
    _write_wav(folder / "you.wav", b"\x01\x02" * 1600)   # the qualifying plain-call sample
    assert owner.learning_offer(result) is None


def test_learning_offer_still_made_for_the_active_engines_voiceprint(tmp_path, monkeypatch):
    """The negative control for the gate above: same meeting shape, same
    engine on both sides -- the offer stands."""
    from tldw_chatbook.Audio.diarizer_local import model_id_for

    _backend_spy(monkeypatch, engine="onnx")
    store = _store(tmp_path, model_id=model_id_for("onnx"))
    owner = _tap_owner(
        tmp_path, monkeypatch, live_diarization=True, diarizer_backend="onnx", voiceprint_store=store,
    )
    session = owner.start()
    folder = Path(session.meta.folder)
    result = owner.stop()
    _write_wav(folder / "you.wav", b"\x01\x02" * 1600)
    offer = owner.learning_offer(result)
    assert offer is not None and offer.kind == "mic_channel"


def test_diarizer_status_reads_the_live_backends_warm_up(tmp_path, monkeypatch):
    """The rail's per-second tick (spec §3): the LIVE meeting's backend first,
    then the worker retained for a learning offer, then None."""
    from tldw_chatbook.Audio.diarizer_local import model_id_for

    backend = FakeBackend()
    backend.warmup_status = "downloading 12 / 35 MB"
    _backend_spy(monkeypatch, backend, engine="onnx")
    owner = _tap_owner(                                     # call mode: an offer qualifies
        tmp_path, monkeypatch, live_diarization=True, diarizer_backend="onnx",
        voiceprint_store=_store(tmp_path, model_id=model_id_for("onnx")),
    )
    assert owner.diarizer_status() is None                  # nothing built yet
    owner.start()
    assert owner.diarizer_status() == "downloading 12 / 35 MB"
    backend.warmup_status = "ready"
    owner.stop()
    assert owner.diarizer_status() == "ready"               # retained for the offer
    owner.dismiss_learning()
    assert owner.diarizer_status() is None


def test_owner_never_imports_an_engine_package(tmp_path):
    """Spec §4: the owner decides availability by `find_spec` alone -- an
    `import torch` here would put the 2 GB stack in the UI process."""
    src = Path(mo.__file__).read_text()
    assert "import sherpa_onnx" not in src and "import torch" not in src


def test_optional_deps_lists_the_onnx_diarization_feature():
    from tldw_chatbook.Utils.optional_deps import OPTIONAL_FEATURES

    feature = OPTIONAL_FEATURES["diarization_onnx"]
    assert feature.package_dependencies == ("sherpa_onnx", "numpy")
    assert feature.label == "Speaker diarization (ONNX)"


def test_prepare_reports_store_unavailable_without_raising(tmp_path, monkeypatch):
    """Review M2: a store that cannot be opened at all is a different repair
    from a record that would not decrypt."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    monkeypatch.setattr(mo, "diarization_requirements", lambda: ())

    def broken_store():
        raise RuntimeError("keyfile permissions")

    owner, _, _ = _owner(tmp_path, live_diarization=True)
    owner._store_factory = broken_store
    assert owner.prepare().voice_match.reason == "store_unavailable"


def test_a_key_file_with_loose_permissions_reports_store_unavailable(tmp_path, monkeypatch):
    """Task 6 review M8: the key provider returned None for a key file the
    user (or an installer) chmod'd wide, which `_load_voiceprint` read as
    "keyring locked" -- so a key-FILE install was told to unlock a keyring it
    does not have. The repair is a chmod, i.e. `store_unavailable`."""
    from tldw_chatbook.Audio import voiceprint as vp

    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    backend, _ = _backend_spy(monkeypatch)
    keyfile = tmp_path / "voiceprint.key"
    store = _store(tmp_path, keys=vp.KeyfileKeyProvider(keyfile))    # mints it at 0o600
    keyfile.chmod(0o644)

    owner, _, _ = _owner(tmp_path, live_diarization=True, voiceprint_store=store)
    owner.start()
    assert owner.voice_match == mo.VoiceMatchState("off", "store_unavailable")
    assert backend.voiceprint is None
    owner.stop()


def test_start_reports_store_unavailable_when_the_store_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    backend, _ = _backend_spy(monkeypatch)

    class Boom:
        def exists(self):
            return True

        def load(self, expected_model_id=None, timeout_s=1.5):
            raise RuntimeError("no key material")

    owner, _, _ = _owner(tmp_path, live_diarization=True)
    owner._store_factory = Boom
    owner.start()
    assert owner.voice_match == mo.VoiceMatchState("off", "store_unavailable")
    assert backend.voiceprint is None
    owner.stop()


def _stopped_owner(tmp_path, monkeypatch, *, backend=None, matched="S1", **over):
    """Run one whole meeting and return (owner, result, backend)."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    made, _ = _backend_spy(monkeypatch, backend)
    owner, _, _ = _owner(tmp_path, live_diarization=True, **over)
    owner.prepare()
    session = owner.start()
    if matched:
        session.meta.matched_self = matched
    result = owner.stop()
    return owner, result, made


def test_learning_offer_once_and_merge_on_accept(tmp_path, monkeypatch):
    store = _store(tmp_path, meetings=1)
    owner, result, backend = _stopped_owner(tmp_path, monkeypatch, voiceprint_store=store)

    offer = owner.learning_offer(result)
    assert offer is not None and offer.kind == "matched_cluster" and offer.cluster_id == "S1"
    assert owner.learning_offer(result) is None          # at most one per meeting
    assert backend.closed == 0                            # kept alive for the export

    assert owner.accept_learning(offer) is True
    assert backend.exports == ["S1"]
    assert store.load().voiceprint.meetings_contributed == 2
    assert backend.closed == 1                            # ... and closed after it


def test_no_offer_for_an_overridden_match(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    backend, _ = _backend_spy(monkeypatch)
    owner, _, _ = _owner(tmp_path, live_diarization=True, voiceprint_store=store)
    owner.prepare()
    session = owner.start()
    session.meta.matched_self = "S1"
    session.meta.matched_self_overridden = True
    result = owner.stop()
    assert owner.learning_offer(result) is None
    assert backend.closed == 1                            # nothing to offer -> released


def test_no_offer_without_a_stored_voiceprint(tmp_path, monkeypatch):
    """Learning MERGES into an enrolled voiceprint; with none stored there is
    nothing to improve (that is what explicit enrollment is for)."""
    owner, result, backend = _stopped_owner(
        tmp_path, monkeypatch, voiceprint_store=_store(tmp_path, enrolled=False),
    )
    assert owner.learning_offer(result) is None
    assert backend.closed == 1


def test_learning_offer_off_never_retains_the_worker(tmp_path, monkeypatch):
    owner, result, backend = _stopped_owner(
        tmp_path, monkeypatch, voiceprint_store=_store(tmp_path), voice_learn_offer=False,
    )
    assert owner.learning_offer(result) is None
    assert backend.closed == 1


def test_learning_in_plain_call_mode_embeds_you_wav(tmp_path, monkeypatch):
    store = _store(tmp_path)
    backend, _ = _backend_spy(monkeypatch)
    owner = _tap_owner(tmp_path, monkeypatch, live_diarization=True, voiceprint_store=store)
    session = owner.start()
    folder = Path(session.meta.folder)
    result = owner.stop()

    pcm = b"\x01\x02" * 1600
    _write_wav(folder / "you.wav", pcm)                   # the mic track, post-finalise
    offer = owner.learning_offer(result)
    assert offer is not None and offer.kind == "mic_channel" and offer.cluster_id is None

    assert owner.accept_learning(offer) is True
    assert backend.enrolled == [(pcm, 16000)]
    assert store.load().voiceprint.meetings_contributed == 2
    assert backend.closed == 1


def test_accepting_an_offer_reports_the_warm_up(tmp_path, monkeypatch):
    """Final review I3: the mic-channel offer's common configuration has NO
    live diarizer, so accepting spawns a fresh worker and blocks on
    `wait_ready` -- up to 120 s, model download included. Without a progress
    ping the Voice row says "Learning from this meeting…" for two minutes and
    reads as a hang."""
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    store = _store(tmp_path)
    monkeypatch.setattr(mo, "build_diarizer", lambda settings, **kw: None)   # the default config
    owner = _tap_owner(tmp_path, monkeypatch, live_diarization=True, voiceprint_store=store)
    session = owner.start()
    folder = Path(session.meta.folder)
    result = owner.stop()
    _write_wav(folder / "you.wav", b"\x01\x02" * 1600)
    offer = owner.learning_offer(result)
    assert offer is not None and offer.kind == "mic_channel"
    assert owner._retained_diarizer is None          # nothing warm to borrow

    spawned = FakeBackend()
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: spawned)
    seen: list[str] = []
    assert owner.accept_learning(offer, progress=seen.append) is True
    assert seen == ["warming up"]                    # the same static word enrollment uses
    assert spawned.closed == 1


def test_accepting_an_offer_without_a_progress_callback_still_works(tmp_path, monkeypatch):
    """`progress` is optional: every existing caller passes nothing."""
    store = _store(tmp_path)
    owner, result, backend = _stopped_owner(tmp_path, monkeypatch, voiceprint_store=store)
    offer = owner.learning_offer(result)
    assert owner.accept_learning(offer) is True


def test_a_raising_session_stop_still_releases_the_meetings_worker(tmp_path, monkeypatch):
    """Final review Minor 5: `_settle_offer` ran after `session.stop()`, so a
    raise there orphaned the live worker for the app's lifetime -- nothing
    else clears `_session_diarizer`, and the next `start()` just overwrote
    the pointer."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    backend, _ = _backend_spy(monkeypatch)
    owner, _, _ = _owner(tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path))
    owner.prepare()
    session = owner.start()
    real_stop = session.stop
    session.stop = lambda reason="user": (_ for _ in ()).throw(RuntimeError("sink exploded"))

    with pytest.raises(RuntimeError, match="sink exploded"):
        owner.stop()
    assert backend.closed == 1 and owner._session_diarizer is None

    session.stop = real_stop
    owner.stop()


def test_decline_and_dismiss_release_the_worker(tmp_path, monkeypatch):
    store = _store(tmp_path)
    owner, result, backend = _stopped_owner(tmp_path, monkeypatch, voiceprint_store=store)
    offer = owner.learning_offer(result)
    owner.decline_learning(offer)
    assert backend.closed == 1
    assert store.load().voiceprint.meetings_contributed == 1     # nothing kept
    owner.decline_learning(offer)                                # no-op, no raise
    owner.dismiss_learning()                                     # no-op, no raise
    assert backend.closed == 1
    assert owner.accept_learning(offer) is False                 # the offer is gone


def test_an_unanswered_offer_lapses_when_the_screen_dismisses_it(tmp_path, monkeypatch):
    owner, result, backend = _stopped_owner(tmp_path, monkeypatch, voiceprint_store=_store(tmp_path))
    owner.learning_offer(result)
    owner.dismiss_learning()
    assert backend.closed == 1


def test_a_new_meeting_lapses_the_previous_offer(tmp_path, monkeypatch):
    owner, result, backend = _stopped_owner(tmp_path, monkeypatch, voiceprint_store=_store(tmp_path))
    owner.learning_offer(result)
    owner.start()                                         # next meeting: the old worker goes
    assert backend.closed == 1
    owner.stop()


def test_shutdown_lapses_the_offer_and_closes_the_worker(tmp_path, monkeypatch):
    owner, result, backend = _stopped_owner(tmp_path, monkeypatch, voiceprint_store=_store(tmp_path))
    owner.learning_offer(result)
    owner.shutdown()
    assert backend.closed == 1


def test_accept_reports_failure_without_raising_when_the_export_fails(tmp_path, monkeypatch):
    store = _store(tmp_path)
    backend = FakeBackend()
    backend.export_centroid = lambda cluster_id: None      # worker not ready / degraded
    owner, result, _ = _stopped_owner(tmp_path, monkeypatch, backend=backend, voiceprint_store=store)
    offer = owner.learning_offer(result)
    assert owner.accept_learning(offer) is False
    assert store.load().voiceprint.meetings_contributed == 1   # untouched
    assert backend.closed == 1


def test_enroll_from_mic_refused_while_meeting_active(tmp_path, monkeypatch):
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    owner, _, _ = _owner(tmp_path, voiceprint_store=_store(tmp_path, enrolled=False))
    owner.prepare()
    owner.start()
    res = owner.enroll_from_mic(seconds=1)
    assert res.ok is False and res.reason == "capture_busy"
    owner.stop()


class PcmRecorder(FakeRecorder):
    """A mic recorder that hands back a fixed buffer, and records how it was
    started (the sample must never be written to a file)."""

    instances: list = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.save_to_file = "unset"
        self.stopped = 0
        PcmRecorder.instances.append(self)

    def start_recording(self, callback=None, save_to_file=None):
        self.save_to_file = save_to_file
        return True

    def stop_recording(self):
        self.stopped += 1
        return b"\x03\x04" * 4000


def test_enroll_from_mic_saves_a_voiceprint_from_memory_only(tmp_path, monkeypatch):
    import tldw_chatbook.Audio.diarizer_local as diarizer_local
    from tldw_chatbook.Audio.diarizer_worker import MODEL_ID

    backend = FakeBackend(centroid=(3.0, 4.0), seconds=27.0)
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: backend)
    PcmRecorder.instances = []
    store = _store(tmp_path, enrolled=False)
    owner, _, _ = _owner(tmp_path, voiceprint_store=store)
    owner._mic_factory = PcmRecorder
    owner._sleep = lambda seconds: None
    seen: list[str] = []

    res = owner.enroll_from_mic(seconds=30, progress=seen.append)

    assert res.ok is True and res.reason is None and res.seconds == 27.0
    assert seen == ["warming up", "recording", "embedding"]
    assert backend.enrolled == [(b"\x03\x04" * 4000, 16000)]
    assert PcmRecorder.instances[-1].save_to_file is None      # memory only
    record = store.load().voiceprint
    assert record.model_id == MODEL_ID and record.centroid == pytest.approx([0.6, 0.8])
    assert backend.closed == 1                                  # the worker is not leaked


def test_enrollment_spawns_the_resolved_engine_and_stamps_its_model_id(tmp_path, monkeypatch):
    """Spec §6's invariant, end to end (review I1): the engine that PRODUCED
    the vector and the model id it is STORED under must be the same one.

    Both halves together, because either alone passes on a bug: an
    `_embedding_diarizer` hard-wired back to SpeechBrain fails the first
    assert while still writing an ONNX id, which is exactly the silent
    corruption -- an ECAPA vector stored as a sherpa-onnx one -- that every
    later match would then read as "not you"."""
    import tldw_chatbook.Audio.diarizer_local as diarizer_local
    from tldw_chatbook.Audio.diarizer_local import model_id_for

    monkeypatch.setattr(mo, "resolve_engine", lambda settings, find_spec=None: ("onnx", ()))
    backend = FakeBackend(centroid=(3.0, 4.0), seconds=27.0)
    captured: dict = {}
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda **kw: captured.update(kw) or backend)
    store = _store(tmp_path, enrolled=False)
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, diarizer_backend="onnx",
        onnx_models_dir=str(tmp_path / "m"), voiceprint_store=store,
    )
    owner._mic_factory = PcmRecorder
    owner._sleep = lambda seconds: None

    assert owner.enroll_from_mic(seconds=1).ok is True
    assert captured["engine"] == "onnx"                       # ... spawned the resolved engine
    assert captured["embedder"] == "titanet_small"
    assert captured["models_dir_override"] == tmp_path / "m"
    assert store.load().voiceprint.model_id == model_id_for("onnx")   # ... and stamped its id


def test_enrollment_waits_out_the_model_fetch_before_starting_the_ready_clock(tmp_path, monkeypatch):
    """Final review I1/M4: the ONNX model fetch has a 600 s budget of its own
    while `READY_TIMEOUT_S` is 120 s, so a first explicit enrollment on a
    fresh install over a slow link gave up MID-DOWNLOAD -- and (before the
    `_closed` guard) leaked the worker that fetch went on to spawn."""
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    class Downloading(FakeBackend):
        """Reports "downloading" for two polls, then warms up and is ready."""

        polls = 0

        @property
        def warmup_status(self):
            return "downloading 3 / 44 MB" if self.polls < 2 else "warming up"

        def wait_ready(self, timeout):
            self.polls += 1
            return self.polls > 2

    backend = Downloading()
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda **kw: backend)
    owner, _, _ = _owner(tmp_path, voiceprint_store=_store(tmp_path, enrolled=False))
    seen: list[str] = []

    spawned, mine = owner._embedding_diarizer(progress=seen.append)

    assert (spawned, mine) == (backend, True)          # ... waited the fetch out
    assert backend.closed == 0                         # ... instead of giving up on it
    assert any(s.startswith("downloading") for s in seen)   # ... and said so


def test_enroll_from_mic_reports_a_failed_embed_without_saving(tmp_path, monkeypatch):
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    backend = FakeBackend()
    backend.enroll_from_pcm = lambda pcm, sr: None
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: backend)
    store = _store(tmp_path, enrolled=False)
    owner, _, _ = _owner(tmp_path, voiceprint_store=store)
    owner._mic_factory = PcmRecorder
    owner._sleep = lambda seconds: None

    res = owner.enroll_from_mic(seconds=1)
    assert res.ok is False and res.reason == "embed_failed"
    assert store.load().voiceprint is None
    assert backend.closed == 1


def test_enroll_from_mic_reports_a_missing_recorder(tmp_path, monkeypatch):
    import tldw_chatbook.Audio.diarizer_local as diarizer_local
    from tldw_chatbook.Audio.recording_service import AudioRecordingError

    backend = FakeBackend()
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: backend)
    owner, _, _ = _owner(tmp_path, voiceprint_store=_store(tmp_path, enrolled=False))

    def no_recorder(**kwargs):
        raise AudioRecordingError("Audio recording functionality requires NumPy\nfor efficient processing.")

    owner._mic_factory = no_recorder
    res = owner.enroll_from_mic(seconds=1)
    assert res.ok is False and res.reason == "Audio recording functionality requires NumPy"


def test_enroll_from_mic_never_opens_the_mic_when_the_worker_is_unavailable(tmp_path, monkeypatch):
    """Review M8: the worker is built FIRST, so giving up on warm-up does not
    leave a just-opened microphone behind."""
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    def no_worker(*a, **kw):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(diarizer_local, "LocalDiarizer", no_worker)
    PcmRecorder.instances = []
    owner, _, _ = _owner(tmp_path, voiceprint_store=_store(tmp_path, enrolled=False))
    owner._mic_factory = PcmRecorder
    owner._sleep = lambda seconds: None

    res = owner.enroll_from_mic(seconds=1)
    assert res.ok is False and res.reason == "diarizer_unavailable"
    assert PcmRecorder.instances == []


def test_enroll_from_mic_uses_the_configured_microphone(tmp_path, monkeypatch):
    """Review I5: enrolling from the default input while meetings record from
    a chosen one is a silent, permanent channel mismatch."""
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    backend = FakeBackend()
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: backend)
    selected: list = []

    class DeviceRecorder(PcmRecorder):
        def get_audio_devices(self):
            return [{"id": 0, "name": "Built-in"}, {"id": 7, "name": "Shure MV7"}]

        def set_device(self, device_id):
            selected.append(device_id)
            return True

    owner, _, _ = _owner(
        tmp_path, mic_device="Shure MV7", voiceprint_store=_store(tmp_path, enrolled=False),
    )
    owner._mic_factory = DeviceRecorder
    owner._sleep = lambda seconds: None
    assert owner.enroll_from_mic(seconds=1).ok is True
    assert selected == [7]


def test_enroll_from_mic_refuses_when_the_configured_microphone_is_gone(tmp_path, monkeypatch):
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    backend = FakeBackend()
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: backend)
    store = _store(tmp_path, enrolled=False)
    owner, _, _ = _owner(tmp_path, mic_device="Shure MV7", voiceprint_store=store)
    owner._mic_factory = PcmRecorder          # enumerates nothing
    owner._sleep = lambda seconds: None

    res = owner.enroll_from_mic(seconds=1)
    assert res.ok is False and res.reason == "mic_device_not_found"
    assert store.load().voiceprint is None


def test_enroll_from_mic_refuses_when_the_recorder_rejects_the_microphone(tmp_path, monkeypatch):
    """Qodo 6, the enrollment half: a device that enumerates but is rejected
    by `set_device` used to enroll from whatever the backend fell back to --
    a voiceprint built from a different input channel than the meetings it is
    matched against, silently and permanently."""
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    backend = FakeBackend()
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: backend)

    class RejectingRecorder(PcmRecorder):
        def get_audio_devices(self):
            return [{"id": 7, "name": "Shure MV7"}]

        def set_device(self, device_id):
            return False

    store = _store(tmp_path, enrolled=False)
    owner, _, _ = _owner(tmp_path, mic_device="Shure MV7", voiceprint_store=store)
    owner._mic_factory = RejectingRecorder
    owner._sleep = lambda seconds: None

    res = owner.enroll_from_mic(seconds=1)
    assert res.ok is False and res.reason == "mic_device_not_found"
    assert store.load().voiceprint is None


def test_enroll_from_mic_can_be_cancelled_mid_recording(tmp_path, monkeypatch):
    """Spec §3.4: the user must be able to abort the sample, not be held for
    the full 30 s (review I7)."""
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    backend = FakeBackend()
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: backend)
    PcmRecorder.instances = []
    store = _store(tmp_path, enrolled=False)
    owner, _, _ = _owner(tmp_path, voiceprint_store=store)
    owner._mic_factory = PcmRecorder
    cancel = threading.Event()
    slices: list[float] = []

    def sleep(seconds):
        slices.append(seconds)
        if len(slices) == 3:
            cancel.set()                       # the user hits Cancel

    owner._sleep = sleep

    res = owner.enroll_from_mic(seconds=30, cancel=cancel)

    assert res.ok is False and res.reason == "cancelled"
    assert max(slices) <= mo.ENROLL_SLICE_S    # sliced, not one 30 s wait
    assert len(slices) < 30 / mo.ENROLL_SLICE_S
    assert PcmRecorder.instances[-1].stopped == 1   # the mic was released
    assert backend.enrolled == []                    # nothing embedded
    assert store.load().voiceprint is None           # nothing kept
    assert backend.closed == 1                       # the spawned worker went


def test_enroll_from_mic_returns_cancelled_before_it_records(tmp_path, monkeypatch):
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    backend = FakeBackend()
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: backend)
    PcmRecorder.instances = []
    owner, _, _ = _owner(tmp_path, voiceprint_store=_store(tmp_path, enrolled=False))
    owner._mic_factory = PcmRecorder
    owner._sleep = lambda seconds: None
    cancel = threading.Event()
    cancel.set()

    res = owner.enroll_from_mic(seconds=1, cancel=cancel)
    assert res.ok is False and res.reason == "cancelled"
    assert PcmRecorder.instances == []


def test_start_refuses_while_an_enrollment_holds_the_mic(tmp_path, monkeypatch):
    """Review I4, the reverse guard: `enroll_from_mic` already refuses during
    a meeting; a Start during an enrollment opened a second recorder on the
    same device."""
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    backend = FakeBackend()
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: backend)
    owner, _, _ = _owner(tmp_path, voiceprint_store=_store(tmp_path, enrolled=False))
    owner._mic_factory = PcmRecorder
    owner.prepare()
    seen: list[bool] = []

    def sleep(seconds):
        if not seen:
            seen.append(owner.is_enrolling)
            with pytest.raises(RuntimeError, match="enrolling"):
                owner.start()

    owner._sleep = sleep
    assert owner.enroll_from_mic(seconds=1).ok is True
    assert seen == [True]
    assert owner.is_enrolling is False          # cleared afterwards
    assert owner.session is None                 # ... and no meeting was opened
    owner.start()                                 # now it is allowed again
    owner.stop()


def test_enroll_from_mic_refuses_a_second_concurrent_enrollment(tmp_path, monkeypatch):
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    backend = FakeBackend()
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: backend)
    owner, _, _ = _owner(tmp_path, voiceprint_store=_store(tmp_path, enrolled=False))
    owner._mic_factory = PcmRecorder
    second: list = []

    def sleep(seconds):
        if not second:
            second.append(owner.enroll_from_mic(seconds=1))

    owner._sleep = sleep
    assert owner.enroll_from_mic(seconds=1).ok is True
    assert second[0].ok is False and second[0].reason == "capture_busy"


# ---- review C1: an offer's release must never touch a live meeting --------

def test_accepting_an_offer_after_a_new_start_leaves_the_live_worker_open(tmp_path, monkeypatch):
    """The reviewer's repro 1: Accept is in flight (an export can take up to
    10 s) when the user starts the next meeting. The lapse closes the offer's
    worker; Accept's cleanup must not then close the NEW meeting's worker.

    `stall_after_s` is generous here (final review I2): the nested
    `owner.start()` below arms a watchdog against the frozen
    `FakeCapture.audio_position_s`, and at the module default (50 ms) it
    stopped the SECOND meeting with `mic_lost` -- closing `second` from a
    background thread -- while `accept_learning` was still inside the real
    store's AES key derivation. That is the product working correctly on a
    capture whose clock never advances; only the test's window was wrong."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    first, second = FakeBackend(), FakeBackend()
    built = iter([first, second])
    monkeypatch.setattr(mo, "build_diarizer", lambda settings, **kw: next(built))
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path), stall_after_s=30.0,
    )
    owner.prepare()
    session = owner.start()
    session.meta.matched_self = "S1"
    result = owner.stop()
    offer = owner.learning_offer(result)
    assert offer is not None

    # The Start lands while the export is in flight. Whether the in-flight
    # sample still merges is not the point (it came from the old worker
    # legitimately); WHICH worker gets closed is.
    first.export_centroid = lambda cluster_id: (owner.start(), ([0.0, 1.0], 5.0))[1]
    owner.accept_learning(offer)
    assert first.closed == 1                             # the lapse closed the OLD worker
    assert second.closed == 0                            # ... and left the live one alone
    assert owner.session._diarizer is second
    assert owner.pending_offer is None
    owner.stop()


def test_dismissing_a_lapsed_offer_leaves_the_live_worker_open(tmp_path, monkeypatch):
    """The reviewer's repro 2 -- the likely Task-5 wiring: a new meeting hides
    the offer card, and the screen calls `dismiss_learning()` afterwards.

    Same generous `stall_after_s` as its sibling above, and for the same
    reason (final review I2): the nested `owner.start()` arms a watchdog
    against a capture clock that never advances."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    first, second = FakeBackend(), FakeBackend()
    built = iter([first, second])
    monkeypatch.setattr(mo, "build_diarizer", lambda settings, **kw: next(built))
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path), stall_after_s=30.0,
    )
    owner.prepare()
    session = owner.start()
    session.meta.matched_self = "S1"
    result = owner.stop()
    owner.learning_offer(result)

    owner.start()                       # the offer lapses here
    owner.dismiss_learning()            # the screen tidies up afterwards
    assert first.closed == 1
    assert second.closed == 0 and owner.session._diarizer is second
    owner.stop()


def test_declining_a_stale_offer_is_a_no_op(tmp_path, monkeypatch):
    """Review M9: `decline_learning` honours its argument."""
    owner, result, backend = _stopped_owner(tmp_path, monkeypatch, voiceprint_store=_store(tmp_path))
    offer = owner.learning_offer(result)
    stale = mo.LearningOffer(kind="matched_cluster", folder=Path(tmp_path), cluster_id="S9")
    owner.decline_learning(stale)
    assert backend.closed == 0 and owner.pending_offer is offer   # untouched
    owner.decline_learning(offer)
    assert backend.closed == 1 and owner.pending_offer is None


@pytest.mark.parametrize("with_pending_offer", [False, True])
def test_dismiss_never_blocks_behind_an_in_flight_stop(tmp_path, monkeypatch, with_pending_offer):
    """Re-review N1: the offer-answer paths run on the UI thread, and a Stop
    holds `_stop_lock` across a `session.stop()` that marshals the ingest
    submit back onto that same UI thread. Sharing a lock between the two hung
    the app permanently -- with or without an offer pending, since the lock
    used to be taken before the "is anything pending?" test."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    backend, _ = _backend_spy(monkeypatch)
    owner, _, _ = _owner(tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path))
    owner.prepare()

    if with_pending_offer:
        first = owner.start()
        first.meta.matched_self = "S1"
        owner.stop()
        assert owner.pending_offer is not None

    session = owner.start()
    held = threading.Event()
    release = threading.Event()
    real_stop = session.stop

    def blocking_stop(reason="user"):
        held.set()
        release.wait(5.0)          # stands in for the blocked call_from_thread
        return real_stop(reason=reason)

    session.stop = blocking_stop
    # Daemon threads and a `finally` release: when this assertion regresses it
    # must FAIL the test, not wedge the suite at exit the way it wedges the app.
    stopper = threading.Thread(target=owner.stop, daemon=True)
    stopper.start()
    try:
        assert held.wait(2.0)      # stop() is inside session.stop(), holding _stop_lock

        dismissed = threading.Event()

        def ui_dismiss() -> None:
            owner.dismiss_learning()   # the UI thread, e.g. hiding the offer card
            dismissed.set()

        threading.Thread(target=ui_dismiss, daemon=True).start()
        assert dismissed.wait(1.0), "dismiss_learning() blocked behind an in-flight stop()"
    finally:
        release.set()
    stopper.join(5.0)
    assert not stopper.is_alive()
    owner.dismiss_learning()       # tidy up whatever the Stop settled


def test_a_refused_start_keeps_the_pending_offer(tmp_path, monkeypatch):
    """Re-review N3: only a meeting that actually starts lapses the previous
    one's offer -- a Start refused during an enrollment used to destroy it on
    its way out."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    offer_backend, next_backend = FakeBackend(), FakeBackend()
    built = iter([offer_backend, next_backend])
    monkeypatch.setattr(mo, "build_diarizer", lambda settings, **kw: next(built))
    owner, _, _ = _owner(tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path))
    owner.prepare()
    session = owner.start()
    session.meta.matched_self = "S1"
    offer = owner.learning_offer(owner.stop())
    assert offer is not None

    owner._enrolling = True                       # an enrollment is recording
    with pytest.raises(RuntimeError, match="enrolling"):
        owner.start()
    owner._enrolling = False
    assert owner.pending_offer is offer and offer_backend.closed == 0

    monkeypatch.setattr(FakeDictation, "start_dictation", lambda self, **kw: False)
    with pytest.raises(RuntimeError, match="failed to start"):
        owner.start()                              # a Start that fails, too
    assert owner.pending_offer is offer and offer_backend.closed == 0
    assert next_backend.closed == 1                # ... its own worker was released

    monkeypatch.undo()
    assert owner.accept_learning(offer) is True    # the offer is still answerable
    assert offer_backend.closed == 1


def test_enrollment_owns_its_worker_while_an_offer_is_answered(tmp_path, monkeypatch):
    """Re-review N2: a 30 s recording is long enough for the user to answer
    the pending offer meanwhile. Borrowing that offer's worker meant the
    finished sample was embedded on a worker the answer had just closed."""
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    store = _store(tmp_path)
    owner, result, offer_backend = _stopped_owner(tmp_path, monkeypatch, voiceprint_store=store)
    offer = owner.learning_offer(result)
    assert offer is not None

    enroll_backend = FakeBackend(centroid=(3.0, 4.0), seconds=27.0)
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: enroll_backend)
    owner._mic_factory = PcmRecorder
    answered: list = []

    def sleep(seconds):
        if not answered:
            answered.append(owner.decline_learning(offer))   # the offer card goes away

    owner._sleep = sleep

    res = owner.enroll_from_mic(seconds=1)

    assert res.ok is True and res.seconds == 27.0            # the sample still lands
    assert enroll_backend.enrolled                            # embedded on its OWN worker
    assert offer_backend.closed == 1                          # the offer's, closed once
    assert enroll_backend.closed == 1                         # enrollment's, closed once
    assert store.load().voiceprint.centroid == pytest.approx([0.6, 0.8])


def test_the_recorder_is_released_when_a_recording_slice_raises(tmp_path, monkeypatch):
    """Re-review N5: once the mic is open, every way out of the loop has to
    close it again -- 120 slices are 120 chances to raise."""
    import tldw_chatbook.Audio.diarizer_local as diarizer_local

    backend = FakeBackend()
    monkeypatch.setattr(diarizer_local, "LocalDiarizer", lambda *a, **kw: backend)
    PcmRecorder.instances = []
    store = _store(tmp_path, enrolled=False)
    owner, _, _ = _owner(tmp_path, voiceprint_store=store)
    owner._mic_factory = PcmRecorder

    def boom(seconds):
        raise OSError("the input device went away")

    owner._sleep = boom

    res = owner.enroll_from_mic(seconds=30)
    assert res.ok is False and res.reason == "OSError"
    assert PcmRecorder.instances[-1].stopped == 1     # the microphone was released
    assert store.load().voiceprint is None
    assert backend.closed == 1


def test_pending_offer_survives_a_screen_remount(tmp_path, monkeypatch):
    """Review M11: `learning_offer()` stays one-shot, but a screen that
    remounted between Stop and the answer can read the offer back."""
    owner, result, _ = _stopped_owner(tmp_path, monkeypatch, voiceprint_store=_store(tmp_path))
    offer = owner.learning_offer(result)
    assert owner.learning_offer(result) is None
    assert owner.pending_offer is offer
    owner.dismiss_learning()
    assert owner.pending_offer is None


def test_stop_without_a_qualifying_sample_never_reads_the_store(tmp_path, monkeypatch):
    """Review I3: a meeting that could never produce a sample must not raise a
    Keychain prompt at Stop just to discover that."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    backend, _ = _backend_spy(monkeypatch)
    keys = FakeKeys()
    owner, _, _ = _owner(
        tmp_path, live_diarization=True, voice_match=False,
        voiceprint_store=_store(tmp_path, keys=keys),
    )
    reads_after_save = keys.reads
    owner.prepare()
    owner.start()                       # room mode, no match, no you.wav
    result = owner.stop()
    assert keys.reads == reads_after_save
    assert owner.learning_offer(result) is None
    assert backend.closed == 1


def test_a_concurrent_second_stop_still_releases_the_worker(tmp_path, monkeypatch):
    """Review M4: `session.stop()` returning None (a genuinely concurrent
    second caller) computed no offer and left the worker running."""
    monkeypatch.setattr(mo, "resolve_effective_config", lambda: SimpleNamespace(provider="p", model="m", language="en"))
    backend, _ = _backend_spy(monkeypatch)
    owner, _, _ = _owner(tmp_path, live_diarization=True, voiceprint_store=_store(tmp_path))
    owner.prepare()
    session = owner.start()
    session.stop = lambda reason="user": None      # the concurrent-caller shape
    assert owner.stop() is None
    assert backend.closed == 1


def test_the_voiceprint_vector_never_reaches_a_log_line(tmp_path, monkeypatch, captured_lines):
    """Spec §6: logs carry modes, counts and exception types -- never the
    vector, a speaker name, or a path."""
    store = _store(tmp_path, centroid=(0.123456, 0.987654))
    owner, result, backend = _stopped_owner(tmp_path, monkeypatch, voiceprint_store=store)
    offer = owner.learning_offer(result)
    owner.accept_learning(offer)
    joined = "\n".join(captured_lines)
    assert "0.123456" not in joined and "0.987654" not in joined
    # ... and no meeting log line carries the recordings path either (the
    # config bootstrap's own startup logs are not this module's).
    meeting_lines = [line for line in captured_lines if "meeting" in line.lower()]
    assert not [line for line in meeting_lines if str(tmp_path) in line]
