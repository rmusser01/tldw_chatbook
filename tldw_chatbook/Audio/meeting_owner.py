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
            base = datetime.now().strftime("%Y-%m-%d_%H%M")
            folder = Path(self.settings.recordings_dir) / base
            suffix = 1
            while folder.exists():
                suffix += 1
                folder = Path(self.settings.recordings_dir) / f"{base}-{suffix}"
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
