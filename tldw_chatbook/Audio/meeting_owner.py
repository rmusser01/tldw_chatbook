"""App-owned meeting session lifecycle (spec §3.4, §7).

Screens are never cached across tab switches, so the running session
lives here. Textual-free: the app hands in `call_from_thread` and the
ingest submit callable; everything else is injectable for tests.

Import-graph rule (final whole-branch review, C1): `app.py` imports this
module at MODULE SCOPE, so nothing imported here at module scope may need
an optional dependency. `meeting_capture` (bare `import numpy`) and
`recording_service` (sounddevice/pyaudio) are therefore imported inside
the functions that need them -- `from __future__ import annotations`
keeps their type hints working. `Tests/Audio/test_meeting_import_safety.py`
pins this.
"""
from __future__ import annotations

import importlib.util
import shutil
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .meeting_session import (
    Diarizer,
    LocalMeetingSink,
    MeetingMeta,
    MeetingResult,
    MeetingSession,
    read_meeting_json,
    update_meeting_json,
)
from .system_audio_tap import TapMode, build_tap, probe
from .wav_writer import HEADER_BYTES, PlaceholderWavWriter, patch_wav_header, wav_needs_patch
from tldw_chatbook.Utils.log_sanitizer import redact_user_paths

MEETINGS_DIRNAME = "meetings"
DIARIZATION_MODULES = ("torch", "torchaudio", "speechbrain", "sklearn")
BYTES_PER_S = 32000.0
#: Ingest job states that will never change again. `done` is the only one
#: that means the raw tracks are safe to delete; the rest simply end the
#: wait (`IngestJobState`, `Library/library_ingest_jobs.py`).
TERMINAL_JOB_STATES = frozenset({"done", "failed", "cancelled", "skipped"})
#: How long Start/prepare will wait for the encrypted voiceprint (spec §3.1).
#: The read runs on its own thread and is JOINED with this timeout, so a
#: locked or prompting Keychain costs the meeting nothing but its match.
VOICEPRINT_LOAD_TIMEOUT_S = 1.5
#: Mic sample rate for explicit enrollment; the whole meeting pipeline is
#: 16 kHz mono PCM16 (`MeetingCapture`, `wav_writer`).
MIC_SAMPLE_RATE = 16000
#: Ceiling on how much of `you.wav` the plain-call-mode learning offer embeds.
# ponytail: the FIRST minute, not the best minute -- a meeting that opens with
# silence learns less. Pick the loudest window if that shows up as a real miss.
LEARN_PCM_MAX_SECONDS = 60.0

if TYPE_CHECKING:  # import-light at runtime: `console_voice_input` pulls config
    from tldw_chatbook.Chat.console_voice_input import EffectiveConfig


def resolve_effective_config() -> "EffectiveConfig | None":
    """Resolve the transcription settings a meeting would actually run with.

    Late import: `console_voice_input` pulls config; keep this module light.

    Returns:
        The effective dictation config, or None when no local provider is
        usable (see `console_voice_input.resolve`).
    """
    from tldw_chatbook.Chat.console_voice_input import resolve

    return resolve()


def meeting_user_display_name(get_display_name: Callable[[str], str] | None = None) -> str:
    """The one shared display name for the mic ("you") channel (task 31746).

    Deliberately NOT a bare `chat_defaults.user_display_name` read: that
    section's own factory default ships as the literal "User" (`config.py`'s
    `DEFAULT_CONFIG_FROM_TOML`), so a fresh install has no way to tell "never
    touched this setting" apart from "chose User" -- returning it unconditio-
    nally would silently turn every untouched install's "You:" rows into
    "User:" rows. Only a value that actually DIFFERS from that shipped
    default counts as a deliberate choice; everything else falls back to
    Meetings' own already-shipped "You".

    Reuses `config.get_chat_defaults_user_display_name` (task 31746 review)
    rather than reading the raw config value: that getter is already the
    validated read for this exact key, normalizing via
    `console_roleplay_identity.normalize_chat_display_name` (strips
    whitespace, rejects control characters, caps length) the same way
    Console does -- so a whitespace-only or hostile configured value can
    never leak through here as a literal display name (a raw comparison
    used to return "   " verbatim).

    Args:
        get_display_name: `(default) -> str` validated getter, injectable
            for tests; defaults to the real
            `config.get_chat_defaults_user_display_name`.

    Returns:
        The configured, normalized `chat_defaults.user_display_name` when it
        differs from the shipped factory default, else "You".
    """
    if get_display_name is None:
        from tldw_chatbook.config import get_chat_defaults_user_display_name

        get_display_name = get_chat_defaults_user_display_name
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    factory_default = DEFAULT_CONFIG_FROM_TOML.get("chat_defaults", {}).get("user_display_name", "User")
    configured = get_display_name(factory_default)
    if configured and configured != factory_default:
        return configured
    return "You"


class MeetingSettings(BaseModel):
    """Validated `[meetings]` configuration for one meeting session.

    Config values are loosely typed (TOML, env, defaults), so they are
    validated here at the boundary rather than cast ad hoc downstream:
    an unusable provider/device/flag raises `ValidationError` naming the
    field instead of surfacing halfway through a recording. Assignment is
    validated too -- `apply_device_choice` writes `mic_device` and
    `system_source` back onto a live instance.
    """

    model_config = ConfigDict(validate_assignment=True)

    provider: str = "auto"
    model: str = ""
    system_source: str = "auto"
    mic_device: str = ""
    recordings_dir: Path
    keep_raw_tracks: bool = True
    post_transcribe: bool = True
    post_diarize: bool = True
    live_diarization: bool = False
    #: Hybrid-room mic diarization (task 31743): in call mode, also diarize
    #: the mic ("you") and overlap ("both") segments instead of leaving them
    #: pre-named as the user. Off by default -- see `meetings.md`.
    diarize_mic_channel: bool = False
    diarizer_backend: str = "local"
    #: Qodo Q7: 0 or a negative value silently disabled the Stop pass (the
    #: clusterer can hold no clusters), so it is refused at the boundary
    #: like every other unusable config value here.
    max_speakers: int = Field(8, ge=1)
    #: Self-voiceprint matching (TASK-31826, spec §5). No effect until a
    #: voiceprint is enrolled; never runs in plain call mode.
    voice_match: bool = True
    #: Cosine-distance ceiling for a `self` match, and the embedded audio a
    #: cluster needs before it can be declared `self`. Bounded here for the
    #: same reason `max_speakers` is: 0 (or > 1) is not a stricter threshold,
    #: it is a silently dead feature.
    voice_match_threshold: float = Field(0.2, gt=0, le=1)
    voice_match_min_seconds: float = Field(4.0, ge=0)
    #: Offer to learn from a qualifying meeting (at most one offer each).
    voice_learn_offer: bool = True

    @field_validator("recordings_dir", mode="before")
    @classmethod
    def _validate_recordings_dir(cls, value: Any) -> Path:
        """Run a configured (string) path through the central validator.

        Args:
            value: A `str` straight from config, or an already-built `Path`.

        Returns:
            The absolute, resolved recordings directory.

        Raises:
            ValueError: Empty, not path-shaped, or rejected by
                `validate_path_simple` (traversal, null bytes, ...).
        """
        from tldw_chatbook.Utils.path_validation import validate_path_simple

        if isinstance(value, str):
            if not value.strip():
                raise ValueError("recordings_dir must not be empty")
            value = validate_path_simple(value)
        elif not isinstance(value, Path):
            raise ValueError("recordings_dir must be a path")
        return Path(value).resolve()

    @classmethod
    def from_config(cls, get_setting: Callable[[str, str, Any], Any], data_dir: Path) -> "MeetingSettings":
        """Build the settings from the `[meetings]` config section.

        Args:
            get_setting: `(section, key, default)` config accessor.
            data_dir: User data dir, parent of the default recordings folder.

        Returns:
            The validated settings.

        Raises:
            pydantic.ValidationError: A configured value is not usable; the
                error names the offending field.
        """
        raw_dir = get_setting("meetings", "recordings_dir", "") or ""
        return cls(
            provider=get_setting("meetings", "provider", "auto") or "auto",
            model=get_setting("meetings", "model", "") or "",
            system_source=get_setting("meetings", "system_source", "auto") or "auto",
            mic_device=get_setting("meetings", "mic_device", "") or "",
            recordings_dir=raw_dir or Path(data_dir) / MEETINGS_DIRNAME,
            keep_raw_tracks=get_setting("meetings", "keep_raw_tracks", True),
            post_transcribe=get_setting("meetings", "post_transcribe", True),
            post_diarize=get_setting("meetings", "post_diarize", True),
            live_diarization=get_setting("meetings", "live_diarization", False),
            diarize_mic_channel=get_setting("meetings", "diarize_mic_channel", False),
            diarizer_backend=get_setting("meetings", "diarizer_backend", "local") or "local",
            max_speakers=get_setting("meetings", "max_speakers", 8),
            voice_match=get_setting("meetings", "voice_match", True),
            voice_match_threshold=get_setting("meetings", "voice_match_threshold", 0.2),
            voice_match_min_seconds=get_setting("meetings", "voice_match_min_seconds", 4.0),
            voice_learn_offer=get_setting("meetings", "voice_learn_offer", True),
        )


@dataclass
class VoiceMatchState:
    """Whether this meeting can tag the user by voice, and why not (spec §3.5).

    `state` is ``"on"`` or ``"off"``; `reason` is a STATIC key the rail turns
    into copy -- never a path, a name, or an exception message. Off-reasons:
    ``"disabled"`` (the setting), ``"plain_call_mode"`` (the mic channel
    already is the user), or the store's own verdict (``"no_voiceprint"``,
    ``"needs_reenrollment"``, ``"cannot_decrypt"``, ``"keyring_locked"``).
    """

    state: str
    reason: str | None = None


@dataclass
class LearningOffer:
    """One post-meeting offer to learn from a clean sample (spec §3.4).

    `kind` is ``"matched_cluster"`` (the matched cluster's batch centroid) or
    ``"mic_channel"`` (plain call mode: the recorded `you.wav`, offered with
    the explicit "was it only you on the mic?" question). At most one per
    meeting, and never more than one outstanding.
    """

    kind: str
    folder: Path
    cluster_id: str | None = None


@dataclass
class EnrollResult:
    """The outcome of an explicit "Enroll my voice" run (spec §3.4).

    `reason` is a static key (``"capture_busy"``, ``"no_audio"``,
    ``"embed_failed"``, ``"diarizer_unavailable"``, ``"store_unavailable"``)
    or a recorder's own "no usable recorder" first line; `seconds` is the
    audio the worker actually embedded.
    """

    ok: bool
    reason: str | None = None
    seconds: float = 0.0


@dataclass
class PrepareResult:
    """Everything the Meetings rail needs to decide what it can offer.

    Produced by `MeetingSessionOwner.prepare()` before any recording starts:
    the resolved system-audio route, the transcriber that would be used,
    whether post-meeting speaker labels are possible, unfinished folders
    from a previous crash, and the input devices to populate the pickers.
    """

    tap_mode: TapMode
    provider: str
    model: str
    diarization_available: bool
    diarization_missing: tuple[str, ...]
    recoverable: tuple[Path, ...]
    input_devices: tuple[str, ...] = ()
    #: Whether `start()` will actually inject a live LOCAL diarizer (spec
    #: §3.4): `diarization_available` alone only says the offline post-meeting
    #: pass is possible -- this also requires `settings.live_diarization` on
    #: AND `settings.diarizer_backend == "local"`, since `build_diarizer`
    #: always returns `None` for the unimplemented "server" backend. Computed
    #: without constructing a diarizer (no subprocess spawn during prepare).
    #: False here still leaves the offline pass available at Stop.
    #: Defaulted (unlike the fields above) so existing positional/keyword
    #: callers built before Task 6 -- notably `Tests/UI/test_meetings_screen
    #: .py`'s `FakeOwner` -- keep constructing a `PrepareResult` unchanged.
    live_diarization_active: bool = False
    #: Set when the mic recorder cannot be built at all (numpy missing, no
    #: audio backend). The rail shows it and keeps Start disabled instead of
    #: offering a Start that can only fail (final whole-branch review, C1).
    capture_error: str | None = None
    #: Whether a meeting started NOW would tag the user by voice, and why not
    #: (TASK-31826). Defaulted so callers built before it -- notably
    #: `Tests/UI/test_meetings_screen.py`'s `FakeOwner` -- keep working;
    #: `prepare()`/`start()` always fill it in.
    voice_match: VoiceMatchState = field(default_factory=lambda: VoiceMatchState("off", None))


def diarization_requirements(find_spec=importlib.util.find_spec) -> tuple[str, ...]:
    """Missing diarization modules, checked WITHOUT importing them (spec §3.5).

    Args:
        find_spec: Module-spec lookup, injectable for tests. Importing these
            modules for real would pull torch into the UI process.

    Returns:
        The names of the `DIARIZATION_MODULES` that are not installed; empty
        when speaker labels can be produced after the meeting.
    """
    missing = []
    for name in DIARIZATION_MODULES:
        try:
            present = find_spec(name) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            missing.append(name)
    return tuple(missing)


def build_diarizer(settings: MeetingSettings, voiceprint: list[float] | None = None) -> Diarizer | None:
    """Build the live diarizer backend named by `settings`, best-effort.

    Import-graph rule (module docstring): `SpeechBrainDiarizer` is imported
    LAZILY here, never at module scope -- this is the only place allowed to
    know it exists, so `app.py` stays torch-free at boot.

    Args:
        settings: The validated meeting settings.
        voiceprint: The enrolled self voiceprint to match against, or None
            (TASK-31826). The vector goes straight to the worker and is never
            logged. The caller decides whether matching applies at all; this
            just forwards what it is handed.

    Returns:
        The diarizer to inject into the session, or `None` when live
        diarization is off, its modules are missing, the backend is not
        implemented, or construction otherwise failed -- a meeting must
        stay startable with coarse (non-diarized) labels either way.
    """
    if not settings.live_diarization or diarization_requirements():
        return None
    try:
        if settings.diarizer_backend == "local":
            from .diarizer_local import SpeechBrainDiarizer

            return SpeechBrainDiarizer(
                max_speakers=settings.max_speakers,
                voiceprint=voiceprint,
                match_threshold=settings.voice_match_threshold,
                match_min_seconds=settings.voice_match_min_seconds,
            )
        raise NotImplementedError(f"diarizer backend {settings.diarizer_backend!r}")
    except Exception as exc:  # noqa: BLE001 - best-effort, never block a meeting start
        logger.warning("meeting: diarizer backend unavailable ({})", type(exc).__name__)
        return None


def scan_recoverable(meetings_dir: Path) -> list[Path]:
    """Find meeting folders left unfinished by a crash.

    A folder qualifies when any of its WAV tracks still carries the
    placeholder header written at creation time (`wav_needs_patch`).

    Args:
        meetings_dir: The recordings directory to scan; a missing directory
            is not an error.

    Returns:
        The recoverable folders, sorted by name (oldest meeting first).
    """
    meetings_dir = Path(meetings_dir)
    if not meetings_dir.exists():
        return []
    found = []
    for folder in sorted(p for p in meetings_dir.iterdir() if p.is_dir()):
        if any(wav_needs_patch(folder / name) for name in ("mixed.wav", "you.wav", "others.wav")):
            found.append(folder)
    return found


def recover_folder(folder: Path) -> dict:
    """Repair one crashed meeting's WAV headers and metadata.

    Patches every track whose header is still a placeholder, recomputes the
    duration from `mixed.wav`, and marks `meeting.json` as recovered (with a
    fallback `ended_at` taken from the recording's mtime).

    Args:
        folder: The meeting folder, as returned by `scan_recoverable`.

    Returns:
        The updated `meeting.json` payload.

    Raises:
        OSError: The folder or its tracks cannot be read or rewritten.
        json.JSONDecodeError: `meeting.json` exists but is truncated or
            malformed; the caller reports it as a failed recovery.
    """
    folder = Path(folder)
    for name in ("mixed.wav", "you.wav", "others.wav"):
        path = folder / name
        if wav_needs_patch(path):
            patch_wav_header(path)
    # From the file size whenever mixed.wav exists, NOT only when it needed
    # patching: writers are closed sequentially, so a crash between closing
    # mixed.wav and closing a raw track leaves a perfectly valid mixed
    # recording that used to be reported as duration 0 (Qodo Q16).
    mixed_path = folder / "mixed.wav"
    data_bytes = max(0, mixed_path.stat().st_size - HEADER_BYTES) if mixed_path.exists() else 0
    duration_s = data_bytes / BYTES_PER_S
    payload = read_meeting_json(folder)
    if not payload.get("ended_at"):
        ended_at = (
            datetime.fromtimestamp(mixed_path.stat().st_mtime)
            if mixed_path.exists()
            else datetime.now()
        )
        payload["ended_at"] = ended_at.isoformat(timespec="seconds")
    payload.update(recovered=True, duration_s=duration_s, stop_reason=payload.get("stop_reason") or "crash")
    # The real writer always persists "folder" in meeting.json (see
    # write_meeting_json call sites); spreading it back in here collides
    # with the positional `folder` argument below (`update_meeting_json()
    # got multiple values for argument 'folder'`). update_meeting_json()
    # re-reads the on-disk payload and merges these fields into it, so the
    # existing "folder" value on disk survives untouched -- dropping it
    # from the spread only removes the duplicate-argument crash.
    payload.pop("folder", None)
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


def _read_wav_pcm(path: Path, max_seconds: float) -> tuple[bytes, int]:
    """Up to `max_seconds` of PCM16 from `path`, for the learning offer.

    Args:
        path: A meeting track (`you.wav`).
        max_seconds: Ceiling on how much audio is read (from the start).

    Returns:
        `(pcm, sample_rate)`; `(b"", 0)` when the file is missing or not a
        readable WAV -- learning is best-effort and never raises at the user.
    """
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            frames = min(handle.getnframes(), int(sample_rate * max_seconds))
            return handle.readframes(frames), sample_rate
    except Exception as exc:  # noqa: BLE001 - type only: the message is a path
        logger.warning("meeting: mic track unreadable for learning ({})", type(exc).__name__)
        return b"", 0


def _missing_recorder_message(exc: BaseException) -> str | None:
    """First line of `exc` when it means "no usable recorder", else None.

    `AudioRecordingError` is raised for a missing numpy and a missing
    backend; anything else (a device-enumeration hiccup on a working
    backend) leaves the pickers empty but must not block Start. The class
    is imported here rather than at module scope: `recording_service`
    pulls sounddevice/pyaudio (see this module's docstring).
    """
    if isinstance(exc, ImportError):
        return str(exc).strip().splitlines()[0] or type(exc).__name__
    try:
        from .recording_service import AudioRecordingError
    except Exception:  # noqa: BLE001 - can't classify it; treat as non-fatal
        return None
    if isinstance(exc, AudioRecordingError):
        return str(exc).strip().splitlines()[0] or type(exc).__name__
    return None


class MeetingSessionOwner:
    """Owns the running meeting for the whole app (spec §3.4, §7).

    Screens are never cached across tab switches, so the session, its
    watchdog and the post-meeting ingest cleanup all live here rather than
    on the Meetings screen: a meeting survives navigating away and is
    re-attached to by the next screen that mounts. Textual-free -- the app
    injects `call_from_thread`, the ingest submit and the registry
    listener hooks; everything else is injectable for tests.
    """

    def __init__(
        self,
        *,
        settings: MeetingSettings,
        call_from_thread: Callable[..., Any],
        submit_ingest: Callable[..., Optional[str]],
        job_state: Callable[[str], Optional[str]] = lambda job_id: None,
        subscribe_jobs: Callable[[Callable[[], None]], None] | None = None,
        unsubscribe_jobs: Callable[[Callable[[], None]], None] | None = None,
        facade_factory: Callable[[], Any] | None = None,
        dictation_factory: Callable[[MeetingCapture, Any, Any], Any] | None = None,
        tap_probe: Callable[..., TapMode] = probe,
        tap_builder: Callable[..., Any] = build_tap,
        mic_recorder_factory: Callable[..., Any] | None = None,
        voiceprint_store_factory: Callable[[], Any] | None = None,
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
        self._subscribe_jobs = subscribe_jobs
        self._unsubscribe_jobs = unsubscribe_jobs
        self._watching_jobs = False
        self._facade_factory = facade_factory or _default_facade_factory
        self._dictation_factory = dictation_factory or _default_dictation_factory
        self._tap_probe = tap_probe
        self._tap_builder = tap_builder
        self._mic_factory = mic_recorder_factory or _default_mic_factory
        self._store_factory = voiceprint_store_factory
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
        self._stop_lock = threading.Lock()
        #: Self-voiceprint state (TASK-31826). `_voice_load` caches the ONE
        #: store read (vector + rail state) until something changes it;
        #: `_retained_diarizer` is the worker kept alive past Stop for a
        #: learning offer, and is closed on accept/decline/lapse.
        self.voice_match = VoiceMatchState("off", None)
        self._voice_load: tuple[list[float] | None, VoiceMatchState] | None = None
        # `Any`, not `Diarizer`: the offer/enrollment paths use the embedding
        # half of the backend (`export_centroid`, `enroll_from_pcm`,
        # `wait_ready`), which is deliberately not in the session's protocol.
        self._retained_diarizer: Any = None
        self._pending_offer: LearningOffer | None = None
        self._offer_handed = False

    # ---- self voiceprint --------------------------------------------------
    def _voiceprint_store(self) -> Any:
        """The encrypted voiceprint store (injectable; lazily imported).

        `Audio/voiceprint.py` pulls the config-encryption stack, so it is
        imported HERE, never at module scope -- boot must not touch it.
        """
        if self._store_factory is not None:
            return self._store_factory()
        from .voiceprint import default_store

        return default_store()

    def invalidate_voiceprint(self) -> None:
        """Forget the cached load (Settings deleted, imported or replaced one)."""
        self._voice_load = None

    def _load_voiceprint(self) -> tuple[list[float] | None, VoiceMatchState]:
        """The stored voiceprint and the rail's match state, read once.

        The read runs on its OWN thread and is joined for at most
        `VOICEPRINT_LOAD_TIMEOUT_S` (spec §3.1): on macOS the first keyring
        access can raise a Keychain prompt, and a meeting Start must never
        wait behind it -- a blocked read simply reports "keyring_locked" and
        the meeting runs without matching.

        Returns:
            `(vector | None, state)`, cached until `invalidate_voiceprint` --
            a FAILED read included, so a locked keyring is not re-prompted on
            every rail refresh. Settings calls `invalidate_voiceprint` after
            an enrollment, import or delete; otherwise the next app run
            re-reads it.
        """
        if self._voice_load is not None:
            return self._voice_load
        outcome: list[Any] = [None]

        def _read() -> None:
            try:
                from .diarizer_worker import MODEL_ID

                outcome[0] = self._voiceprint_store().load(
                    expected_model_id=MODEL_ID, timeout_s=VOICEPRINT_LOAD_TIMEOUT_S
                )
            except Exception as exc:  # noqa: BLE001 - never raises into a meeting
                logger.warning("meeting: voiceprint load failed ({})", type(exc).__name__)

        thread = threading.Thread(target=_read, daemon=True, name="meeting-voiceprint-load")
        thread.start()
        thread.join(VOICEPRINT_LOAD_TIMEOUT_S)
        result = outcome[0]
        if result is None:
            # Still running -> the key read is blocked; finished with nothing
            # -> the store itself raised, which the record cannot recover from.
            reason = "keyring_locked" if thread.is_alive() else "cannot_decrypt"
            loaded: tuple[list[float] | None, VoiceMatchState] = (None, VoiceMatchState("off", reason))
        elif result.voiceprint is None:
            loaded = (None, VoiceMatchState("off", result.reason or "no_voiceprint"))
        else:
            loaded = (list(result.voiceprint.centroid), VoiceMatchState("on", None))
        self._voice_load = loaded
        return loaded

    def _voice_match_for(self, mode: str) -> tuple[list[float] | None, VoiceMatchState]:
        """The vector to hand the diarizer for `mode`, plus the state to show.

        Args:
            mode: The meeting's effective mode (``"room"`` / ``"call"``).

        Returns:
            `(vector | None, state)`. Matching applies only when the setting
            is on AND the mode allows it -- in PLAIN call mode the mic channel
            already is the user, so remote clusters are never compared (spec
            §3.4, the largest false-positive path), and the store is not even
            read, so no Keychain prompt is raised for a meeting that could
            never match.
        """
        if not self.settings.voice_match:
            return None, VoiceMatchState("off", "disabled")
        if mode == "call" and not self.settings.diarize_mic_channel:
            return None, VoiceMatchState("off", "plain_call_mode")
        return self._load_voiceprint()

    # ---- prepare ----------------------------------------------------------
    def prepare(self) -> PrepareResult:
        """Probe everything a meeting needs, without touching the recorder.

        Safe to call repeatedly; `apply_device_choice` clears the cached
        result so the next call re-probes the new source.

        Returns:
            The probe outcome, also stored on `self.prepared`.
        """
        cfg = resolve_effective_config()
        provider = self.settings.provider if self.settings.provider != "auto" else getattr(cfg, "provider", "auto")
        model = self.settings.model or (getattr(cfg, "model", "") or "")
        self._cfg = type("Cfg", (), {"provider": provider, "model": model or None, "language": getattr(cfg, "language", "en")})()
        if self._facade is None:
            self._facade = self._facade_factory()
        tap_mode = self._tap_probe(system_source=self.settings.system_source)
        missing = diarization_requirements()
        recoverable = tuple(scan_recoverable(self.settings.recordings_dir))
        devices: tuple[str, ...] = ()
        capture_error: str | None = None
        try:
            probe_recorder = self._mic_factory(use_vad=False, retain_audio=False, chunk_size=320)
            devices = tuple(str(d.get("name", "")) for d in probe_recorder.get_audio_devices() if d.get("name"))
        except Exception as exc:  # noqa: BLE001 - no backend: pickers stay empty
            # task-31748: `str(exc)` can embed a filesystem path (a missing
            # backend module's path, say) -- redact it.
            logger.info("meeting device enumeration unavailable: {}", redact_user_paths(str(exc)))
            capture_error = _missing_recorder_message(exc)
        # The tap decides the mode a meeting started now would run in, which
        # is what says whether voice matching applies (spec §3.4). `start()`
        # re-reads it from the capture, which is authoritative once the tap
        # has actually come up.
        self.voice_match = self._voice_match_for("room" if tap_mode.kind == "unavailable" else "call")[1]
        self.prepared = PrepareResult(
            tap_mode=tap_mode, provider=provider, model=model or "",
            diarization_available=not missing, diarization_missing=missing,
            live_diarization_active=(
                self.settings.live_diarization and not missing
                and self.settings.diarizer_backend == "local"
            ),
            recoverable=recoverable, input_devices=devices, capture_error=capture_error,
            voice_match=self.voice_match,
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
        """Create the meeting folder and start recording.

        Returns:
            The running session (also stored on `self.session`).

        Raises:
            RuntimeError: A meeting is already running, or the session
                failed to start -- in which case the folder, its writers and
                the transcript sink are cleaned up first.
        """
        # Late import (C1): `meeting_capture` needs numpy, and `app.py`
        # imports this module at module scope. A meeting is the only thing
        # that needs the mixer, so nothing pays for numpy until Start.
        from .meeting_capture import MeetingCapture

        if self.prepared is None:
            self.prepare()
        # Held for the whole body, OUTSIDE `self._lock`: a Start landing
        # during an in-flight stop() blocks here until that stop has fully
        # finalised the old session, instead of racing it to open a second
        # one. stop() takes `_lock` only briefly and releases it before
        # taking `_stop_lock`, so there is no lock-order cycle.
        with self._stop_lock:
            # A new meeting lapses the previous one's unanswered learning
            # offer (spec §3.4) and releases the worker kept for it. INSIDE
            # `_stop_lock`: an in-flight stop() is still driving that worker's
            # batch pass, and closing it from under that pass would cost the
            # previous meeting its authoritative labels.
            self._clear_offer()
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
                try:
                    capture = MeetingCapture(
                        mic_recorder_factory=self._mic_factory, tap=tap, writers=writers,
                        vad_factory=self._vad_factory,
                        mic_device_name=self.settings.mic_device or None,
                    )
                except Exception:
                    # The constructor resolves numpy; a raise here would
                    # otherwise leak the folder and three open WAV handles.
                    for writer in writers.values():
                        writer.close()
                    shutil.rmtree(folder, ignore_errors=True)
                    raise
                meta = MeetingMeta(
                    folder=folder, mode=capture.mode,
                    started_at=datetime.now().isoformat(timespec="seconds"),
                    mic_device=self.settings.mic_device or "default",
                    system_source=self.prepared.tap_mode.reason,
                    provider=self.prepared.provider, model=self.prepared.model,
                    user_display_name=meeting_user_display_name(),
                    diarize_mic_channel=self.settings.diarize_mic_channel,
                )
                # Two independent mechanisms, deliberately NOT conflated
                # (Qodo Q12): the live backend's authoritative Stop pass is
                # driven by `MeetingSession`'s own `_diarizer` (below), while
                # the sink's `post_diarize` only asks the Library ingest for a
                # SECOND, offline diarization of mixed.wav that knows nothing
                # of the live cluster ids or the user's renames. Forcing the
                # latter on because a live diarizer exists overrode an
                # explicit `post_diarize = false` and could relabel the
                # Library copy with generic ids.
                # Self-voiceprint matching (spec §3.4): the mode the capture
                # actually settled on decides whether the vector is handed
                # over at all, and the rail's live verdict is refreshed from
                # it (a tap that failed to start downgrades call -> room).
                voiceprint, self.voice_match = self._voice_match_for(capture.mode)
                if self.prepared is not None:
                    self.prepared.voice_match = self.voice_match
                diarizer = build_diarizer(self.settings, voiceprint=voiceprint)
                # The learning offer needs this worker alive AFTER Stop to
                # export the matched cluster's centroid, so the owner takes
                # over the close whenever an offer could plausibly follow
                # (`_clear_offer` then closes it, on every path out).
                retain = bool(
                    self.settings.voice_learn_offer and diarizer is not None and self._store_readable()
                )
                self._retained_diarizer = diarizer if retain else None
                self.local_sink = LocalMeetingSink(
                    folder, submit=self._submit_on_ui_thread,
                    post_transcribe=self.settings.post_transcribe,
                    post_diarize=self.settings.post_diarize,
                )
                facade, cfg = self._facade, self._cfg
                session = MeetingSession(
                    meta=meta, capture=capture,
                    dictation_factory=lambda cap: self._dictation_factory(cap, facade, cfg),
                    sinks=[self.local_sink],
                    diarizer=diarizer,
                    close_diarizer_on_stop=not retain,
                )
                self.session = session
                # A RAISING start() has to run the same cleanup as a False
                # one: `self.session` is already assigned, so leaving it set
                # would wedge the owner -- `is_active` stays False (the
                # session is in "error"/"idle"), yet a later stop() would
                # drive a session that never started (final review, I3).
                try:
                    started = session.start()
                    failure = "meeting failed to start (see log)"
                except Exception as exc:  # noqa: BLE001 - re-raised below with cleanup done
                    started = False
                    failure = f"meeting failed to start: {exc}"
                if not started:
                    capture.stop_recording()  # closes the writers; tolerates a never-started mic
                    self.local_sink.close()   # the JSONL handle, if on_started got that far
                    shutil.rmtree(folder, ignore_errors=True)
                    self.session = None
                    self.local_sink = None
                    # A meeting that never started has no Stop to close the
                    # diarizer the owner just claimed -- do it here or leak
                    # the worker subprocess.
                    self._close_retained_diarizer()
                    raise RuntimeError(failure)
                self._start_watchdog()
                return session

    def pause(self) -> None:
        """Pause the running meeting, if there is one."""
        if self.session is not None:
            self.session.pause()

    def resume(self) -> None:
        """Resume a paused meeting, if there is one."""
        if self.session is not None:
            self.session.resume()

    def apply_device_choice(self, kind: str, value: str) -> None:
        """Persist a rail picker choice and force the next `prepare()` to re-probe.

        Args:
            kind: ``"mic"`` or ``"system"``.
            value: The device name; ``"default"`` for the mic means "no
                explicit device" and is stored as an empty string.
        """
        from tldw_chatbook.config import save_setting_to_cli_config

        key = "mic_device" if kind == "mic" else "system_source"
        value = "" if (kind == "mic" and value == "default") else value
        setattr(self.settings, key, value)
        save_setting_to_cli_config("meetings", key, value)
        self.prepared = None   # next prepare() re-probes with the new source

    def stop(self, reason: str = "user") -> MeetingResult | None:
        """Finalise the running meeting; idempotent for sequential callers.

        Args:
            reason: Why it ended (``"user"``, ``"mic_lost"``,
                ``"disk_error"``, ``"shutdown"``), recorded in the result.

        Returns:
            The meeting's result, or the last one when no session is running.
        """
        # ponytail: claim under the owner RLock, then run the (possibly
        # blocking, cross-thread) session.stop() outside it -- serialized by
        # a separate plain lock so a UI-thread callback that needs
        # self._lock during the ingest submit can never deadlock against it.
        with self._lock:
            session = self.session
            if session is None:
                return self.last_result
        with self._stop_lock:
            self._watchdog_stop.set()
            previous = self.last_result
            result = session.stop(reason=reason)  # idempotent for sequential callers
            if result is not None:
                self.last_result = result
                if result is not previous:
                    # One offer decision per MEETING: a second stop() hands
                    # back the same cached result and must not resurrect an
                    # offer the user already answered.
                    self._pending_offer = self._offer_for(result)
                    self._offer_handed = False
                    if self._pending_offer is None:
                        self._close_retained_diarizer()
            self._watch_ingest_job()
            return result if result is not None else self.last_result

    def shutdown(self) -> None:
        """App quit: finalise files, skip the ingest submit (spec §3.4)."""
        session = self.session
        if session is not None and self.is_active:
            sink = self.local_sink
            if sink is not None:
                sink._submit = lambda **kwargs: None
            self.stop(reason="shutdown")
        if self.local_sink is not None:
            self.local_sink.close()
        # App quit: an unanswered offer lapses and its worker goes with it.
        self._clear_offer()
        self._unwatch_ingest_job()

    # ---- learning offer ---------------------------------------------------
    def _store_readable(self) -> bool:
        """False only once a load has PROVED the store unreadable.

        An unattempted load (plain call mode, matching off) reads as "maybe":
        the offer path loads for itself when it gets there.
        """
        loaded = self._voice_load
        return loaded is None or loaded[1].reason not in ("cannot_decrypt", "keyring_locked")

    def _close_retained_diarizer(self) -> None:
        """Close the worker kept for a learning offer, if any. Idempotent."""
        diarizer, self._retained_diarizer = self._retained_diarizer, None
        if diarizer is None:
            return
        try:
            diarizer.close()
        except Exception as exc:  # noqa: BLE001 - best-effort teardown
            logger.warning("meeting: diarizer close failed ({})", type(exc).__name__)

    def _clear_offer(self) -> None:
        """Drop any pending offer and release the worker held for it."""
        self._pending_offer = None
        self._offer_handed = False
        self._close_retained_diarizer()

    def _offer_for(self, result: MeetingResult) -> LearningOffer | None:
        """Whether this meeting produced a sample worth learning from (§3.4).

        Args:
            result: The finished meeting.

        Returns:
            The offer, or None. Learning MERGES into an enrolled voiceprint,
            so with nothing stored there is nothing to improve -- that is what
            the explicit "Enroll my voice" flow is for.
        """
        if not self.settings.voice_learn_offer:
            return None
        if self._load_voiceprint()[0] is None:
            return None
        meta = result.meta
        folder = Path(meta.folder)
        if meta.matched_self and not meta.matched_self_overridden:
            return LearningOffer(kind="matched_cluster", folder=folder, cluster_id=meta.matched_self)
        if meta.mode == "call" and not meta.diarize_mic_channel and (folder / "you.wav").exists():
            # Plain call mode: the mic channel was never diarized, so there is
            # no cluster to export -- the recorded track is the sample, and
            # the screen asks "was it only you on the mic?" before accepting.
            return LearningOffer(kind="mic_channel", folder=folder)
        return None

    def learning_offer(self, result: MeetingResult) -> LearningOffer | None:
        """The one learning offer for `result`, or None.

        Idempotent per meeting (spec §3.4): the first call after a qualifying
        meeting returns the offer, every later call returns None.

        Args:
            result: The finished meeting the screen is showing.

        Returns:
            The offer to present, or None when this meeting has none (or its
            offer was already taken, answered, or lapsed).
        """
        offer = self._pending_offer
        if offer is None or self._offer_handed or Path(result.meta.folder) != offer.folder:
            return None
        self._offer_handed = True
        return offer

    def accept_learning(self, offer: LearningOffer) -> bool:
        """Merge the meeting's clean sample into the stored voiceprint.

        Args:
            offer: The offer `learning_offer` handed out.

        Returns:
            True when the voiceprint was updated. False -- never an exception
            -- when the offer is stale, the worker could not produce a
            centroid, or the store refused the merge (spec §6: the previous
            voiceprint stays intact either way). The worker kept for the offer
            is released on every path out.
        """
        if offer is None or offer is not self._pending_offer:
            return False
        try:
            sample = self._sample_for(offer)
            if sample is None:
                return False
            centroid, seconds = sample
            from .diarizer_worker import MODEL_ID

            self._voiceprint_store().merge_sample(centroid, weight=seconds, model_id=MODEL_ID)
            self._voice_load = None      # the vector moved; the next meeting re-reads it
            return True
        except Exception as exc:  # noqa: BLE001 - the offer reports failure (spec §6)
            logger.warning("meeting: voiceprint learning failed ({})", type(exc).__name__)
            return False
        finally:
            self._clear_offer()

    def _sample_for(self, offer: LearningOffer) -> tuple[list[float], float] | None:
        """The centroid (and its seconds) this offer would merge, or None."""
        diarizer = self._embedding_diarizer()
        if diarizer is None:
            return None
        if offer.kind == "matched_cluster":
            return diarizer.export_centroid(offer.cluster_id)
        pcm, sample_rate = _read_wav_pcm(offer.folder / "you.wav", LEARN_PCM_MAX_SECONDS)
        if not pcm:
            return None
        return diarizer.enroll_from_pcm(pcm, sample_rate)

    def decline_learning(self, offer: LearningOffer | None = None) -> None:
        """The user said no: keep nothing, release the worker."""
        self.dismiss_learning()

    def dismiss_learning(self) -> None:
        """The offer lapsed (the screen hid it): same cleanup as a decline."""
        self._clear_offer()

    # ---- explicit enrollment ----------------------------------------------
    def _embedding_diarizer(self, progress: Callable[[str], None] | None = None) -> Any | None:
        """A diarizer able to embed audio, spawning (and warming) one if needed.

        Reuses the worker kept from the last meeting when there is one --
        that one is already warm. A freshly spawned worker is stored as the
        retained diarizer too, so `_clear_offer` closes it on every path and
        the subprocess is never leaked.

        Args:
            progress: Optional `(status) -> None` for the warm-up indicator.

        Returns:
            The diarizer, or None when one cannot be built or never warms up.
        """
        diarizer = self._retained_diarizer
        if diarizer is not None and hasattr(diarizer, "enroll_from_pcm"):
            return diarizer
        try:
            from .diarizer_local import READY_TIMEOUT_S, SpeechBrainDiarizer

            spawned = SpeechBrainDiarizer(max_speakers=self.settings.max_speakers)
        except Exception as exc:  # noqa: BLE001 - best-effort, never raises at the user
            logger.warning("meeting: diarizer unavailable for embedding ({})", type(exc).__name__)
            return None
        self._retained_diarizer = spawned
        self._report(progress, "warming up")
        if not spawned.wait_ready(READY_TIMEOUT_S):
            # First run downloads the ECAPA model; giving up here is the only
            # honest answer -- the embed op would silently return None anyway.
            self._close_retained_diarizer()
            return None
        return spawned

    @staticmethod
    def _report(progress: Callable[[str], None] | None, status: str) -> None:
        """Best-effort progress ping; a raising callback never fails a run."""
        if progress is None:
            return
        try:
            progress(status)
        except Exception as exc:  # noqa: BLE001
            logger.debug("meeting: enrollment progress callback failed ({})", type(exc).__name__)

    def enroll_from_mic(
        self, seconds: float = 30.0, progress: Callable[[str], None] | None = None
    ) -> EnrollResult:
        """Record a mic sample and store it as the user's voiceprint (§3.4).

        Synchronous -- the caller runs it on a worker thread. The sample lives
        in memory only: it is never written to a file, and neither it nor the
        resulting vector is ever logged. This is where the store's key is
        first minted, deliberately: the user initiated this flow, so a
        Keychain prompt is expected here rather than at a meeting Start.

        Args:
            seconds: How long to record.
            progress: Optional `(status) -> None` receiving short, static
                strings ("warming up", "recording", "embedding") -- never a
                path, a level, or anything derived from the audio.

        Returns:
            `EnrollResult(ok=True, seconds=<embedded>)`, or `ok=False` with a
            static reason. Refused outright while a meeting is running.
        """
        if self.is_active:
            return EnrollResult(ok=False, reason="capture_busy")
        try:
            recorder = self._mic_factory(use_vad=False, retain_audio=True, chunk_size=320)
        except Exception as exc:  # noqa: BLE001 - no usable recorder: say so
            return EnrollResult(ok=False, reason=_missing_recorder_message(exc) or type(exc).__name__)
        diarizer = self._embedding_diarizer(progress=progress)
        if diarizer is None:
            return EnrollResult(ok=False, reason="diarizer_unavailable")
        try:
            self._report(progress, "recording")
            try:
                if not recorder.start_recording(callback=None, save_to_file=None):   # memory only
                    return EnrollResult(ok=False, reason="capture_failed")
                self._sleep(max(0.0, float(seconds)))
                pcm = recorder.stop_recording() or b""
            except Exception as exc:  # noqa: BLE001
                return EnrollResult(ok=False, reason=_missing_recorder_message(exc) or type(exc).__name__)
            if not pcm:
                return EnrollResult(ok=False, reason="no_audio")
            self._report(progress, "embedding")
            sample = diarizer.enroll_from_pcm(pcm, MIC_SAMPLE_RATE)
            if sample is None:
                return EnrollResult(ok=False, reason="embed_failed")
            centroid, embedded_s = sample
            try:
                from .diarizer_worker import MODEL_ID
                from .voiceprint import Voiceprint, unit_normalise

                now = datetime.now().isoformat(timespec="seconds")
                self._voiceprint_store().save(Voiceprint(
                    model_id=MODEL_ID, centroid=unit_normalise(centroid),
                    sample_count=float(embedded_s), meetings_contributed=0,
                    created_at=now, updated_at=now,
                    threshold_used=float(self.settings.voice_match_threshold),
                ))
            except Exception as exc:  # noqa: BLE001 - types only, never the vector
                logger.warning("meeting: voiceprint save failed ({})", type(exc).__name__)
                return EnrollResult(ok=False, reason="store_unavailable")
            self._voice_load = None      # the next meeting reads the new print
            return EnrollResult(ok=True, seconds=float(embedded_s))
        finally:
            # The worker used here (kept or freshly spawned) goes now, and an
            # explicit enrollment supersedes any pending learning offer.
            self._clear_offer()

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
                # A disk fault names the meeting folder under the user's
                # recordings dir; redact before it reaches the log file.
                logger.error("meeting watchdog: capture fault {}", redact_user_paths(str(capture.fault)))
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
    def _watch_ingest_job(self) -> None:
        """Watch the ingest registry until this meeting's job settles.

        `keep_raw_tracks = false` is only honoured once the Library job that
        consumed `mixed.wav` finishes, which happens long after `stop()`
        returns -- and the Meetings screen may be gone by then. The owner
        outlives the screen, so the wait lives here (Qodo Q12).
        """
        if self.settings.keep_raw_tracks or self._subscribe_jobs is None or self._watching_jobs:
            return
        sink = self.local_sink
        if sink is None or not sink.job_id:
            return
        self._watching_jobs = True
        self._subscribe_jobs(self._on_ingest_jobs_changed)

    def _unwatch_ingest_job(self) -> None:
        """Drop the registry listener, if one is registered."""
        if not self._watching_jobs:
            return
        self._watching_jobs = False
        if self._unsubscribe_jobs is not None:
            self._unsubscribe_jobs(self._on_ingest_jobs_changed)

    def _on_ingest_jobs_changed(self) -> None:
        """Registry listener (UI thread): clean up once the job is terminal."""
        sink = self.local_sink
        job_id = getattr(sink, "job_id", None)
        state = self._job_state(job_id) if job_id else None
        if state not in TERMINAL_JOB_STATES:
            return
        self.cleanup_raw_tracks_if_done()   # a no-op unless the state is "done"
        self._unwatch_ingest_job()

    def cleanup_raw_tracks_if_done(self) -> bool:
        """Delete you/others once the ingest job is done (best effort, spec §5).

        Returns:
            True when the job was done and the deletion pass ran. Individual
            unlink failures are logged and skipped: a raw track that cannot
            be removed is a disk-space problem, never a lost meeting.
        """
        if self.settings.keep_raw_tracks or self.last_result is None or self.local_sink is None:
            return False
        job_id = self.local_sink.job_id
        if not job_id or self._job_state(job_id) != "done":
            return False
        folder = Path(self.last_result.meta.folder)
        for name in ("you.wav", "others.wav"):   # never mixed.wav: that is the recording
            path = folder / name
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                logger.warning("meeting raw track cleanup failed: {}", redact_user_paths(str(exc)))
        return True


def _config_accessors():
    """Late import seam (tests monkeypatch this)."""
    from tldw_chatbook.config import get_cli_setting, get_user_data_dir

    return get_cli_setting, get_user_data_dir


def build_meeting_session_owner(app: Any) -> "MeetingSessionOwner":
    """Wire the owner to a `TldwCli`: config, ingest registry, UI-thread marshalling.

    Args:
        app: The running `TldwCli`. Its ingest registry is UI-thread-only, so
            every call into it (submit, state read, listener registration) is
            marshalled through `app.call_from_thread`.

    Returns:
        The owner, built from the `[meetings]` config section.

    Raises:
        pydantic.ValidationError: The `[meetings]` config is unusable.
    """
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

    # The ingest registry is UI-thread-only, listener registration included.
    def subscribe_jobs(listener: Callable[[], None]) -> None:
        marshal(app.library_ingest_jobs.add_listener, listener)

    def unsubscribe_jobs(listener: Callable[[], None]) -> None:
        marshal(app.library_ingest_jobs.remove_listener, listener)

    return MeetingSessionOwner(
        settings=settings, call_from_thread=marshal, submit_ingest=submit_ingest, job_state=job_state,
        subscribe_jobs=subscribe_jobs, unsubscribe_jobs=unsubscribe_jobs,
    )
