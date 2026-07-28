"""Headless Console voice-dictation controller.

Deliberately free of Textual imports: the widget layer owns rendering and
threading policy, this module owns availability, provider resolution, and the
dictation state machine. That split is what makes the state machine unit
testable without a running app.
"""

from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from ..config import get_cli_setting

# Capture backends, in preference order. AudioRecordingService picks between
# them itself; we only need to know whether at least one exists.
CAPTURE_MODULES: tuple[str, ...] = ("pyaudio", "sounddevice")

# Provider id -> import name. Only providers that run locally: the dictation
# service's privacy mode rejects everything else (see resolve()).
LOCAL_PROVIDER_MODULES: dict[str, str] = {
    "parakeet-mlx": "parakeet_mlx",
    "faster-whisper": "faster_whisper",
    "lightning-whisper-mlx": "lightning_whisper_mlx",
}

CAPTURE_REASON = "No microphone backend installed."
CAPTURE_REMEDY = (
    "Microphone support isn't installed. "
    "Install with: pip install 'tldw_chatbook[speech_recording]'"
)
PROVIDER_REASON = "No speech-to-text provider installed."
PROVIDER_REMEDY = (
    "No speech-to-text provider installed. "
    "Install with: pip install 'tldw_chatbook[transcription_faster_whisper]'"
)


@dataclass(frozen=True)
class Availability:
    """Whether dictation can run, and what to do about it if not."""

    ok: bool
    kind: str = "ok"  # "ok" | "missing-capture" | "missing-provider"
    reason: str = ""
    remedy: str = ""


def _module_installed(module_name: str) -> bool:
    """Return True when `module_name` is importable, without importing it.

    `find_spec` is required here rather than `optional_deps.check_dependency`,
    which really imports the module and would drag torch/NeMo into app start.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        # A namespace package with a broken parent raises rather than
        # returning None; treat that as "not usable".
        return False


def capture_available() -> bool:
    """Return True when at least one audio capture backend is installed."""
    return any(_module_installed(name) for name in CAPTURE_MODULES)


def installed_local_providers() -> tuple[str, ...]:
    """Return the local transcription providers that are actually installed."""
    return tuple(
        provider
        for provider, module_name in LOCAL_PROVIDER_MODULES.items()
        if _module_installed(module_name)
    )


def probe() -> Availability:
    """Report whether dictation is usable, distinguishing the two failures."""
    if not capture_available():
        logger.debug("Console dictation unavailable: no capture backend")
        return Availability(
            ok=False,
            kind="missing-capture",
            reason=CAPTURE_REASON,
            remedy=CAPTURE_REMEDY,
        )
    if not installed_local_providers():
        logger.debug("Console dictation unavailable: no transcription provider")
        return Availability(
            ok=False,
            kind="missing-provider",
            reason=PROVIDER_REASON,
            remedy=PROVIDER_REMEDY,
        )
    return Availability(ok=True)


DEFAULT_LANGUAGE = "en"


@dataclass(frozen=True)
class EffectiveConfig:
    """The transcription settings dictation will actually run with."""

    provider: str
    model: str | None
    language: str
    configured_provider: str
    was_overridden: bool


def resolve() -> EffectiveConfig | None:
    """Choose the provider before the dictation service gets the chance.

    `LazyLiveDictationService._initialize_streaming_transcriber` rewrites the
    provider to `parakeet-mlx` whenever privacy mode is on and the configured
    provider is not on its allowlist -- silently, and to an Apple-Silicon-only
    provider. Resolving here means the service is always handed a provider that
    is both local and installed, so that branch never fires.

    Returns:
        The settings to run with, or None when no local provider is installed.
    """
    installed = installed_local_providers()
    if not installed:
        return None

    # Key names matter and are easy to get wrong: the [transcription] section
    # uses `default_provider`/`default_model`/`default_language` (config.py:3333),
    # and the raw TOML section `STTSettings` is stored in the loaded config under
    # `STT_settings` (config.py:1548). Reading `provider`/`model`/`language` or
    # `STTSettings` silently returns the default and defeats this whole function.
    configured = get_cli_setting(
        "transcription", "default_provider", None
    ) or get_cli_setting("STT_settings", "default_stt_provider", "")
    configured = str(configured or "")

    if configured in installed:
        provider = configured
    else:
        # Preference order is LOCAL_PROVIDER_MODULES' declaration order.
        provider = installed[0]
        if configured:
            logger.info(
                "Console dictation provider '{}' unavailable; using '{}'",
                configured,
                provider,
            )

    model = get_cli_setting("transcription", "default_model", None)
    language = get_cli_setting("transcription", "default_language", DEFAULT_LANGUAGE)

    return EffectiveConfig(
        provider=provider,
        model=str(model) if model else None,
        language=str(language or DEFAULT_LANGUAGE),
        configured_provider=configured,
        was_overridden=bool(configured) and provider != configured,
    )


STATE_UNAVAILABLE = "unavailable"
STATE_IDLE = "idle"
STATE_PREPARING = "preparing"
STATE_LISTENING = "listening"
STATE_FINISHING = "finishing"
STATE_ERROR = "error"


@dataclass(frozen=True)
class VoicePartial:
    """In-flight recognizer text; superseded by the next partial or final."""

    text: str


@dataclass(frozen=True)
class VoiceFinal:
    """A segment the recognizer finalized on the silence threshold."""

    text: str


@dataclass(frozen=True)
class VoiceStateChanged:
    state: str


@dataclass(frozen=True)
class VoiceFailed:
    reason: str
    remedy: str = ""


@dataclass(frozen=True)
class VoiceProviderOverridden:
    configured: str
    effective: str


def default_service_factory(**kwargs: Any) -> Any:
    """Build a LazyLiveDictationService, importing it as late as possible.

    The import lives in the function body on purpose: `tldw_chatbook.Audio`
    (the package) chains to `transcription_service`, which imports
    faster-whisper and NeMo at module scope. Importing the submodule directly,
    at call time, keeps that cost off app start entirely.
    """
    from ..Audio.dictation_service_lazy import LazyLiveDictationService

    return LazyLiveDictationService(**kwargs)


class ConsoleVoiceInputController:
    """Own the dictation lifecycle without touching the UI.

    Threading policy lives in the caller: `spawn` runs a thunk off the UI
    thread (a Textual worker in the app, a direct call in tests), because both
    `start_dictation()` (cold model load) and `stop_dictation()` (a 2s thread
    join) block.
    """

    def __init__(
        self,
        *,
        emit: Callable[[Any], None],
        spawn: Callable[[Callable[[], None]], None],
        service_factory: Callable[..., Any] = default_service_factory,
    ) -> None:
        self._emit = emit
        self._spawn = spawn
        self._service_factory = service_factory
        self._service: Any | None = None
        self._state = STATE_IDLE
        self._state_lock = threading.Lock()
        self._override_announced = False
        self.save_audio_requested = False
        # One-way latch: once `abandon()` has run, an in-flight `_begin()`
        # (still building/starting a service on another thread, a cold model
        # load can take tens of seconds) must release what it built instead
        # of transitioning to `listening`. Never reset -- `abandon()` is a
        # teardown path (unmount, app quit); the controller is not expected
        # to `start()` again afterward.
        self._abandoned = False

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_active(self) -> bool:
        """True while a microphone is or is about to be live."""
        return self._state in (STATE_PREPARING, STATE_LISTENING, STATE_FINISHING)

    def _set_state(self, state: str) -> None:
        self._state = state
        self._emit(VoiceStateChanged(state))

    def _fail(self, reason: str, remedy: str = "") -> None:
        # Mutate first so a throwing `emit` cannot leave the machine wedged,
        # but keep VoiceFailed ahead of VoiceStateChanged(idle): the UI clears
        # its pending-send on the failure and fires it on the idle transition,
        # so reversing these would send the message on a failed dictation.
        self._state = STATE_IDLE
        self._emit(VoiceFailed(reason=reason, remedy=remedy))
        self._emit(VoiceStateChanged(STATE_IDLE))

    def start(self) -> None:
        """Begin capture. Rejected unless currently idle."""
        with self._state_lock:
            if self._state != STATE_IDLE:
                logger.debug("Console dictation start ignored in state {}", self._state)
                return
            self._state = STATE_PREPARING
        self._emit(VoiceStateChanged(STATE_PREPARING))

        try:
            availability = probe()
            if not availability.ok:
                self._fail(availability.reason, availability.remedy)
                return

            effective = resolve()
            if effective is None:
                self._fail(PROVIDER_REASON, PROVIDER_REMEDY)
                return
        except Exception as exc:  # noqa: BLE001 - a probe/resolve crash must not wedge preparing
            logger.opt(exception=True).warning("Console dictation availability check failed")
            self._fail(str(exc))
            return

        if effective.was_overridden and not self._override_announced:
            self._override_announced = True
            self._emit(
                VoiceProviderOverridden(
                    configured=effective.configured_provider,
                    effective=effective.provider,
                )
            )

        self._spawn(lambda: self._begin(effective))

    def _begin(self, effective: EffectiveConfig) -> None:
        """Blocking half of start(); always runs via `spawn`."""
        try:
            service = self._service_factory(
                transcription_provider=effective.provider,
                transcription_model=effective.model,
                language=effective.language,
                enable_commands=False,  # V2 owns voice commands, not V1
            )
            started = service.start_dictation(
                on_partial_transcript=lambda text: self._emit(VoicePartial(text)),
                on_final_transcript=lambda text: self._emit(VoiceFinal(text)),
                on_state_change=lambda _state: None,  # our state machine is authoritative
                on_error=lambda error: self._fail(str(error)),
                save_audio=self.save_audio_requested,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            logger.opt(exception=True).warning("Console dictation failed to start")
            self._fail(str(exc))
            return

        # Claim the freshly built service unless `abandon()` won the race
        # while the factory/`start_dictation()` call (a cold model load can
        # take tens of seconds) was still in flight. That check happened
        # against no service to release, so it's on us to release this one.
        with self._state_lock:
            if self._abandoned:
                claimed = False
            else:
                self._service = service
                claimed = True

        if not claimed:
            self._release(service)
            return

        if not started:
            self._service = None
            self._fail("Could not start the microphone.")
            return

        self._set_state(STATE_LISTENING)

    def stop(self) -> None:
        """End capture and commit. No-op unless currently listening."""
        with self._state_lock:
            if self._state != STATE_LISTENING:
                logger.debug("Console dictation stop ignored in state {}", self._state)
                return
            self._state = STATE_FINISHING
        self._emit(VoiceStateChanged(STATE_FINISHING))
        self._spawn(self._finish)

    def _finish(self) -> None:
        """Blocking half of stop(); always runs via `spawn`."""
        service, self._service = self._service, None
        try:
            if service is not None:
                service.stop_dictation()
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning("Console dictation failed to stop")
            self._fail(str(exc))
            return
        self._set_state(STATE_IDLE)

    def abandon(self) -> None:
        """Release the microphone without waiting on the 2s join.

        For teardown paths (unmount, app quit) where blocking would show up
        as a hang. Best effort by design. Safe to call from any state,
        including mid-`preparing`: sets a one-way latch that `_begin()`
        checks after it finishes building/starting a service, so a service
        that only comes into existence after this call still gets released
        instead of handed off to `listening`.
        """
        with self._state_lock:
            self._abandoned = True
            service, self._service = self._service, None
            self._state = STATE_IDLE
        if service is not None:
            self._release(service)

    def _release(self, service: Any) -> None:
        """Best-effort microphone release used by `abandon()`. Never raises.

        Args:
            service: The dictation service instance to release.
        """
        try:
            audio = getattr(service, "_audio_service", None)
            if audio is not None and hasattr(audio, "stop_recording"):
                audio.stop_recording()
        except Exception:  # noqa: BLE001 - teardown must never raise
            logger.opt(exception=True).debug("Console dictation abandon failed")
