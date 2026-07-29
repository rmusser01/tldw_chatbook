"""Headless Console voice-dictation controller.

Deliberately free of Textual imports: the widget layer owns rendering and
threading policy, this module owns availability, provider resolution, and the
dictation state machine. That split is what makes the state machine unit
testable without a running app.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from ..config import get_cli_setting

# Capture backends, in preference order. AudioRecordingService picks between
# them itself; we only need to know whether at least one exists.
CAPTURE_MODULES: tuple[str, ...] = ("pyaudio", "sounddevice")

# Provider id -> required import name(s), ALL of which must resolve for the
# provider to count as installed. Mirrors `get_available_providers()` in
# `Local_Ingestion/transcription_service.py` -- same providers, same
# declaration order (also the resolve() fallback preference below), same
# detection rule per provider:
#   - parakeet-onnx           find_spec("onnx_asr")
#   - parakeet-mlx            find_spec("parakeet_mlx"), darwin only
#   - lightning-whisper-mlx   find_spec("lightning_whisper_mlx"), darwin only
#   - faster-whisper          find_spec("faster_whisper")
#   - qwen2audio              find_spec("torch") AND find_spec("transformers")
#   - parakeet / canary       find_spec("nemo") (NVIDIA NeMo)
#
# `remote-whisper` is deliberately excluded even though the service lists it:
# it needs only `requests`, so it would always resolve as "installed", and
# the dictation service's privacy mode (local_only, default True) rejects
# non-local providers outright. Adding it here would let resolve() hand it
# back as the chosen provider, which the service would then silently swap
# out for something else -- exactly the silent-substitution bug this module
# exists to prevent. Do not "complete the set" by adding it.
LOCAL_PROVIDER_MODULES: dict[str, tuple[str, ...]] = {
    "parakeet-onnx": ("onnx_asr",),
    "parakeet-mlx": ("parakeet_mlx",),
    "lightning-whisper-mlx": ("lightning_whisper_mlx",),
    "faster-whisper": ("faster_whisper",),
    "qwen2audio": ("torch", "transformers"),
    "parakeet": ("nemo",),
    "canary": ("nemo",),
}

# Providers usable only on Apple Silicon. Mirrors
# `transcription_service._optional_module_available()`'s
# `sys.platform == "darwin"` gate: a force-installed package on Linux must
# not be reported as usable, or the button would light up and then fail at
# capture time.
DARWIN_ONLY_PROVIDERS: frozenset[str] = frozenset(
    {"parakeet-mlx", "lightning-whisper-mlx"}
)

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


def _provider_installed(provider: str, module_names: tuple[str, ...]) -> bool:
    """Return True when `provider`'s required module(s) all resolve.

    Darwin-only providers additionally require `sys.platform == "darwin"`,
    checked before touching `find_spec` at all so a non-darwin platform never
    even looks at whether the module happens to be importable.
    """
    if provider in DARWIN_ONLY_PROVIDERS and sys.platform != "darwin":
        return False
    return all(_module_installed(name) for name in module_names)


def installed_local_providers() -> tuple[str, ...]:
    """Return the local transcription providers that are actually installed."""
    return tuple(
        provider
        for provider, module_names in LOCAL_PROVIDER_MODULES.items()
        if _provider_installed(provider, module_names)
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
        # Per-attempt (not per-instance) latch: set when the service reports a
        # real cause through `on_error`, cleared at the top of every
        # `_run_begin()` so a failed attempt can never silence a later one.
        self._error_reported = False

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
        """Begin capture. Rejected unless currently idle and never abandoned."""
        with self._state_lock:
            if self._abandoned or self._state != STATE_IDLE:
                logger.debug(
                    "Console dictation start ignored (abandoned={}, state={})",
                    self._abandoned,
                    self._state,
                )
                return
            self._state = STATE_PREPARING
        self._emit(VoiceStateChanged(STATE_PREPARING))

        # Each `try` below covers only the call that can crash unexpectedly,
        # never the `_fail()` that handles its result: `_fail()`'s own emit
        # can itself raise (that's the whole point of Finding 2), and if that
        # raise were caught by one of these `except` blocks it would trigger
        # a second, mislabeled `_fail()` describing the plumbing exception
        # instead of the real cause.
        try:
            availability = probe()
        except Exception as exc:  # noqa: BLE001 - a probe crash must not wedge preparing
            logger.opt(exception=True).warning("Console dictation availability probe crashed")
            self._fail(str(exc))
            return

        if not availability.ok:
            self._fail(availability.reason, availability.remedy)
            return

        try:
            effective = resolve()
        except Exception as exc:  # noqa: BLE001 - a resolve crash must not wedge preparing
            logger.opt(exception=True).warning("Console dictation provider resolution crashed")
            self._fail(str(exc))
            return

        if effective is None:
            self._fail(PROVIDER_REASON, PROVIDER_REMEDY)
            return

        try:
            if effective.was_overridden and not self._override_announced:
                self._override_announced = True
                self._emit(
                    VoiceProviderOverridden(
                        configured=effective.configured_provider,
                        effective=effective.provider,
                    )
                )

            self._spawn(lambda: self._begin(effective))
        except Exception as exc:  # noqa: BLE001 - override-announce/spawn must not wedge preparing
            logger.opt(exception=True).warning("Console dictation could not be spawned")
            self._fail(str(exc))
            return

    def _begin(self, effective: EffectiveConfig) -> None:
        """Blocking half of start(); always runs via `spawn`.

        A thread boundary: when `spawn` is inline -- the default in nearly
        every test, and any future ad-hoc caller -- this method runs
        synchronously inside `start()`'s own try/except around the `spawn()`
        call. That guard exists to catch a real `spawn()` failing to
        *schedule* work and must stay in place, so nothing raised in here may
        propagate back through `spawn()` into it: `_run_begin()`'s own
        `_fail()` calls have a raising emit as their whole reason for
        existing (Finding 2), and letting that reach `start()`'s guard would
        fire a second, mislabeled `VoiceFailed` describing this method's
        plumbing instead of the real cause -- the exact cascade N1 fixed in
        `start()`, recurring one call frame deeper.
        """
        try:
            self._run_begin(effective)
        except Exception:  # noqa: BLE001 - nothing may escape _begin(); see docstring
            logger.opt(exception=True).warning("Console dictation _begin() raised unexpectedly")

    def _run_begin(self, effective: EffectiveConfig) -> None:
        """The actual work of `_begin()`, shielded from its caller by `_begin()`."""
        # Cleared per attempt, before anything can set it: `on_error` fires
        # synchronously from inside `start_dictation()` (see
        # `_report_service_error`), and a latch left over from an earlier
        # failed attempt would silence this attempt's fallback report.
        self._error_reported = False
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
                on_error=self._report_service_error,
                save_audio=self.save_audio_requested,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            logger.opt(exception=True).warning("Console dictation failed to start")
            # `on_error` is invoked from *inside* `start_dictation()`, i.e.
            # from inside the `try` above, so this `exc` can be the real
            # cause's own `_fail()` emit raising rather than a start failure.
            # Reporting again would bury the real cause under plumbing --
            # the same latch `_fail_not_started()` consults, for the same
            # reason.
            if self._error_reported:
                logger.debug(
                    "Console dictation start crashed after the service reported the cause"
                )
                return
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
            self._claim_service()  # drop it; the service already cleaned up
            self._fail_not_started()
            return

        self._enter_listening()

    def _report_service_error(self, error: Any) -> None:
        """Turn a service-reported error into a failure, recording that we did.

        `LazyLiveDictationService` reports through this callback
        *synchronously, from inside `start_dictation()`*, and then returns
        `False` rather than raising -- all three of its failure branches do
        (`dictation_service_lazy.py` lines 285-290, 323-329 and 332-335, each
        `self._notify_error(...)` followed by `return False`). Without the
        latch, that one failure produces two `VoiceFailed` events: the real
        cause from here, then `_fail_not_started()`'s generic one, which
        arrives *last* and buries the actionable diagnostic in the UI.

        Args:
            error: The exception the service reported.
        """
        # Set before `_fail()`, which emits and can therefore raise: the
        # report has happened either way, and the service's own
        # `_notify_error()` only logs whatever escapes this callback.
        self._error_reported = True
        # Claim and release the service *before* reporting: a mid-session
        # error (state LISTENING, `self._service` already claimed by
        # `_run_begin()`) must not leave a live recorder behind an idle
        # machine, and must not be silently orphaned when a retry claims a
        # second service. During the startup path there is nothing to claim
        # yet (`_claim_service()` returns None), so this is a no-op there.
        # `_release()` never raises, so it cannot disturb the `_fail()`
        # raising-emit contract this callback depends on (see
        # `_run_begin`'s `except` and the `_error_reported` latch above).
        service = self._claim_service()
        if service is not None:
            self._release(service)
        self._fail(str(error))

    def _fail_not_started(self) -> None:
        """Report that `start_dictation()` returned `False`.

        Stays quiet in two cases. If `abandon()` landed in the narrow window
        between the claim above and this check, the controller is already
        idle and torn down, so a `VoiceFailed`/`VoiceStateChanged(idle)` pair
        here would be noise on top of teardown. And if the service already
        told us *why* it could not start (see `_report_service_error`), this
        generic message would land second and bury that real cause.
        """
        if self._abandoned:
            return
        if self._error_reported:
            logger.debug(
                "Console dictation start failed; real cause already reported by the service"
            )
            return
        self._fail("Could not start the microphone.")

    def _enter_listening(self) -> None:
        """Atomically transition to `listening`, re-checking abandonment.

        Between claiming the service above (under `_state_lock`) and this
        call, `abandon()` may have run on another thread -- a real one in
        production, since `_begin()` runs on a worker thread while `abandon()`
        fires from the UI thread -- and already released the microphone and
        returned the machine to idle. Re-checking `_abandoned` here, under
        the same lock, closes that window instead of stomping the state back
        to `listening` with no service behind it.
        """
        with self._state_lock:
            if self._abandoned:
                return
            self._state = STATE_LISTENING
        self._emit(VoiceStateChanged(STATE_LISTENING))

    def stop(self) -> None:
        """End capture and commit. No-op unless currently listening."""
        with self._state_lock:
            if self._state != STATE_LISTENING:
                logger.debug("Console dictation stop ignored in state {}", self._state)
                return
            self._state = STATE_FINISHING
        # Same guard `start()` carries, for the same reason: nothing else
        # unwinds `finishing`, so a raising emit or a `spawn()` that fails to
        # schedule would wedge the machine there forever with `is_active`
        # true. `_finish()` is an exception boundary (see its docstring), so
        # with an inline `spawn` this `try` cannot transitively swallow
        # `_finish()`'s own `_fail()` and cascade a mislabeled failure.
        try:
            self._emit(VoiceStateChanged(STATE_FINISHING))
            self._spawn(self._finish)
        except Exception as exc:  # noqa: BLE001 - finishing must never wedge
            logger.opt(exception=True).warning("Console dictation could not be finished")
            # The microphone is live and no worker will ever run `_finish()`
            # now, so drop it here rather than leave it recording behind an
            # idle state machine.
            service = self._claim_service()
            if service is not None:
                self._release(service)
            self._fail(str(exc))
            return

    def _finish(self) -> None:
        """Blocking half of stop(); always runs via `spawn`.

        An exception boundary, exactly like `_begin()`: with an inline
        `spawn` this runs synchronously inside `stop()`'s try around the
        `spawn()` call, and `_run_finish()`'s `_fail()` has a raising emit as
        its whole reason for existing -- letting that reach `stop()`'s guard
        would re-fire a second, mislabeled `VoiceFailed` describing this
        method's plumbing instead of the real cause.
        """
        try:
            self._run_finish()
        except Exception:  # noqa: BLE001 - nothing may escape _finish(); see docstring
            logger.opt(exception=True).warning("Console dictation _finish() raised unexpectedly")

    def _run_finish(self) -> None:
        """The actual work of `_finish()`, shielded from its caller by `_finish()`."""
        service = self._claim_service()
        try:
            if service is not None:
                service.stop_dictation()
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning("Console dictation failed to stop")
            self._fail(str(exc))
            return
        # Belt-and-braces: `LazyLiveDictationService.stop_dictation()` has
        # historically returned successfully without releasing capture, so
        # the Console does not trust the dependency alone. `stop_recording()`
        # early-returns when not already recording, so releasing again here
        # cannot double-stop a service that already released itself
        # correctly -- it logs a warning at worst.
        if service is not None:
            self._release(service)
        self._enter_idle()

    def _enter_idle(self) -> None:
        """Atomically return to `idle`, re-checking abandonment.

        The mirror of `_enter_listening()`, and needed for the same reason:
        `_run_finish()` runs on a worker thread while `abandon()` fires from
        the UI thread, so teardown can complete while this is still in
        flight. Announcing `idle` again afterwards would emit a state change
        for a controller that has already been torn down -- and a later task
        treats `VoiceStateChanged(idle)` as the trigger to send a deferred
        message.
        """
        with self._state_lock:
            if self._abandoned:
                return
            self._state = STATE_IDLE
        self._emit(VoiceStateChanged(STATE_IDLE))

    def _claim_service(self) -> Any | None:
        """Take sole ownership of the current service, under `_state_lock`.

        Every other read-and-clear of `self._service` is serialized against
        `abandon()` this way. Without the lock, `abandon()` on the UI thread
        and `_run_finish()` on a worker can both come away with the same
        service (double release), or `_run_finish()` can call
        `stop_dictation()` on one `abandon()` has already released -- which
        lands in its `except` and reports a spurious failure after teardown.

        Returns:
            The service that was held, or None if there was none to take.
        """
        with self._state_lock:
            service, self._service = self._service, None
        return service

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
        """Best-effort microphone release, skipping the 2s join. Never raises.

        Used by `abandon()` at teardown, by `_run_begin()` when `abandon()`
        won the race, and by `stop()` when no worker will ever run
        `_finish()`.

        Args:
            service: The dictation service instance to release.
        """
        try:
            audio = getattr(service, "_audio_service", None)
            if audio is not None and hasattr(audio, "stop_recording"):
                audio.stop_recording()
        except Exception:  # noqa: BLE001 - teardown must never raise
            logger.opt(exception=True).debug("Console dictation abandon failed")
