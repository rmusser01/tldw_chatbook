# Console Voice Dictation (voice2voice V1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add push-to-talk voice dictation to the Console composer — `alt+r` toggles a microphone, finalized speech segments land in the draft as editable text, and an inline chip shows recording state and the in-flight partial.

**Architecture:** A headless `ConsoleVoiceInputController` (`Chat/console_voice_input.py`, zero Textual imports) owns availability probing, provider resolution, and the state machine. It receives three injected callables — `emit` (events out), `spawn` (run this thunk off the UI thread), `service_factory` (build the dictation service) — so it is fully unit-testable with a fake. `ConsoleComposerBar` owns the controller instance, wraps emitted dataclasses in Textual `Message`s via the thread-safe `post_message`, renders the chip, and inserts into the draft. `ChatScreen` owns only cross-cutting concerns: the `alt+r` action, the mic button press, the deferred send, and the shutdown triggers.

**Tech Stack:** Python 3.11+, Textual 8.2.7, `LazyLiveDictationService` (pyaudio/sounddevice capture + faster-whisper/parakeet-mlx/lightning-whisper-mlx transcription), pytest + `app.run_test()`.

**Source spec:** `Docs/superpowers/specs/2026-07-27-console-voice-dictation-design.md`

## Global Constraints

These apply to every task. Violating any one of them is grounds to reject a task.

- **Never import `tldw_chatbook.Audio` (the package).** It chains to `Local_Ingestion/transcription_service`, which at module scope runs `from faster_whisper import WhisperModel` and `import nemo.collections.asr`. Import `tldw_chatbook.Audio.dictation_service_lazy` **directly**, and only **inside a function body** — never at module scope.
- **`probe()` must never import a provider module.** Use `importlib.util.find_spec` only. Do **not** use `optional_deps.check_dependency()` — it calls `__import__(module_name)` (`Utils/optional_deps.py:539`) and would defeat the entire rule above.
- **Never use `call_from_thread`.** It blocks the calling thread, which here is the audio path. Use `post_message`, which is thread-safe (`textual/message_pump.py:882`).
- **Escape every transcript string** with `rich.markup.escape` before it reaches a `Static`, Button label, or tooltip. Whisper emits `[BLANK_AUDIO]`, `[Music]`, `[silence]`; Rich parses `[...]` as markup and raises `MarkupError`.
- **Workers:** `run_worker(..., thread=True, group="console-dictation", exit_on_error=False)`. Never `exclusive=True` without `group=`.
- **CSS goes in `tldw_chatbook/css/components/_agentic_terminal.tcss`** (the source). The bundle `tldw_chatbook/css/tldw_cli_modular.tcss` regenerates at boot — never hand-edit it.
- **Test markers:** files under `Tests/UI/` run wholesale in CI. Files anywhere else, including `Tests/Chat/`, need `pytestmark = pytest.mark.unit` or CI (`pytest -m unit`) never runs them.
- **Run pytest in the foreground** via the venv: `.venv/bin/python -m pytest ...`.
- Never edit the legacy voice widgets: `Widgets/voice_input_button.py`, `Widgets/voice_input_widget.py`, `UI/Chat_Modules/chat_voice_handler.py`, `UI/Dictation_Window*.py`.

---

### Task 1: Availability probe

Answers "can we dictate at all?" without importing anything heavy.

**Files:**
- Create: `tldw_chatbook/Chat/console_voice_input.py`
- Test: `Tests/Chat/test_console_voice_input.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Availability(ok: bool, kind: str, reason: str, remedy: str)` where `kind` is one of `"ok"`, `"missing-capture"`, `"missing-provider"`; module constants `CAPTURE_MODULES: tuple[str, ...]`, `LOCAL_PROVIDER_MODULES: dict[str, str]`; free function `capture_available() -> bool`; `installed_local_providers() -> tuple[str, ...]`.

- [ ] **Step 1: Write the failing test**

Create `Tests/Chat/test_console_voice_input.py`:

```python
"""Console voice dictation controller tests."""

from __future__ import annotations

import sys

import pytest

from tldw_chatbook.Chat import console_voice_input as cvi

pytestmark = pytest.mark.unit


def test_probe_reports_missing_capture(monkeypatch):
    """No pyaudio and no sounddevice means no microphone backend."""
    monkeypatch.setattr(cvi, "capture_available", lambda: False)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))

    availability = cvi.probe()

    assert availability.ok is False
    assert availability.kind == "missing-capture"
    assert "speech_recording" in availability.remedy


def test_probe_reports_missing_provider(monkeypatch):
    """Capture present but no transcription provider is a different remedy."""
    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ())

    availability = cvi.probe()

    assert availability.ok is False
    assert availability.kind == "missing-provider"
    assert "transcription_faster_whisper" in availability.remedy
    assert "speech_recording" not in availability.remedy


def test_probe_ok_when_both_present(monkeypatch):
    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))

    availability = cvi.probe()

    assert availability.ok is True
    assert availability.kind == "ok"


def test_probe_does_not_import_transcription_service():
    """Probing must stay cheap: no faster-whisper, no NeMo, no torch."""
    heavy = "tldw_chatbook.Local_Ingestion.transcription_service"
    sys.modules.pop(heavy, None)

    cvi.probe()

    assert heavy not in sys.modules
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_voice_input.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.console_voice_input'`

- [ ] **Step 3: Write minimal implementation**

Create `tldw_chatbook/Chat/console_voice_input.py`:

```python
"""Headless Console voice-dictation controller.

Deliberately free of Textual imports: the widget layer owns rendering and
threading policy, this module owns availability, provider resolution, and the
dictation state machine. That split is what makes the state machine unit
testable without a running app.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from loguru import logger

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_voice_input.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_voice_input.py Tests/Chat/test_console_voice_input.py
git commit -m "feat(console): add voice dictation availability probe"
```

---

### Task 2: Provider resolution

Neutralizes the silent `parakeet-mlx` swap by deciding the provider before the service can.

**Files:**
- Modify: `tldw_chatbook/Chat/console_voice_input.py`
- Test: `Tests/Chat/test_console_voice_input.py`

**Interfaces:**
- Consumes: `installed_local_providers()`, `Availability` from Task 1.
- Produces: `EffectiveConfig(provider: str, model: str | None, language: str, configured_provider: str, was_overridden: bool)`; `resolve() -> EffectiveConfig | None` (None when no usable provider).

- [ ] **Step 1: Write the failing test**

Append to `Tests/Chat/test_console_voice_input.py`:

```python
def _stub_settings(monkeypatch, values: dict[str, object]) -> None:
    """Route console_voice_input's config reads through a dict."""

    def fake_get(section, key=None, default=None):
        if key is not None and not isinstance(key, str):
            default = key
            key = None
        lookup = section if key is None else f"{section}.{key}"
        return values.get(lookup, default)

    monkeypatch.setattr(cvi, "get_cli_setting", fake_get)


def test_resolve_keeps_configured_provider_when_installed(monkeypatch):
    monkeypatch.setattr(
        cvi, "installed_local_providers", lambda: ("faster-whisper", "parakeet-mlx")
    )
    _stub_settings(
        monkeypatch,
        {
            "transcription.provider": "faster-whisper",
            "transcription.model": "base",
            "transcription.language": "en",
        },
    )

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider == "faster-whisper"
    assert effective.model == "base"
    assert effective.was_overridden is False


def test_resolve_flags_override_instead_of_swapping_silently(monkeypatch):
    """A configured provider that is not installed is replaced, and it shows."""
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.provider": "parakeet-mlx"})

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider == "faster-whisper"
    assert effective.configured_provider == "parakeet-mlx"
    assert effective.was_overridden is True


def test_resolve_never_returns_an_uninstalled_provider(monkeypatch):
    """This is the guard against the service's parakeet-mlx rewrite."""
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.provider": "lightning-whisper-mlx"})

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider in cvi.installed_local_providers()


def test_resolve_returns_none_when_nothing_installed(monkeypatch):
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ())
    _stub_settings(monkeypatch, {"transcription.provider": "faster-whisper"})

    assert cvi.resolve() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_voice_input.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'resolve'`

- [ ] **Step 3: Write minimal implementation**

Add the import at the top of `console_voice_input.py`, below the `loguru` import:

```python
from ..config import get_cli_setting
```

Then append:

```python
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

    configured = get_cli_setting("transcription", "provider", None) or get_cli_setting(
        "STTSettings", "default_stt_provider", ""
    )
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

    model = get_cli_setting("transcription", "model", None)
    language = get_cli_setting("transcription", "language", DEFAULT_LANGUAGE)

    return EffectiveConfig(
        provider=provider,
        model=str(model) if model else None,
        language=str(language or DEFAULT_LANGUAGE),
        configured_provider=configured,
        was_overridden=bool(configured) and provider != configured,
    )
```

Then update `probe()` to use it, replacing the `installed_local_providers()` check:

```python
    if not installed_local_providers():
```

stays as-is — `resolve()` and `probe()` share `installed_local_providers()` as the single source of truth, so they cannot disagree.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_voice_input.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_voice_input.py Tests/Chat/test_console_voice_input.py
git commit -m "feat(console): resolve dictation provider before the service can swap it"
```

---

### Task 3: Controller state machine

The heart of the feature, and the only part that is cheap to test exhaustively.

**Files:**
- Modify: `tldw_chatbook/Chat/console_voice_input.py`
- Test: `Tests/Chat/test_console_voice_input.py`

**Interfaces:**
- Consumes: `probe()`, `resolve()`, `EffectiveConfig` from Tasks 1–2.
- Produces:
  - State constants `STATE_IDLE = "idle"`, `STATE_PREPARING = "preparing"`, `STATE_LISTENING = "listening"`, `STATE_FINISHING = "finishing"`, `STATE_ERROR = "error"`, `STATE_UNAVAILABLE = "unavailable"`.
  - Events `VoicePartial(text: str)`, `VoiceFinal(text: str)`, `VoiceStateChanged(state: str)`, `VoiceFailed(reason: str, remedy: str)`, `VoiceProviderOverridden(configured: str, effective: str)`.
  - `ConsoleVoiceInputController(*, emit, spawn, service_factory=default_service_factory)` with `.state`, `.is_active`, `.start()`, `.stop()`, `.abandon()`.
  - `default_service_factory(**kwargs)` — imports `dictation_service_lazy` inside the function body.

- [ ] **Step 1: Write the failing test**

Append to `Tests/Chat/test_console_voice_input.py`:

```python
import threading


class FakeDictationService:
    """Stands in for LazyLiveDictationService, recording how it was built."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.start_result = True
        self.start_error: Exception | None = None
        self._callbacks: dict[str, object] = {}

    def start_dictation(self, **callbacks):
        if self.start_error is not None:
            raise self.start_error
        self._callbacks = callbacks
        self.started = True
        return self.start_result

    def stop_dictation(self):
        self.stopped = True
        return None

    # -- test drivers -------------------------------------------------
    def emit_partial(self, text):
        self._callbacks["on_partial_transcript"](text)

    def emit_final(self, text):
        self._callbacks["on_final_transcript"](text)

    def emit_error(self, error):
        self._callbacks["on_error"](error)


def _controller(monkeypatch, service=None, spawn=None):
    """Build a controller with a fake service.

    `spawn` defaults to running the thunk inline, which is what makes the
    state machine testable without an event loop. Pass a deferring spawn to
    freeze the controller mid-`preparing`.
    """
    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.provider": "faster-whisper"})

    service = service or FakeDictationService()
    events = []
    controller = cvi.ConsoleVoiceInputController(
        emit=events.append,
        spawn=spawn or (lambda thunk: thunk()),
        service_factory=lambda **kwargs: service,
    )
    return controller, events, service


def test_start_transitions_preparing_then_listening(monkeypatch):
    controller, events, _ = _controller(monkeypatch)

    controller.start()

    states = [e.state for e in events if isinstance(e, cvi.VoiceStateChanged)]
    assert states == [cvi.STATE_PREPARING, cvi.STATE_LISTENING]
    assert controller.is_active is True


def test_second_start_while_listening_is_a_no_op(monkeypatch):
    """Rejected by our own state, not left to the service's lock."""
    controller, events, _ = _controller(monkeypatch)
    controller.start()
    assert controller.state == cvi.STATE_LISTENING
    events.clear()

    controller.start()

    assert events == []


def test_start_while_still_preparing_is_a_no_op(monkeypatch):
    """The preparing window is real once `spawn` is a worker, so cover it.

    A deferring spawn captures the thunk instead of running it, which is the
    only way to observe the controller mid-`preparing`. With the inline spawn
    used elsewhere, `start()` has already reached `listening` on return.
    """
    pending = []
    controller, events, _ = _controller(monkeypatch, spawn=pending.append)

    controller.start()
    assert controller.state == cvi.STATE_PREPARING
    events.clear()

    controller.start()

    assert events == []
    assert len(pending) == 1  # the second start never queued more work


def test_stop_returns_to_idle(monkeypatch):
    controller, events, service = _controller(monkeypatch)
    controller.start()
    events.clear()

    controller.stop()

    states = [e.state for e in events if isinstance(e, cvi.VoiceStateChanged)]
    assert states == [cvi.STATE_FINISHING, cvi.STATE_IDLE]
    assert service.stopped is True
    assert controller.is_active is False


def test_stop_from_idle_is_a_no_op(monkeypatch):
    controller, events, _ = _controller(monkeypatch)

    controller.stop()

    assert events == []


def test_failed_start_returns_to_idle_not_stuck_on_preparing(monkeypatch):
    service = FakeDictationService()
    service.start_error = RuntimeError("no input device")
    controller, events, _ = _controller(monkeypatch, service)

    controller.start()

    failures = [e for e in events if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert "no input device" in failures[0].reason
    assert controller.state == cvi.STATE_IDLE


def test_service_built_with_commands_and_audio_saving_off(monkeypatch):
    """V2 behaviour must not leak into V1, and no audio is retained."""
    controller, _, service = _controller(monkeypatch)

    controller.start()

    assert service.kwargs["enable_commands"] is False
    assert controller.save_audio_requested is False


def test_transcript_callbacks_survive_a_foreign_thread(monkeypatch):
    """Callbacks arrive on the service's worker thread; nothing may block."""
    controller, events, service = _controller(monkeypatch)
    controller.start()
    events.clear()

    thread = threading.Thread(target=lambda: service.emit_final("hello there"))
    thread.start()
    thread.join(timeout=5)

    finals = [e for e in events if isinstance(e, cvi.VoiceFinal)]
    assert [e.text for e in finals] == ["hello there"]


def test_unavailable_start_emits_remedy(monkeypatch):
    controller, events, _ = _controller(monkeypatch)
    monkeypatch.setattr(cvi, "capture_available", lambda: False)

    controller.start()

    failures = [e for e in events if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert "speech_recording" in failures[0].remedy
    assert controller.state == cvi.STATE_IDLE


def test_provider_override_is_announced_once(monkeypatch):
    controller, events, _ = _controller(monkeypatch)
    _stub_settings(monkeypatch, {"transcription.provider": "parakeet-mlx"})

    controller.start()
    controller.stop()
    controller.start()

    overrides = [e for e in events if isinstance(e, cvi.VoiceProviderOverridden)]
    assert len(overrides) == 1
    assert overrides[0].effective == "faster-whisper"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_voice_input.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'ConsoleVoiceInputController'`

- [ ] **Step 3: Write minimal implementation**

Append to `console_voice_input.py`:

```python
import threading
from typing import Any, Callable

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
        self._emit(VoiceFailed(reason=reason, remedy=remedy))
        self._set_state(STATE_IDLE)

    def start(self) -> None:
        """Begin capture. Rejected unless currently idle."""
        with self._state_lock:
            if self._state != STATE_IDLE:
                logger.debug("Console dictation start ignored in state {}", self._state)
                return
            self._state = STATE_PREPARING
        self._emit(VoiceStateChanged(STATE_PREPARING))

        availability = probe()
        if not availability.ok:
            self._fail(availability.reason, availability.remedy)
            return

        effective = resolve()
        if effective is None:
            self._fail(PROVIDER_REASON, PROVIDER_REMEDY)
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
            self._service = self._service_factory(
                transcription_provider=effective.provider,
                transcription_model=effective.model,
                language=effective.language,
                enable_commands=False,  # V2 owns voice commands, not V1
            )
            started = self._service.start_dictation(
                on_partial_transcript=lambda text: self._emit(VoicePartial(text)),
                on_final_transcript=lambda text: self._emit(VoiceFinal(text)),
                on_state_change=lambda _state: None,  # our state machine is authoritative
                on_error=lambda error: self._fail(str(error)),
                save_audio=self.save_audio_requested,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            logger.opt(exception=True).warning("Console dictation failed to start")
            self._service = None
            self._fail(str(exc))
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

        For teardown paths (unmount, app quit) where blocking would show up as
        a hang. Best effort by design.
        """
        service, self._service = self._service, None
        self._state = STATE_IDLE
        if service is None:
            return
        try:
            audio = getattr(service, "_audio_service", None)
            if audio is not None and hasattr(audio, "stop_recording"):
                audio.stop_recording()
        except Exception:  # noqa: BLE001 - teardown must never raise
            logger.opt(exception=True).debug("Console dictation abandon failed")
```

Move the `import threading` and `from typing import Any, Callable` lines up to the module's import block rather than leaving them mid-file.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_voice_input.py -v`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_voice_input.py Tests/Chat/test_console_voice_input.py
git commit -m "feat(console): add voice dictation state machine"
```

---

### Task 4: Composer chip and mic button

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py` (compose at `:1824`, actions row at `:1909`)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (after the `.console-send-disabled-reason` block at `:2681`)
- Test: `Tests/UI/test_console_voice_chip.py`

**Interfaces:**
- Consumes: `STATE_*` constants from Task 3.
- Produces: `ConsoleComposerBar.set_voice_status(state: str, *, partial: str = "", elapsed_seconds: int = 0, message: str = "") -> None`; widget ids `#console-voice-status` and `#console-voice-toggle`; class constant `VOICE_CHIP_MIN_WIDTH = 24`.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_console_voice_chip.py`:

```python
"""Console composer voice-status chip tests."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Static

from tldw_chatbook.Chat.console_voice_input import (
    STATE_ERROR,
    STATE_IDLE,
    STATE_LISTENING,
    STATE_PREPARING,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar


class ComposerApp(App):
    def compose(self):
        yield ConsoleComposerBar(id="console-native-composer")


def _visible(widget) -> bool:
    """True only when the widget and every ancestor are displayed.

    `renderable` having text proves nothing: #console-composer-status carries
    `console-hidden-control` (display: none) and would happily hold a string
    no user can see.
    """
    node = widget
    while node is not None:
        if not getattr(node, "display", True):
            return False
        node = node.parent
    return True


@pytest.mark.asyncio
async def test_idle_collapses_a_chip_that_was_showing():
    """Show it first: asserting width==0 on a never-shown chip proves nothing.

    The chip starts at width 0 from `compose()`, so a bare idle assertion
    would pass even if `set_voice_status` were a no-op.
    """
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(STATE_LISTENING, partial="hello", elapsed_seconds=1)
        chip = composer.query_one("#console-voice-status", Static)
        assert chip.styles.width.value > 0

        composer.set_voice_status(STATE_IDLE)

        assert chip.styles.width.value == 0
        assert str(chip.renderable) == ""


@pytest.mark.asyncio
async def test_chip_is_actually_visible_while_listening():
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(
            STATE_LISTENING, partial="and compare them to", elapsed_seconds=7
        )
        chip = composer.query_one("#console-voice-status", Static)
        assert _visible(chip)
        assert chip.styles.width.value > 0
        assert "0:07" in str(chip.renderable)


@pytest.mark.asyncio
async def test_whisper_bracket_tokens_render_literally():
    """[BLANK_AUDIO] is routine Whisper output and is not Rich markup."""
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(STATE_LISTENING, partial="[BLANK_AUDIO] [Music]")
        chip = composer.query_one("#console-voice-status", Static)
        assert "[BLANK_AUDIO]" in str(chip.renderable)


@pytest.mark.asyncio
async def test_narrow_terminal_drops_the_partial_not_the_draft():
    app = ComposerApp()
    async with app.run_test(size=(30, 12)):
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(
            STATE_LISTENING, partial="a very long partial transcript", elapsed_seconds=3
        )
        chip = composer.query_one("#console-voice-status", Static)
        assert "very long partial" not in str(chip.renderable)
        assert "●" in str(chip.renderable)


@pytest.mark.asyncio
async def test_preparing_and_error_states_render_their_message():
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.set_voice_status(STATE_PREPARING, message="Loading model…")
        chip = composer.query_one("#console-voice-status", Static)
        assert "Loading model…" in str(chip.renderable)

        composer.set_voice_status(STATE_ERROR, message="No microphone access.")
        assert "No microphone access." in str(chip.renderable)


@pytest.mark.asyncio
async def test_mic_button_exists_in_the_actions_row():
    from textual.widgets import Button

    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        button = composer.query_one("#console-voice-toggle", Button)
        assert _visible(button)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_voice_chip.py -v`
Expected: FAIL — `NoMatches: No nodes match '#console-voice-status'`

- [ ] **Step 3: Write minimal implementation**

In `console_composer_bar.py`, add the class constant next to `MAX_DRAFT_ROWS` (`:69`):

```python
    VOICE_CHIP_MIN_WIDTH = 24
    VOICE_CHIP_MAX_WIDTH = 42
```

In `compose()`, immediately after the `recovery` block (`:1857`, the `yield recovery` line), add:

```python
            voice_status = Static(
                "",
                id="console-voice-status",
                classes="console-voice-status",
            )
            voice_status.styles.display = "none"
            voice_status.styles.width = 0
            voice_status.styles.min_width = 0
            voice_status.styles.height = 0
            voice_status.styles.min_height = 0
            yield voice_status
```

In the `#console-composer-actions` block, immediately before the `#console-attach-context` button (`:1935`), add:

```python
                yield self._bounded_button(
                    "🎤 Mic",
                    width=8,
                    id="console-voice-toggle",
                )
```

Add the rendering method next to `set_pending_attachment_label` (`:1789`):

```python
    def set_voice_status(
        self,
        state: str,
        *,
        partial: str = "",
        elapsed_seconds: int = 0,
        message: str = "",
    ) -> None:
        """Render the dictation state into the inline voice chip.

        Args:
            state: One of the `STATE_*` constants from `console_voice_input`.
            partial: In-flight recognizer text. Truncated from the left so the
                newest words stay visible, and dropped entirely on narrow
                terminals so the 1fr draft never collapses.
            elapsed_seconds: Recording duration, rendered as m:ss.
            message: Status or failure text for non-listening states.
        """
        try:
            chip = self.query_one("#console-voice-status", Static)
        except NoMatches:
            return

        if state in ("idle", "unavailable"):
            chip.styles.display = "none"
            chip.styles.width = 0
            chip.styles.min_width = 0
            chip.update("")
            return

        # `size` is (0, 0) before the first layout; fall back to the ceiling
        # rather than computing a zero width and rendering an invisible chip.
        total_width = self.size.width or self.VOICE_CHIP_MAX_WIDTH * 2
        available = max(0, total_width - self.VOICE_CHIP_MIN_WIDTH)
        width = min(self.VOICE_CHIP_MAX_WIDTH, available)

        if state == "listening":
            head = f"● {elapsed_seconds // 60}:{elapsed_seconds % 60:02d}"
            room = width - len(head) - 3
            if partial and room > 8:
                tail = partial[-room:]
                body = f"{head}  {tail}"
            else:
                # Below the floor the counter alone still proves the mic is live.
                body = head
                width = min(width, len(head) + 2)
        else:
            body = message or state
            width = min(width, len(body) + 2)

        chip.styles.display = "block"
        chip.styles.width = max(width, 1)
        chip.styles.min_width = 0
        chip.styles.height = 1
        chip.styles.min_height = 1
        chip.set_class(state == "error", "console-voice-status-error")
        chip.update(escape(body))
```

Add to `_agentic_terminal.tcss` after the `.console-send-disabled-reason` block:

```css
.console-voice-status {
    width: 0;
    min-width: 0;
    margin: 0 1 0 0;
    padding: 0 1;
    background: $ds-surface-raised;
    color: $ds-status-running;
    text-style: bold;
    text-overflow: ellipsis;
    text-wrap: nowrap;
}

.console-voice-status-error {
    color: $ds-status-error;
}

#console-voice-toggle {
    width: 8;
    min-width: 8;
    height: 1;
    min-height: 1;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_voice_chip.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_composer_bar.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_console_voice_chip.py
git commit -m "feat(console): add voice status chip and mic button to the composer"
```

---

### Task 5: Dictated text insertion

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Test: `Tests/UI/test_console_voice_chip.py`

**Interfaces:**
- Consumes: `insert_text()` (`:1017`), `draft_text()` (`:135`).
- Produces: `ConsoleComposerBar.insert_dictated_text(text: str) -> bool` — returns True when the draft changed.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_voice_chip.py`:

```python
@pytest.mark.asyncio
async def test_dictated_segments_join_with_single_spaces():
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_dictated_text("Summarize the RAG audit")
        composer.insert_dictated_text("and compare the findings")
        assert composer.draft_text() == "Summarize the RAG audit and compare the findings"


@pytest.mark.asyncio
async def test_no_leading_space_on_an_empty_draft():
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.insert_dictated_text("hello")
        assert composer.draft_text() == "hello"


@pytest.mark.asyncio
async def test_no_double_space_after_existing_whitespace():
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        composer.load_draft("typed text ")
        composer.insert_dictated_text("dictated")
        assert composer.draft_text() == "typed text dictated"


@pytest.mark.asyncio
async def test_long_dictation_is_literal_text_not_a_paste_token():
    """Must not route through insert_pasted_text and trip paste-collapse."""
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        long_text = "word " * 400
        composer.insert_dictated_text(long_text.strip())
        assert composer.draft_text().startswith("word word")
        assert len(composer.draft_text()) > 1000


@pytest.mark.asyncio
async def test_empty_segment_is_ignored():
    app = ComposerApp()
    async with app.run_test():
        composer = app.query_one(ConsoleComposerBar)
        assert composer.insert_dictated_text("   ") is False
        assert composer.draft_text() == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_voice_chip.py -v -k dictated`
Expected: FAIL — `AttributeError: 'ConsoleComposerBar' object has no attribute 'insert_dictated_text'`

- [ ] **Step 3: Write minimal implementation**

Add next to `insert_text` (`:1017`) in `console_composer_bar.py`:

```python
    def insert_dictated_text(self, text: str) -> bool:
        """Insert a finalized dictation segment at the caret.

        Routes through `insert_text` rather than `insert_pasted_text` on
        purpose: a long utterance must stay literal, editable text and must
        never trip `PASTE_COLLAPSE_THRESHOLD` into a collapsed token.

        Args:
            text: A finalized recognizer segment.

        Returns:
            True when the draft changed, False for empty or whitespace-only
            segments (which recognizers do emit on silence).
        """
        segment = text.strip()
        if not segment:
            return False
        existing = self.draft_text()
        if existing and not existing[-1].isspace():
            segment = f" {segment}"
        self.insert_text(segment)
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_voice_chip.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_composer_bar.py Tests/UI/test_console_voice_chip.py
git commit -m "feat(console): insert dictated segments into the composer draft"
```

---

### Task 6: Wire the controller into the composer

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Test: `Tests/UI/test_console_voice_chip.py`

**Interfaces:**
- Consumes: `ConsoleVoiceInputController`, all `Voice*` events, `STATE_*` (Task 3); `set_voice_status` (Task 4); `insert_dictated_text` (Task 5).
- Produces: nested message classes `ConsoleComposerBar.VoiceStateChanged(state: str)`, `.VoiceTick()`, `.VoiceFailure(reason: str, remedy: str)`, `.VoiceOverride(configured: str, effective: str)`; methods `toggle_dictation() -> None`, `stop_dictation() -> None`, `abandon_dictation() -> None`, property `dictation_active: bool`.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_voice_chip.py`:

```python
@pytest.mark.asyncio
async def test_toggle_starts_and_final_segments_reach_the_draft(monkeypatch):
    from tldw_chatbook.Chat import console_voice_input as cvi

    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))

    class Service:
        def __init__(self, **kwargs):
            self.callbacks = {}

        def start_dictation(self, **callbacks):
            self.callbacks = callbacks
            return True

        def stop_dictation(self):
            return None

    service = Service()
    monkeypatch.setattr(cvi, "default_service_factory", lambda **kw: service)

    app = ComposerApp()
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.toggle_dictation()
        await pilot.pause()
        assert composer.dictation_active is True

        service.callbacks["on_final_transcript"]("hello world")
        await pilot.pause()
        assert composer.draft_text() == "hello world"

        composer.stop_dictation()
        await pilot.pause()
        assert composer.dictation_active is False


@pytest.mark.asyncio
async def test_elapsed_timer_stops_when_dictation_stops(monkeypatch):
    from tldw_chatbook.Chat import console_voice_input as cvi

    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))

    class Service:
        def start_dictation(self, **callbacks):
            return True

        def stop_dictation(self):
            return None

    monkeypatch.setattr(cvi, "default_service_factory", lambda **kw: Service())

    app = ComposerApp()
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.toggle_dictation()
        await pilot.pause()
        assert composer._voice_timer is not None

        composer.stop_dictation()
        await pilot.pause()
        assert composer._voice_timer is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_voice_chip.py -v -k toggle`
Expected: FAIL — `AttributeError: 'ConsoleComposerBar' object has no attribute 'toggle_dictation'`

- [ ] **Step 3: Write minimal implementation**

Add to the imports at the top of `console_composer_bar.py`:

```python
from textual.message import Message
```

Add to `ConsoleComposerBar.__init__` (alongside the other instance attributes):

```python
        self._voice: Any | None = None
        self._voice_timer: Any | None = None
        self._voice_elapsed = 0
        self._voice_partial = ""
        self._voice_state = "idle"
```

Add the message classes inside `ConsoleComposerBar`, next to the other nested classes:

```python
    class VoiceStateChanged(Message):
        """Dictation entered a NEW state. Also handled by ChatScreen.

        Only posted on a real transition -- ChatScreen's deferred send keys
        off this message, so a repaint must not masquerade as a transition.
        """

        def __init__(self, state: str) -> None:
            super().__init__()
            self.state = state

    class VoiceTick(Message):
        """The chip needs repainting; the state did not change."""

    class VoiceFailure(Message):
        """Dictation could not start or stop."""

        def __init__(self, reason: str, remedy: str = "") -> None:
            super().__init__()
            self.reason = reason
            self.remedy = remedy

    class VoiceOverride(Message):
        """The configured provider was not usable; another was selected."""

        def __init__(self, configured: str, effective: str) -> None:
            super().__init__()
            self.configured = configured
            self.effective = effective
```

Add the controller plumbing:

```python
    @property
    def dictation_active(self) -> bool:
        """True while the microphone is or is about to be live."""
        return self._voice is not None and self._voice.is_active

    def _ensure_voice_controller(self) -> Any:
        """Build the controller on first use, importing lazily.

        The import is inside the method because `console_voice_input`'s
        service factory reaches the dictation stack; keeping it out of module
        scope keeps Console import cost unchanged.
        """
        if self._voice is not None:
            return self._voice

        from ...Chat.console_voice_input import ConsoleVoiceInputController

        self._voice = ConsoleVoiceInputController(
            emit=self._on_voice_event,
            spawn=self._spawn_voice_work,
        )
        return self._voice

    def _spawn_voice_work(self, thunk) -> None:
        """Run blocking dictation work off the UI thread.

        Both `start_dictation()` (cold model load) and `stop_dictation()` (a 2s
        join) block; running them inline would freeze the TUI.
        """
        self.run_worker(
            thunk,
            thread=True,
            group="console-dictation",
            exit_on_error=False,
        )

    def _on_voice_event(self, event: Any) -> None:
        """Translate controller dataclasses into Textual messages.

        Called from the dictation worker thread. `post_message` is thread-safe
        (it hops to the event loop via `call_soon_threadsafe`); `call_from_thread`
        would block this thread and stall transcription.
        """
        from ...Chat.console_voice_input import (
            VoiceFailed,
            VoiceFinal,
            VoicePartial,
            VoiceProviderOverridden,
            VoiceStateChanged as ControllerStateChanged,
        )

        if isinstance(event, VoicePartial):
            self._voice_partial = event.text
            self.post_message(self.VoiceTick())
        elif isinstance(event, VoiceFinal):
            self._voice_partial = ""
            self.post_message(self.VoiceTick())
            # `call_later` is safe from this worker thread: it wraps the
            # callback in an events.Callback and hands it to `post_message`
            # (message_pump.py:504), so ordering with the message above is
            # preserved and nothing touches a widget off-thread.
            self.app.call_later(self.insert_dictated_text, event.text)
        elif isinstance(event, ControllerStateChanged):
            self._voice_state = event.state
            self.post_message(self.VoiceStateChanged(event.state))
        elif isinstance(event, VoiceFailed):
            self.post_message(self.VoiceFailure(event.reason, event.remedy))
        elif isinstance(event, VoiceProviderOverridden):
            self.post_message(self.VoiceOverride(event.configured, event.effective))

    def toggle_dictation(self) -> None:
        """Start dictation, or stop it if already listening."""
        controller = self._ensure_voice_controller()
        if controller.is_active:
            self.stop_dictation()
            return
        self._voice_elapsed = 0
        self._voice_partial = ""
        controller.start()

    def stop_dictation(self) -> None:
        """Stop dictation and commit whatever was heard."""
        if self._voice is None:
            return
        self._voice.stop()

    def abandon_dictation(self) -> None:
        """Release the microphone without blocking. For teardown paths only."""
        self._stop_voice_timer()
        if self._voice is not None:
            self._voice.abandon()
        self.set_voice_status("idle")

    def _start_voice_timer(self) -> None:
        if self._voice_timer is not None:
            return
        self._voice_timer = self.set_interval(1.0, self._tick_voice_elapsed)

    def _stop_voice_timer(self) -> None:
        if self._voice_timer is None:
            return
        self._voice_timer.stop()
        self._voice_timer = None

    def _tick_voice_elapsed(self) -> None:
        self._voice_elapsed += 1
        self._render_voice_chip()

    def _render_voice_chip(self, message: str = "") -> None:
        self.set_voice_status(
            self._voice_state,
            partial=self._voice_partial,
            elapsed_seconds=self._voice_elapsed,
            message=message,
        )

    @on(VoiceStateChanged)
    def _handle_voice_state(self, event: "ConsoleComposerBar.VoiceStateChanged") -> None:
        """Keep the chip and the elapsed timer in step with the state."""
        if event.state == "listening":
            self._start_voice_timer()
        else:
            self._stop_voice_timer()
        messages = {
            "preparing": "Preparing microphone…",
            "finishing": "Finishing…",
        }
        self._render_voice_chip(messages.get(event.state, ""))

    @on(VoiceTick)
    def _handle_voice_tick(self, event: "ConsoleComposerBar.VoiceTick") -> None:
        """Repaint the chip for a new partial without faking a transition."""
        event.stop()
        self._render_voice_chip()

    @on(VoiceFailure)
    def _handle_voice_failure(self, event: "ConsoleComposerBar.VoiceFailure") -> None:
        self._stop_voice_timer()
        self._voice_state = "error"
        self._render_voice_chip(event.reason)
```

Note: `VoiceStateChanged` messages must **not** be stopped here — `ChatScreen` also handles them for the deferred send in Task 8. Textual's `@on` handlers do not stop propagation unless you call `event.stop()`, so leave them alone.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_voice_chip.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_composer_bar.py Tests/UI/test_console_voice_chip.py
git commit -m "feat(console): wire the voice controller into the composer"
```

---

### Task 7: Screen binding, action, and mic button routing

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (BINDINGS at `:556-608`; actions near `:623`; `@on(Button.Pressed)` handlers near `:9950`)
- Test: `Tests/UI/test_console_dictation.py`

**Interfaces:**
- Consumes: `composer.toggle_dictation()`, `composer.dictation_active` (Task 6); `_console_setup_modal_blocking()` (`:6708`); `_set_console_composer_collapsed()` (`:627`).
- Produces: `ChatScreen.action_toggle_console_dictation() -> None`; binding `alt+r`.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_console_dictation.py`:

```python
"""Console screen dictation wiring tests."""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _binding_keys() -> set[str]:
    return {binding.key for binding in ChatScreen.BINDINGS if hasattr(binding, "key")}


def test_alt_r_is_bound_for_dictation():
    assert "alt+r" in _binding_keys()


def test_dictation_binding_does_not_collide():
    """alt+v is paste-image and escape is doubly bound; neither may be reused."""
    keys = [b.key for b in ChatScreen.BINDINGS if hasattr(b, "key")]
    assert keys.count("alt+r") == 1


def test_toggle_action_exists():
    assert callable(getattr(ChatScreen, "action_toggle_console_dictation", None))


@pytest.mark.asyncio
async def test_toggle_is_refused_while_the_setup_modal_blocks(monkeypatch):
    calls = []

    class FakeComposer:
        collapsed = False
        dictation_active = False

        def toggle_dictation(self):
            calls.append("toggled")

    screen = ChatScreen.__new__(ChatScreen)
    monkeypatch.setattr(
        ChatScreen, "_console_setup_modal_blocking", lambda self: True, raising=False
    )
    monkeypatch.setattr(
        ChatScreen, "_console_composer_or_none", lambda self: FakeComposer(), raising=False
    )

    ChatScreen.action_toggle_console_dictation(screen)

    assert calls == []


@pytest.mark.asyncio
async def test_collapsed_composer_is_expanded_before_starting(monkeypatch):
    expanded = []
    toggled = []

    class FakeComposer:
        collapsed = True
        dictation_active = False

        def toggle_dictation(self):
            toggled.append("toggled")

    screen = ChatScreen.__new__(ChatScreen)
    monkeypatch.setattr(
        ChatScreen, "_console_setup_modal_blocking", lambda self: False, raising=False
    )
    monkeypatch.setattr(
        ChatScreen, "_console_composer_or_none", lambda self: FakeComposer(), raising=False
    )
    monkeypatch.setattr(
        ChatScreen,
        "_set_console_composer_collapsed",
        lambda self, value: expanded.append(value),
        raising=False,
    )

    ChatScreen.action_toggle_console_dictation(screen)

    assert expanded == [False]
    assert toggled == ["toggled"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_dictation.py -v`
Expected: FAIL — `assert 'alt+r' in {...}`

- [ ] **Step 3: Write minimal implementation**

Add to `ChatScreen.BINDINGS`, after the `alt+v` entry (`:587`):

```python
        Binding("alt+r", "toggle_console_dictation", "Dictate", show=True),
```

Add the action next to `action_expand_collapsed_console_composer` (`:623`):

```python
    def action_toggle_console_dictation(self) -> None:
        """Start or stop voice dictation into the Console composer.

        `alt+r` reaches this even while typing: `on_key` is a whitelist that
        stops only named keys and printable characters, so unmatched
        non-printable keys fall through to these bindings.
        """
        if self._console_setup_modal_blocking():
            return
        composer = self._console_composer_or_none()
        if composer is None:
            return
        if composer.collapsed and not composer.dictation_active:
            # Dictation writes into the draft, so the draft must be on screen.
            self._set_console_composer_collapsed(False)
        composer.toggle_dictation()
```

Add the button handler next to the other composer button handlers (`:9950`):

```python
    @on(Button.Pressed, "#console-voice-toggle")
    def _handle_console_voice_toggle(self, event: Button.Pressed) -> None:
        """Route the mic button through the same path as alt+r."""
        event.stop()
        self.action_toggle_console_dictation()
```

Add the notification handlers for the composer's messages:

```python
    @on(ConsoleComposerBar.VoiceFailure)
    def _handle_console_voice_failure(
        self, event: ConsoleComposerBar.VoiceFailure
    ) -> None:
        """Surface the failure as a toast as well as in the chip.

        The remedy must never live only in a hover target.
        """
        event.stop()
        detail = f"{event.reason} {event.remedy}".strip()
        self.app_instance.notify(detail, severity="warning")

    @on(ConsoleComposerBar.VoiceOverride)
    def _handle_console_voice_override(
        self, event: ConsoleComposerBar.VoiceOverride
    ) -> None:
        event.stop()
        self.app_instance.notify(
            f"Speech provider '{event.configured}' is unavailable; "
            f"using '{event.effective}'.",
            severity="information",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_dictation.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_dictation.py
git commit -m "feat(console): bind alt+r and the mic button to dictation"
```

---

### Task 8: Deferred send on Enter

The highest-value task in the plan: without it, the last words spoken land in the *next* message.

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`on_key` enter branch at `:11368`)
- Test: `Tests/UI/test_console_dictation.py`

**Interfaces:**
- Consumes: `composer.dictation_active`, `composer.stop_dictation()`, `ConsoleComposerBar.VoiceStateChanged` (Task 6).
- Produces: `ChatScreen._console_pending_voice_send: bool`; `ChatScreen._press_console_send() -> None`.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_dictation.py`:

```python
def test_enter_while_listening_defers_the_send(monkeypatch):
    """Stop is async; sending in the same tick would drop the last words."""
    sent = []
    stopped = []

    class FakeComposer:
        collapsed = False
        dictation_active = True

        def stop_dictation(self):
            stopped.append("stopped")

    screen = ChatScreen.__new__(ChatScreen)
    screen._console_pending_voice_send = False
    monkeypatch.setattr(
        ChatScreen, "_console_composer_or_none", lambda self: FakeComposer(), raising=False
    )
    monkeypatch.setattr(
        ChatScreen, "_press_console_send", lambda self: sent.append("sent"), raising=False
    )

    handled = ChatScreen._defer_send_for_dictation(screen)

    assert handled is True
    assert stopped == ["stopped"]
    assert sent == []
    assert screen._console_pending_voice_send is True


def test_pending_send_fires_once_dictation_reaches_idle(monkeypatch):
    sent = []
    screen = ChatScreen.__new__(ChatScreen)
    screen._console_pending_voice_send = True
    monkeypatch.setattr(
        ChatScreen, "_press_console_send", lambda self: sent.append("sent"), raising=False
    )

    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    event = ConsoleComposerBar.VoiceStateChanged("idle")
    ChatScreen._handle_console_voice_state(screen, event)

    assert sent == ["sent"]
    assert screen._console_pending_voice_send is False


def test_pending_send_is_cleared_without_sending_on_failure(monkeypatch):
    sent = []
    screen = ChatScreen.__new__(ChatScreen)
    screen._console_pending_voice_send = True
    monkeypatch.setattr(
        ChatScreen, "_press_console_send", lambda self: sent.append("sent"), raising=False
    )
    monkeypatch.setattr(
        ChatScreen, "app_instance", type("A", (), {"notify": lambda *a, **k: None})()
    )

    from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar

    ChatScreen._handle_console_voice_failure(
        screen, ConsoleComposerBar.VoiceFailure("mic died", "")
    )

    assert sent == []
    assert screen._console_pending_voice_send is False


def test_repeated_enter_while_pending_is_a_no_op(monkeypatch):
    stopped = []

    class FakeComposer:
        collapsed = False
        dictation_active = True

        def stop_dictation(self):
            stopped.append("stopped")

    screen = ChatScreen.__new__(ChatScreen)
    screen._console_pending_voice_send = True
    monkeypatch.setattr(
        ChatScreen, "_console_composer_or_none", lambda self: FakeComposer(), raising=False
    )

    handled = ChatScreen._defer_send_for_dictation(screen)

    assert handled is True
    assert stopped == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_dictation.py -v -k defer`
Expected: FAIL — `AttributeError: type object 'ChatScreen' has no attribute '_defer_send_for_dictation'`

- [ ] **Step 3: Write minimal implementation**

Initialise the flag in `ChatScreen.__init__`:

```python
        self._console_pending_voice_send = False
```

Add the helpers near the other console composer helpers:

```python
    def _press_console_send(self) -> None:
        """Press the composer's Send button, matching the Enter key path."""
        try:
            self.query_one("#console-send-message", Button).press()
        except QueryError:
            self.app_instance.notify("Console send is unavailable.", severity="error")

    def _defer_send_for_dictation(self) -> bool:
        """Hold the send until dictation has committed its final segment.

        `stop_dictation()` is asynchronous -- it runs on a worker and joins for
        up to two seconds. Sending in the same tick would ship the message
        before the last recognized segment reaches the draft, putting the tail
        of the sentence into the *next* message. Returns True when the send was
        deferred and the caller should do nothing further.
        """
        composer = self._console_composer_or_none()
        if composer is None or not composer.dictation_active:
            return False
        if self._console_pending_voice_send:
            # Already waiting; a second Enter must not stop twice.
            return True
        self._console_pending_voice_send = True
        composer.stop_dictation()
        return True
```

Add the state handler:

```python
    @on(ConsoleComposerBar.VoiceStateChanged)
    def _handle_console_voice_state(
        self, event: ConsoleComposerBar.VoiceStateChanged
    ) -> None:
        """Fire a deferred send once dictation has fully settled."""
        if event.state != "idle" or not self._console_pending_voice_send:
            return
        self._console_pending_voice_send = False
        self._press_console_send()
```

Extend the failure handler from Task 7 so a failed stop never sends:

```python
    @on(ConsoleComposerBar.VoiceFailure)
    def _handle_console_voice_failure(
        self, event: ConsoleComposerBar.VoiceFailure
    ) -> None:
        event.stop()
        self._console_pending_voice_send = False
        detail = f"{event.reason} {event.remedy}".strip()
        self.app_instance.notify(detail, severity="warning")
```

Change the `enter` branch of `on_key` (`:11368`) so its first lines become:

```python
        if event.key == "enter":
            if composer.activate_focused_paste_token():
                event.stop()
                event.prevent_default()
                return
            event.stop()
            event.prevent_default()
            if self._defer_send_for_dictation():
                return
            self._press_console_send()
            return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_dictation.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_dictation.py
git commit -m "feat(console): defer send until dictation commits its final segment"
```

---

### Task 9: Shutdown triggers and the session cap

A microphone left live because a screen changed is a privacy bug.

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (session-switch `load_draft` sites at `:3083`, `:7594`, `:11645`, `:11664`; `on_unmount` at `:7658`; the removed-suspend note at `:12438`)
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Test: `Tests/UI/test_console_dictation.py`

**Interfaces:**
- Consumes: `composer.dictation_active`, `composer.stop_dictation()`, `composer.abandon_dictation()` (Task 6).
- Produces: `ChatScreen._stop_dictation_for_session_change() -> None`; `ChatScreen.on_screen_suspend() -> None`; composer constant `VOICE_MAX_SESSION_SECONDS_DEFAULT = 300` and config key `dictation.max_session_seconds`.

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_console_dictation.py`:

```python
def test_session_change_stops_dictation(monkeypatch):
    """The draft is per-session; finals must not land in another session."""
    stopped = []

    class FakeComposer:
        collapsed = False
        dictation_active = True

        def stop_dictation(self):
            stopped.append("stopped")

    screen = ChatScreen.__new__(ChatScreen)
    screen._console_pending_voice_send = False
    monkeypatch.setattr(
        ChatScreen, "_console_composer_or_none", lambda self: FakeComposer(), raising=False
    )

    ChatScreen._stop_dictation_for_session_change(screen)

    assert stopped == ["stopped"]


def test_screen_suspend_stops_dictation(monkeypatch):
    stopped = []

    class FakeComposer:
        collapsed = False
        dictation_active = True

        def stop_dictation(self):
            stopped.append("stopped")

    screen = ChatScreen.__new__(ChatScreen)
    screen._console_pending_voice_send = False
    monkeypatch.setattr(
        ChatScreen, "_console_composer_or_none", lambda self: FakeComposer(), raising=False
    )

    ChatScreen.on_screen_suspend(screen)

    assert stopped == ["stopped"]


def test_screen_suspend_is_cheap_when_not_recording(monkeypatch):
    """The previous suspend override was removed for cost; stay O(1)."""
    looked_up = []

    class FakeComposer:
        collapsed = False
        dictation_active = False

        def stop_dictation(self):
            looked_up.append("stopped")

    screen = ChatScreen.__new__(ChatScreen)
    monkeypatch.setattr(
        ChatScreen, "_console_composer_or_none", lambda self: FakeComposer(), raising=False
    )

    ChatScreen.on_screen_suspend(screen)

    assert looked_up == []
```

And in `Tests/UI/test_console_voice_chip.py`:

```python
@pytest.mark.asyncio
async def test_session_cap_stops_dictation(monkeypatch):
    from tldw_chatbook.Chat import console_voice_input as cvi

    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))

    class Service:
        def start_dictation(self, **callbacks):
            return True

        def stop_dictation(self):
            return None

    monkeypatch.setattr(cvi, "default_service_factory", lambda **kw: Service())

    app = ComposerApp()
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer._voice_max_seconds = 2
        composer.toggle_dictation()
        await pilot.pause()

        composer._voice_elapsed = 2
        composer._tick_voice_elapsed()
        await pilot.pause()

        assert composer.dictation_active is False


@pytest.mark.asyncio
async def test_abandon_releases_without_blocking(monkeypatch):
    from tldw_chatbook.Chat import console_voice_input as cvi

    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))

    stops = []

    class Service:
        def start_dictation(self, **callbacks):
            return True

        def stop_dictation(self):
            stops.append("joined")

    monkeypatch.setattr(cvi, "default_service_factory", lambda **kw: Service())

    app = ComposerApp()
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.toggle_dictation()
        await pilot.pause()

        composer.abandon_dictation()
        await pilot.pause()

        assert stops == []  # teardown must not go through the 2s join
        assert composer._voice_timer is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_dictation.py Tests/UI/test_console_voice_chip.py -v -k "session or suspend or cap or abandon"`
Expected: FAIL — `AttributeError: type object 'ChatScreen' has no attribute '_stop_dictation_for_session_change'`

- [ ] **Step 3: Write minimal implementation**

In `console_composer_bar.py`, add the cap constant next to `VOICE_CHIP_MIN_WIDTH`:

```python
    VOICE_MAX_SESSION_SECONDS_DEFAULT = 300
```

In `__init__`, alongside the other voice attributes:

```python
        self._voice_max_seconds = coerce_int_setting(
            get_cli_setting("dictation", "max_session_seconds", None),
            self.VOICE_MAX_SESSION_SECONDS_DEFAULT,
        )
```

Add `get_cli_setting` to the existing `from ...config import (...)` block.

Change `_tick_voice_elapsed` to enforce the cap:

```python
    def _tick_voice_elapsed(self) -> None:
        self._voice_elapsed += 1
        if self._voice_elapsed >= self._voice_max_seconds:
            # A hot microphone is a privacy problem, not just an untidy one.
            self.stop_dictation()
            self.post_message(
                self.VoiceFailure(
                    f"Dictation stopped after {self._voice_max_seconds}s.", ""
                )
            )
            return
        self._render_voice_chip()
```

In `chat_screen.py`, add:

```python
    def _stop_dictation_for_session_change(self) -> None:
        """Stop dictation before the composer's draft is swapped.

        The draft is per-session and replaced by `load_draft`; leaving the
        microphone live would route the next finalized segment into a
        different session's draft.
        """
        composer = self._console_composer_or_none()
        if composer is None or not composer.dictation_active:
            return
        self._console_pending_voice_send = False
        composer.stop_dictation()

    def on_screen_suspend(self) -> None:
        """Release the microphone when navigating away from the Console.

        Deliberately O(1) when not recording: a previous `on_screen_suspend`
        override here was removed for doing expensive work on every tab switch
        (see the task-247 note below). No `super()` call -- `BaseAppScreen`
        does not define this hook.
        """
        composer = self._console_composer_or_none()
        if composer is None or not composer.dictation_active:
            return
        self._console_pending_voice_send = False
        composer.stop_dictation()
```

Call `self._stop_dictation_for_session_change()` immediately before each of the four `composer.load_draft(...)` session-switch sites (`:3083`, `:7594`, `:11645`, `:11664`). Do **not** add it to the `load_draft` at `:8562` — that one inserts a suggested prompt into the current session and is not a session change.

In `on_unmount` (`:7658`), add as the first statement:

```python
        composer = self._console_composer_or_none()
        if composer is not None:
            composer.abandon_dictation()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_dictation.py Tests/UI/test_console_voice_chip.py -v`
Expected: 27 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_composer_bar.py Tests/UI/test_console_dictation.py Tests/UI/test_console_voice_chip.py
git commit -m "feat(console): release the microphone on every teardown path"
```

---

### Task 10: Startup-cost guard, live verification, and follow-ups

**Files:**
- Test: `Tests/Chat/test_console_voice_input.py`
- Create: three files in `backlog/tasks/`

**Interfaces:**
- Consumes: everything above.
- Produces: no new code interfaces.

- [ ] **Step 1: Write the failing test**

Append to `Tests/Chat/test_console_voice_input.py`:

```python
def test_importing_the_controller_does_not_pull_in_the_audio_stack():
    """The Console imports this module; it must stay cheap.

    `tldw_chatbook.Audio` chains to `transcription_service`, which imports
    faster-whisper and NeMo at module scope. If this ever regresses, Console
    startup gains seconds and nothing else fails.
    """
    import subprocess
    import sys as _sys

    code = (
        "import sys;"
        "import tldw_chatbook.Chat.console_voice_input as m;"
        "m.probe();"
        "banned = [n for n in sys.modules"
        " if n.startswith('tldw_chatbook.Audio')"
        " or n.endswith('transcription_service')"
        " or n == 'faster_whisper' or n.startswith('nemo')];"
        "print(banned)"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Temporarily add `from ..Audio.dictation_service_lazy import LazyLiveDictationService` at module scope in `console_voice_input.py`, then run:

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_voice_input.py -v -k audio_stack`
Expected: FAIL — the banned list is non-empty. **Remove the temporary import before continuing.**

- [ ] **Step 3: Verify the test passes against the real implementation**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_voice_input.py -v -k audio_stack`
Expected: PASS

- [ ] **Step 4: Run the full affected suites and compare to baseline**

```bash
.venv/bin/python -m pytest Tests/Chat Tests/UI -q > /tmp/after.txt 2>&1
git stash
.venv/bin/python -m pytest Tests/Chat Tests/UI -q > /tmp/before.txt 2>&1
git stash pop
grep -E "^(FAILED|ERROR)" /tmp/before.txt | sort > /tmp/before-names.txt
grep -E "^(FAILED|ERROR)" /tmp/after.txt | sort > /tmp/after-names.txt
comm -13 /tmp/before-names.txt /tmp/after-names.txt
```

Expected: the `comm` output is empty. Compare **names**, never counts — a count match can hide one test fixed and another broken.

- [ ] **Step 5: Live verification**

Launch the app per the `verify` skill and confirm by hand, because none of the above exercises a real microphone:

1. `alt+r` while the composer has focus starts dictation; the chip appears and is readable.
2. Speech produces finalized text in the draft, and **Send becomes enabled**.
3. `alt+r` again stops; text remains and is editable.
4. Enter while listening sends a message that **includes the last thing you said**.
5. Navigating to another screen mid-dictation releases the microphone (the OS mic indicator clears).
6. With `speech_recording` uninstalled, the mic button is visible, disabled, and its tooltip names the install command.

Record the outcome of each in the task's Implementation Notes.

- [ ] **Step 6: File the follow-up tasks**

Assign IDs by scanning **all** worktrees with a Python `os.listdir` + regex scan against `origin/dev` — `git ls-tree | uniq` misses em-dash filenames — and re-verify the IDs immediately before writing the files, since IDs collide routinely here. Then create:

1. **Fix the `lightning-whisper` allowlist mismatch** — `dictation_service_lazy.py:341` lists `"lightning-whisper"`, but the real provider id is `"lightning-whisper-mlx"`, so lightning users are silently rewritten to `parakeet-mlx`. AC: the allowlist matches the ids used by `transcription_service`; a test covers a lightning-whisper-mlx user keeping their provider.
2. **Delete `Widgets/voice_input_button.py`** — zero callers, and it touches widgets from the transcription worker thread (`_on_partial` → `_set_status` → `query_one`). AC: file removed; its macOS permission remedy copy preserved wherever it is still used; no import breaks.
3. **Composer undo/redo (`ctrl+z` / `ctrl+shift+z`)** — covering typing, paste, file segments, and dictation uniformly. AC: keys registered in `ChatScreen.on_key`'s whitelist next to `ctrl+u`, **not** in `BINDINGS`; undo pops the last dictation insertion.

- [ ] **Step 7: Commit**

```bash
git add Tests/Chat/test_console_voice_input.py backlog/tasks/
git commit -m "test(console): guard dictation import cost; file voice follow-ups"
```

---

## Self-review notes

Checked against the spec:

- **Spec coverage.** Every constraint (1–7) maps to a task: press-and-hold → Task 7's toggle; `post_message` → Task 6; hidden status element → Task 4's `_visible` helper; import cost → Tasks 1, 3, 10; blocking calls → Task 6's `_spawn_voice_work` and Task 9's `abandon`; provider swap → Task 2; markup → Task 4. Interaction model → Tasks 6–9. Failure and safety → Tasks 1, 7, 9. The one spec item with no task is the input-level meter, which the spec explicitly excludes.
- **Type consistency.** `set_voice_status(state, *, partial, elapsed_seconds, message)` is defined in Task 4 and called with those exact keywords in Task 6. `insert_dictated_text(text) -> bool` is defined in Task 5 and consumed in Task 6. `STATE_*` constants are defined in Task 3 and imported by name in Task 4's test. `_press_console_send()` is introduced in Task 8 and reused by the `enter` branch.
- **Known refinement to the spec.** The spec lists two injected callables (`emit`, `service_factory`); the plan adds a third, `spawn`. A headless controller cannot call `run_worker`, so threading policy has to be injected for `start()`/`stop()` to be non-blocking as specified. Tests pass a synchronous `spawn`, which is also what makes the state machine testable without an event loop.
