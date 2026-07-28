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


def test_capture_available_true_with_only_one_backend_installed(monkeypatch):
    """any(), not all(): a single resolvable backend is enough.

    Drives the real `capture_available()`/`_module_installed()` by patching
    the `importlib.util.find_spec` seam, rather than patching
    `capture_available` itself, so an `any()` -> `all()` mutation would fail
    this test.
    """
    def fake_find_spec(name, *args, **kwargs):
        return object() if name == "sounddevice" else None

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)

    assert cvi.capture_available() is True


def test_capture_available_false_with_no_backend_installed(monkeypatch):
    """Neither pyaudio nor sounddevice resolves -> no capture backend."""
    monkeypatch.setattr(cvi.importlib.util, "find_spec", lambda name, *a, **k: None)

    assert cvi.capture_available() is False


def test_installed_local_providers_returns_subset_in_declared_order(monkeypatch):
    """Only installed providers are returned, in LOCAL_PROVIDER_MODULES order.

    `faster_whisper` is deliberately excluded from `installed` so the result
    is a proper subset. The two that remain (`parakeet-mlx`,
    `lightning-whisper-mlx`) are alphabetically out of order relative to each
    other, so a stray `sorted()` in the implementation would also fail this
    test. Patches `find_spec` directly so a real, potentially
    machine-installed `parakeet_mlx` cannot leak into the result.
    """
    installed = {"parakeet_mlx", "lightning_whisper_mlx"}

    def fake_find_spec(name, *args, **kwargs):
        return object() if name in installed else None

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)

    assert cvi.installed_local_providers() == ("parakeet-mlx", "lightning-whisper-mlx")


@pytest.mark.parametrize("exc", [ImportError, ValueError])
def test_module_installed_returns_false_when_find_spec_raises(monkeypatch, exc):
    """A broken namespace package raises rather than returning None from
    `find_spec`; `_module_installed` must swallow that and report False
    instead of propagating.
    """
    def fake_find_spec(name, *args, **kwargs):
        raise exc("broken namespace package")

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)

    assert cvi._module_installed("broken.namespace.package") is False


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
    """The configured provider must NOT be at index 0.

    With `configured` first, deleting the entire honor-configured branch and
    always taking `installed[0]` would still pass -- the fallback happens to
    produce the same answer. Ordering it second is what makes this test
    detect that deletion.
    """
    monkeypatch.setattr(
        cvi, "installed_local_providers", lambda: ("parakeet-mlx", "faster-whisper")
    )
    _stub_settings(
        monkeypatch,
        {
            "transcription.default_provider": "faster-whisper",
            "transcription.default_model": "base",
            # Deliberately not "en": DEFAULT_LANGUAGE is also "en", so an "en"
            # stub can't tell a correct `default_language` read apart from a
            # mutated `language` read silently falling back to the default.
            "transcription.default_language": "fr",
        },
    )

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider == "faster-whisper"
    assert effective.model == "base"
    assert effective.language == "fr"
    assert effective.was_overridden is False


def test_resolve_flags_override_instead_of_swapping_silently(monkeypatch):
    """A configured provider that is not installed is replaced, and it shows."""
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.default_provider": "parakeet-mlx"})

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider == "faster-whisper"
    assert effective.configured_provider == "parakeet-mlx"
    assert effective.was_overridden is True


def test_resolve_never_returns_an_uninstalled_provider(monkeypatch):
    """This is the guard against the service's parakeet-mlx rewrite."""
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.default_provider": "lightning-whisper-mlx"})

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider in cvi.installed_local_providers()


def test_resolve_fallback_prefers_the_first_declared_provider(monkeypatch):
    """With several installed and none configured, declaration order decides.

    Task 1 pins `installed_local_providers()`' order; this pins that `resolve()`
    consumes it as a preference order rather than sorting it. A single-element
    `installed` tuple cannot detect `sorted(installed)[0]`.
    """
    monkeypatch.setattr(
        cvi,
        "installed_local_providers",
        lambda: ("parakeet-mlx", "faster-whisper", "lightning-whisper-mlx"),
    )
    _stub_settings(monkeypatch, {"transcription.default_provider": "qwen2audio"})

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider == "parakeet-mlx"
    assert effective.was_overridden is True


def test_resolve_reads_the_real_config_key_names(monkeypatch):
    """Guards the exact bug this task shipped once already.

    [transcription] uses default_provider (config.py:3333), and the raw TOML
    section STTSettings lands under STT_settings (config.py:1548). Reading
    "provider" or "STTSettings" silently yields the default, so `configured`
    is always "" and resolve() degrades to always-fallback.
    """
    monkeypatch.setattr(
        cvi, "installed_local_providers", lambda: ("parakeet-mlx", "faster-whisper")
    )
    _stub_settings(monkeypatch, {"transcription.default_provider": "faster-whisper"})

    effective = cvi.resolve()

    assert effective is not None
    assert effective.configured_provider == "faster-whisper"
    assert effective.was_overridden is False


def test_resolve_falls_back_to_stt_settings_section_name(monkeypatch):
    """Pins the STT_settings key name specifically, not just default_provider.

    `test_resolve_reads_the_real_config_key_names` always stubs
    `transcription.default_provider`, so the `or get_cli_setting("STT_settings", ...)`
    fallback branch is never reached there and a `STT_settings` -> `STTSettings`
    mutation would pass unnoticed. This test leaves `transcription.default_provider`
    unset so only the fallback section name can produce the expected result.
    """
    monkeypatch.setattr(
        cvi, "installed_local_providers", lambda: ("parakeet-mlx", "faster-whisper")
    )
    _stub_settings(
        monkeypatch, {"STT_settings.default_stt_provider": "faster-whisper"}
    )

    effective = cvi.resolve()

    assert effective is not None
    assert effective.configured_provider == "faster-whisper"
    assert effective.provider == "faster-whisper"
    assert effective.was_overridden is False


def test_resolve_returns_none_when_nothing_installed(monkeypatch):
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ())
    _stub_settings(monkeypatch, {"transcription.default_provider": "faster-whisper"})

    assert cvi.resolve() is None


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


class FakeAudioService:
    """Stands in for LazyLiveDictationService._audio_service.

    `abandon()` reaches for this private attribute on purpose: it is the
    teardown path that releases the microphone without going through
    `stop_dictation()`'s 2s thread join.
    """

    def __init__(self, raise_on_stop: Exception | None = None):
        self.stop_called = False
        self._raise_on_stop = raise_on_stop

    def stop_recording(self):
        self.stop_called = True
        if self._raise_on_stop is not None:
            raise self._raise_on_stop


def _controller(monkeypatch, service=None, spawn=None):
    """Build a controller with a fake service.

    `spawn` defaults to running the thunk inline, which is what makes the
    state machine testable without an event loop. Pass a deferring spawn to
    freeze the controller mid-`preparing`.
    """
    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.default_provider": "faster-whisper"})

    service = service or FakeDictationService()

    def _service_factory(**kwargs):
        # Same identity every call (so the test's `service` reference stays
        # valid across `start()`), but record the kwargs the controller
        # actually built it with -- a bare `lambda **kwargs: service` would
        # silently drop them and `service.kwargs` would stay `{}` forever.
        service.kwargs = kwargs
        return service

    events = []
    controller = cvi.ConsoleVoiceInputController(
        emit=events.append,
        spawn=spawn or (lambda thunk: thunk()),
        service_factory=_service_factory,
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
    _stub_settings(monkeypatch, {"transcription.default_provider": "parakeet-mlx"})

    controller.start()
    controller.stop()
    controller.start()

    overrides = [e for e in events if isinstance(e, cvi.VoiceProviderOverridden)]
    assert len(overrides) == 1
    assert overrides[0].effective == "faster-whisper"


# -- Fix round 1: wedge-proofing (probe/resolve crash, throwing emit, --------
# -- abandon()/`_begin()` race) ----------------------------------------------


def test_start_returns_to_idle_when_probe_raises(monkeypatch):
    """A probe()/resolve() crash -- e.g. a corrupt config -- must not wedge
    the state machine in `preparing` forever: every later start() would
    no-op (state != idle) and every stop() would no-op (state != listening).
    """
    controller, events, _ = _controller(monkeypatch)

    def boom():
        raise RuntimeError("config read blew up")

    monkeypatch.setattr(cvi, "probe", boom)

    controller.start()

    failures = [e for e in events if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert "config read blew up" in failures[0].reason
    assert controller.state == cvi.STATE_IDLE


def test_fail_leaves_state_idle_even_if_emit_raises(monkeypatch):
    """A throwing `emit` (plausible: a Textual `post_message` racing widget
    teardown) must not leave the internal state wedged, even though the
    `VoiceFailed` event itself is lost when that emit throws.

    Only `VoiceFailed` raises here, not every event: the earlier
    `VoiceStateChanged(preparing)` emit in `start()` must go through
    normally, so this isolates `_fail()`'s own mutate-before-notify ordering
    rather than the (out of scope) question of a globally-raising `emit`.
    """
    monkeypatch.setattr(cvi, "capture_available", lambda: False)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))

    def emit_raising_on_failure(event):
        if isinstance(event, cvi.VoiceFailed):
            raise RuntimeError("widget torn down")

    controller = cvi.ConsoleVoiceInputController(
        emit=emit_raising_on_failure,
        spawn=lambda thunk: thunk(),
        service_factory=lambda **kwargs: FakeDictationService(**kwargs),
    )

    with pytest.raises(RuntimeError):
        controller.start()

    assert controller.state == cvi.STATE_IDLE


def test_fail_emits_voicefailed_before_voicestatechanged_idle(monkeypatch):
    """Event order is load-bearing: a later task defers a pending "send"
    until it observes `VoiceStateChanged(idle)`, and clears that pending
    flag on `VoiceFailed`. Reversing the order would send the user's message
    on a failed dictation.
    """
    service = FakeDictationService()
    service.start_error = RuntimeError("no input device")
    controller, events, _ = _controller(monkeypatch, service)

    controller.start()

    failed_index = next(i for i, e in enumerate(events) if isinstance(e, cvi.VoiceFailed))
    idle_index = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, cvi.VoiceStateChanged) and e.state == cvi.STATE_IDLE
    )
    assert failed_index < idle_index


def test_abandon_while_idle_is_a_no_op(monkeypatch):
    controller, events, _ = _controller(monkeypatch)

    controller.abandon()  # must not raise

    assert controller.state == cvi.STATE_IDLE
    assert events == []


def test_abandon_while_listening_releases_without_stop_dictation(monkeypatch):
    """The 2s thread join inside stop_dictation() is exactly what abandon()
    exists to avoid at teardown."""
    service = FakeDictationService()
    audio = FakeAudioService()
    service._audio_service = audio
    controller, events, _ = _controller(monkeypatch, service)
    controller.start()
    assert controller.state == cvi.STATE_LISTENING

    controller.abandon()

    assert controller.state == cvi.STATE_IDLE
    assert service.stopped is False
    assert audio.stop_called is True
    assert controller.is_active is False


def test_abandon_mid_preparing_releases_service_built_after_abandon(monkeypatch):
    """Guards the race: user quits while the model is still loading.

    `abandon()` runs before the deferred `_begin()` thunk (the captured
    `spawn` call) has executed, so `controller._service` is still None at
    that point -- a naive `abandon()` sees nothing to release. The in-flight
    `_begin()` must still notice, after it finishes building/starting the
    service, that the controller was abandoned meanwhile, and release the
    microphone instead of transitioning to `listening`.
    """
    pending = []
    audio = FakeAudioService()
    service = FakeDictationService()
    service._audio_service = audio
    controller, events, _ = _controller(monkeypatch, service=service, spawn=pending.append)

    controller.start()
    assert controller.state == cvi.STATE_PREPARING
    assert len(pending) == 1

    controller.abandon()
    assert controller.state == cvi.STATE_IDLE

    # Simulate the factory/start_dictation() call finishing after abandon().
    pending[0]()

    assert controller.state == cvi.STATE_IDLE
    assert controller.is_active is False
    assert audio.stop_called is True


def test_abandon_swallows_a_raising_release(monkeypatch):
    """Teardown must never raise, even if the audio backend's own
    stop_recording() raises."""
    service = FakeDictationService()
    service._audio_service = FakeAudioService(raise_on_stop=RuntimeError("device gone"))
    controller, events, _ = _controller(monkeypatch, service)
    controller.start()
    assert controller.state == cvi.STATE_LISTENING

    controller.abandon()  # must not raise

    assert controller.state == cvi.STATE_IDLE


# -- Fix round 2: cascading double-VoiceFailed, ghost-listening race, --------
# -- abandon-then-start, unguarded override-emit/spawn -----------------------


def test_probe_failure_does_not_cascade_into_a_second_voicefailed(monkeypatch):
    """`_fail()`'s own second emit (`VoiceStateChanged(idle)`) raising must
    not be caught by the outer probe/resolve guard in `start()` and re-fire
    `_fail()` with the plumbing exception's message instead of the real,
    original unavailability reason.

    Reproduces the exact scenario Finding 2 exists for -- an `emit` that
    raises partway through `_fail()`'s two calls -- one level up, at the
    call site that wraps `probe()`/`resolve()`.
    """
    monkeypatch.setattr(cvi, "capture_available", lambda: False)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))

    recorded = []

    def emit_raising_on_third_event(event):
        recorded.append(event)
        if len(recorded) == 3:  # VoiceStateChanged(idle), _fail()'s 2nd emit
            raise RuntimeError("widget torn down mid-delivery")

    controller = cvi.ConsoleVoiceInputController(
        emit=emit_raising_on_third_event,
        spawn=lambda thunk: thunk(),
        service_factory=lambda **kwargs: FakeDictationService(**kwargs),
    )

    with pytest.raises(RuntimeError):
        controller.start()

    failures = [e for e in recorded if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert failures[0].reason == cvi.CAPTURE_REASON


def test_fail_emits_voicefailed_before_voicestatechanged_idle_after_restructure(monkeypatch):
    """Re-confirms the Finding 2 invariant survives the try/except
    restructuring done for NEW BREAKAGE 1: state mutated first, then
    VoiceFailed, then VoiceStateChanged(idle)."""
    service = FakeDictationService()
    service.start_error = RuntimeError("no input device")
    controller, events, _ = _controller(monkeypatch, service)

    controller.start()

    failed_index = next(i for i, e in enumerate(events) if isinstance(e, cvi.VoiceFailed))
    idle_index = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, cvi.VoiceStateChanged) and e.state == cvi.STATE_IDLE
    )
    assert failed_index < idle_index
    assert controller.state == cvi.STATE_IDLE


def test_start_returns_to_idle_when_spawn_raises(monkeypatch):
    """A raising spawn() (e.g. a Textual run_worker() call itself failing)
    must not wedge `preparing` forever, same as a probe()/resolve() crash."""

    def raising_spawn(thunk):
        raise RuntimeError("worker pool exhausted")

    controller, events, _ = _controller(monkeypatch, spawn=raising_spawn)

    controller.start()

    failures = [e for e in events if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert "worker pool exhausted" in failures[0].reason
    assert controller.state == cvi.STATE_IDLE


def test_ghost_listening_race_is_closed_by_the_atomic_recheck(monkeypatch):
    """Reproduces the narrow gap between claiming the service (locked, in
    `_begin()`) and transitioning to `listening` (originally unlocked): if
    `abandon()` lands in that exact gap, the in-flight `_begin()` must not
    stomp the state back to `listening` with the microphone already
    released. `_enter_listening()` is the seam: it is invoked only after the
    service is claimed, so patching it to run `abandon()` first simulates
    `abandon()` firing on the UI thread at the worst possible instant.
    """
    audio = FakeAudioService()
    service = FakeDictationService()
    service._audio_service = audio
    controller, events, _ = _controller(monkeypatch, service=service)

    real_enter_listening = controller._enter_listening

    def enter_listening_after_concurrent_abandon():
        controller.abandon()
        real_enter_listening()

    monkeypatch.setattr(controller, "_enter_listening", enter_listening_after_concurrent_abandon)

    controller.start()

    assert controller.state == cvi.STATE_IDLE
    assert controller.is_active is False
    assert controller._service is None
    assert audio.stop_called is True


def test_start_after_abandon_never_constructs_a_service(monkeypatch):
    """`abandon()` is a one-way, terminal latch: once torn down, a later
    `start()` on the same controller must not engage the microphone at all,
    even briefly -- not construct a service, not call start_dictation(), not
    emit anything.
    """
    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.default_provider": "faster-whisper"})

    def factory_must_not_be_called(**kwargs):
        raise AssertionError("service_factory must not be called after abandon()")

    events = []
    controller = cvi.ConsoleVoiceInputController(
        emit=events.append,
        spawn=lambda thunk: thunk(),
        service_factory=factory_must_not_be_called,
    )
    controller.abandon()
    events.clear()

    controller.start()

    assert events == []
    assert controller.state == cvi.STATE_IDLE
    assert controller.is_active is False


# -- Fix round 3: N1's cascade recurring inside _begin() (via the S4 fix's --
# -- try/except around an inline spawn), plus zero coverage of the --------
# -- not-started path that let it slip past four targeted tests -----------


def test_begin_when_start_dictation_returns_false(monkeypatch):
    """`_begin()`'s `if not started:` branch had ZERO test coverage before
    this round -- exactly how the round-2 mutation checks and four targeted
    tests missed the cascade this round fixes. Plain coverage, no raising
    emit involved.
    """
    service = FakeDictationService()
    service.start_result = False
    controller, events, _ = _controller(monkeypatch, service)

    controller.start()

    failures = [e for e in events if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert failures[0].reason == "Could not start the microphone."
    assert controller.state == cvi.STATE_IDLE
    assert controller._service is None


def test_begin_not_started_failure_does_not_cascade_into_a_second_voicefailed(monkeypatch):
    """The S4 fix (wrapping `self._spawn(...)` in `start()`) reintroduced the
    N1 cascade one call frame deeper: with the default inline `spawn`, that
    try transitively covers all of `_begin()`, so `_begin()`'s own `_fail()`
    call (for a `start_dictation()` that returns `False`) had its raising
    emit re-caught by `start()`'s guard and re-fired as a second, mislabeled
    `VoiceFailed` describing the plumbing exception instead of the real
    cause. `_begin()` is now a thread boundary that swallows anything raised
    inside it, so `start()` must not raise at all here.
    """
    service = FakeDictationService()
    service.start_result = False

    recorded = []

    def emit_raising_on_third_event(event):
        recorded.append(event)
        if len(recorded) == 3:  # VoiceStateChanged(idle), _fail()'s 2nd emit
            raise RuntimeError("widget torn down mid-delivery")

    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.default_provider": "faster-whisper"})

    controller = cvi.ConsoleVoiceInputController(
        emit=emit_raising_on_third_event,
        spawn=lambda thunk: thunk(),
        service_factory=lambda **kwargs: service,
    )

    controller.start()  # must NOT raise: _begin() is a thread boundary

    failures = [e for e in recorded if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert failures[0].reason == "Could not start the microphone."
    assert controller.state == cvi.STATE_IDLE


def test_begin_not_started_stays_quiet_when_abandoned_in_the_gap(monkeypatch):
    """SECONDARY fix: if `abandon()` lands between `_begin()` claiming the
    service and its `if not started:` check, the controller is already idle
    and torn down; `_begin()` must not pile a spurious
    `VoiceFailed`/`VoiceStateChanged(idle)` pair on top of that.
    `_fail_not_started()` is the seam -- patching it to run `abandon()`
    before delegating simulates `abandon()` landing at exactly that instant,
    the same technique used for the round-2 ghost-listening race test.
    """
    service = FakeDictationService()
    service.start_result = False
    controller, events, _ = _controller(monkeypatch, service)

    real_fail_not_started = controller._fail_not_started

    def fail_not_started_after_concurrent_abandon():
        controller.abandon()
        real_fail_not_started()

    monkeypatch.setattr(controller, "_fail_not_started", fail_not_started_after_concurrent_abandon)

    controller.start()

    failures = [e for e in events if isinstance(e, cvi.VoiceFailed)]
    assert failures == []
    assert controller.state == cvi.STATE_IDLE


# -- Fix round 4: the real service's synchronous on_error contract (one -----
# -- failure, two VoiceFailed events), and the stop() path never getting ----
# -- the hardening the start() path did ------------------------------------


class NotifyingFakeDictationService(FakeDictationService):
    """Mirrors `LazyLiveDictationService.start_dictation()`'s real contract.

    All three of its failure branches call `self._notify_error(e)`
    **synchronously, before returning** and then `return False` rather than
    raising: `dictation_service_lazy.py` lines 285-290 (audio/transcription
    init failure), 323-329 (`start_recording()` returned `False`) and 332-335
    (catch-all). `_notify_error()` (702-714) wraps the `on_error(...)` call in
    its own log-only try/except, so an exception escaping the controller's
    callback is swallowed there -- it never turns into a raise out of
    `start_dictation()`.

    `FakeDictationService` models the *raising* shape instead, which is why
    the double-`VoiceFailed` this class reproduces went unnoticed.
    """

    def __init__(self, error: Exception | None, shield_callback: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.start_result = False
        self.sync_error = error
        # `_notify_error()` shields itself from a raising `on_error`. A
        # service that does not is the only way our callback's own failure
        # can reach `_run_begin()`'s `try`, so it is worth covering.
        self.shield_callback = shield_callback

    def start_dictation(self, **callbacks):
        self._callbacks = callbacks
        if self.sync_error is not None:
            if not self.shield_callback:
                callbacks["on_error"](self.sync_error)
            else:
                try:
                    callbacks["on_error"](self.sync_error)
                except Exception:  # noqa: BLE001 - mirrors _notify_error()'s log-only catch
                    pass
        return self.start_result


class LockAuditingController(cvi.ConsoleVoiceInputController):
    """Records whether `_state_lock` was held on each `_service` touch.

    `self._service` is read-and-cleared from a worker thread (`_run_finish`)
    and from the UI thread (`abandon`), so every touch has to happen under
    `_state_lock`. A plain instance attribute offers no seam to observe that,
    hence the property. `_state_lock` does not exist yet when the base
    `__init__` makes its first assignment, which is why the recorder tolerates
    its absence.
    """

    def __init__(self, **kwargs):
        self.service_touches: list[bool] = []
        self._service_value = None
        super().__init__(**kwargs)

    def _lock_held(self) -> bool:
        lock = self.__dict__.get("_state_lock")
        return bool(lock.locked()) if lock is not None else True

    @property
    def _service(self):
        self.service_touches.append(self._lock_held())
        return self._service_value

    @_service.setter
    def _service(self, value):
        self.service_touches.append(self._lock_held())
        self._service_value = value


def test_synchronous_on_error_is_the_only_failure_reported(monkeypatch):
    """One real failure must produce exactly one `VoiceFailed`, carrying the
    real cause.

    Against the *real* dependency this needs no exception and no race: the
    service reports through `on_error` synchronously and then returns `False`,
    so the controller's `on_error` fires `_fail(real cause)` and the
    `if not started:` path then fired `_fail("Could not start the
    microphone.")` on top. The generic one arrived **last**, so a UI showing
    the latest failure buried the actionable diagnostic.
    """
    service = NotifyingFakeDictationService(RuntimeError("no input device"))
    controller, events, _ = _controller(monkeypatch, service)

    controller.start()

    failures = [e for e in events if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert "no input device" in failures[0].reason
    assert controller.state == cvi.STATE_IDLE
    assert controller._service is None
    # The load-bearing ordering still holds on this path too.
    failed_index = next(i for i, e in enumerate(events) if isinstance(e, cvi.VoiceFailed))
    idle_index = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, cvi.VoiceStateChanged) and e.state == cvi.STATE_IDLE
    )
    assert failed_index < idle_index


def test_not_started_fallback_is_per_attempt_not_per_instance(monkeypatch):
    """A failed attempt must not silence the next one.

    Attempt 1 reports the real cause through `on_error`, which sets the
    suppression latch. Attempt 2 fails the other way -- `start_dictation()`
    just returns `False`, with no `on_error` at all -- so the generic fallback
    is the only report available and must still be emitted.
    """
    service = NotifyingFakeDictationService(RuntimeError("no input device"))
    controller, events, _ = _controller(monkeypatch, service)

    controller.start()
    assert controller.state == cvi.STATE_IDLE
    events.clear()

    service.sync_error = None  # this time the service reports nothing at all
    controller.start()

    failures = [e for e in events if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert failures[0].reason == "Could not start the microphone."
    assert controller.state == cvi.STATE_IDLE


def test_synchronous_on_error_does_not_cascade_through_run_begin(monkeypatch):
    """The last place a `_fail()` still sits transitively inside a `try`.

    `on_error` is invoked from *inside* `start_dictation()`, which
    `_run_begin()` wraps -- so if our callback's own `_fail()` emit raises and
    the service does not shield the callback, `_run_begin()`'s `except` would
    catch it and fire a second `VoiceFailed` carrying the plumbing message.
    The same per-attempt latch closes it: the real cause has already been
    reported, so the fallback stays quiet.
    """
    service = NotifyingFakeDictationService(
        RuntimeError("no input device"), shield_callback=False
    )

    recorded = []

    def emit_raising_on_idle(event):
        recorded.append(event)
        if isinstance(event, cvi.VoiceStateChanged) and event.state == cvi.STATE_IDLE:
            raise RuntimeError("widget torn down mid-delivery")

    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.default_provider": "faster-whisper"})

    controller = cvi.ConsoleVoiceInputController(
        emit=emit_raising_on_idle,
        spawn=lambda thunk: thunk(),
        service_factory=lambda **kwargs: service,
    )

    controller.start()  # must NOT raise: _begin() is a thread boundary

    failures = [e for e in recorded if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert "no input device" in failures[0].reason
    assert controller.state == cvi.STATE_IDLE


def test_stop_returns_to_idle_when_spawn_raises(monkeypatch):
    """`stop()` never got `start()`'s wedge guard: a `spawn()` that fails to
    schedule left `finishing` set forever, with `is_active` true and nothing
    else able to unwind it. The microphone must not be left recording behind
    the resulting idle state either.
    """
    calls = {"n": 0}

    def spawn(thunk):
        calls["n"] += 1
        if calls["n"] == 1:
            thunk()  # start()'s _begin() runs inline, reaching `listening`
            return
        raise RuntimeError("worker pool exhausted")

    audio = FakeAudioService()
    service = FakeDictationService()
    service._audio_service = audio
    controller, events, _ = _controller(monkeypatch, service=service, spawn=spawn)
    controller.start()
    assert controller.state == cvi.STATE_LISTENING
    events.clear()

    controller.stop()

    failures = [e for e in events if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert "worker pool exhausted" in failures[0].reason
    assert controller.state == cvi.STATE_IDLE
    assert controller.is_active is False
    assert audio.stop_called is True
    assert controller._service is None


def test_stop_does_not_wedge_finishing_when_the_finishing_emit_raises(monkeypatch):
    """The other half of the same wedge: the `VoiceStateChanged(finishing)`
    emit itself raising (a Textual `post_message` racing widget teardown).
    """
    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.default_provider": "faster-whisper"})

    recorded = []

    def emit_raising_on_finishing(event):
        recorded.append(event)
        if isinstance(event, cvi.VoiceStateChanged) and event.state == cvi.STATE_FINISHING:
            raise RuntimeError("widget torn down")

    service = FakeDictationService()
    controller = cvi.ConsoleVoiceInputController(
        emit=emit_raising_on_finishing,
        spawn=lambda thunk: thunk(),
        service_factory=lambda **kwargs: service,
    )
    controller.start()
    assert controller.state == cvi.STATE_LISTENING

    controller.stop()

    assert controller.state == cvi.STATE_IDLE
    assert controller.is_active is False
    failures = [e for e in recorded if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert "widget torn down" in failures[0].reason


def test_finish_failure_does_not_cascade_into_a_second_voicefailed(monkeypatch):
    """`stop()`'s new guard must not re-absorb `_finish()`'s own `_fail()`.

    The exact trap that recurred three times: with the default inline `spawn`,
    the guard around `self._spawn(self._finish)` transitively covers all of
    `_finish()`, so `_fail()`'s second emit raising would be caught there and
    re-fired as a second `VoiceFailed` carrying the plumbing message instead
    of the real cause. `_finish()` is an exception boundary, so `stop()` must
    not raise and must report exactly once.
    """
    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.default_provider": "faster-whisper"})

    recorded = []

    def emit_raising_on_idle(event):
        recorded.append(event)
        if isinstance(event, cvi.VoiceStateChanged) and event.state == cvi.STATE_IDLE:
            raise RuntimeError("widget torn down mid-delivery")

    def stop_dictation_boom():
        raise RuntimeError("device gone")

    service = FakeDictationService()
    controller = cvi.ConsoleVoiceInputController(
        emit=emit_raising_on_idle,
        spawn=lambda thunk: thunk(),
        service_factory=lambda **kwargs: service,
    )
    controller.start()
    assert controller.state == cvi.STATE_LISTENING
    monkeypatch.setattr(service, "stop_dictation", stop_dictation_boom)
    recorded.clear()

    controller.stop()  # must NOT raise: _finish() is a thread boundary

    failures = [e for e in recorded if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    assert "device gone" in failures[0].reason
    assert controller.state == cvi.STATE_IDLE


def test_stale_finish_after_abandon_does_not_emit_a_spurious_idle(monkeypatch):
    """`_run_finish()` must re-check abandonment before announcing `idle`.

    Freeze `_finish()` mid-flight (deferring spawn), tear the controller down
    with `abandon()`, then let the stale thunk run: it must not emit a state
    change for a controller that is already torn down -- a later task treats
    `VoiceStateChanged(idle)` as the trigger to send a deferred message. It
    must also not call `stop_dictation()` on the service `abandon()` already
    released, which would land in `_run_finish()`'s `except` and report a
    spurious failure after teardown.
    """
    pending = []
    defer = {"on": False}

    def spawn(thunk):
        if defer["on"]:
            pending.append(thunk)
        else:
            thunk()

    audio = FakeAudioService()
    service = FakeDictationService()
    service._audio_service = audio
    controller, events, _ = _controller(monkeypatch, service=service, spawn=spawn)

    controller.start()
    assert controller.state == cvi.STATE_LISTENING

    defer["on"] = True
    controller.stop()
    assert controller.state == cvi.STATE_FINISHING
    assert len(pending) == 1

    controller.abandon()
    assert controller.state == cvi.STATE_IDLE
    assert audio.stop_called is True
    events.clear()

    pending[0]()  # the frozen _finish() thunk finally runs, after teardown

    assert events == []
    assert controller.state == cvi.STATE_IDLE
    assert service.stopped is False  # abandon() had already taken the service


def test_finish_claims_the_service_under_the_state_lock(monkeypatch):
    """The read-and-clear of `self._service` in the stop path must be locked.

    Every other touch of that attribute is serialized against `abandon()`;
    `_finish()`'s was not, so a UI-thread `abandon()` during teardown could
    come away with the same service (double release) or leave `_run_finish()`
    calling `stop_dictation()` on one already released.
    """
    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.default_provider": "faster-whisper"})

    service = FakeDictationService()
    events = []
    controller = LockAuditingController(
        emit=events.append,
        spawn=lambda thunk: thunk(),
        service_factory=lambda **kwargs: service,
    )
    controller.start()
    assert controller.state == cvi.STATE_LISTENING
    controller.service_touches.clear()

    controller.stop()

    assert controller.service_touches  # the claim really happened...
    assert all(controller.service_touches)  # ...and every touch held the lock
    assert service.stopped is True
    assert controller.state == cvi.STATE_IDLE
