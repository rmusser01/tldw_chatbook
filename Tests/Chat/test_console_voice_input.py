"""Console voice dictation controller tests."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time

import pytest

from tldw_chatbook.Chat import console_voice_input as cvi

pytestmark = pytest.mark.unit


# The in-process `test_probe_does_not_import_transcription_service` below only
# proves `console_voice_input.probe()` stays cheap, and it runs inside a test
# session that has already imported an unpredictable pile of other modules --
# a real leak elsewhere could hide behind something pytest itself dragged in
# first. This script runs in a *clean* interpreter and imports the actual
# screen module Console mounts, so it also covers the path this file's other
# test does not: `chat_screen`'s own imports (`from ...Chat import
# console_voice_input`, `default_service_factory`, etc.) never reaching into
# `tldw_chatbook.Audio` at module scope.
_IMPORT_COST_GUARD_SCRIPT = """
import sys

import tldw_chatbook.UI.Screens.chat_screen  # noqa: F401

banned_exact = {"tldw_chatbook.Audio", "torch"}
banned_prefixes = ("tldw_chatbook.Audio.", "torch.")

leaked = []
for name in sys.modules:
    if name in banned_exact or any(name.startswith(p) for p in banned_prefixes):
        leaked.append(name)
        continue
    if name.endswith("transcription_service"):
        leaked.append(name)
        continue
    if name == "faster_whisper" or name.startswith("faster_whisper."):
        leaked.append(name)
        continue
    if name == "nemo" or name.startswith("nemo."):
        leaked.append(name)
        continue
    if name == "parakeet_mlx" or name.startswith("parakeet_mlx."):
        leaked.append(name)
        continue

print(chr(10).join(sorted(leaked)))
"""


def test_screen_import_does_not_load_transcription_stack():
    """Mounting the Console screen must never pull in the heavy STT stack.

    `tldw_chatbook.Audio` (the package) chains to `transcription_service`,
    which imports `faster_whisper` and `nemo.collections.asr` at module
    scope -- seconds of startup cost for anyone with those extras installed.
    Runs in a subprocess so the result reflects only what importing the
    screen actually triggers, not whatever this test session happened to
    import first.
    """
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_COST_GUARD_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"import of tldw_chatbook.UI.Screens.chat_screen failed:\n{result.stderr}"
    )
    leaked = [line for line in result.stdout.splitlines() if line.strip()]
    assert leaked == [], (
        f"heavy transcription modules leaked into sys.modules: {leaked}\n"
        f"stderr:\n{result.stderr}"
    )


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
    this test. NumPy is also resolved here because `capture_available()`
    unconditionally requires it (`CAPTURE_REQUIRED_MODULES`); pyaudio is
    deliberately left unresolved so this still proves any(), not all(), over
    `CAPTURE_MODULES` specifically.
    """
    def fake_find_spec(name, *args, **kwargs):
        return object() if name in {"sounddevice", "numpy"} else None

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)

    assert cvi.capture_available() is True


def test_capture_available_false_with_no_backend_installed(monkeypatch):
    """Neither pyaudio nor sounddevice resolves -> no capture backend."""
    monkeypatch.setattr(cvi.importlib.util, "find_spec", lambda name, *a, **k: None)

    assert cvi.capture_available() is False


def test_capture_available_false_when_numpy_missing_even_with_backend_present(monkeypatch):
    """A resolvable capture backend is not enough on its own.

    `AudioRecordingService.__init__` raises `AudioRecordingError` when NumPy
    is absent (`Audio/recording_service.py:127`), regardless of which backend
    was chosen. Without this check, `probe()` would report OK and the Mic
    button would light up for a start that deterministically fails.
    """
    def fake_find_spec(name, *args, **kwargs):
        return object() if name == "pyaudio" else None  # numpy stays unresolved

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)

    assert cvi.capture_available() is False


def test_capture_available_true_when_backend_and_numpy_both_present(monkeypatch):
    """Positive case, pinned so the NumPy check can't pass vacuously.

    Only "pyaudio" and "numpy" resolve here (everything else, including
    "sounddevice", stays unresolved), proving both are genuinely consulted
    rather than the fake happening to resolve everything.
    """
    def fake_find_spec(name, *args, **kwargs):
        return object() if name in {"pyaudio", "numpy"} else None

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)

    assert cvi.capture_available() is True


def test_probe_reports_missing_capture_when_numpy_missing(monkeypatch):
    """`probe()` must route a numpy-less machine through the same
    "missing-capture" outcome as a machine with no backend at all -- the
    underlying failure (`AudioRecordingService` refuses to construct) is
    identical either way.
    """
    def fake_find_spec(name, *args, **kwargs):
        return object() if name == "sounddevice" else None  # numpy stays unresolved

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)

    availability = cvi.probe()

    assert availability.ok is False
    assert availability.kind == "missing-capture"


def test_installed_local_providers_returns_subset_in_declared_order(monkeypatch):
    """Only installed providers are returned, in LOCAL_PROVIDER_MODULES order.

    `faster_whisper` is deliberately excluded from `installed` so the result
    is a proper subset. The two that remain (`parakeet-mlx`,
    `lightning-whisper-mlx`) are alphabetically out of order relative to each
    other, so a stray `sorted()` in the implementation would also fail this
    test. Patches `find_spec` directly so a real, potentially
    machine-installed `parakeet_mlx` cannot leak into the result. Both
    remaining providers are Apple-Silicon only, so `sys.platform` is pinned
    to darwin rather than trusting whatever this machine happens to be.
    """
    installed = {"parakeet_mlx", "lightning_whisper_mlx"}

    def fake_find_spec(name, *args, **kwargs):
        return object() if name in installed else None

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(cvi.sys, "platform", "darwin")

    assert cvi.installed_local_providers() == ("parakeet-mlx", "lightning-whisper-mlx")


def test_installed_local_providers_includes_parakeet_onnx_when_present(monkeypatch):
    """parakeet-onnx is the provider the shipping one-shot dictation used --
    the worst omission this task fixes. Cross-platform: no darwin gate.
    """
    def fake_find_spec(name, *args, **kwargs):
        return object() if name == "onnx_asr" else None

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)

    assert cvi.installed_local_providers() == ("parakeet-onnx",)


def test_installed_local_providers_excludes_parakeet_onnx_when_absent(monkeypatch):
    monkeypatch.setattr(cvi.importlib.util, "find_spec", lambda name, *a, **k: None)

    assert "parakeet-onnx" not in cvi.installed_local_providers()


@pytest.mark.parametrize(
    ("provider", "module_name"),
    [
        ("faster-whisper", "faster_whisper"),
        ("parakeet", "nemo"),
        ("canary", "nemo"),
    ],
)
def test_installed_local_providers_detects_cross_platform_providers(
    monkeypatch, provider, module_name
):
    """Providers with no darwin gate are detected purely by find_spec."""
    def fake_find_spec(name, *args, **kwargs):
        return object() if name == module_name else None

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)

    assert provider in cvi.installed_local_providers()


def test_parakeet_and_canary_both_come_from_nemo(monkeypatch):
    """A single NeMo install makes both NeMo-backed providers available."""
    def fake_find_spec(name, *args, **kwargs):
        return object() if name == "nemo" else None

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)

    installed = cvi.installed_local_providers()
    assert "parakeet" in installed
    assert "canary" in installed


def test_qwen2audio_requires_both_torch_and_transformers(monkeypatch):
    def fake_find_spec(name, *args, **kwargs):
        return object() if name in {"torch", "transformers"} else None

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)

    assert "qwen2audio" in cvi.installed_local_providers()


@pytest.mark.parametrize("present_module", ["torch", "transformers"])
def test_qwen2audio_unavailable_with_only_one_of_two_modules(monkeypatch, present_module):
    """Present with only one of the two required modules -> not available."""
    def fake_find_spec(name, *args, **kwargs):
        return object() if name == present_module else None

    monkeypatch.setattr(cvi.importlib.util, "find_spec", fake_find_spec)

    assert "qwen2audio" not in cvi.installed_local_providers()


@pytest.mark.parametrize(
    ("provider", "module_name"),
    [("parakeet-mlx", "parakeet_mlx"), ("lightning-whisper-mlx", "lightning_whisper_mlx")],
)
def test_mlx_providers_require_darwin_even_if_module_resolves(
    monkeypatch, provider, module_name
):
    """A force-installed MLX package on a non-darwin platform must not be
    reported as usable -- it would light up the button and then fail at
    capture time.
    """
    monkeypatch.setattr(
        cvi.importlib.util, "find_spec", lambda name, *a, **k: object() if name == module_name else None
    )
    monkeypatch.setattr(cvi.sys, "platform", "linux")

    assert provider not in cvi.installed_local_providers()


@pytest.mark.parametrize(
    ("provider", "module_name"),
    [("parakeet-mlx", "parakeet_mlx"), ("lightning-whisper-mlx", "lightning_whisper_mlx")],
)
def test_mlx_providers_available_on_darwin_when_module_resolves(
    monkeypatch, provider, module_name
):
    monkeypatch.setattr(
        cvi.importlib.util, "find_spec", lambda name, *a, **k: object() if name == module_name else None
    )
    monkeypatch.setattr(cvi.sys, "platform", "darwin")

    assert provider in cvi.installed_local_providers()


def test_remote_whisper_is_never_returned(monkeypatch):
    """Always-available in the service, but privacy-mode-incompatible here:
    including it would let resolve() hand it back and have the service
    silently swap it out later -- the exact bug this module exists to
    prevent.
    """
    monkeypatch.setattr(cvi.importlib.util, "find_spec", lambda name, *a, **k: object())
    monkeypatch.setattr(cvi.sys, "platform", "darwin")

    assert "remote-whisper" not in cvi.installed_local_providers()


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

    `installed_local_providers()`'s order now leads with `parakeet-onnx`
    (added for the provider-coverage fix); this pins that `resolve()`
    consumes that order as a preference rather than sorting it. A
    single-element `installed` tuple cannot detect `sorted(installed)[0]`,
    and the elements here are not alphabetical, so a stray `sorted()` would
    also be caught.
    """
    monkeypatch.setattr(
        cvi,
        "installed_local_providers",
        lambda: (
            "parakeet-onnx",
            "parakeet-mlx",
            "faster-whisper",
            "lightning-whisper-mlx",
        ),
    )
    _stub_settings(monkeypatch, {"transcription.default_provider": "qwen2audio"})

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider == "parakeet-onnx"
    assert effective.was_overridden is True


def test_resolve_honours_a_configured_parakeet_onnx(monkeypatch):
    """The regression this task fixes: `parakeet-onnx` is the provider the
    shipping one-shot dictation used, so a user with it configured and
    installed must get it back verbatim, not overridden to whatever used to
    be `installed[0]`.
    """
    monkeypatch.setattr(
        cvi,
        "installed_local_providers",
        lambda: ("parakeet-onnx", "faster-whisper"),
    )
    _stub_settings(monkeypatch, {"transcription.default_provider": "parakeet-onnx"})

    effective = cvi.resolve()

    assert effective is not None
    assert effective.provider == "parakeet-onnx"
    assert effective.was_overridden is False


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


def test_release_stops_the_real_processing_thread(monkeypatch):
    """`_release()` must stop the `DictationProcessor` daemon thread, not
    just the microphone.

    `LazyLiveDictationService._processing_loop` only ever gets told to exit
    via `stop_processing.set()`, which today only happens inside
    `stop_dictation()`'s blocking 2s-join path -- exactly what `_release()`
    exists to skip. Without setting the event here, the thread (and the
    service instance it holds a bound-method reference to) survives every
    abandoned or mid-session-failed capture forever. Drives a *real*
    `_processing_loop` thread (no fakes for the loop itself) so a fix that
    merely no-ops instead of actually stopping it would be caught.
    """
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    service = LazyLiveDictationService.__new__(LazyLiveDictationService)
    service._audio_service = FakeAudioService()
    service.state_lock = threading.Lock()
    service.stop_processing = threading.Event()
    service.processing_queue = queue.Queue()
    service.buffer_lock = threading.Lock()
    service.audio_buffer = []
    service.buffer_duration_ms = 500
    service.last_speech_time = 0
    service.privacy_settings = {"auto_clear_buffer": True, "save_history": False}
    service.processing_thread = threading.Thread(
        target=service._processing_loop, daemon=True, name="DictationProcessor"
    )
    service.processing_thread.start()
    assert service.processing_thread.is_alive()

    controller, _events, _ = _controller(monkeypatch)
    controller._release(service)  # must not join, must not raise

    service.processing_thread.join(timeout=2.0)
    assert not service.processing_thread.is_alive()


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


# -- Fix round 5: the dependency's stop_dictation() never released capture, --
# -- and the controller trusted it anyway ------------------------------------


def test_stop_releases_capture_even_if_the_service_forgets(monkeypatch):
    """The Console does not trust the dependency to release the microphone.

    LazyLiveDictationService.stop_dictation() historically returned without
    stopping capture, so the controller verifies it independently.
    """
    released = []

    class ForgetfulService:
        def __init__(self, **kwargs):
            self._audio_service = type(
                "R", (), {"stop_recording": lambda s: released.append("stopped")}
            )()

        def start_dictation(self, **callbacks):
            return True

        def stop_dictation(self):
            return None  # deliberately does NOT release capture

    controller, events, _ = _controller(monkeypatch, service=ForgetfulService())
    controller.start()
    controller.stop()

    assert released == ["stopped"]
    assert controller.state == cvi.STATE_IDLE


def test_mid_session_error_releases_capture(monkeypatch):
    """An error while listening must not leave a live recorder behind idle.

    Without this, a retry claims a second service and orphans the first --
    two simultaneously live recorders.
    """
    released = []

    class ErroringService:
        def __init__(self, **kwargs):
            self._audio_service = type(
                "R", (), {"stop_recording": lambda s: released.append("stopped")}
            )()
            self._on_error = None

        def start_dictation(self, **callbacks):
            self._on_error = callbacks["on_error"]
            return True

        def stop_dictation(self):
            return None

    service = ErroringService()
    controller, events, _ = _controller(monkeypatch, service=service)
    controller.start()
    released.clear()

    service._on_error(RuntimeError("transcription model missing"))

    assert released == ["stopped"]
    assert controller.state == cvi.STATE_IDLE


# --------------------------------------------------------------------------
# Model preparation happens in `preparing`, before the microphone opens
#
# The defect this locks in against, measured on a live capture: the model load
# happened lazily on the *first audio chunk*, so a fresh machine spent 155s
# downloading 1.4 GB while the user was already speaking, and then lost the
# whole capture to the stop-side thread join. The microphone was flawless; the
# message blamed it anyway.
# --------------------------------------------------------------------------


class _WarmableService:
    """A dictation service exposing a `transcription_service`, like the real one.

    The property lives on the *class* (as it does on
    `LazyLiveDictationService`), because that is what `warmup_target()` checks
    before touching anything.
    """

    def __init__(self, transcriber=None, gate=None, build_error=None, **kwargs):
        self.kwargs = kwargs
        self.calls: list[str] = []
        self.started = False
        self.start_result = True
        self._transcriber = transcriber if transcriber is not None else _Transcriber()
        self._gate = gate
        self._build_error = build_error
        self._audio_service = type(
            "R", (), {"stop_recording": lambda s: None}
        )()

    @property
    def transcription_service(self):
        self.calls.append("build-transcriber")
        if self._build_error is not None:
            raise self._build_error
        if self._gate is not None:
            self._gate.wait(timeout=5)
        return self._transcriber

    def start_dictation(self, **callbacks):
        self.calls.append("start_dictation")
        self.started = True
        return self.start_result

    def stop_dictation(self):
        return None


class _Transcriber:
    """Records the warm-up transcription the controller performs."""

    def __init__(self, error=None, gate=None, entered=None):
        self.buffer_calls: list[dict] = []
        self._error = error
        self._gate = gate
        self.entered = entered

    def transcribe_buffer(
        self,
        audio_data,
        sample_rate,
        channels=1,
        sample_width=2,
        provider=None,
        model=None,
        language=None,
        **kwargs,
    ):
        self.buffer_calls.append(
            {
                "audio_data": audio_data,
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": sample_width,
                "provider": provider,
                "model": model,
                "language": language,
            }
        )
        if self.entered is not None:
            self.entered.set()
        if self._gate is not None:
            self._gate.wait(timeout=5)
        if self._error is not None:
            raise self._error
        return {"text": ""}


@pytest.fixture(autouse=True)
def _forget_warmed_models():
    """Each test starts as if nothing had ever been warmed in this process."""
    cvi.reset_model_warmup_state()
    yield
    cvi.reset_model_warmup_state()


def test_the_model_is_warmed_before_the_microphone_opens(monkeypatch):
    """Order is the whole fix: warm, *then* start capturing."""
    service = _WarmableService()
    controller, _events, _ = _controller(monkeypatch, service=service)

    controller.start()

    assert service.calls == ["build-transcriber", "start_dictation"]
    assert len(service._transcriber.buffer_calls) == 1
    assert controller.state == cvi.STATE_LISTENING


def test_the_warm_up_uses_the_resolved_provider_model_and_language(monkeypatch):
    """A warm-up against a different model warms the wrong cache entry."""
    service = _WarmableService()
    monkeypatch.setattr(
        cvi,
        "resolve",
        lambda: cvi.EffectiveConfig(
            provider="faster-whisper",
            model="distil-large-v3",
            language="fr",
            configured_provider="faster-whisper",
            was_overridden=False,
        ),
    )
    controller, _events, _ = _controller(monkeypatch, service=service)

    controller.start()

    call = service._transcriber.buffer_calls[0]
    assert call["provider"] == "faster-whisper"
    assert call["model"] == "distil-large-v3"
    assert call["language"] == "fr"
    # Silence, at the capture's own PCM format -- never real microphone audio.
    assert set(call["audio_data"]) == {0}
    assert call["sample_rate"] == cvi.WARMUP_SAMPLE_RATE
    assert call["sample_width"] == cvi.WARMUP_SAMPLE_WIDTH


def test_the_warm_up_targets_the_service_own_transcriber(monkeypatch):
    """`_model_cache` is per instance; warming a throwaway helps nothing.

    The object warmed must be the identical one `_process_audio_buffer()` will
    reach for, or every press still pays a full model load on its first chunk.
    """
    transcriber = _Transcriber()
    service = _WarmableService(transcriber=transcriber)
    controller, _events, _ = _controller(monkeypatch, service=service)

    controller.start()

    assert service.transcription_service is transcriber
    assert transcriber.buffer_calls, "the service's own transcriber was never warmed"


def test_a_slow_warm_announces_itself_before_it_blocks(monkeypatch):
    """Minutes of silence with no explanation is indistinguishable from a hang."""
    gate = threading.Event()
    entered = threading.Event()
    transcriber = _Transcriber(gate=gate, entered=entered)
    service = _WarmableService(transcriber=transcriber)
    controller, events, _ = _controller(
        monkeypatch, service=service, spawn=lambda thunk: threading.Thread(
            target=thunk, daemon=True
        ).start()
    )

    controller.start()
    assert entered.wait(timeout=5), "the warm-up never ran"

    # The notice is already out while the warm-up is still blocked.
    preparing = [e for e in events if isinstance(e, cvi.VoiceModelPreparing)]
    assert len(preparing) == 1
    assert preparing[0].first_run is True
    assert "speech model" in preparing[0].message.lower()
    assert service.started is False, "capture started before the model was ready"

    gate.set()


def test_the_first_run_message_differs_from_later_presses(monkeypatch):
    """A first-run download warrants different copy from a 1s disk load."""
    service = _WarmableService()
    controller, events, _ = _controller(monkeypatch, service=service)

    controller.start()
    controller.stop()
    first = [e for e in events if isinstance(e, cvi.VoiceModelPreparing)][0]

    events.clear()
    controller.start()
    second = [e for e in events if isinstance(e, cvi.VoiceModelPreparing)][0]

    assert first.first_run is True
    assert second.first_run is False
    assert first.message != second.message
    assert first.message == cvi.WARMUP_MESSAGE_FIRST_RUN
    assert second.message == cvi.WARMUP_MESSAGE
    # The duration warning is the whole reason a separate first-run string
    # exists, and it does not fit in the chip -- so it rides in `detail`.
    assert "minutes" in first.detail
    assert second.detail == ""


@pytest.mark.parametrize(
    "message", [cvi.WARMUP_MESSAGE_FIRST_RUN, cvi.WARMUP_MESSAGE]
)
def test_every_preparing_message_fits_the_one_row_chip(message):
    """The chip is 42 cells and one row; longer copy is cut mid-sentence.

    An earlier draft ended on "…(first run may" and lost the duration warning
    entirely. The "◌ " prefix the screen adds counts against the budget.
    """
    assert len(f"◌ {message}") <= cvi.WARMUP_MESSAGE_MAX_CELLS


def test_the_first_run_detail_says_what_the_chip_cannot(monkeypatch):
    """It is only worth splitting if the long half carries the real warning."""
    assert "minutes" in cvi.WARMUP_DETAIL_FIRST_RUN
    assert "recorded" in cvi.WARMUP_DETAIL_FIRST_RUN
    # And it is genuinely too long for the chip -- otherwise merge them.
    assert len(cvi.WARMUP_DETAIL_FIRST_RUN) > cvi.WARMUP_MESSAGE_MAX_CELLS


def test_a_failed_warm_up_degrades_instead_of_disabling_dictation(monkeypatch):
    """A warm-up that fails after the service was built must not be fatal.

    The Console warms on *every* press, so a fatal warm-up would turn one
    transient error -- a provider that dislikes digital silence, a blip with
    the weights already on disk -- into permanently unusable dictation. The
    capture goes ahead; that is safe now only because an empty result can no
    longer be misreported as a dead microphone.
    """
    transcriber = _Transcriber(error=RuntimeError("could not download model weights"))
    service = _WarmableService(transcriber=transcriber)
    controller, events, _ = _controller(monkeypatch, service=service)

    controller.start()

    assert [e for e in events if isinstance(e, cvi.VoiceFailed)] == []
    warnings = [e for e in events if isinstance(e, cvi.VoiceModelWarmupFailed)]
    assert len(warnings) == 1
    text = f"{warnings[0].reason} {warnings[0].remedy}".lower()
    assert "model" in text
    assert "could not download model weights" in text
    assert "microphone" not in text
    # The capture still opened, and the machine is live.
    assert service.started is True
    assert controller.state == cvi.STATE_LISTENING


def test_a_transcriber_that_cannot_be_built_is_a_fatal_model_problem(monkeypatch):
    """`LazyLiveDictationService.transcription_service` raises on missing models.

    This half stays fatal: models genuinely absent is strong evidence, and
    opening a microphone against a transcriber that cannot exist would record
    into a void -- the original defect.
    """
    service = _WarmableService(build_error=RuntimeError("models are not installed"))
    controller, events, _ = _controller(monkeypatch, service=service)

    controller.start()

    failures = [e for e in events if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
    text = f"{failures[0].reason} {failures[0].remedy}".lower()
    assert "model" in text
    assert "microphone" not in text
    assert service.started is False
    assert controller.state == cvi.STATE_IDLE


def test_a_fatal_warm_up_failure_still_ends_idle_with_failed_first(monkeypatch):
    """The ordering invariant the UI's deferred send depends on."""
    service = _WarmableService(build_error=RuntimeError("boom"))
    controller, events, _ = _controller(monkeypatch, service=service)

    controller.start()

    kinds = [
        type(e).__name__
        for e in events
        if isinstance(e, (cvi.VoiceFailed, cvi.VoiceStateChanged))
    ]
    assert kinds[-2:] == ["VoiceFailed", "VoiceStateChanged"]
    assert events[-1].state == cvi.STATE_IDLE
    assert controller.state == cvi.STATE_IDLE


def test_warming_can_be_turned_off_entirely(monkeypatch):
    """`dictation.warm_model_before_capture = false` is the escape hatch."""
    service = _WarmableService()
    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(
        monkeypatch,
        {
            "transcription.default_provider": "faster-whisper",
            "dictation.warm_model_before_capture": False,
        },
    )
    events = []
    controller = cvi.ConsoleVoiceInputController(
        emit=events.append,
        spawn=lambda thunk: thunk(),
        service_factory=lambda **kwargs: service,
    )

    controller.start()

    assert service._transcriber.buffer_calls == []
    assert service.calls == ["start_dictation"]
    assert [e for e in events if isinstance(e, cvi.VoiceModelPreparing)] == []
    assert controller.state == cvi.STATE_LISTENING


@pytest.mark.parametrize("value", [True, "true", "yes", 1, None])
def test_warming_stays_on_for_anything_but_an_explicit_off(monkeypatch, value):
    """Default and truthy values must all warm; only an explicit off disables."""
    service = _WarmableService()
    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    settings = {"transcription.default_provider": "faster-whisper"}
    if value is not None:
        settings["dictation.warm_model_before_capture"] = value
    _stub_settings(monkeypatch, settings)
    controller = cvi.ConsoleVoiceInputController(
        emit=lambda _e: None,
        spawn=lambda thunk: thunk(),
        service_factory=lambda **kwargs: service,
    )

    controller.start()

    assert service._transcriber.buffer_calls, "warm-up was skipped"


def test_the_warm_up_never_blocks_the_calling_thread_to_the_end(monkeypatch):
    """The load runs on a daemon thread so quitting mid-download can exit.

    `_run_begin` runs on the default asyncio executor, and `asyncio.run()`
    joins that executor at shutdown -- a 155s download directly on it means a
    dead terminal until it finishes. Abandoning must release this frame while
    the load is still running.
    """
    gate = threading.Event()
    entered = threading.Event()
    transcriber = _Transcriber(gate=gate, entered=entered)
    service = _WarmableService(transcriber=transcriber)
    controller, _events, _ = _controller(
        monkeypatch,
        service=service,
        spawn=lambda thunk: threading.Thread(target=thunk, daemon=True).start(),
    )
    warm_threads_before = {t.name for t in threading.enumerate()}

    controller.start()
    assert entered.wait(timeout=5)

    # The load is on a *daemon* thread, so the interpreter can exit past it.
    warm_threads = [
        t
        for t in threading.enumerate()
        if t.name not in warm_threads_before and "Warmup" in t.name
    ]
    assert warm_threads, "the warm-up did not get its own thread"
    assert all(t.daemon for t in warm_threads)

    controller.abandon()
    # Abandon returns while the load is still blocked -- that is the point.
    assert not gate.is_set()
    gate.set()


def test_a_failed_warm_up_is_still_a_first_run_next_time(monkeypatch):
    """Only a model that actually loaded may downgrade the first-run message."""
    transcriber = _Transcriber(error=RuntimeError("boom"))
    service = _WarmableService(transcriber=transcriber)
    controller, events, _ = _controller(monkeypatch, service=service)

    controller.start()
    controller.stop()
    events.clear()
    controller.start()

    preparing = [e for e in events if isinstance(e, cvi.VoiceModelPreparing)]
    assert preparing and preparing[0].first_run is True


def test_a_service_without_a_transcriber_is_warmed_silently(monkeypatch):
    """Services that expose no transcriber (fakes) must still start normally."""
    controller, events, service = _controller(monkeypatch)

    controller.start()

    assert service.started is True
    assert [e for e in events if isinstance(e, cvi.VoiceModelPreparing)] == []
    assert controller.state == cvi.STATE_LISTENING


def test_abandon_during_a_long_warm_up_never_opens_the_microphone(monkeypatch):
    """A minutes-long first run gives unmount plenty of room to land."""
    gate = threading.Event()
    entered = threading.Event()
    transcriber = _Transcriber(gate=gate, entered=entered)
    service = _WarmableService(transcriber=transcriber)
    controller, _events, _ = _controller(
        monkeypatch,
        service=service,
        spawn=lambda thunk: threading.Thread(target=thunk, daemon=True).start(),
    )

    controller.start()
    assert entered.wait(timeout=5)
    controller.abandon()
    gate.set()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and controller.state != cvi.STATE_IDLE:
        time.sleep(0.01)
    time.sleep(0.1)

    assert service.started is False
    assert controller.state == cvi.STATE_IDLE


def test_a_raising_preparing_emit_does_not_abort_the_start(monkeypatch):
    """Progress copy is cosmetic; a plumbing error in it must not cost the capture."""
    service = _WarmableService()
    monkeypatch.setattr(cvi, "capture_available", lambda: True)
    monkeypatch.setattr(cvi, "installed_local_providers", lambda: ("faster-whisper",))
    _stub_settings(monkeypatch, {"transcription.default_provider": "faster-whisper"})

    def emit(event):
        if isinstance(event, cvi.VoiceModelPreparing):
            raise RuntimeError("chip exploded")

    controller = cvi.ConsoleVoiceInputController(
        emit=emit,
        spawn=lambda thunk: thunk(),
        service_factory=lambda **kwargs: service,
    )

    controller.start()

    assert service.started is True
    assert controller.state == cvi.STATE_LISTENING


# --------------------------------------------------------------------------
# The stop-side outcome the caller needs to tell three failures apart
# --------------------------------------------------------------------------


class _ReportingService(_WarmableService):
    """Returns a `DictationResult`-shaped object from `stop_dictation()`."""

    def __init__(self, result=None, **kwargs):
        super().__init__(**kwargs)
        self._result = result

    def stop_dictation(self):
        return self._result


def test_capture_outcome_defaults_to_unknown_when_nothing_is_reported(monkeypatch):
    controller, _events, _ = _controller(monkeypatch, service=_WarmableService())

    controller.start()
    controller.stop()

    outcome = controller.last_capture_outcome
    assert outcome.captured_bytes is None
    assert outcome.transcription_complete is True


def test_capture_outcome_carries_the_services_byte_count_and_completion(monkeypatch):
    from types import SimpleNamespace

    service = _ReportingService(
        result=SimpleNamespace(captured_bytes=51840, transcription_complete=False)
    )
    controller, _events, _ = _controller(monkeypatch, service=service)

    controller.start()
    controller.stop()

    outcome = controller.last_capture_outcome
    assert outcome.captured_bytes == 51840
    assert outcome.transcription_complete is False


def test_capture_outcome_is_reset_by_the_next_start(monkeypatch):
    """A stale "did not finish" must not condemn the following capture."""
    from types import SimpleNamespace

    service = _ReportingService(
        result=SimpleNamespace(captured_bytes=10, transcription_complete=False)
    )
    controller, _events, _ = _controller(monkeypatch, service=service)
    controller.start()
    controller.stop()
    assert controller.last_capture_outcome.transcription_complete is False

    controller.start()

    assert controller.last_capture_outcome.transcription_complete is True
    assert controller.last_capture_outcome.captured_bytes is None


def test_a_raising_stop_dictation_still_releases_the_microphone(monkeypatch):
    """The claimed service must not be orphaned when `stop_dictation()` throws.

    `_run_finish()` claims the service before stopping it, so nothing else will
    ever release it. `Thread.join(timeout=nan)` -> ValueError is one real way
    to land here, and it used to leave a live recorder behind an idle machine.
    """
    released = []

    class _ThrowingStopService(_WarmableService):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._audio_service = type(
                "R", (), {"stop_recording": lambda s: released.append("stopped")}
            )()

        def stop_dictation(self):
            raise ValueError("timeout must be a non-negative number")

    controller, events, service = _controller(
        monkeypatch, service=_ThrowingStopService()
    )
    controller.start()
    released.clear()
    events.clear()

    controller.stop()

    assert released == ["stopped"]
    assert controller.state == cvi.STATE_IDLE
    failures = [e for e in events if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1
