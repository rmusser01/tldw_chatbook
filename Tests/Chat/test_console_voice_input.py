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
