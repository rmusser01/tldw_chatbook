"""Privacy-mode provider allowlist tests.

Two layers once kept two lists of "which providers are local", and they drifted
twice with the same symptom -- the user's provider silently rewritten:

* the allowlist spelled a provider `"lightning-whisper"` while the rest of the
  app dispatched on `"lightning-whisper-mlx"`;
* the allowlist held three providers while the Console's resolver had grown to
  seven, so the Console would resolve, warm and *announce* `parakeet-onnx` and
  `start_dictation()` would then rewrite it to `parakeet-mlx` -- a first press
  downloading model A for minutes and transcribing with model B, and on Linux
  with only `onnx_asr` installed, failing every chunk.

The fix is one catalogue (`Utils/local_stt_providers`) consumed by both, so
these tests now guard that there is still only one.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_the_privacy_check_has_no_provider_list_of_its_own():
    """A literal list here is the bug; the shared catalogue is the fix.

    Structural, not cosmetic: every drift incident above began with somebody
    maintaining a second copy of this fact inside this method.
    """
    import inspect

    from tldw_chatbook.Audio import dictation_service_lazy

    source = inspect.getsource(
        dictation_service_lazy.LazyLiveDictationService._initialize_streaming_transcriber
    )

    assert "allowed_providers" not in source
    assert "provider_is_local(" in source
    for provider_id in ("parakeet-mlx", "faster-whisper", "lightning-whisper-mlx"):
        assert f'"{provider_id}"' not in source


def test_the_resolver_and_the_privacy_check_read_the_same_catalogue():
    """Same object, so they cannot drift apart again."""
    from tldw_chatbook.Audio import dictation_service_lazy
    from tldw_chatbook.Chat import console_voice_input
    from tldw_chatbook.Utils import local_stt_providers

    assert (
        dictation_service_lazy.LOCAL_STT_PROVIDERS
        is local_stt_providers.LOCAL_STT_PROVIDERS
    )
    assert (
        console_voice_input.LOCAL_PROVIDER_MODULES
        is local_stt_providers.LOCAL_PROVIDER_MODULES
    )
    assert set(local_stt_providers.LOCAL_STT_PROVIDERS) == set(
        local_stt_providers.LOCAL_PROVIDER_MODULES
    )


def test_the_catalogue_uses_real_provider_ids():
    """`lightning-whisper` matches nothing; `lightning-whisper-mlx` dispatches."""
    from tldw_chatbook.Utils.local_stt_providers import LOCAL_STT_PROVIDERS

    assert "lightning-whisper-mlx" in LOCAL_STT_PROVIDERS
    assert "lightning-whisper" not in LOCAL_STT_PROVIDERS


def test_remote_whisper_is_never_local():
    """The one provider that sends audio off the machine stays out."""
    from tldw_chatbook.Utils.local_stt_providers import (
        LOCAL_STT_PROVIDERS,
        provider_is_local,
    )

    assert "remote-whisper" not in LOCAL_STT_PROVIDERS
    assert provider_is_local("remote-whisper") is False


def _privacy_service(provider: str):
    """A service with just enough state for `_initialize_streaming_transcriber`."""
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    service = LazyLiveDictationService.__new__(LazyLiveDictationService)
    service.privacy_settings = {"local_only": True}
    service.transcription_provider = provider
    service.transcription_model = None
    service.language = "en"
    service.streaming_transcriber = None
    return service


@pytest.fixture
def _no_streaming(monkeypatch):
    """Force the buffer-transcription path (no streaming transcriber)."""
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    class _NoStreaming:
        def create_streaming_transcriber(self, **kwargs):
            return None

    monkeypatch.setattr(
        LazyLiveDictationService,
        "transcription_service",
        property(lambda self: _NoStreaming()),
    )


@pytest.mark.parametrize(
    "provider",
    [
        "parakeet-onnx",
        "parakeet-mlx",
        "lightning-whisper-mlx",
        "faster-whisper",
        "qwen2audio",
        "parakeet",
        "canary",
    ],
)
def test_every_provider_the_resolver_can_choose_survives_privacy_mode(
    provider, _no_streaming
):
    """The defect, generalised: whatever the Console warmed must be what runs.

    All seven are local, so privacy mode has no business rewriting any of them.
    Before the shared catalogue, four of these were rewritten to `parakeet-mlx`
    *after* the Console had already downloaded and announced something else.
    """
    service = _privacy_service(provider)

    service._initialize_streaming_transcriber()

    assert service.transcription_provider == provider


def test_a_non_local_provider_is_still_replaced_under_privacy_mode(_no_streaming):
    """The allowlist still does its actual job: no audio leaves the machine."""
    from tldw_chatbook.Utils.local_stt_providers import provider_is_local

    service = _privacy_service("remote-whisper")

    service._initialize_streaming_transcriber()

    assert service.transcription_provider != "remote-whisper"
    assert provider_is_local(service.transcription_provider)


def test_the_privacy_fallback_prefers_an_installed_provider(monkeypatch, _no_streaming):
    """The old hard-coded `parakeet-mlx` fallback is Apple-only.

    On Linux it swapped a non-local provider for one that cannot run at all.
    """
    from tldw_chatbook.Audio import dictation_service_lazy

    monkeypatch.setattr(
        dictation_service_lazy,
        "installed_local_providers",
        lambda: ("faster-whisper",),
    )
    service = _privacy_service("remote-whisper")

    service._initialize_streaming_transcriber()

    assert service.transcription_provider == "faster-whisper"


def test_privacy_mode_off_leaves_even_a_non_local_provider_alone(_no_streaming):
    service = _privacy_service("remote-whisper")
    service.privacy_settings = {"local_only": False}

    service._initialize_streaming_transcriber()

    assert service.transcription_provider == "remote-whisper"
