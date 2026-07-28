"""Privacy-mode provider allowlist tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_allowlist_uses_real_provider_ids():
    """The allowlist must use the ids the rest of the app uses.

    `lightning-whisper` matches nothing: transcription_service dispatches on
    `lightning-whisper-mlx`, and that is what console_voice_input resolves to.
    A mismatch here silently rewrites the user's provider to parakeet-mlx.
    """
    import inspect

    from tldw_chatbook.Audio import dictation_service_lazy

    source = inspect.getsource(
        dictation_service_lazy.LazyLiveDictationService._initialize_streaming_transcriber
    )

    assert '"lightning-whisper-mlx"' in source
    assert '"lightning-whisper",' not in source


def test_lightning_whisper_mlx_survives_privacy_mode(monkeypatch):
    """A resolved lightning-whisper-mlx must not be rewritten."""
    from tldw_chatbook.Audio.dictation_service_lazy import LazyLiveDictationService

    service = LazyLiveDictationService.__new__(LazyLiveDictationService)
    service.privacy_settings = {"local_only": True}
    service.transcription_provider = "lightning-whisper-mlx"
    service.transcription_model = None
    service.language = "en"
    service.streaming_transcriber = None

    class _NoStreaming:
        def create_streaming_transcriber(self, **kwargs):
            return None

    monkeypatch.setattr(
        type(service),
        "transcription_service",
        property(lambda self: _NoStreaming()),
    )

    service._initialize_streaming_transcriber()

    assert service.transcription_provider == "lightning-whisper-mlx"
