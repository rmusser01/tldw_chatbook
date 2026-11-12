"""Deferred selections stop at real admission boundaries before effects."""

import builtins

import pytest


@pytest.mark.parametrize(
    "provider", ["lightning-whisper-mlx", "lightning-whisper", "auto"]
)
def test_lazy_start_refuses_before_recorder_or_transcription_construction(
    monkeypatch, provider
):
    from tldw_chatbook.Audio import dictation_service_lazy as lazy

    monkeypatch.setattr(
        lazy,
        "get_cli_setting",
        lambda key, default=None: (
            "lightning-whisper" if key == "transcription.default_provider" else default
        ),
    )
    service = lazy.LazyLiveDictationService(
        transcription_provider=provider,
        recorder_factory=lambda **kwargs: pytest.fail("recorder constructed"),
        transcription_service_factory=lambda: pytest.fail("transcription constructed"),
    )
    errors = []
    states = []
    try:
        assert (
            service.start_dictation(
                on_error=errors.append, on_state_change=states.append
            )
            is False
        )
        assert len(errors) == 1
        assert "Your saved settings have not been changed." in str(errors[0])
        assert states[-1] is lazy.DictationState.ERROR
        assert service.transcription_provider == provider
        assert service._audio_service is None
        assert service._transcription_service is None
        assert service.processing_thread is None
    finally:
        service.abandon()


@pytest.mark.parametrize("provider", ["lightning-whisper-mlx", "lightning-whisper"])
def test_voice_process_refuses_before_binding_any_worker_factory(monkeypatch, provider):
    from tldw_chatbook.Audio import parakeet_voice_worker as worker

    monkeypatch.setattr(
        worker, "partial", lambda *args, **kwargs: pytest.fail("worker factory bound")
    )
    with pytest.raises(ValueError, match="Lightning Whisper MLX is deferred"):
        worker.ParakeetVoiceProcess(model=None, language="en", provider=provider)


@pytest.mark.parametrize("provider", ["lightning-whisper-mlx", "lightning-whisper"])
def test_child_runtime_refuses_before_importing_transcription_bootstrap(
    monkeypatch, provider
):
    from tldw_chatbook.Audio import parakeet_voice_worker as worker

    original_import = builtins.__import__

    def reject_transcription_import(name, *args, **kwargs):
        if name == "tldw_chatbook.Local_Ingestion.transcription_service":
            pytest.fail("transcription bootstrap imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_transcription_import)
    with pytest.raises(ValueError, match="Lightning Whisper MLX is deferred"):
        worker._load_runtime(None, "en", provider=provider)


@pytest.mark.parametrize("provider", ["lightning-whisper-mlx", "lightning-whisper"])
def test_batch_and_library_refuse_deferred_identity_before_other_option_validation(
    provider,
):
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import resolve_batch_stt_route
    from tldw_chatbook.Utils.input_validation import (
        validate_batch_transcription_provider,
    )

    with pytest.raises(ValueError, match="Your saved settings have not been changed"):
        resolve_batch_stt_route(provider=provider, language=object())
    with pytest.raises(ValueError, match="Your saved settings have not been changed"):
        validate_batch_transcription_provider(provider)
