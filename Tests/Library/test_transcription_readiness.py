"""Download-free selected-backend checks before Library queue creation."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Library.transcription_readiness import (
    TranscriptionReadinessError,
    check_transcription_readiness,
)


def _runtime(**attributes):
    return SimpleNamespace(**attributes)


@pytest.mark.parametrize(
    "provider,module,extra",
    [
        ("default", "faster_whisper", "transcription_faster_whisper"),
        ("faster-whisper", "faster_whisper", "transcription_faster_whisper"),
        ("parakeet-onnx", "onnx_asr", "transcription_parakeet_onnx"),
        ("transcribe-cpp", "transcribe_cpp", "transcription_transcribe_cpp"),
    ],
)
def test_missing_selected_package_names_its_install_remedy(provider, module, extra):
    def missing(name):
        assert name == module
        raise ModuleNotFoundError(name)

    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": provider}, import_module=missing
        )
    assert caught.value.reason == "package_missing"
    assert extra in str(caught.value)
    assert "Python environment" in str(caught.value)


def test_whisper_checks_exact_selected_cache_without_downloading(tmp_path):
    calls = []
    model = tmp_path / "cached"
    model.mkdir()
    (model / "model.bin").write_bytes(b"weights")
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")

    def cached(name, *, local_files_only):
        calls.append((name, local_files_only))
        return str(model)

    def importer(name):
        assert name in {"faster_whisper", "faster_whisper.utils"}
        return _runtime(download_model=cached)

    check_transcription_readiness(
        {"transcription_provider": "faster-whisper", "transcription_model": "small.en"},
        import_module=importer,
    )
    assert calls == [("small.en", True)]


@pytest.mark.parametrize(
    "absent", ["cache", "model.bin", "config.json", "tokenizer.json"]
)
def test_whisper_missing_or_incomplete_cache_has_model_remedy(tmp_path, absent):
    for filename in ("model.bin", "config.json", "tokenizer.json"):
        if filename != absent:
            (tmp_path / filename).write_text("test")

    def cached(*args, **kwargs):
        if absent == "cache":
            raise FileNotFoundError("/private/cache/token")
        return str(tmp_path)

    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": "faster-whisper", "transcription_model": "base"},
            import_module=lambda name: _runtime(download_model=cached),
        )
    assert caught.value.reason == "model_missing"
    assert "base" in str(caught.value)
    assert "download" in str(caught.value).lower()
    assert str(tmp_path) not in str(caught.value)
    assert "/private" not in str(caught.value)


def test_parakeet_uses_exact_language_precision_and_source_service():
    calls = []
    service = _runtime(resolve=lambda key, **kwargs: calls.append((key, kwargs)))
    check_transcription_readiness(
        {
            "transcription_provider": "parakeet-onnx",
            "language": "de",
            "transcription_precision": "f32",
        },
        import_module=lambda name: _runtime(),
        parakeet_source_service=service,
    )
    assert calls[0][0].value == "v3_f32"


def test_external_parakeet_preserves_existing_folder_and_vad_consent_flow():
    def unexpected(*args, **kwargs):
        raise AssertionError(
            "external verification belongs to its existing consent flow"
        )

    check_transcription_readiness(
        {
            "transcription_provider": "parakeet-onnx",
            "transcription_model_dir": "/private/model",
        },
        import_module=lambda name: _runtime(),
        parakeet_source_service=_runtime(resolve=unexpected),
    )


def test_missing_parakeet_bundle_is_not_called_incompatible():
    def missing(*args, **kwargs):
        raise FileNotFoundError("/private/not-installed")

    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": "parakeet-onnx"},
            import_module=lambda name: _runtime(),
            parakeet_source_service=_runtime(resolve=missing),
        )
    assert caught.value.reason == "model_missing"
    assert "Install" in str(caught.value)
    assert "Parakeet" in str(caught.value)
    assert "/private" not in str(caught.value)


def test_missing_managed_vad_names_vad_instead_of_the_asr_model():
    from tldw_chatbook.STT.parakeet_sources import (
        ParakeetSourceError,
        ParakeetSourceErrorCode,
    )

    def missing(*args, **kwargs):
        raise ParakeetSourceError(ParakeetSourceErrorCode.VAD_UNAVAILABLE)

    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": "parakeet-onnx"},
            import_module=lambda name: _runtime(),
            parakeet_source_service=_runtime(resolve=missing),
        )
    assert caught.value.reason == "vad_missing"
    assert "Silero VAD" in str(caught.value)
    assert "Install" in str(caught.value)


def test_transcribe_cpp_requires_a_selected_gguf_before_queueing():
    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": "transcribe-cpp"},
            import_module=lambda name: _runtime(),
        )
    assert caught.value.reason == "model_missing"
    assert "GGUF" in str(caught.value)


def test_dispatch_classifies_absent_parakeet_files_as_missing_model():
    from tldw_chatbook.app_ingest_queue import LibraryIngestQueueMixin
    from tldw_chatbook.STT.contracts import TranscriptionFailureCode

    code, _ = LibraryIngestQueueMixin._classify_local_stt_dispatch_error(
        "parakeet-onnx", FileNotFoundError("/private/missing")
    )
    assert code is TranscriptionFailureCode.MODEL_NOT_INSTALLED


@pytest.mark.parametrize("provider", ["parakeet-onnx", "transcribe-cpp"])
def test_closed_translate_checkbox_does_not_change_selected_route(provider):
    # The form retains checkbox values when their provider gate closes.
    with pytest.raises(TranscriptionReadinessError) as caught:
        check_transcription_readiness(
            {"transcription_provider": provider, "translate_to_english": True},
            import_module=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
        )
    assert caught.value.reason == "package_missing"


def test_explicit_translation_target_wins_over_checkbox():
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import BatchSTTRoutingError

    with pytest.raises(BatchSTTRoutingError, match="only supports target en"):
        check_transcription_readiness(
            {
                "transcription_provider": "faster-whisper",
                "translate_to_english": True,
                "translation_target_language": "de",
            },
            import_module=lambda name: _runtime(),
        )
