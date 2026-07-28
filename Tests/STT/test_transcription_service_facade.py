"""Compatibility tests for the public transcription service facade."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

import tldw_chatbook.Local_Ingestion.transcription_service as service_module
from tldw_chatbook.Local_Ingestion.transcription_service import (
    TranscriptionService,
    _LegacyTranscriptionBackend,
)


def _signature_shape(callable_object: object) -> tuple[tuple[str, object, object], ...]:
    return tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(callable_object).parameters.values()
    )


def test_public_constructor_remains_zero_argument() -> None:
    assert _signature_shape(TranscriptionService) == ()


def test_public_method_signatures_match_the_legacy_contract() -> None:
    empty = inspect.Parameter.empty
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    variadic_keywords = inspect.Parameter.VAR_KEYWORD

    expected = {
        "cleanup": (("self", positional, empty),),
        "transcribe": (
            ("self", positional, empty),
            ("audio_path", positional, empty),
            ("provider", positional, None),
            ("model", positional, None),
            ("language", positional, None),
            ("source_lang", positional, None),
            ("target_lang", positional, None),
            ("vad_filter", positional, False),
            ("diarize", positional, False),
            ("progress_callback", positional, None),
            ("batch_route_resolved", positional, False),
            ("kwargs", variadic_keywords, empty),
        ),
        "transcribe_buffer": (
            ("self", positional, empty),
            ("audio_data", positional, empty),
            ("sample_rate", positional, empty),
            ("channels", positional, 1),
            ("sample_width", positional, 2),
            ("provider", positional, None),
            ("model", positional, None),
            ("language", positional, None),
            ("kwargs", variadic_keywords, empty),
        ),
        "get_available_providers": (("self", positional, empty),),
        "list_available_models": (
            ("self", positional, empty),
            ("provider", positional, None),
        ),
        "get_device_info": (("self", positional, empty),),
        "is_diarization_available": (("self", positional, empty),),
        "get_diarization_requirements": (("self", positional, empty),),
        "format_segments_with_timestamps": (
            ("self", positional, empty),
            ("segments", positional, empty),
            ("include_timestamps", positional, True),
            ("include_speakers", positional, True),
        ),
        "create_streaming_transcriber": (
            ("self", positional, empty),
            ("provider", positional, None),
            ("model", positional, None),
            ("source_lang", positional, None),
            ("kwargs", variadic_keywords, empty),
        ),
    }

    for method_name, signature in expected.items():
        assert _signature_shape(getattr(TranscriptionService, method_name)) == signature


def test_public_service_can_still_be_constructed_without_arguments() -> None:
    with (
        patch(
            "tldw_chatbook.Local_Ingestion.transcription_service.get_cli_setting",
            side_effect=lambda _key, default=None: default,
        ),
        patch.object(
            TranscriptionService,
            "get_available_providers",
            return_value=["remote-whisper"],
        ),
    ):
        service = TranscriptionService()

    assert service.config["default_language"] == "en"


class _Bridge:
    def __init__(self) -> None:
        self.config = {
            "default_provider": "faster-whisper",
            "default_language": "fr",
        }
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.result: object = {"text": "ok", "segments": []}
        self.error: Exception | None = None

    def _record(
        self,
        method: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        self.calls.append((method, args, kwargs))
        if self.error is not None:
            raise self.error
        return self.result

    def transcribe_legacy(self, *args: object, **kwargs: object) -> object:
        return self._record("transcribe", *args, **kwargs)

    def transcribe_buffer_legacy(self, *args: object, **kwargs: object) -> object:
        return self._record("transcribe_buffer", *args, **kwargs)

    def cleanup_legacy(self) -> object:
        return self._record("cleanup")

    def get_available_providers_legacy(self) -> object:
        return self._record("get_available_providers")

    def list_available_models_legacy(self, provider: str | None = None) -> object:
        return self._record("list_available_models", provider)

    def get_device_info_legacy(self) -> object:
        return self._record("get_device_info")

    def is_diarization_available_legacy(self) -> object:
        return self._record("is_diarization_available")

    def get_diarization_requirements_legacy(self) -> object:
        return self._record("get_diarization_requirements")

    def format_segments_with_timestamps_legacy(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        return self._record("format_segments_with_timestamps", *args, **kwargs)

    def create_streaming_transcriber_legacy(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        return self._record("create_streaming_transcriber", *args, **kwargs)


@pytest.fixture
def facade(monkeypatch: pytest.MonkeyPatch) -> tuple[TranscriptionService, _Bridge]:
    bridge = _Bridge()
    monkeypatch.setattr(
        service_module,
        "LegacyTranscriptionBridge",
        lambda _backend_factory: bridge,
        raising=False,
    )
    return service_module.TranscriptionService(), bridge


def test_facade_preserves_omitted_provider_and_language(
    facade: tuple[TranscriptionService, _Bridge],
) -> None:
    service, bridge = facade

    result = service.transcribe("audio.wav", model="base", custom_option=7)

    assert result is bridge.result
    assert bridge.calls == [
        (
            "transcribe",
            ("audio.wav", None, "base", None, None, None, False, False, None, False),
            {"custom_option": 7},
        )
    ]


def test_facade_does_not_activate_semantic_default_routing(
    facade: tuple[TranscriptionService, _Bridge],
) -> None:
    service, bridge = facade

    service.transcribe("audio.wav", provider="default", language="en")

    assert bridge.calls[0][1][1] == "default"
    assert bridge.calls[0][1][3] == "en"


def test_facade_preserves_buffer_arguments_and_provider_specific_kwargs(
    facade: tuple[TranscriptionService, _Bridge],
) -> None:
    service, bridge = facade

    result = service.transcribe_buffer(
        b"\x00\x00",
        16_000,
        2,
        1,
        "parakeet-onnx",
        "nemo-parakeet-tdt-0.6b-v2",
        "en",
        model_dir="/models/parakeet",
    )

    assert result is bridge.result
    assert bridge.calls == [
        (
            "transcribe_buffer",
            (
                b"\x00\x00",
                16_000,
                2,
                1,
                "parakeet-onnx",
                "nemo-parakeet-tdt-0.6b-v2",
                "en",
            ),
            {"model_dir": "/models/parakeet"},
        )
    ]


@pytest.mark.parametrize(
    ("method_name", "arguments", "keywords", "bridge_method"),
    [
        ("cleanup", (), {}, "cleanup"),
        ("get_available_providers", (), {}, "get_available_providers"),
        ("list_available_models", ("faster-whisper",), {}, "list_available_models"),
        ("get_device_info", (), {}, "get_device_info"),
        ("is_diarization_available", (), {}, "is_diarization_available"),
        ("get_diarization_requirements", (), {}, "get_diarization_requirements"),
        (
            "format_segments_with_timestamps",
            ([{"text": "hello"}], False, False),
            {},
            "format_segments_with_timestamps",
        ),
        (
            "create_streaming_transcriber",
            ("parakeet-mlx", "model-a", "en"),
            {"precision": "bf16"},
            "create_streaming_transcriber",
        ),
    ],
)
def test_facade_explicitly_forwards_each_public_helper(
    facade: tuple[TranscriptionService, _Bridge],
    method_name: str,
    arguments: tuple[object, ...],
    keywords: dict[str, object],
    bridge_method: str,
) -> None:
    service, bridge = facade

    result = getattr(service, method_name)(*arguments, **keywords)

    assert result is bridge.result
    assert bridge.calls[0] == (bridge_method, arguments, keywords)


def test_facade_preserves_backend_config_without_forwarding_private_attributes(
    facade: tuple[TranscriptionService, _Bridge],
) -> None:
    service, bridge = facade

    assert service.config is bridge.config
    assert not hasattr(service, "_model_cache")
    assert "__getattr__" not in type(service).__dict__


def test_facade_config_can_be_replaced_explicitly(
    facade: tuple[TranscriptionService, _Bridge],
) -> None:
    service, bridge = facade
    replacement = {"default_provider": "remote-whisper"}

    service.config = replacement

    assert service.config is replacement
    assert bridge.config is replacement


def test_facade_preserves_backend_exception_identity(
    facade: tuple[TranscriptionService, _Bridge],
) -> None:
    service, bridge = facade
    failure = RuntimeError("legacy public failure")
    bridge.error = failure

    with pytest.raises(RuntimeError) as caught:
        service.transcribe("audio.wav")

    assert caught.value is failure


def test_real_facade_keeps_configured_legacy_provider_and_language() -> None:
    settings = {
        "transcription.default_provider": "faster-whisper",
        "transcription.default_model": "base",
        "transcription.default_language": "en",
        "transcription.default_source_language": "fr",
    }
    legacy_result = {"text": "bonjour", "segments": [], "language": "fr"}

    with (
        patch(
            "tldw_chatbook.Local_Ingestion.transcription_service.get_cli_setting",
            side_effect=lambda key, default=None: settings.get(key, default),
        ),
        patch.object(
            _LegacyTranscriptionBackend,
            "get_available_providers",
            return_value=["faster-whisper"],
        ),
        patch.object(
            _LegacyTranscriptionBackend,
            "_transcribe_with_faster_whisper",
            return_value=legacy_result,
        ) as transcribe,
    ):
        result = TranscriptionService().transcribe("audio.wav")

    assert result is legacy_result
    assert transcribe.call_args.args[:6] == (
        "audio.wav",
        "base",
        "fr",
        False,
        "fr",
        None,
    )
