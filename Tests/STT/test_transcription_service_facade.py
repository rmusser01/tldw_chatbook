"""Compatibility tests for the public transcription service facade."""

from __future__ import annotations

import inspect
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints
from unittest.mock import patch

import pytest

import tldw_chatbook.Local_Ingestion.transcription_service as service_module
from tldw_chatbook.Local_Ingestion.transcription_service import (
    TranscriptionError,
    TranscriptionService,
    _LegacyTranscriptionBackend,
)


def _signature_shape(callable_object: object) -> tuple[tuple[str, object, object], ...]:
    return tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(callable_object).parameters.values()
    )


def test_public_constructor_adds_only_a_keyword_dispatcher() -> None:
    keyword_only = inspect.Parameter.KEYWORD_ONLY

    assert _signature_shape(TranscriptionService) == (
        ("local_stt_dispatcher", keyword_only, None),
        ("parakeet_source_service", keyword_only, None),
    )


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

    def cleanup_legacy(self) -> None:
        self._record("cleanup")

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
        "faster-whisper",
        "base.en",
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
                "faster-whisper",
                "base.en",
                "en",
            ),
            {"model_dir": "/models/parakeet"},
        )
    ]


@pytest.mark.parametrize(
    ("provider_kwargs", "expected_local_only"),
    [({}, False), ({"local_files_only": True}, True)],
)
def test_retained_faster_whisper_buffer_honors_explicit_local_only_without_changing_default(
    monkeypatch: pytest.MonkeyPatch,
    provider_kwargs: dict[str, object],
    expected_local_only: bool,
) -> None:
    constructor_calls: list[dict[str, object]] = []

    class _WhisperModel:
        def __init__(self, _model: str, **kwargs: object) -> None:
            constructor_calls.append(kwargs)

        def transcribe(self, _audio: object, **_kwargs: object) -> object:
            return iter(()), SimpleNamespace(language="en")

    monkeypatch.setattr(service_module, "WhisperModel", _WhisperModel)
    backend = object.__new__(_LegacyTranscriptionBackend)
    backend.config = {
        "default_model": "base",
        "default_language": "en",
        "device": "cpu",
        "compute_type": "int8",
    }
    backend._model_cache = {}
    backend._model_cache_lock = threading.Lock()

    backend._transcribe_buffer_with_faster_whisper(
        b"\x00\x00" * 16,
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        model="base",
        language="en",
        **provider_kwargs,
    )

    assert constructor_calls[0]["local_files_only"] is expected_local_only


class _Dispatcher:
    def __init__(self) -> None:
        self.buffer_calls: list[dict[str, object]] = []
        self.dictation_calls: list[dict[str, object]] = []
        self.buffer_result = {
            "text": "shared result",
            "segments": [],
            "language": "en",
        }
        self.handle = object()
        self.closed = False

    def transcribe_buffer(self, **kwargs: object) -> object:
        self.buffer_calls.append(kwargs)
        return self.buffer_result

    def begin_dictation(self, **kwargs: object) -> object:
        self.dictation_calls.append(kwargs)
        return self.handle

    def close(self) -> None:
        self.closed = True


class _SourceService:
    def __init__(self, dispatch: object | None = None) -> None:
        self.dispatch = dispatch if dispatch is not None else object()
        self.calls: list[dict[str, object]] = []
        self.error: Exception | None = None
        self.closed = False

    def resolve(self, key: object, **kwargs: object) -> object:
        self.calls.append({"key": key, **kwargs})
        if self.error is not None:
            raise self.error
        return self.dispatch

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("provider", "configured"),
    [
        pytest.param("parakeet-onnx", "faster-whisper", id="explicit"),
        pytest.param(None, "parakeet-onnx", id="configured-default"),
    ],
)
def test_parakeet_buffer_uses_the_shared_dispatcher_and_compatibility_result(
    monkeypatch: pytest.MonkeyPatch,
    provider: str | None,
    configured: str,
) -> None:
    dispatcher = _Dispatcher()
    dispatch = object()
    source_service = _SourceService(dispatch)
    bridge = _Bridge()
    bridge.config["default_provider"] = configured
    monkeypatch.setattr(
        service_module,
        "LegacyTranscriptionBridge",
        lambda _backend_factory: bridge,
    )
    service = TranscriptionService(
        local_stt_dispatcher=dispatcher,
        parakeet_source_service=source_service,
    )
    result = service.transcribe_buffer(
        b"\x00\x01" * 4,
        24_000,
        2,
        1,
        provider,
        None,
        None,
    )

    assert result is dispatcher.buffer_result
    assert bridge.calls == []
    call = dispatcher.buffer_calls[0]
    source = call["source"]
    assert source.audio == b"\x00\x01" * 4
    assert (source.sample_rate, source.channels, source.sample_width) == (24_000, 2, 1)
    assert call == {"source": source, "dispatch": dispatch, "language": "en"}
    assert len(source_service.calls) == 1


@pytest.mark.parametrize(
    ("configured_precision", "expected"),
    [(None, "int8"), (" F32 ", "f32")],
)
def test_parakeet_buffer_resolves_the_configured_precision_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    configured_precision: str | None,
    expected: str,
) -> None:
    dispatcher = _Dispatcher()
    source_service = _SourceService()
    bridge = _Bridge()
    monkeypatch.setattr(
        service_module,
        "LegacyTranscriptionBridge",
        lambda _backend_factory: bridge,
    )
    monkeypatch.setattr(
        service_module,
        "get_cli_setting",
        lambda key, default=None: (
            configured_precision
            if key == "transcription.default_precision"
            else default
        ),
    )

    TranscriptionService(
        local_stt_dispatcher=dispatcher,
        parakeet_source_service=source_service,
    ).transcribe_buffer(
        b"\x00\x00",
        16_000,
        provider="parakeet-onnx",
    )

    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    assert source_service.calls == [
        {
            "key": ParakeetSourceKey.from_values("nemo-parakeet-tdt-0.6b-v2", expected),
            "override": None,
        }
    ]


def test_invalid_parakeet_precision_fails_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    source_service = _SourceService()
    bridge = _Bridge()
    monkeypatch.setattr(
        service_module,
        "LegacyTranscriptionBridge",
        lambda _backend_factory: bridge,
    )
    monkeypatch.setattr(
        service_module,
        "get_cli_setting",
        lambda key, default=None: (
            "fp16" if key == "transcription.default_precision" else default
        ),
    )

    with pytest.raises(ValueError, match="unsupported Parakeet model and precision"):
        TranscriptionService(
            local_stt_dispatcher=dispatcher,
            parakeet_source_service=source_service,
        ).transcribe_buffer(
            b"\x00\x00",
            16_000,
            provider="parakeet-onnx",
        )

    assert source_service.calls == []
    assert dispatcher.buffer_calls == []
    assert bridge.calls == []


def test_parakeet_buffer_requires_the_shared_dispatcher(
    facade: tuple[TranscriptionService, _Bridge],
) -> None:
    service, bridge = facade

    with pytest.raises(
        TranscriptionError,
        match="requires the shared local executor",
    ):
        service.transcribe_buffer(
            b"\x00\x00",
            16_000,
            provider="parakeet-onnx",
        )

    assert bridge.calls == []


def test_parakeet_streaming_reports_unsupported_without_consulting_the_bridge(
    facade: tuple[TranscriptionService, _Bridge],
) -> None:
    service, bridge = facade

    assert service.create_streaming_transcriber(provider="parakeet-onnx") is None
    assert bridge.calls == []


def test_facade_cleanup_does_not_close_the_app_owned_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    source_service = _SourceService()
    bridge = _Bridge()
    monkeypatch.setattr(
        service_module,
        "LegacyTranscriptionBridge",
        lambda _backend_factory: bridge,
    )
    service = TranscriptionService(
        local_stt_dispatcher=dispatcher,
        parakeet_source_service=source_service,
    )

    service.cleanup()

    assert bridge.calls == [("cleanup", (), {})]
    assert dispatcher.closed is False
    assert source_service.closed is False


def test_facade_cleanup_closes_only_its_download_free_default_source_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _Bridge()
    owned = _SourceService()
    constructed: list[object] = []
    monkeypatch.setattr(
        service_module,
        "LegacyTranscriptionBridge",
        lambda _backend_factory: bridge,
    )

    def _create() -> _SourceService:
        constructed.append(owned)
        return owned

    monkeypatch.setattr(service_module, "ParakeetSourceService", _create, raising=False)

    service = TranscriptionService()
    service.cleanup()

    assert constructed == [owned]
    assert owned.closed is True


def test_begin_dictation_capture_forwards_the_resolved_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    bridge = _Bridge()
    dispatch = object()
    source_service = _SourceService(dispatch)

    def callback(_sequence: int, _text: str) -> None:
        return None

    monkeypatch.setattr(
        service_module,
        "LegacyTranscriptionBridge",
        lambda _backend_factory: bridge,
    )
    handle = TranscriptionService(
        local_stt_dispatcher=dispatcher,
        parakeet_source_service=source_service,
    ).begin_dictation_capture(
        capture_generation=7,
        model=None,
        language="en",
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        on_logical_segment=callback,
    )

    assert handle is dispatcher.handle
    assert dispatcher.dictation_calls == [
        {
            "capture_generation": 7,
            "dispatch": dispatch,
            "sample_rate": 16_000,
            "channels": 1,
            "sample_width": 2,
            "language": "en",
            "on_logical_segment": callback,
        }
    ]


def test_begin_dictation_uses_persistent_service_resolution_not_bridge_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = _Dispatcher()
    bridge = _Bridge()
    bridge.config["parakeet_onnx_model_dir"] = "/legacy/direct-path"
    source_service = _SourceService()

    monkeypatch.setattr(
        service_module,
        "LegacyTranscriptionBridge",
        lambda _backend_factory: bridge,
    )

    TranscriptionService(
        local_stt_dispatcher=dispatcher,
        parakeet_source_service=source_service,
    ).begin_dictation_capture(
        capture_generation=7,
        model=None,
        language="en",
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        on_logical_segment=lambda _sequence, _text: None,
    )

    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    assert source_service.calls == [
        {"key": ParakeetSourceKey.V2_INT8, "override": None}
    ]


def test_source_failure_is_structured_path_private_and_never_uses_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.STT.parakeet_sources import (
        ParakeetSourceError,
        ParakeetSourceErrorCode,
    )

    dispatcher = _Dispatcher()
    bridge = _Bridge()
    source_service = _SourceService()
    source_service.error = ParakeetSourceError(ParakeetSourceErrorCode.VAD_UNAVAILABLE)
    selected_path = "/private/user/parakeet"
    monkeypatch.setattr(
        service_module,
        "LegacyTranscriptionBridge",
        lambda _backend_factory: bridge,
    )
    monkeypatch.setattr(
        service_module,
        "resolve_parakeet_dispatch",
        lambda **_kwargs: pytest.fail("legacy fallback must not run"),
        raising=False,
    )

    with pytest.raises(ParakeetSourceError) as caught:
        TranscriptionService(
            local_stt_dispatcher=dispatcher,
            parakeet_source_service=source_service,
        ).transcribe_buffer(
            b"\x00\x00",
            16_000,
            provider="parakeet-onnx",
            model_dir=selected_path,
        )

    assert caught.value.code is ParakeetSourceErrorCode.VAD_UNAVAILABLE
    assert selected_path not in str(caught.value)
    assert dispatcher.buffer_calls == []
    assert bridge.calls == []


@pytest.mark.parametrize(
    ("method_name", "arguments", "keywords", "bridge_method"),
    [
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


def test_facade_cleanup_has_truthful_none_return_contract(
    facade: tuple[TranscriptionService, _Bridge],
) -> None:
    service, bridge = facade

    service.cleanup()

    assert get_type_hints(TranscriptionService.cleanup)["return"] is type(None)
    assert bridge.calls == [("cleanup", (), {})]


def test_facade_transcribe_documents_its_public_contract() -> None:
    docstring = inspect.getdoc(TranscriptionService.transcribe)

    assert docstring is not None
    assert "Args:" in docstring
    assert "Returns:" in docstring
    assert "Raises:" in docstring


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


@pytest.mark.parametrize(
    ("audio_path", "cause_text"),
    [
        pytest.param("missing;unsafe.wav", "dangerous pattern", id="unsafe"),
        pytest.param("non_existent_file.wav", "Path does not exist", id="missing"),
    ],
)
def test_real_facade_rejects_invalid_audio_path_before_provider_dispatch(
    audio_path: str,
    cause_text: str,
) -> None:
    settings = {
        "transcription.default_provider": "faster-whisper",
        "transcription.default_model": "base",
    }

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
        ) as transcribe,
    ):
        with pytest.raises(
            TranscriptionError,
            match="Invalid audio file path",
        ) as exc_info:
            TranscriptionService().transcribe(
                audio_path,
                provider="faster-whisper",
            )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert cause_text in str(exc_info.value.__cause__)
    transcribe.assert_not_called()


def test_real_facade_keeps_configured_legacy_provider_and_language(
    tmp_path: Path,
) -> None:
    settings = {
        "transcription.default_provider": "faster-whisper",
        "transcription.default_model": "base",
        "transcription.default_language": "en",
        "transcription.default_source_language": "fr",
    }
    legacy_result = {"text": "bonjour", "segments": [], "language": "fr"}
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()

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
        result = TranscriptionService().transcribe(str(audio_path))

    assert result is legacy_result
    assert transcribe.call_args.args[:6] == (
        str(audio_path),
        "base",
        "fr",
        False,
        "fr",
        None,
    )
