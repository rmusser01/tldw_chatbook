import copy
import pickle
from typing import get_args

import pytest

from tldw_chatbook.TTS import adapter_types
from tldw_chatbook.TTS.adapter_types import TTSAudioResponse, TTSRequest


@pytest.mark.asyncio
async def test_audio_response_closes_stream_and_callbacks_once() -> None:
    events: list[str] = []

    async def stream():
        try:
            yield b"first"
            yield b"second"
        finally:
            events.append("stream")

    async def cleanup() -> None:
        events.append("cleanup")

    response = TTSAudioResponse(
        provider_id="openai",
        model_id="tts-1",
        audio_format="mp3",
        content_type="audio/mpeg",
        byte_stream=stream(),
        cleanup=cleanup,
    )
    assert await anext(response.byte_stream) == b"first"

    await response.aclose()
    await response.aclose()

    assert events == ["stream", "cleanup"]


@pytest.mark.asyncio
async def test_audio_response_context_manager_closes_after_consumer_failure() -> None:
    closed = False

    async def stream():
        nonlocal closed
        try:
            yield b"audio"
        finally:
            closed = True

    with pytest.raises(RuntimeError, match="consumer"):
        async with TTSAudioResponse(
            provider_id="openai",
            model_id="tts-1",
            audio_format="mp3",
            content_type="audio/mpeg",
            byte_stream=stream(),
        ) as response:
            assert await anext(response.byte_stream) == b"audio"
            raise RuntimeError("consumer")

    assert closed is True


def test_tts_request_copies_options_at_the_boundary() -> None:
    source = {"temperature": 0.5}
    request = TTSRequest(
        provider_id="chatterbox",
        model_id="chatterbox",
        text="hello",
        voice="default",
        response_format="wav",
        speed=1.0,
        options=source,
    )
    source["temperature"] = 1.0

    assert request.options == {"temperature": 0.5}
    with pytest.raises(TypeError):
        request.options["temperature"] = 0.2  # type: ignore[index]


def test_audio_response_copies_metadata_at_the_boundary() -> None:
    async def stream():
        yield b"audio"

    source = {
        "operation_id": "op-test",
        "generation_ms": 12.5,
        "sample_count": 42,
        "cached": False,
        "upstream_timing": None,
    }
    response = TTSAudioResponse(
        provider_id="audio_cpp",
        model_id="model",
        audio_format="wav",
        content_type="audio/wav",
        byte_stream=stream(),
        metadata=source,
    )
    source["operation_id"] = "changed"

    assert response.metadata == {
        "operation_id": "op-test",
        "generation_ms": 12.5,
        "sample_count": 42,
        "cached": False,
        "upstream_timing": None,
    }
    with pytest.raises(TypeError):
        response.metadata["operation_id"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    "nested_value",
    (
        {"nested": "value"},
        ["nested"],
        bytearray(b"nested"),
    ),
)
def test_audio_response_rejects_non_scalar_metadata_values(
    nested_value: object,
) -> None:
    async def stream():
        yield b"audio"

    with pytest.raises(
        TypeError,
        match="TTS audio response metadata values must be immutable scalars",
    ) as error:
        TTSAudioResponse(
            provider_id="audio_cpp",
            model_id="model",
            audio_format="wav",
            content_type="audio/wav",
            byte_stream=stream(),
            metadata={"unsafe": nested_value},
        )

    assert "nested" not in str(error.value)


def test_tts_operation_code_contains_only_stable_values() -> None:
    assert get_args(adapter_types.TTSOperationCode) == (
        "configuration_invalid",
        "connection_unavailable",
        "contract_incompatible",
        "not_configured",
        "request_invalid",
        "model_invalid",
        "server_busy",
        "generation_failed",
        "audio_response_invalid",
        "generation_timeout",
    )


def test_tts_operation_error_exposes_only_safe_immutable_fields() -> None:
    error = adapter_types.TTSOperationError(
        code="server_busy",
        message="The audio.cpp server is busy",
        retryable=True,
        operation_id="op-test",
        recovery_action="retry",
    )

    assert error.code == "server_busy"
    assert str(error) == "The audio.cpp server is busy"
    assert (
        error.retryable,
        error.operation_id,
        error.recovery_action,
    ) == (
        True,
        "op-test",
        "retry",
    )
    assert error.args == ("The audio.cpp server is busy",)
    assert not hasattr(error, "message")
    assert vars(error) == {}
    assert error.__cause__ is None
    assert not hasattr(error, "remote_body")
    assert not hasattr(error, "remote_url")
    for field_name in (
        "code",
        "retryable",
        "operation_id",
        "recovery_action",
    ):
        with pytest.raises(AttributeError):
            setattr(error, field_name, "changed")
    with pytest.raises(AttributeError):
        error.remote_body = "secret"  # type: ignore[attr-defined]

    with pytest.raises(TypeError):
        adapter_types.TTSOperationError(
            code="server_busy",
            message="safe",
            retryable=True,
            operation_id="op-test",
            remote_body="secret",  # type: ignore[call-arg]
        )


def test_tts_operation_error_supports_standard_exception_notes() -> None:
    error = adapter_types.TTSOperationError(
        code="generation_failed",
        message="Audio generation failed",
        retryable=False,
        operation_id="op-test",
    )

    error.add_note("TTS cleanup also failed while preserving the original error")

    assert error.__notes__ == [
        "TTS cleanup also failed while preserving the original error"
    ]
    assert str(error) == "Audio generation failed"


def test_tts_operation_error_copy_and_pickle_preserve_safe_contract() -> None:
    error = adapter_types.TTSOperationError(
        code="generation_timeout",
        message="Audio generation timed out",
        retryable=True,
        operation_id="op-test",
        recovery_action="retry",
    )
    error.add_note("safe cleanup note")

    copies = (
        copy.copy(error),
        pickle.loads(pickle.dumps(error)),
    )

    for restored in copies:
        assert isinstance(restored, adapter_types.TTSOperationError)
        assert restored is not error
        assert restored.code == "generation_timeout"
        assert restored.retryable is True
        assert restored.operation_id == "op-test"
        assert restored.recovery_action == "retry"
        assert str(restored) == "Audio generation timed out"
        assert restored.__notes__ == ["safe cleanup note"]
