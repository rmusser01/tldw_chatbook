import pytest

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
