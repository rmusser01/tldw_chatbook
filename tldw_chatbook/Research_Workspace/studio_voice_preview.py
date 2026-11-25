"""Synthesize a fixed sample through the selected speech owner."""

from tldw_chatbook.tldw_api.audio_schemas import OpenAISpeechRequest

from .studio_models import ResearchGenerationConfig, StudioSpeechChoice


async def synthesize_voice_preview(
    scope, mode: str, choice: StudioSpeechChoice, speed: float
) -> bytes:
    ResearchGenerationConfig(speech=choice, speech_speed=speed)
    request = OpenAISpeechRequest(
        backend=choice.provider,
        model=choice.model,
        voice=choice.voice,
        input="This is a Research Studio voice preview."[: choice.max_characters],
        response_format=choice.format,
        stream=False,
        allow_fallback=False,
        normalization_options=None,
        **({"speed": speed} if choice.supports_speed else {}),
    )
    sample = await scope.preview_audio_speech(mode=mode, request_data=request)
    content = sample.get("content")
    if (
        not isinstance(content, bytes)
        or not 0 < len(content) <= 4 * 1024 * 1024
        or not str(sample.get("content_type", "")).startswith("audio/")
    ):
        raise ValueError("The selected speech owner returned no playable voice sample")
    return content
