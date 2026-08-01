"""Tests for per-turn TTS synthesis across the exact and legacy provider paths.

`synthesize_turn` is the piece that turns one script turn -- one speaker,
one resolved voice, one block of text -- into WAV bytes, on any configured
TTS provider. The only faked seam is the TTS service itself (a stub
exposing `synthesize_exact` and `generate_audio_stream`, mirroring
`TTSService`'s own call shape); everything else, including the real
`TTSAudioResponse`, `TextChunker`, and `concat_wav_segments` (via `pydub`),
is exercised as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import pytest
from pydub import AudioSegment

from tldw_chatbook.Subscriptions.briefing_audio import (
    MAX_TURN_CHARS,
    TurnSynthesisError,
    synthesize_turn,
)
from tldw_chatbook.Subscriptions.briefing_voices import VoiceSelection
from tldw_chatbook.TTS.adapter_types import TTSAudioResponse, TTSRequest
from tldw_chatbook.TTS.playground_types import TTSRequestedSelectionSnapshot

pytestmark = pytest.mark.unit

_DEFAULT_GAP_MS = 350


def _silence_wav(duration_ms: int = 100, frame_rate: int = 22050) -> bytes:
    """Build a real WAV payload of `duration_ms` of silence, in-process."""
    segment = AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate)
    buffer = BytesIO()
    segment.export(buffer, format="wav")
    return buffer.getvalue()


def _decode(payload: bytes) -> AudioSegment:
    return AudioSegment.from_file(BytesIO(payload), format="wav")


def _exact_selection(
    *,
    speaker: str = "Host",
    model_id: str = "model-a",
    voice_id: str | None = "voice-a",
    speed: float = 1.0,
    options: dict[str, Any] | None = None,
) -> VoiceSelection:
    return VoiceSelection(
        speaker=speaker,
        provider_id="audio_cpp",
        model_id=model_id,
        voice_id=voice_id,
        response_format="wav",
        speed=speed,
        options={} if options is None else options,
        profile_id="11111111-1111-4111-8111-111111111111",
        profile_revision=1,
    )


def _legacy_selection(
    *,
    speaker: str = "Host",
    provider_id: str = "kokoro",
    model_id: str = "model-a",
    voice_id: str | None = "bella",
    speed: float = 1.0,
    options: dict[str, Any] | None = None,
) -> VoiceSelection:
    return VoiceSelection(
        speaker=speaker,
        provider_id=provider_id,
        model_id=model_id,
        voice_id=voice_id,
        response_format="wav",
        speed=speed,
        options={} if options is None else options,
        profile_id="22222222-2222-4222-8222-222222222222",
        profile_revision=1,
    )


def _exact_snapshot(
    *,
    model_id: str = "model-a",
    voice_id: str | None = "voice-a",
) -> TTSRequestedSelectionSnapshot:
    return TTSRequestedSelectionSnapshot(
        provider_id="audio_cpp",
        model_id=model_id,
        voice_id=voice_id,
        response_format="wav",
        speed=1.0,
        options={},
        configuration_revision=1,
    )


@dataclass
class _ExactPlan:
    """One programmed `synthesize_exact` call's outcome."""

    snapshot: TTSRequestedSelectionSnapshot
    chunks: list[bytes]
    stream_error: BaseException | None = None


@dataclass
class _LegacyPlan:
    """One programmed `generate_audio_stream` call's outcome."""

    chunks: list[bytes]


@dataclass
class _FakeTTSService:
    """The one faked seam: mirrors `TTSService`'s exact/legacy call shape.

    `exact_plans`/`legacy_plans` are queues, one entry consumed per call --
    a long turn (chunked into several pieces) makes several calls, so each
    can hand back its own audio and snapshot. `legacy_repeat_chunk`, when
    set, makes every `generate_audio_stream` call yield the same fixed
    chunk instead, for a long-turn test that does not care how many pieces
    `TextChunker` produces, only that there is more than one.
    """

    exact_plans: list[_ExactPlan] = field(default_factory=list)
    legacy_plans: list[_LegacyPlan] = field(default_factory=list)
    legacy_repeat_chunk: bytes | None = None
    exact_requests: list[TTSRequest] = field(default_factory=list, init=False)
    legacy_requests: list[tuple[Any, str]] = field(default_factory=list, init=False)
    aclose_count: int = field(default=0, init=False)

    async def synthesize_exact(self, request: TTSRequest, progress_sink: Any = None):
        self.exact_requests.append(request)
        plan = self.exact_plans.pop(0)

        async def _stream():
            for chunk in plan.chunks:
                yield chunk
            if plan.stream_error is not None:
                raise plan.stream_error

        response = TTSAudioResponse(
            provider_id=request.provider_id,
            model_id=request.model_id,
            audio_format="wav",
            content_type="audio/wav",
            byte_stream=_stream(),
        )
        response.add_cleanup(self._count_aclose)
        return response, plan.snapshot

    async def _count_aclose(self) -> None:
        self.aclose_count += 1

    async def generate_audio_stream(
        self,
        request: Any,
        internal_model_id: str,
        progress_sink: Any = None,
    ):
        self.legacy_requests.append((request, internal_model_id))
        if self.legacy_repeat_chunk is not None:
            yield self.legacy_repeat_chunk
            return
        plan = self.legacy_plans.pop(0)
        for chunk in plan.chunks:
            yield chunk


# --------------------------------------------------------------------------
# Exact path (audio_cpp)
# --------------------------------------------------------------------------


async def test_exact_path_builds_request_and_closes_the_response_once() -> None:
    wav = _silence_wav(120)
    selection = _exact_selection(model_id="model-a", voice_id="voice-a", speed=1.0)
    service = _FakeTTSService(
        exact_plans=[_ExactPlan(snapshot=_exact_snapshot(), chunks=[wav])]
    )

    result = await synthesize_turn(service, selection, "hello", turn_index=0)

    assert result == wav
    assert service.aclose_count == 1
    [request] = service.exact_requests
    assert request.provider_id == "audio_cpp"
    assert request.model_id == "model-a"
    assert request.voice == "voice-a"
    assert request.response_format == "wav"
    assert request.speed == 1.0
    assert dict(request.options) == {}


async def test_exact_path_closes_response_exactly_once_when_the_drain_raises() -> None:
    """The leak this guards: a failed drain must still release the response.

    A failed drain that skips `aclose()` would leave the exact provider's
    registry lease held -- see the module docstring's cross-reference to
    `TTSEventHandler._generate_tts`'s `finally`.
    """
    selection = _exact_selection()
    boom = RuntimeError("stream broke mid-drain")
    service = _FakeTTSService(
        exact_plans=[
            _ExactPlan(
                snapshot=_exact_snapshot(),
                chunks=[b"partial-bytes"],
                stream_error=boom,
            )
        ]
    )

    with pytest.raises(RuntimeError, match="stream broke mid-drain"):
        await synthesize_turn(service, selection, "hello", turn_index=0)

    assert service.aclose_count == 1


async def test_exact_path_contract_violation_names_speaker_index_and_both_voices() -> (
    None
):
    selection = _exact_selection(speaker="Narrator", voice_id="voice-a")
    mismatched_snapshot = _exact_snapshot(voice_id="voice-b")
    service = _FakeTTSService(
        exact_plans=[
            _ExactPlan(snapshot=mismatched_snapshot, chunks=[_silence_wav(80)])
        ]
    )

    with pytest.raises(TurnSynthesisError) as caught:
        await synthesize_turn(service, selection, "hello", turn_index=3)

    message = str(caught.value)
    assert "Narrator" in message
    assert "turn 3" in message
    assert "voice-a" in message
    assert "voice-b" in message
    # The response must still be released even though validation failed.
    assert service.aclose_count == 1


# --------------------------------------------------------------------------
# Legacy path (everything else)
# --------------------------------------------------------------------------


async def test_legacy_path_builds_request_via_the_shared_builder_and_joins_chunks() -> (
    None
):
    wav = _silence_wav(90)
    midpoint = len(wav) // 2
    selection = _legacy_selection(provider_id="kokoro", model_id="ignored", voice_id="bella")
    service = _FakeTTSService(
        legacy_plans=[_LegacyPlan(chunks=[wav[:midpoint], wav[midpoint:]])]
    )

    result = await synthesize_turn(service, selection, "hello", turn_index=0)

    assert result == wav
    [(request, internal_model_id)] = service.legacy_requests
    assert internal_model_id == "local_kokoro_default_onnx"
    assert request.voice == "bella"
    assert request.model == "kokoro"
    assert request.response_format == "wav"


async def test_legacy_path_zero_byte_result_raises_naming_speaker_and_index() -> None:
    selection = _legacy_selection(speaker="Guest")
    service = _FakeTTSService(legacy_plans=[_LegacyPlan(chunks=[])])

    with pytest.raises(TurnSynthesisError) as caught:
        await synthesize_turn(service, selection, "hello", turn_index=5)

    message = str(caught.value)
    assert "Guest" in message
    assert "turn 5" in message


async def test_legacy_path_non_wav_payload_raises_naming_speaker_and_index() -> None:
    selection = _legacy_selection(speaker="Guest")
    service = _FakeTTSService(legacy_plans=[_LegacyPlan(chunks=[b"not audio at all"])])

    with pytest.raises(TurnSynthesisError) as caught:
        await synthesize_turn(service, selection, "hello", turn_index=2)

    message = str(caught.value)
    assert "Guest" in message
    assert "turn 2" in message


async def test_legacy_path_rejects_an_unmapped_provider_id_naming_it() -> None:
    """Closes Task 2's carried finding: an unmapped provider must never reach
    the shared builder, since its `model_id` could collide with a reserved
    internal id (e.g. `"local_kokoro_default_onnx"`) and silently mis-route.
    """
    selection = _legacy_selection(
        speaker="Guest",
        provider_id="mystery_tts",
        model_id="local_kokoro_default_onnx",
    )
    service = _FakeTTSService()

    with pytest.raises(TurnSynthesisError) as caught:
        await synthesize_turn(service, selection, "hello", turn_index=1)

    message = str(caught.value)
    assert "Guest" in message
    assert "turn 1" in message
    assert "mystery_tts" in message
    # The builder (and the service) must never have been reached.
    assert service.legacy_requests == []


async def test_legacy_path_rejects_an_empty_voice_naming_speaker_and_index() -> None:
    selection = _legacy_selection(speaker="Guest", voice_id=None)
    service = _FakeTTSService()

    with pytest.raises(TurnSynthesisError) as caught:
        await synthesize_turn(service, selection, "hello", turn_index=4)

    message = str(caught.value)
    assert "Guest" in message
    assert "turn 4" in message
    assert service.legacy_requests == []


# --------------------------------------------------------------------------
# Long-turn chunking (both paths funnel through the same stitcher)
# --------------------------------------------------------------------------


async def test_long_turn_text_is_chunked_and_stitched_covering_all_pieces() -> None:
    sentence = "This sentence repeats to build a turn longer than the character gate. "
    text = sentence * 40
    assert len(text) > MAX_TURN_CHARS

    chunk_wav = _silence_wav(120)
    selection = _legacy_selection(provider_id="kokoro")
    service = _FakeTTSService(legacy_repeat_chunk=chunk_wav)

    result = await synthesize_turn(service, selection, text, turn_index=0)

    piece_count = len(service.legacy_requests)
    assert piece_count > 1

    decoded = _decode(result)
    expected_ms = piece_count * 120 + (piece_count - 1) * _DEFAULT_GAP_MS
    assert decoded.frame_count() == pytest.approx(
        decoded.frame_rate * expected_ms / 1000.0, rel=0.01
    )
    assert len(decoded) == pytest.approx(expected_ms, abs=50)


async def test_short_turn_is_not_chunked_and_needs_only_one_provider_call() -> None:
    wav = _silence_wav(100)
    selection = _legacy_selection(provider_id="kokoro")
    service = _FakeTTSService(legacy_plans=[_LegacyPlan(chunks=[wav])])

    text = "A short turn well under the character gate."
    assert len(text) <= MAX_TURN_CHARS

    result = await synthesize_turn(service, selection, text, turn_index=0)

    assert result == wav
    assert len(service.legacy_requests) == 1
