"""Response-eligibility seam for the streaming PCM sink.

Pure, provider-agnostic decision: given a TTS response's declared audio
format, sample rate, and (for containerized formats) the first bytes of the
body, decide whether the response is eligible for the streaming PCM sink and,
if so, produce a `SinkPlan` describing how to consume it.

The seam branches only on the response's own shape (format / rate / bytes),
never on provider identity -- callers pass plain values, not a provider
object, so this stays true regardless of whether the caller holds an adapter
`TTSAudioResponse` or a backend path's raw format string.

Import weight: `audio_cpp_contract` was read before adding this module-scope
import -- it pulls in only stdlib (`json`, `math`, `re`, `struct`,
`unicodedata`, `dataclasses`, `decimal`, `types`, `typing`), so no lazy-import
is needed. This module itself imports nothing else: no sounddevice, no
Textual, no other TTS submodules. That said, `import tldw_chatbook.TTS.pcm_stream`
still runs `tldw_chatbook/TTS/__init__.py` first (Python always imports the
package before a submodule), which eagerly pulls in the full TTS package --
including, transitively, `textual.*`. That's pre-existing package structure,
not introduced here, but it means `tldw_chatbook/Audio/streaming_sink.py`
(which is constrained to zero Textual imports) must never import this module
at module scope, even though this module's own direct imports are light. The
two meet only in the consumer that wires them together (the TTS event
handling code), never in `streaming_sink.py` itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from math import gcd
import struct
from dataclasses import dataclass
from typing import Literal

from tldw_chatbook.TTS.audio_cpp_contract import (
    AudioCppContractError,
    validate_pcm16_wav,
)

_FORMAT_RAW_PCM = "pcm"
_FORMAT_WAV = "wav"
_DEFAULT_CHANNELS = 1
_MIN_SOURCE_SAMPLE_RATE = 8_000
_PROCESSING_SAMPLE_RATE = 48_000
_MAX_SOURCE_CHANNELS = 2
_MAX_OUTPUT_FRAMES_PER_SOURCE_BLOCK = 10
_MAX_SOURCE_BLOCK_BYTES = 19_200
_MAX_RAW_CHUNK_BYTES = 256 * 1024
_MAX_RAW_DURATION_SECONDS = 30
_MAX_RAW_STREAM_BYTES = (
    _PROCESSING_SAMPLE_RATE * _MAX_SOURCE_CHANNELS * 2 * _MAX_RAW_DURATION_SECONDS
)
_MAX_WAV_BYTES = 32 * 1024 * 1024

__all__ = [
    "PcmStreamError",
    "SinkPlan",
    "iter_normalized_pcm_frames",
    "sink_plan",
]


class PcmStreamError(RuntimeError):
    """Typed speech-only failure while preparing duplex render PCM."""

    def __init__(
        self,
        code: Literal["unsupported_audio_format", "invalid_audio_stream"],
    ) -> None:
        self.code = code
        self.recoverable = True
        super().__init__(code.replace("_", " "))


@dataclass(frozen=True, slots=True)
class SinkPlan:
    """Parameters the streaming PCM sink needs to consume an eligible response."""

    sample_rate: int
    channels: int
    skip_bytes: int
    data_bytes: int | None = None


def sink_plan(
    audio_format: str,
    sample_rate: int | None,
    first_bytes: bytes | None,
    channels: int | None = None,
) -> SinkPlan | None:
    """Decide whether a TTS response is eligible for the streaming PCM sink.

    Returns `None` for anything the sink can't safely consume -- compressed
    formats, raw PCM with no declared rate, or a WAV body that fails
    structural validation -- so the caller can fall back to the legacy
    whole-file path unconditionally.

    Args:
        audio_format: The response's declared audio format, e.g. "pcm",
            "wav", "mp3". Compared case-sensitively against the two formats
            the sink understands; anything else is ineligible.
        sample_rate: The response's declared sample rate. Required for
            "pcm" -- returns `None` when absent OR when present but not an
            `int` (fix-round F8: an adapter/legacy-bridge bug supplying a
            wrong-typed value must fail closed here, not raise deep inside
            `StreamingPcmSink.open`'s `sample_rate * blocksize_ms // 1000`
            arithmetic, where it would otherwise be swallowed by
            `_generate_tts`'s generic exception handling) -- ignored for
            "wav", where the validated header is authoritative.
        first_bytes: The response body (or its head). Required for "wav" --
            validated as a canonical PCM16 RIFF/WAVE body via
            `validate_pcm16_wav`. Unused for "pcm". Must be the COMPLETE WAV
            body: `validate_pcm16_wav` fails closed on any truncated or
            internally-inconsistent buffer (declared RIFF size not matching
            the buffer's actual length, or a `data` chunk claiming more
            bytes than are present), so a caller that only has the first N
            bytes of a still-arriving WAV response will get `None` here,
            not a partial plan. audio.cpp's single-response delivery (one
            complete WAV body per response) satisfies this; a streaming WAV
            source would need to buffer the whole body before calling this.
        channels: Explicit channel count for raw "pcm"; defaults to 1 when
            omitted. Ignored for "wav", where the validated header is
            authoritative.

    Returns:
        A `SinkPlan` if the response is eligible, else `None`. For "wav",
        `SinkPlan.data_bytes` bounds the audio payload: callers must read
        only `first_bytes[skip_bytes : skip_bytes + data_bytes]` (or the
        equivalent window from the full body), never everything from
        `skip_bytes` onward -- a WAV may have trailing chunks after `data`,
        and their bytes are not audio.
    """
    if audio_format == _FORMAT_RAW_PCM:
        if sample_rate is None or type(sample_rate) is not int:
            return None
        resolved_channels = _DEFAULT_CHANNELS if channels is None else channels
        return SinkPlan(
            sample_rate=sample_rate,
            channels=resolved_channels,
            skip_bytes=0,
            data_bytes=None,  # no container length; the decoder enforces runtime caps
        )

    if audio_format == _FORMAT_WAV:
        return _wav_plan(first_bytes)

    return None


def _wav_plan(first_bytes: bytes | None) -> SinkPlan | None:
    """Validate a WAV body and derive the plan, or `None` if it's not usable."""
    if not first_bytes:
        return None
    try:
        info = validate_pcm16_wav(first_bytes)
    except AudioCppContractError:
        return None

    # Pcm16WavInfo (tldw_chatbook/TTS/audio_cpp_contract.py) carries no
    # chunk-offset field -- only sample_rate/channels/frame_count/data_size/
    # byte_rate/block_align/bits_per_sample. `validate_pcm16_wav` accepts
    # ancillary RIFF chunks generically, in ANY position, including AFTER
    # `data` (see its module docstring) -- so `len(first_bytes) -
    # info.data_size` is wrong whenever something trails `data`: it counts
    # that trailing content as if it were header, over-skipping into the
    # real audio (task-3 review, finding I1). Locating `data`'s payload
    # offset directly, by walking the chunk structure, is correct
    # regardless of what precedes or follows it.
    return SinkPlan(
        sample_rate=info.sample_rate,
        channels=info.channels,
        skip_bytes=_data_chunk_offset(first_bytes),
        data_bytes=info.data_size,
    )


def _data_chunk_offset(body: bytes) -> int:
    """Return the byte offset of the `data` chunk's payload in `body`.

    Only called after `validate_pcm16_wav` has already accepted `body`, so
    this walk can simply assert its way to `data` -- the validator is the
    acceptance gate; it already proved `body` is a well-formed RIFF/WAVE
    stream (12-byte RIFF header, then a sequence of 8-byte chunk headers
    each followed by a word-aligned payload) containing exactly one `data`
    chunk. This mirrors that same chunk walk, just to find where audio
    playback should actually start.
    """
    position = 12
    while True:
        chunk_id = body[position : position + 4]
        chunk_size = struct.unpack_from("<I", body, position + 4)[0]
        payload_start = position + 8
        if chunk_id == b"data":
            return payload_start
        position = payload_start + chunk_size
        if chunk_size % 2:
            position += 1


async def iter_normalized_pcm_frames(
    *,
    audio_format: str,
    sample_rate: int | None,
    channels: int,
    byte_stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Yield 48 kHz mono PCM16 ten-millisecond frames from one TTS response.

    Raw PCM remains incremental across arbitrary provider chunk boundaries. WAV
    is the declared complete-body fallback and is bounded before validation.
    The response owner, not this decoder, closes ``byte_stream``.
    """

    if audio_format == _FORMAT_RAW_PCM:
        async for frame in _iter_raw_pcm_frames(
            sample_rate=sample_rate,
            channels=channels,
            byte_stream=byte_stream,
        ):
            yield frame
        return

    if audio_format != _FORMAT_WAV:
        raise PcmStreamError("unsupported_audio_format")

    body = bytearray()
    async for chunk in byte_stream:
        _validate_stream_chunk(chunk)
        if len(body) + len(chunk) > _MAX_WAV_BYTES:
            raise PcmStreamError("invalid_audio_stream")
        body.extend(chunk)
    plan = sink_plan(_FORMAT_WAV, None, bytes(body))
    if plan is None or plan.data_bytes is None:
        raise PcmStreamError("invalid_audio_stream")
    pcm = bytes(body[plan.skip_bytes : plan.skip_bytes + plan.data_bytes])
    for frame in _iter_normalized_padded_pcm(
        pcm,
        sample_rate=plan.sample_rate,
        channels=plan.channels,
    ):
        yield frame


async def _iter_raw_pcm_frames(
    *,
    sample_rate: int | None,
    channels: int,
    byte_stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    block_bytes, output_frames_per_block = _source_block_shape(
        sample_rate,
        channels,
    )
    if sample_rate is None:  # narrowed by _source_block_shape; keep arithmetic typed
        raise PcmStreamError("invalid_audio_stream")
    sample_frame_bytes = channels * 2
    lookahead_blocks = 1 if sample_rate == _PROCESSING_SAMPLE_RATE else 2
    required_bytes = block_bytes * lookahead_blocks
    stream_byte_limit = min(
        _MAX_RAW_STREAM_BYTES,
        sample_rate * sample_frame_bytes * _MAX_RAW_DURATION_SECONDS,
    )
    pending = bytearray()
    observed = False
    total_bytes = 0
    async for chunk in byte_stream:
        _validate_stream_chunk(chunk)
        chunk_bytes = len(chunk)
        if (
            chunk_bytes > _MAX_RAW_CHUNK_BYTES
            or total_bytes + chunk_bytes > stream_byte_limit
        ):
            raise PcmStreamError("invalid_audio_stream")
        total_bytes += chunk_bytes
        observed = observed or bool(chunk)
        offset = 0
        while offset < chunk_bytes:
            copied = min(required_bytes - len(pending), chunk_bytes - offset)
            pending.extend(memoryview(chunk)[offset : offset + copied])
            offset += copied
            if len(pending) < required_bytes:
                continue
            source = bytes(pending)
            del pending[:block_bytes]
            from_frames = _normalize_complete_pcm(
                source,
                sample_rate=sample_rate,
                channels=channels,
            )
            for frame in from_frames[:output_frames_per_block]:
                yield frame
    if not observed:
        raise PcmStreamError("invalid_audio_stream")
    if len(pending) % sample_frame_bytes:
        raise PcmStreamError("invalid_audio_stream")
    if pending:
        padded_bytes = block_bytes if len(pending) <= block_bytes else block_bytes * 2
        pending.extend(bytes(padded_bytes - len(pending)))
        for frame in _normalize_complete_pcm(
            bytes(pending),
            sample_rate=sample_rate,
            channels=channels,
        ):
            yield frame


def _normalize_complete_pcm(
    pcm16: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> tuple[bytes, ...]:
    try:
        from tldw_chatbook.Audio.voice_preprocessor import normalize_pcm16_frames

        return normalize_pcm16_frames(
            pcm16,
            sample_rate=sample_rate,
            channels=channels,
        )
    except (TypeError, ValueError):
        raise PcmStreamError("invalid_audio_stream") from None


def _iter_normalized_padded_pcm(
    pcm16: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> Iterator[bytes]:
    """Normalize complete source blocks and zero-pad one valid short tail."""

    block_bytes, output_frames_per_block = _source_block_shape(
        sample_rate,
        channels,
    )
    sample_frame_bytes = channels * 2
    if not pcm16 or len(pcm16) % sample_frame_bytes:
        raise PcmStreamError("invalid_audio_stream")

    block_count = (len(pcm16) + block_bytes - 1) // block_bytes
    for block_index in range(block_count - 1):
        offset = block_index * block_bytes
        pair = pcm16[offset : offset + block_bytes * 2]
        if len(pair) < block_bytes * 2:
            pair += bytes(block_bytes * 2 - len(pair))
        from_frames = _normalize_complete_pcm(
            pair,
            sample_rate=sample_rate,
            channels=channels,
        )
        yield from from_frames[:output_frames_per_block]
    final_block = pcm16[(block_count - 1) * block_bytes :]
    final_block += bytes(block_bytes - len(final_block))
    yield from _normalize_complete_pcm(
        final_block,
        sample_rate=sample_rate,
        channels=channels,
    )


def _source_block_shape(
    sample_rate: int | None,
    channels: int,
) -> tuple[int, int]:
    if (
        type(sample_rate) is not int
        or not _MIN_SOURCE_SAMPLE_RATE <= sample_rate <= _PROCESSING_SAMPLE_RATE
        or type(channels) is not int
        or not 1 <= channels <= _MAX_SOURCE_CHANNELS
    ):
        raise PcmStreamError("invalid_audio_stream")
    output_frames = 100 // gcd(sample_rate, 100)
    source_frames = sample_rate * output_frames // 100
    block_bytes = source_frames * channels * 2
    if (
        output_frames > _MAX_OUTPUT_FRAMES_PER_SOURCE_BLOCK
        or block_bytes > _MAX_SOURCE_BLOCK_BYTES
    ):
        raise PcmStreamError("invalid_audio_stream")
    return block_bytes, output_frames


def _validate_stream_chunk(chunk: object) -> None:
    if type(chunk) is not bytes:
        raise PcmStreamError("invalid_audio_stream")
