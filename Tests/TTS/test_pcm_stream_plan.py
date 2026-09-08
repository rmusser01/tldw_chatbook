import asyncio
from collections.abc import AsyncIterator
import struct

import pytest

from tldw_chatbook.TTS.pcm_stream import (
    PcmStreamError,
    SinkPlan,
    iter_normalized_pcm_frames,
    sink_plan,
)


def _wav_header(rate=22050, channels=1, data=b"\x00\x00" * 64):
    hdr = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH", 16, 1, channels, rate, rate * channels * 2, channels * 2, 16
        )
        + b"data"
        + struct.pack("<I", len(data))
    )
    return hdr + data


def _wav_with_trailing_chunk(
    rate=22050, channels=1, data=None, trailer_id=b"LIST", trailer_payload=b"INFOtest"
):
    # A well-formed WAV whose RIFF size correctly accounts for a chunk placed
    # AFTER `data` -- a shape validate_pcm16_wav structurally accepts (it
    # only special-cases `fmt `/`data`; any other chunk id is skipped
    # generically regardless of position). Reproduces the reviewer's
    # trailing-LIST-chunk finding (task-3-review.md, I1).
    if data is None:
        data = bytes(range(128))
    trailer = trailer_id + struct.pack("<I", len(trailer_payload)) + trailer_payload
    declared_size = 36 + len(data) + len(trailer)
    hdr = (
        b"RIFF"
        + struct.pack("<I", declared_size)
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH", 16, 1, channels, rate, rate * channels * 2, channels * 2, 16
        )
        + b"data"
        + struct.pack("<I", len(data))
    )
    return hdr + data + trailer


def test_raw_pcm_with_rate_is_eligible():
    assert sink_plan("pcm", 24000, None) == SinkPlan(24000, 1, 0)


def test_raw_pcm_without_rate_is_not():
    assert sink_plan("pcm", None, None) is None


def test_raw_pcm_with_a_wrong_typed_rate_is_not():
    # Fix-round F8 (task-4 review): a wrong-typed sample_rate must fail
    # closed HERE, not raise deep inside StreamingPcmSink.open()'s
    # `sample_rate * blocksize_ms // 1000` arithmetic (str) or silently
    # produce a nonsensical plan (bool is an int subclass in Python, so
    # True/False would otherwise pass an `isinstance` check).
    assert sink_plan("pcm", "24000", None) is None
    assert sink_plan("pcm", 24000.0, None) is None
    assert sink_plan("pcm", True, None) is None


def test_valid_pcm16_wav_is_eligible_with_header_skip():
    data = b"\x00\x00" * 64
    plan = sink_plan("wav", None, _wav_header(data=data))
    assert plan is not None
    assert plan.sample_rate == 22050 and plan.channels == 1 and plan.skip_bytes == 44
    assert plan.data_bytes == len(data)


def test_wav_with_trailing_chunk_after_data_uses_true_data_offset():
    # Task-3 review I1: skip_bytes must be the TRUE data-chunk payload
    # offset, not `len(first_bytes) - data_size` -- the latter over-skips
    # when a chunk (e.g. LIST) trails `data`, dropping real audio and
    # exposing the trailer's own bytes as if they were PCM samples.
    data = bytes(range(128))
    body = _wav_with_trailing_chunk(data=data)
    plan = sink_plan("wav", None, body)
    assert plan is not None
    assert plan.skip_bytes == 44
    assert plan.data_bytes == len(data)
    assert body[plan.skip_bytes : plan.skip_bytes + plan.data_bytes] == data


def test_invalid_wav_falls_back():
    assert sink_plan("wav", None, b"RIFFgarbage") is None


def test_compressed_formats_fall_back():
    for fmt in ("mp3", "opus", "aac", "flac", ""):
        assert sink_plan(fmt, 24000, None) is None


async def _stream(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def _must_not_consume() -> AsyncIterator[bytes]:
    pytest.fail("invalid PCM metadata must fail before stream consumption")
    yield b""  # pragma: no cover


@pytest.mark.asyncio
async def test_raw_pcm_chunks_are_reassembled_into_normalized_ten_ms_frames():
    frame = b"\x34\x12" * 480

    normalized = [
        chunk
        async for chunk in iter_normalized_pcm_frames(
            audio_format="pcm",
            sample_rate=48_000,
            channels=1,
            byte_stream=_stream(frame[:317], frame[317:701], frame[701:]),
        )
    ]

    assert normalized == [frame]


@pytest.mark.asyncio
async def test_native_rate_raw_pcm_yields_before_the_next_provider_chunk() -> None:
    frame = b"\x34\x12" * 480
    release = asyncio.Event()

    async def delayed_tail() -> AsyncIterator[bytes]:
        yield frame
        await release.wait()

    decoded = iter_normalized_pcm_frames(
        audio_format="pcm",
        sample_rate=48_000,
        channels=1,
        byte_stream=delayed_tail(),
    )
    try:
        assert await asyncio.wait_for(anext(decoded), timeout=0.1) == frame
    finally:
        release.set()
        await decoded.aclose()


@pytest.mark.asyncio
async def test_raw_pcm_rejects_an_oversized_provider_chunk_before_first_yield() -> None:
    decoded = iter_normalized_pcm_frames(
        audio_format="pcm",
        sample_rate=48_000,
        channels=1,
        byte_stream=_stream(bytes(1_000_000)),
    )

    try:
        with pytest.raises(PcmStreamError) as captured:
            await anext(decoded)
    finally:
        await decoded.aclose()

    assert captured.value.code == "invalid_audio_stream"


@pytest.mark.asyncio
async def test_raw_pcm_rejects_many_small_chunks_past_phrase_duration_cap() -> None:
    frame = b"\x34\x12" * 480
    small_chunk = frame * 64

    async def too_long() -> AsyncIterator[bytes]:
        for _ in range(47):
            yield small_chunk

    yielded = 0
    with pytest.raises(PcmStreamError) as captured:
        async for _ in iter_normalized_pcm_frames(
            audio_format="pcm",
            sample_rate=48_000,
            channels=1,
            byte_stream=too_long(),
        ):
            yielded += 1

    assert captured.value.code == "invalid_audio_stream"
    assert yielded == 46 * 64


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sample_rate", "channels"),
    ((1, 1), (48_000, 1_000_000), (8_000, 0), (48_001, 1), (8_000, 3)),
)
async def test_raw_pcm_rejects_unrealistic_metadata_before_consuming_bytes(
    sample_rate: int,
    channels: int,
) -> None:
    with pytest.raises(PcmStreamError) as captured:
        _ = [
            chunk
            async for chunk in iter_normalized_pcm_frames(
                audio_format="pcm",
                sample_rate=sample_rate,
                channels=channels,
                byte_stream=_must_not_consume(),
            )
        ]

    assert captured.value.code == "invalid_audio_stream"


@pytest.mark.asyncio
async def test_streaming_resampler_matches_whole_input_across_block_boundaries():
    from tldw_chatbook.Audio.voice_preprocessor import normalize_pcm16_frames

    samples = tuple(range(-240, 240))
    pcm = struct.pack(f"<{len(samples)}h", *samples)
    expected = normalize_pcm16_frames(pcm, sample_rate=24_000, channels=1)

    raw = tuple(
        [
            chunk
            async for chunk in iter_normalized_pcm_frames(
                audio_format="pcm",
                sample_rate=24_000,
                channels=1,
                byte_stream=_stream(pcm[:317], pcm[317:701], pcm[701:]),
            )
        ]
    )
    wav = tuple(
        [
            chunk
            async for chunk in iter_normalized_pcm_frames(
                audio_format="wav",
                sample_rate=None,
                channels=1,
                byte_stream=_stream(_wav_header(rate=24_000, data=pcm)),
            )
        ]
    )

    assert raw == expected
    assert wav == expected


@pytest.mark.asyncio
async def test_wav_fallback_decodes_stereo_and_normalizes_to_48k_mono():
    stereo_silence = b"\xe8\x03\x18\xfc" * 240
    wav = _wav_header(rate=24_000, channels=2, data=stereo_silence)

    normalized = [
        chunk
        async for chunk in iter_normalized_pcm_frames(
            audio_format="wav",
            sample_rate=None,
            channels=1,
            byte_stream=_stream(wav[:31], wav[31:]),
        )
    ]

    assert normalized == [bytes(960)]


@pytest.mark.asyncio
async def test_wav_fallback_zero_pads_a_partial_final_ten_ms_frame():
    five_ms_silence = bytes(120 * 2)
    wav = _wav_header(rate=24_000, channels=1, data=five_ms_silence)

    normalized = [
        chunk
        async for chunk in iter_normalized_pcm_frames(
            audio_format="wav",
            sample_rate=None,
            channels=1,
            byte_stream=_stream(wav),
        )
    ]

    assert normalized == [bytes(960)]


@pytest.mark.asyncio
async def test_wav_fallback_rejects_a_malformed_partial_sample_frame():
    wav = _wav_header(rate=24_000, channels=1, data=b"\x00")

    with pytest.raises(PcmStreamError) as captured:
        _ = [
            chunk
            async for chunk in iter_normalized_pcm_frames(
                audio_format="wav",
                sample_rate=None,
                channels=1,
                byte_stream=_stream(wav),
            )
        ]

    assert captured.value.code == "invalid_audio_stream"


@pytest.mark.asyncio
async def test_unsupported_audio_format_is_a_typed_recoverable_failure():
    with pytest.raises(PcmStreamError) as captured:
        _ = [
            chunk
            async for chunk in iter_normalized_pcm_frames(
                audio_format="mp3",
                sample_rate=24_000,
                channels=1,
                byte_stream=_stream(b"compressed"),
            )
        ]

    assert captured.value.code == "unsupported_audio_format"
    assert captured.value.recoverable is True


@pytest.mark.asyncio
async def test_truncated_wav_is_a_typed_recoverable_failure():
    with pytest.raises(PcmStreamError) as captured:
        _ = [
            chunk
            async for chunk in iter_normalized_pcm_frames(
                audio_format="wav",
                sample_rate=None,
                channels=1,
                byte_stream=_stream(b"RIFFgarbage"),
            )
        ]

    assert captured.value.code == "invalid_audio_stream"
    assert captured.value.recoverable is True
