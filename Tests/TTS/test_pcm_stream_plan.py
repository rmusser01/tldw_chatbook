import struct

from tldw_chatbook.TTS.pcm_stream import SinkPlan, sink_plan


def _wav_header(rate=22050, channels=1, data=b"\x00\x00" * 64):
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + \
        struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * channels * 2,
                    channels * 2, 16) + b"data" + struct.pack("<I", len(data))
    return hdr + data


def _wav_with_trailing_chunk(rate=22050, channels=1, data=None,
                              trailer_id=b"LIST", trailer_payload=b"INFOtest"):
    # A well-formed WAV whose RIFF size correctly accounts for a chunk placed
    # AFTER `data` -- a shape validate_pcm16_wav structurally accepts (it
    # only special-cases `fmt `/`data`; any other chunk id is skipped
    # generically regardless of position). Reproduces the reviewer's
    # trailing-LIST-chunk finding (task-3-review.md, I1).
    if data is None:
        data = bytes(range(128))
    trailer = trailer_id + struct.pack("<I", len(trailer_payload)) + trailer_payload
    declared_size = 36 + len(data) + len(trailer)
    hdr = b"RIFF" + struct.pack("<I", declared_size) + b"WAVEfmt " + \
        struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * channels * 2,
                    channels * 2, 16) + b"data" + struct.pack("<I", len(data))
    return hdr + data + trailer


def test_raw_pcm_with_rate_is_eligible():
    assert sink_plan("pcm", 24000, None) == SinkPlan(24000, 1, 0)


def test_raw_pcm_without_rate_is_not():
    assert sink_plan("pcm", None, None) is None


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
