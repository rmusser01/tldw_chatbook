import struct

from tldw_chatbook.TTS.pcm_stream import SinkPlan, sink_plan


def _wav_header(rate=22050, channels=1, data=b"\x00\x00" * 64):
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + \
        struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * channels * 2,
                    channels * 2, 16) + b"data" + struct.pack("<I", len(data))
    return hdr + data


def test_raw_pcm_with_rate_is_eligible():
    assert sink_plan("pcm", 24000, None) == SinkPlan(24000, 1, 0)


def test_raw_pcm_without_rate_is_not():
    assert sink_plan("pcm", None, None) is None


def test_valid_pcm16_wav_is_eligible_with_header_skip():
    plan = sink_plan("wav", None, _wav_header())
    assert plan is not None
    assert plan.sample_rate == 22050 and plan.channels == 1 and plan.skip_bytes == 44


def test_invalid_wav_falls_back():
    assert sink_plan("wav", None, b"RIFFgarbage") is None


def test_compressed_formats_fall_back():
    for fmt in ("mp3", "opus", "aac", "flac", ""):
        assert sink_plan(fmt, 24000, None) is None
