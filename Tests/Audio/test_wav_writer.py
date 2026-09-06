"""Task 2: crash-safe WAV files (spec §4, §7)."""
from __future__ import annotations

import wave

import pytest

from tldw_chatbook.Audio.wav_writer import (
    PlaceholderWavWriter,
    patch_wav_header,
    wav_header,
    wav_needs_patch,
)

pytestmark = pytest.mark.unit

FRAME = b"\x10\x00" * 320  # 20 ms of a constant sample


def test_header_is_44_bytes_and_encodes_sizes():
    header = wav_header(640)
    assert len(header) == 44
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    assert int.from_bytes(header[40:44], "little") == 640
    assert int.from_bytes(header[4:8], "little") == 36 + 640


def test_writer_streams_and_patches_on_close(tmp_path):
    path = tmp_path / "mixed.wav"
    writer = PlaceholderWavWriter(path)
    writer.write(FRAME)
    writer.write(FRAME)
    assert writer.bytes_written == 1280
    assert writer.audio_position_s == pytest.approx(0.04)
    writer.close()
    assert writer.closed

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        assert handle.getnframes() == 640


def test_unclosed_file_is_detected_and_patched(tmp_path):
    path = tmp_path / "you.wav"
    writer = PlaceholderWavWriter(path)
    writer.write(FRAME)
    writer._handle.flush()  # simulate a crash: never close()

    assert wav_needs_patch(path)
    assert patch_wav_header(path) == 640
    assert not wav_needs_patch(path)
    with wave.open(str(path), "rb") as handle:
        assert handle.getnframes() == 320


def test_write_after_close_raises(tmp_path):
    writer = PlaceholderWavWriter(tmp_path / "x.wav")
    writer.close()
    with pytest.raises(ValueError):
        writer.write(FRAME)


def test_writer_is_a_context_manager_and_close_is_idempotent(tmp_path):
    """Qodo Q3: the handle must not depend on a caller remembering close()."""
    path = tmp_path / "ctx.wav"
    with PlaceholderWavWriter(path) as writer:
        writer.write(FRAME)
    assert writer.closed and not wav_needs_patch(path)
    with wave.open(str(path), "rb") as handle:
        assert handle.getnframes() == 320
    writer.close()  # idempotent


def test_writer_context_manager_closes_on_exception(tmp_path):
    path = tmp_path / "boom.wav"
    with pytest.raises(RuntimeError):
        with PlaceholderWavWriter(path) as writer:
            writer.write(FRAME)
            raise RuntimeError("crash mid-meeting")
    assert writer.closed and not wav_needs_patch(path)


def test_needs_patch_false_for_missing_or_tiny_file(tmp_path):
    assert not wav_needs_patch(tmp_path / "absent.wav")
    (tmp_path / "tiny.wav").write_bytes(b"RIFF")
    assert not wav_needs_patch(tmp_path / "tiny.wav")
