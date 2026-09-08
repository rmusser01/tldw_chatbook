"""Containerize known PCM for file playback without changing its source artifact."""

from __future__ import annotations

import os
import wave
from pathlib import Path

from tldw_chatbook.Utils.secure_temp_files import (
    create_secure_temp_file,
    secure_delete_file,
)

_COPY_BYTES = 64 * 1024


def create_pcm16_wav_copy(
    source: Path,
    sample_rate: object,
    channels: object = 1,
) -> Path:
    """Copy declared PCM16 into an owner-only WAV file using bounded reads.

    Args:
        source: Complete caller-owned raw PCM artifact, retained while copying.
        sample_rate: Explicit response rate; unknown or invalid rates are refused.
        channels: Declared mono or stereo channel count.

    Returns:
        Temporary WAV path. The caller owns its cleanup and the source is unchanged.

    Raises:
        ValueError: If sample metadata or the complete PCM frame shape is invalid.
        OSError: If an owned artifact cannot be read or written.
    """
    if (
        type(sample_rate) is not int
        or type(channels) is not int
        or channels not in (1, 2)
        or not 0 < sample_rate <= 0xFFFFFFFF // (2 * channels)
    ):
        raise ValueError("PCM playback requires a known sample rate and channel count")
    with source.open("rb") as raw:
        size = os.fstat(raw.fileno()).st_size
        if not 0 < size <= 0xFFFFFFFF - 36 or size % (2 * channels):
            raise ValueError("PCM playback requires complete signed 16-bit frames")
        destination = Path(
            create_secure_temp_file(b"", suffix=".wav", prefix="tts_pcm_playback_")
        )
        try:
            with wave.open(str(destination), "wb") as output:
                output.setnchannels(channels)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                remaining = size
                while remaining:
                    chunk = raw.read(min(_COPY_BYTES, remaining))
                    if not chunk:
                        raise ValueError("PCM source changed before playback")
                    output.writeframesraw(chunk)
                    remaining -= len(chunk)
                if raw.read(1):
                    raise ValueError("PCM source changed before playback")
            return destination
        except BaseException:
            secure_delete_file(destination)
            raise
