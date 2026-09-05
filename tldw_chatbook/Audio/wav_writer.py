"""Crash-safe WAV writing for meeting recordings.

The stdlib ``wave`` module writes its header only on ``close()``, so a crash
mid-meeting would leave an unreadable file. This writer puts a 44-byte
header with a zero data length first, appends raw PCM as it arrives, and
patches the header on close. A header still reading zero length marks an
unfinished file; ``patch_wav_header`` repairs it from the file size.

No Textual, no numpy: stdlib only.
"""
from __future__ import annotations

import struct
from pathlib import Path

HEADER_BYTES = 44


def wav_header(
    data_bytes: int,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Return a canonical 44-byte PCM WAV header for ``data_bytes`` of audio."""
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8,
        b"data",
        data_bytes,
    )


class PlaceholderWavWriter:
    """Append-only WAV writer whose header is finalised on ``close()``."""

    def __init__(
        self,
        path: Path,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> None:
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.bytes_written = 0
        self._closed = False
        self._handle = open(self.path, "wb")  # noqa: SIM115 - long-lived stream
        self._handle.write(
            wav_header(0, sample_rate=sample_rate, channels=channels, sample_width=sample_width)
        )

    def __enter__(self) -> "PlaceholderWavWriter":
        """Return the writer itself so ``with PlaceholderWavWriter(p) as w`` works."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Finalise the header on the way out, exception or not."""
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def audio_position_s(self) -> float:
        return self.bytes_written / float(self.sample_rate * self.channels * self.sample_width)

    def write(self, pcm: bytes) -> None:
        if self._closed:
            raise ValueError(f"{self.path.name} is closed")
        self._handle.write(pcm)
        self.bytes_written += len(pcm)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._handle.seek(0)
        self._handle.write(
            wav_header(
                self.bytes_written,
                sample_rate=self.sample_rate,
                channels=self.channels,
                sample_width=self.sample_width,
            )
        )
        self._handle.close()


def wav_needs_patch(path: Path) -> bool:
    """True when ``path`` has a placeholder header but audio bytes after it."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= HEADER_BYTES:
        return False
    with open(path, "rb") as handle:
        head = handle.read(HEADER_BYTES)
    if len(head) < HEADER_BYTES or head[:4] != b"RIFF" or head[36:40] != b"data":
        return False
    return int.from_bytes(head[40:44], "little") == 0


def patch_wav_header(
    path: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> int:
    """Rewrite the header from the file size; return the data byte count."""
    path = Path(path)
    data_bytes = max(0, path.stat().st_size - HEADER_BYTES)
    with open(path, "r+b") as handle:
        handle.seek(0)
        handle.write(
            wav_header(
                data_bytes, sample_rate=sample_rate, channels=channels, sample_width=sample_width
            )
        )
    return data_bytes
