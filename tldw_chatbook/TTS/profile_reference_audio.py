"""Bounded, source-independent PCM16 WAV canonicalization for clone references."""

from __future__ import annotations

import os
import stat
import struct
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Literal, cast

from tldw_chatbook.TTS.profile_errors import ProfileValidationError
from tldw_chatbook.TTS.profile_reference_types import (
    MAX_REFERENCE_CANONICAL_BYTES,
    MAX_REFERENCE_DURATION_MS,
    MAX_REFERENCE_SAMPLE_RATE_HZ,
    MAX_REFERENCE_SOURCE_BYTES,
    MIN_REFERENCE_SAMPLE_RATE_HZ,
    REFERENCE_SAMPLE_ENCODING,
    CanonicalTTSCloneReference,
    validate_reference_text,
)

_READ_CHUNK_BYTES = 1024 * 1024
_RIFF_HEADER_BYTES = 12
_CHUNK_HEADER_BYTES = 8
_PCM16_WIDTH_BYTES = 2


def _invalid() -> ProfileValidationError:
    return ProfileValidationError("reference_invalid")


@dataclass(frozen=True, slots=True)
class TTSCloneReferenceAudioMetadata:
    """Validated public-safe metadata for one canonical reference WAV."""

    byte_length: int
    duration_ms: int
    sample_rate_hz: int
    channels: int
    sample_encoding: Literal["pcm_s16le"]


@dataclass(frozen=True, slots=True)
class _ParsedWave:
    metadata: TTSCloneReferenceAudioMetadata
    canonical_bytes: bytes


def _canonical_wave(
    *,
    channels: int,
    sample_rate_hz: int,
    frames: bytes,
) -> bytes:
    block_align = channels * _PCM16_WIDTH_BYTES
    fmt = struct.pack(
        "<HHIIHH",
        1,
        channels,
        sample_rate_hz,
        sample_rate_hz * block_align,
        block_align,
        16,
    )
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(frames))
        + frames
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _parse_wave(payload: bytes) -> _ParsedWave:
    parse_error: BaseException | None = None
    try:
        if (
            type(payload) is not bytes
            or len(payload) < _RIFF_HEADER_BYTES
            or len(payload) > MAX_REFERENCE_SOURCE_BYTES
            or payload[:4] != b"RIFF"
            or payload[8:12] != b"WAVE"
            or struct.unpack_from("<I", payload, 4)[0] != len(payload) - 8
        ):
            raise ValueError

        fmt_payload: bytes | None = None
        frames: bytes | None = None
        offset = _RIFF_HEADER_BYTES
        while offset < len(payload):
            if len(payload) - offset < _CHUNK_HEADER_BYTES:
                raise ValueError
            chunk_id = payload[offset : offset + 4]
            chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
            content_start = offset + _CHUNK_HEADER_BYTES
            content_end = content_start + chunk_size
            padded_end = content_end + (chunk_size & 1)
            if content_end < content_start or padded_end > len(payload):
                raise ValueError
            chunk_payload = payload[content_start:content_end]
            if chunk_id == b"fmt ":
                if fmt_payload is not None:
                    raise ValueError
                fmt_payload = chunk_payload
            elif chunk_id == b"data":
                if frames is not None:
                    raise ValueError
                frames = chunk_payload
            offset = padded_end
        if offset != len(payload) or fmt_payload is None or frames is None:
            raise ValueError
        if len(fmt_payload) != 16 or not frames:
            raise ValueError

        (
            encoding,
            channels,
            sample_rate_hz,
            byte_rate,
            block_align,
            bits_per_sample,
        ) = struct.unpack("<HHIIHH", fmt_payload)
        expected_block_align = channels * _PCM16_WIDTH_BYTES
        if (
            encoding != 1
            or channels not in (1, 2)
            or not MIN_REFERENCE_SAMPLE_RATE_HZ
            <= sample_rate_hz
            <= MAX_REFERENCE_SAMPLE_RATE_HZ
            or bits_per_sample != 16
            or block_align != expected_block_align
            or byte_rate != sample_rate_hz * expected_block_align
            or len(frames) % expected_block_align != 0
        ):
            raise ValueError
        frame_count = len(frames) // expected_block_align
        if frame_count <= 0:
            raise ValueError
        duration_ms = (frame_count * 1_000 + sample_rate_hz - 1) // sample_rate_hz
        if not 0 < duration_ms <= MAX_REFERENCE_DURATION_MS:
            raise ValueError

        canonical = _canonical_wave(
            channels=channels,
            sample_rate_hz=sample_rate_hz,
            frames=frames,
        )
        if len(canonical) > MAX_REFERENCE_CANONICAL_BYTES:
            raise ValueError
        metadata = TTSCloneReferenceAudioMetadata(
            byte_length=len(canonical),
            duration_ms=duration_ms,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            sample_encoding="pcm_s16le",
        )
        return _ParsedWave(metadata=metadata, canonical_bytes=canonical)
    except BaseException as error:
        parse_error = error
    assert parse_error is not None
    if not isinstance(parse_error, Exception):
        raise parse_error
    if isinstance(parse_error, ProfileValidationError):
        raise ProfileValidationError(parse_error.code)
    raise _invalid()


def _read_canonical_stream(stream: BinaryIO) -> bytes:
    error: BaseException | None = None
    payload: bytes | None = None
    try:
        parts: list[bytes] = []
        total = 0
        while True:
            chunk = stream.read(
                min(_READ_CHUNK_BYTES, MAX_REFERENCE_CANONICAL_BYTES + 1 - total)
            )
            if type(chunk) is not bytes:
                raise ValueError
            if not chunk:
                break
            parts.append(chunk)
            total += len(chunk)
            if total > MAX_REFERENCE_CANONICAL_BYTES:
                raise ValueError
        payload = b"".join(parts)
    except BaseException as caught_error:
        error = caught_error

    if error is not None and not isinstance(error, Exception):
        raise error
    if error is not None or payload is None:
        raise _invalid() from None
    return payload


def validate_canonical_reference_wav(
    payload: bytes | BinaryIO,
) -> TTSCloneReferenceAudioMetadata:
    """Validate and describe the exact metadata-free canonical WAV shape."""

    exact_payload = (
        payload
        if type(payload) is bytes
        else _read_canonical_stream(cast(BinaryIO, payload))
    )
    parsed = _parse_wave(exact_payload)
    if parsed.canonical_bytes != exact_payload:
        raise _invalid()
    return parsed.metadata


def _source_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _source_open_flags() -> int:
    return (
        int(getattr(os, "O_RDONLY", 0))
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NONBLOCK", 0))
        | int(getattr(os, "O_NOFOLLOW", 0))
        | int(getattr(os, "O_BINARY", 0))
    )


def _read_regular_source(source_path: Path) -> bytes:
    descriptor: int | None = None
    primary_error: BaseException | None = None
    close_error: BaseException | None = None
    payload: bytes | None = None
    try:
        path_state = os.lstat(source_path)
        path_identity = _source_identity(path_state)
        if (
            not stat.S_ISREG(path_state.st_mode)
            or path_state.st_size <= 0
            or path_state.st_size > MAX_REFERENCE_SOURCE_BYTES
        ):
            raise ValueError
        descriptor = os.open(source_path, _source_open_flags())
        descriptor_state = os.fstat(descriptor)
        if _source_identity(descriptor_state) != path_identity:
            raise ValueError

        parts: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES, MAX_REFERENCE_SOURCE_BYTES + 1 - total),
            )
            if not chunk:
                break
            parts.append(chunk)
            total += len(chunk)
            if total > MAX_REFERENCE_SOURCE_BYTES:
                raise ValueError
        if total != path_state.st_size:
            raise ValueError
        final_descriptor_state = os.fstat(descriptor)
        final_path_state = os.lstat(source_path)
        if (
            _source_identity(final_descriptor_state) != path_identity
            or _source_identity(final_path_state) != path_identity
        ):
            raise ValueError
        payload = b"".join(parts)
    except BaseException as caught_error:
        primary_error = caught_error

    if descriptor is not None:
        try:
            os.close(descriptor)
        except BaseException as caught_error:
            close_error = caught_error

    for pending_error in (primary_error, close_error):
        if pending_error is not None and not isinstance(pending_error, Exception):
            raise pending_error
    if primary_error is not None or close_error is not None or payload is None:
        raise _invalid() from None
    return payload


def canonicalize_reference_wav(
    source_path: Path,
    reference_text: str,
) -> CanonicalTTSCloneReference:
    """Pin and canonicalize one bounded regular WAV without retaining its path."""

    if not isinstance(source_path, Path):
        raise _invalid()
    text = validate_reference_text(reference_text)
    payload = _read_regular_source(source_path)
    parsed = _parse_wave(payload)
    metadata = parsed.metadata
    return CanonicalTTSCloneReference(
        wav_bytes=parsed.canonical_bytes,
        reference_text=text,
        sha256=sha256(parsed.canonical_bytes).hexdigest(),
        byte_length=metadata.byte_length,
        duration_ms=metadata.duration_ms,
        sample_rate_hz=metadata.sample_rate_hz,
        channels=metadata.channels,
        sample_encoding=REFERENCE_SAMPLE_ENCODING,
    )


__all__ = [
    "TTSCloneReferenceAudioMetadata",
    "canonicalize_reference_wav",
    "validate_canonical_reference_wav",
]
