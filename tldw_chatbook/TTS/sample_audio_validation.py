"""Bounded structural validation for completed TTS audio samples."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from tldw_chatbook.Utils.optional_deps import get_safe_import
from tldw_chatbook.Utils.path_validation import validate_path

MAX_PLAYABLE_AUDIO_BYTES = 8 * 1024 * 1024

CONTENT_TYPES_BY_FORMAT = MappingProxyType(
    {
        "aac": frozenset({"audio/aac"}),
        "flac": frozenset({"audio/flac"}),
        "mp3": frozenset({"audio/mpeg", "audio/mp3"}),
        "opus": frozenset({"audio/ogg", "audio/opus"}),
        "pcm": frozenset({"audio/l16", "audio/pcm"}),
        "wav": frozenset({"audio/wav", "audio/wave", "audio/x-wav"}),
    }
)


@dataclass(frozen=True, slots=True)
class ValidatedPlayableAudio:
    """Immutable identity of bytes validated from one stable regular file."""

    byte_length: int
    sha256: str


def wav_has_complete_frames(body: bytes) -> bool:
    if (
        len(body) < 44
        or body[:4] != b"RIFF"
        or body[8:12] != b"WAVE"
        or int.from_bytes(body[4:8], "little") + 8 != len(body)
    ):
        return False
    try:
        with wave.open(io.BytesIO(body), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            frame_rate = audio.getframerate()
            frame_count = audio.getnframes()
            if (
                audio.getcomptype() != "NONE"
                or channels <= 0
                or sample_width <= 0
                or frame_rate <= 0
                or frame_count <= 0
            ):
                return False
            frames = audio.readframes(frame_count)
            return len(frames) == frame_count * channels * sample_width
    except (EOFError, wave.Error):
        return False


def compressed_audio_has_decodable_frame(body: bytes, response_format: str) -> bool:
    """Decode at most one bounded audio frame, failing closed without PyAV."""

    av = get_safe_import("av", "av")
    if av is None:
        return False

    container_format, expected_codecs = {
        "mp3": ("mp3", frozenset({"mp3", "mp3float"})),
        "opus": ("ogg", frozenset({"opus"})),
        "flac": ("flac", frozenset({"flac"})),
        "aac": ("aac", frozenset({"aac"})),
    }[response_format]
    try:
        with av.open(
            io.BytesIO(body),
            mode="r",
            format=container_format,
        ) as container:
            streams = tuple(container.streams.audio)
            if len(streams) != 1:
                return False
            stream = streams[0]
            codec_name = str(getattr(stream.codec_context, "name", "")).lower()
            if codec_name not in expected_codecs:
                return False
            for packet_index, packet in enumerate(container.demux(stream)):
                if packet_index >= 64:
                    return False
                for frame in packet.decode():
                    sample_rate = getattr(frame, "sample_rate", 0)
                    samples = getattr(frame, "samples", 0)
                    layout = getattr(frame, "layout", None)
                    channels = len(getattr(layout, "channels", ()))
                    return bool(
                        type(sample_rate) is int
                        and 1 <= sample_rate <= 384_000
                        and type(samples) is int
                        and 1 <= samples <= sample_rate
                        and 1 <= channels <= 8
                    )
            return False
    except Exception:  # noqa: BLE001 - malformed or unsupported audio
        return False


def audio_body_matches_format(
    body: bytes,
    response_format: str,
    *,
    content_type: str | None = None,
    sample_rate_hz: int | None = None,
    channels: int | None = None,
    sample_width_bytes: int | None = None,
    max_bytes: int = MAX_PLAYABLE_AUDIO_BYTES,
) -> bool:
    """Return whether bounded bytes contain playable audio of the claimed type."""

    if (
        type(body) is not bytes
        or type(response_format) is not str
        or type(max_bytes) is not int
        or max_bytes <= 0
        or not 0 < len(body) <= max_bytes
    ):
        return False
    response_format = response_format.lower()
    if content_type is not None and (
        type(content_type) is not str
        or content_type.split(";", 1)[0].strip().lower()
        not in CONTENT_TYPES_BY_FORMAT.get(response_format, frozenset())
    ):
        return False
    if response_format == "wav":
        return wav_has_complete_frames(body)
    if response_format in {"mp3", "opus", "flac", "aac"}:
        return compressed_audio_has_decodable_frame(body, response_format)
    if response_format == "pcm":
        if (
            type(sample_rate_hz) is not int
            or not 1 <= sample_rate_hz <= 384_000
            or type(channels) is not int
            or not 1 <= channels <= 8
            or type(sample_width_bytes) is not int
            or sample_width_bytes not in {1, 2, 3, 4}
        ):
            return False
        frame_size = channels * sample_width_bytes
        return len(body) >= frame_size and len(body) % frame_size == 0
    return False


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return bool(
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _read_bounded_regular_file(
    path: Path,
    max_bytes: int,
) -> tuple[bytes, os.stat_result] | None:
    try:
        validate_path(
            path,
            path.parent,
            redact_paths=True,
            allow_hidden=True,
        )
    except (AttributeError, TypeError, ValueError):
        return None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= max_bytes:
            return None
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_file(before, opened):
            return None
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        body = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            not _same_file(opened, after)
            or len(body) != opened.st_size
            or not 0 < len(body) <= max_bytes
        ):
            return None
        return body, after
    except (OSError, ValueError):
        return None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def validate_playable_audio_file(
    path: Path,
    response_format: str,
    content_type: str,
    metadata: Mapping[str, object],
    *,
    max_bytes: int = MAX_PLAYABLE_AUDIO_BYTES,
) -> ValidatedPlayableAudio | None:
    """Validate one exact artifact-owned file without following replacements.

    Args:
        path: Absolute path to the generated audio artifact.
        response_format: Expected audio format and filename suffix.
        content_type: Expected response content type for the format.
        metadata: Provider metadata used to validate raw PCM parameters.
        max_bytes: Maximum artifact size accepted for bounded reading.

    Returns:
        The validated byte length and SHA-256 identity, or ``None`` when the
        path, file identity, size, metadata, or audio structure is invalid.
    """

    if (
        type(path) is not type(Path())
        or not path.is_absolute()
        or ".." in path.parts
        or type(response_format) is not str
        or path.suffix.lower() != f".{response_format.lower()}"
        or type(content_type) is not str
        or not isinstance(metadata, Mapping)
        or type(max_bytes) is not int
        or max_bytes <= 0
    ):
        return None
    read = _read_bounded_regular_file(path, max_bytes)
    if read is None:
        return None
    body, opened = read
    try:
        current = os.lstat(path)
    except OSError:
        return None
    if not _same_file(opened, current):
        return None
    sample_rate_hz = metadata.get("sample_rate")
    channels = metadata.get("channels")
    sample_width_bytes = metadata.get("sample_width_bytes")
    if sample_width_bytes is None:
        bits_per_sample = metadata.get("bits_per_sample")
        if type(bits_per_sample) is int and bits_per_sample % 8 == 0:
            sample_width_bytes = bits_per_sample // 8
    if not audio_body_matches_format(
        body,
        response_format,
        content_type=content_type,
        sample_rate_hz=(sample_rate_hz if type(sample_rate_hz) is int else None),
        channels=channels if type(channels) is int else None,
        sample_width_bytes=(
            sample_width_bytes if type(sample_width_bytes) is int else None
        ),
        max_bytes=max_bytes,
    ):
        return None
    return ValidatedPlayableAudio(
        byte_length=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
    )
