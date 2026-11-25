"""Admission for the qualified Windows default-render/ffplay WAV profile."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import struct
import sys
from typing import BinaryIO

from tldw_chatbook.Utils.windows_files import WindowsReadPin

RATE = 48000
PADDING_FRAMES = 12000
POST_REAP_SETTLE_S = 0.250
PADDING_POLICY = "pcm16-stereo-48k-zero-tail-250ms-v1"
QUALIFIED_SHA256 = "f6536c4caf9bac0fee1e8ffaafa5b13b6ad3e386a9a9f0b6613360e8c6d59919"
_ROUTE_KEYS = ("SDL_AUDIODRIVER", "SDL_AUDIO_DEVICE_NAME", "AUDIODEV")


class ControlledWavError(ValueError):
    """Bounded diagnostic code safe to show without source paths or stderr."""


def wait_for_predecessor(previous: tuple[int, int]) -> None:
    """Observe the exact local process and wait for its qualified device tail."""
    import time
    from tldw_chatbook.Utils.windows_job import predecessor_retired

    try:
        confirmed = os.name == "nt" and predecessor_retired(*previous)
    except (OSError, RuntimeError):
        confirmed = False
    if not confirmed:
        raise ControlledWavError("previous_player_unconfirmed")
    time.sleep(POST_REAP_SETTLE_S)


@dataclass(frozen=True)
class RetainedWav:
    path: Path
    sha256: str
    sample_rate: int
    content_frames: int
    file_frames: int
    padding_policy: str = PADDING_POLICY


@dataclass(frozen=True)
class WavBookmark:
    sha256: str
    sample_rate: int
    frame: int


def check_route(env: dict[str, str]) -> None:
    if sys.platform != "win32":
        raise ControlledWavError("unsupported_platform")
    if any(env.get(key) for key in _ROUTE_KEYS):
        raise ControlledWavError("unsupported_audio_route")


def _pin_executable(explicit: Path | None) -> tuple[Path, WindowsReadPin]:
    check_route(dict(os.environ))
    install = Path(os.environ.get("ProgramData", "C:/ProgramData")) / "chocolatey"
    custom = Path(os.environ.get("ChocolateyInstall", str(install)))
    shim_dirs = {root / "bin" for root in (install, custom)}
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = [
            candidate
            for root in dict.fromkeys((custom, install))
            for candidate in sorted((root / "lib").glob("ffmpeg*/tools/**/ffplay.exe"))
        ]
        candidates.extend(
            Path(entry.strip('"')) / "ffplay.exe"
            for entry in os.get_exec_path()
            if entry
        )
    for candidate in dict.fromkeys(candidates):
        if (
            not isinstance(candidate, Path)
            or candidate.suffix.lower() != ".exe"
            or candidate.parent in shim_dirs
        ):
            continue
        pin = WindowsReadPin()
        matched = False
        try:
            pin.open(candidate)
            matched = (
                hashlib.file_digest(pin.stream, "sha256").hexdigest()
                == QUALIFIED_SHA256
            )
            if matched:
                # CreateProcessW accepts the device spelling of this exact
                # pinned volume. The WAV consumer keeps its extended spelling.
                return Path(
                    str(pin.physical_path).replace("\\\\?\\", "\\\\.\\", 1)
                ), pin
        except (OSError, ValueError):
            pass
        finally:
            if not matched and not pin.close():
                raise ControlledWavError("pin_retirement_unconfirmed")
    raise ControlledWavError("qualified_ffplay_unavailable")


async def controlled_wav_readiness(*, executable: Path | None = None, custody=None) -> str | None:
    """Return a bounded refusal code or None for the exact supported backend."""

    def check():
        try:
            path, pin = _pin_executable(executable)
            try:
                if custody is not None:
                    custody.recover_predecessor()
                    from ._controlled_wav_windows import check_contained_version

                    check_contained_version(path)
            finally:
                if not pin.close():
                    raise ControlledWavError("pin_retirement_unconfirmed")
            return None
        except ControlledWavError as error:
            return str(error)
        except (OSError, ValueError, RuntimeError):
            return (
                "player_containment_unavailable"
                if custody is not None
                else "qualified_ffplay_unavailable"
            )

    return await asyncio.to_thread(check)


def _check_wave(stream: BinaryIO, frames: int) -> None:
    """Check actual RIFF extents, PCM sample layout and every padding byte."""
    length = stream.seek(0, os.SEEK_END)
    stream.seek(0)
    magic, declared, kind = struct.unpack("<4sI4s", stream.read(12))
    valid = magic == b"RIFF" and kind == b"WAVE" and declared + 8 == length
    found_format = found_data = False
    while valid and stream.tell() < length:
        chunk, size = struct.unpack("<4sI", stream.read(8))
        start = stream.tell()
        end = start + size + (size % 2)
        if end > length:
            break
        if chunk == b"fmt ":
            if found_format or size not in (16, 18):
                break
            # PCM tag, stereo, sample rate, byte rate, block alignment, bit depth.
            params = struct.unpack("<HHIIHH", stream.read(16))
            if params != (1, 2, RATE, RATE * 4, 4, 16) or (
                size == 18 and stream.read(2) != bytes(2)
            ):
                break
            found_format = True
        elif chunk == b"data":
            if not found_format or found_data or size != frames * 4:
                break
            stream.seek(start + size - PADDING_FRAMES * 4)
            if stream.read(PADDING_FRAMES * 4) != bytes(PADDING_FRAMES * 4):
                break
            found_data = True
        stream.seek(end)
    else:
        if valid and found_format and found_data:
            return
    raise ControlledWavError("invalid_retained_wave")


# Storage validates the same bytes using the physical frame count.
validate_controlled_wav = _check_wave


def _validate(
    artifact: RetainedWav, bookmark: WavBookmark, pin: WindowsReadPin
) -> None:
    if (
        not isinstance(artifact.path, Path)
        or not isinstance(artifact.sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", artifact.sha256) is None
        or artifact.sha256 != bookmark.sha256
        or any(
            type(n) is not int
            for n in (
                artifact.sample_rate,
                artifact.content_frames,
                artifact.file_frames,
                bookmark.sample_rate,
                bookmark.frame,
            )
        )
        or artifact.sample_rate != RATE
        or bookmark.sample_rate != RATE
        or artifact.content_frames <= 0
        or artifact.file_frames != artifact.content_frames + PADDING_FRAMES
        or artifact.padding_policy != PADDING_POLICY
        or not 0 <= bookmark.frame <= artifact.content_frames
    ):
        raise ControlledWavError("invalid_artifact_or_bookmark")
    pin.open(artifact.path)
    # Validate the existing private source without changing its ACL. Every
    # component is now pinned; the byte reads below use only the pinned leaf.
    pin.verify_private()
    if hashlib.file_digest(pin.stream, "sha256").hexdigest() != artifact.sha256:
        raise ControlledWavError("artifact_digest_mismatch")
    _check_wave(pin.stream, artifact.file_frames)


def prepare_files(
    artifact: RetainedWav, bookmark: WavBookmark, executable: Path | None
):
    """Acquire both immutable pins or clean up all partial acquisitions."""
    path, binary_pin = _pin_executable(executable)
    wav_pin = WindowsReadPin()
    try:
        _validate(artifact, bookmark, wav_pin)
        return path, binary_pin, wav_pin
    except Exception as error:
        wave_closed = wav_pin.close()
        binary_closed = binary_pin.close()
        if not (wave_closed and binary_closed):
            raise ControlledWavError("pin_retirement_unconfirmed") from None
        reason = (
            str(error)
            if isinstance(error, ControlledWavError)
            else "invalid_retained_wave"
        )
    raise ControlledWavError(reason)
