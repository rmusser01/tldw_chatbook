"""Tests for bounded no-follow clone-reference WAV canonicalization."""

from __future__ import annotations

import os
import struct
from io import BytesIO
from hashlib import sha256
from pathlib import Path

import pytest

import tldw_chatbook.TTS.profile_reference_audio as reference_audio
from tldw_chatbook.TTS.profile_errors import ProfileValidationError
from tldw_chatbook.TTS.profile_reference_audio import (
    canonicalize_reference_wav,
    validate_canonical_reference_wav,
)
from tldw_chatbook.TTS.profile_reference_types import (
    MAX_REFERENCE_CANONICAL_BYTES,
    MAX_REFERENCE_SOURCE_BYTES,
)


def _chunk(name: bytes, payload: bytes) -> bytes:
    assert len(name) == 4
    return (
        name
        + struct.pack("<I", len(payload))
        + payload
        + (b"\x00" if len(payload) % 2 else b"")
    )


def _fmt(
    *, channels: int = 1, rate: int = 24_000, width: int = 2, encoding: int = 1
) -> bytes:
    block_align = channels * width
    return struct.pack(
        "<HHIIHH",
        encoding,
        channels,
        rate,
        rate * block_align,
        block_align,
        width * 8,
    )


def _riff(*chunks: bytes, declared_size: int | None = None) -> bytes:
    body = b"WAVE" + b"".join(chunks)
    size = len(body) if declared_size is None else declared_size
    return b"RIFF" + struct.pack("<I", size) + body


def _pcm_frames(*, channels: int = 1, frames: int = 2_400) -> bytes:
    frame = struct.pack("<h", 1_000) * channels
    return frame * frames


def _valid_wav(*, channels: int = 1, rate: int = 24_000, frames: int = 2_400) -> bytes:
    return _riff(
        _chunk(b"fmt ", _fmt(channels=channels, rate=rate)),
        _chunk(b"data", _pcm_frames(channels=channels, frames=frames)),
    )


def _safe_error() -> pytest.RaisesExc[ProfileValidationError]:
    return pytest.raises(
        ProfileValidationError,
        match=r"^TTS profile validation failed: reference_invalid$",
    )


def test_canonicalizer_strips_metadata_and_normalizes_chunk_order(
    tmp_path: Path,
) -> None:
    frames = _pcm_frames()
    source = _riff(
        _chunk(b"JUNK", b"private/path/metadata"),
        _chunk(b"data", frames),
        _chunk(b"LIST", b"private transcript metadata"),
        _chunk(b"fmt ", _fmt()),
    )
    source_path = tmp_path / "reference.wav"
    source_path.write_bytes(source)

    canonical = canonicalize_reference_wav(source_path, "  Exact transcript.  ")

    expected = _valid_wav()
    assert canonical.wav_bytes == expected
    assert canonical.sha256 == sha256(expected).hexdigest()
    assert canonical.byte_length == len(expected)
    assert canonical.duration_ms == 100
    assert canonical.sample_rate_hz == 24_000
    assert canonical.channels == 1
    assert canonical.sample_encoding == "pcm_s16le"
    assert canonical.reference_text == "Exact transcript."
    assert b"private" not in canonical.wav_bytes
    assert repr(canonical) == "CanonicalTTSCloneReference(<private>)"


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("rate", [8_000, 24_000, 48_000, 96_000])
def test_canonical_validator_accepts_supported_pcm16_shapes(
    channels: int, rate: int
) -> None:
    payload = _valid_wav(channels=channels, rate=rate, frames=rate // 10)

    metadata = validate_canonical_reference_wav(payload)

    assert metadata.byte_length == len(payload)
    assert metadata.duration_ms == 100
    assert metadata.sample_rate_hz == rate
    assert metadata.channels == channels
    assert metadata.sample_encoding == "pcm_s16le"


def test_canonical_validator_accepts_a_binary_stream() -> None:
    payload = _valid_wav()

    metadata = validate_canonical_reference_wav(BytesIO(payload))

    assert metadata.byte_length == len(payload)


def test_canonicalizer_accepts_nonzero_unknown_chunk_padding(tmp_path: Path) -> None:
    source_path = tmp_path / "reference.wav"
    odd_unknown_chunk = b"JUNK" + struct.pack("<I", 1) + b"x" + b"\xff"
    source_path.write_bytes(
        _riff(
            odd_unknown_chunk,
            _chunk(b"fmt ", _fmt()),
            _chunk(b"data", _pcm_frames()),
        )
    )

    canonical = canonicalize_reference_wav(source_path, "Text")

    assert canonical.wav_bytes == _valid_wav()


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"NOPE" + _valid_wav()[4:],
        _valid_wav()[:-1],
        _riff(_chunk(b"fmt ", _fmt())),
        _riff(_chunk(b"data", _pcm_frames())),
        _riff(
            _chunk(b"fmt ", _fmt()),
            _chunk(b"fmt ", _fmt()),
            _chunk(b"data", _pcm_frames()),
        ),
        _riff(
            _chunk(b"fmt ", _fmt()),
            _chunk(b"data", _pcm_frames()),
            _chunk(b"data", b"\x00\x00"),
        ),
        _riff(_chunk(b"fmt ", _fmt(width=1)), _chunk(b"data", b"\x00" * 2_400)),
        _riff(_chunk(b"fmt ", _fmt(encoding=3)), _chunk(b"data", _pcm_frames())),
        _riff(
            _chunk(b"fmt ", _fmt(channels=3)), _chunk(b"data", _pcm_frames(channels=3))
        ),
        _riff(_chunk(b"fmt ", _fmt(rate=7_999)), _chunk(b"data", _pcm_frames())),
        _riff(_chunk(b"fmt ", _fmt()), _chunk(b"data", b"\x00")),
        _riff(_chunk(b"fmt ", _fmt()), _chunk(b"data", b"")),
        _riff(_chunk(b"fmt ", _fmt()), _chunk(b"data", _pcm_frames()), declared_size=4),
        _valid_wav() + b"trailing",
    ],
)
def test_canonical_validator_rejects_malformed_or_unsupported_wav(
    payload: bytes,
) -> None:
    with _safe_error():
        validate_canonical_reference_wav(payload)


def test_canonicalizer_rejects_source_over_global_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "large.wav"
    source_path.write_bytes(_valid_wav())
    real_lstat = reference_audio.os.lstat

    def oversized(path: os.PathLike[str] | str) -> os.stat_result:
        result = real_lstat(path)
        values = list(result)
        values[6] = MAX_REFERENCE_SOURCE_BYTES + 1
        return os.stat_result(values)

    monkeypatch.setattr(reference_audio.os, "lstat", oversized)

    with _safe_error():
        canonicalize_reference_wav(source_path, "Text")


def test_canonical_validator_rejects_canonical_payload_over_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_wav()
    monkeypatch.setattr(
        reference_audio, "MAX_REFERENCE_CANONICAL_BYTES", len(payload) - 1
    )

    with _safe_error():
        validate_canonical_reference_wav(payload)


def test_canonicalizer_rejects_symlink_directory_and_fifo(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(_valid_wav())
    symlink = tmp_path / "link.wav"
    symlink.symlink_to(source)
    directory = tmp_path / "directory.wav"
    directory.mkdir()
    candidates = [symlink, directory]
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "fifo.wav"
        os.mkfifo(fifo)
        candidates.append(fifo)

    for candidate in candidates:
        with _safe_error():
            canonicalize_reference_wav(candidate, "Text")


def test_canonicalizer_rejects_source_replaced_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "reference.wav"
    source_path.write_bytes(_valid_wav())
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(_valid_wav(channels=2))
    real_read = reference_audio.os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        payload = real_read(descriptor, size)
        if not replaced:
            replaced = True
            os.replace(replacement, source_path)
        return payload

    monkeypatch.setattr(reference_audio.os, "read", replacing_read)

    with _safe_error():
        canonicalize_reference_wav(source_path, "Text")


def test_canonicalizer_rejects_source_mutated_in_place_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "reference.wav"
    source_path.write_bytes(_valid_wav())
    real_read = reference_audio.os.read
    mutated = False

    def mutating_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        payload = real_read(descriptor, size)
        if payload and not mutated:
            mutated = True
            with source_path.open("r+b") as source:
                source.seek(-1, os.SEEK_END)
                final_byte = source.read(1)
                source.seek(-1, os.SEEK_END)
                source.write(bytes((final_byte[0] ^ 0xFF,)))
                source.flush()
                os.fsync(source.fileno())
        return payload

    monkeypatch.setattr(reference_audio.os, "read", mutating_read)

    with _safe_error():
        canonicalize_reference_wav(source_path, "Text")


def test_canonicalizer_public_error_graph_contains_no_source_path(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "PRIVATE-reference-name.wav"
    source_path.write_bytes(b"not a wav")

    with _safe_error() as caught:
        canonicalize_reference_wav(source_path, "PRIVATE transcript")

    error: BaseException | None = caught.value
    while error is not None:
        rendered = f"{error!r} {error}"
        assert str(source_path) not in rendered
        assert "PRIVATE" not in rendered
        error = error.__cause__ or error.__context__


def test_canonicalizer_preserves_control_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "reference.wav"
    source_path.write_bytes(_valid_wav())

    def cancelled(*_args: object) -> bytes:
        raise KeyboardInterrupt

    monkeypatch.setattr(reference_audio.os, "read", cancelled)

    with pytest.raises(KeyboardInterrupt):
        canonicalize_reference_wav(source_path, "Text")


def test_canonicalizer_closes_descriptor_after_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "reference.wav"
    source_path.write_bytes(_valid_wav())
    real_close = reference_audio.os.close
    closed: list[int] = []

    def failed_read(*_args: object) -> bytes:
        raise OSError("PRIVATE source failure")

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(reference_audio.os, "read", failed_read)
    monkeypatch.setattr(reference_audio.os, "close", tracked_close)

    with _safe_error():
        canonicalize_reference_wav(source_path, "Text")

    assert len(closed) == 1


def test_test_fixture_stays_below_product_bounds() -> None:
    assert len(_valid_wav()) < MAX_REFERENCE_CANONICAL_BYTES
