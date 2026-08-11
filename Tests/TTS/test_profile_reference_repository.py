"""Atomic repository contracts for private clone-reference payloads."""

from __future__ import annotations

import asyncio
import sqlite3
import struct
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

import tldw_chatbook.TTS.profile_repository as profile_repository
import tldw_chatbook.TTS.profile_reference_storage as reference_storage
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_reference_types import CanonicalTTSCloneReference
from tldw_chatbook.TTS.profile_types import CharacterRef, TTSProfileDraft


PROFILE_A = UUID("20000000-0000-4000-8000-000000000001")
PROFILE_B = UUID("20000000-0000-4000-8000-000000000002")
REFERENCE_A = UUID("30000000-0000-4000-8000-000000000001")
REFERENCE_B = UUID("30000000-0000-4000-8000-000000000002")
REFERENCE_C = UUID("30000000-0000-4000-8000-000000000003")
NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


class _ControlFlow(BaseException):
    """Test-only operation-lane control flow."""


def _canonical(*, sample: int = 1, frames: int = 32) -> CanonicalTTSCloneReference:
    channels = 1
    sample_rate_hz = 16_000
    pcm = struct.pack("<h", sample) * frames
    byte_rate = sample_rate_hz * channels * 2
    fmt = struct.pack("<HHIIHH", 1, channels, sample_rate_hz, byte_rate, 2, 16)
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )
    wav = b"RIFF" + struct.pack("<I", len(body)) + body
    return CanonicalTTSCloneReference(
        wav_bytes=wav,
        reference_text=f"Reference {sample}",
        sha256=sha256(wav).hexdigest(),
        byte_length=len(wav),
        duration_ms=(frames * 1_000 + sample_rate_hz - 1) // sample_rate_hz,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        sample_encoding="pcm_s16le",
    )


def _draft(name: str) -> TTSProfileDraft:
    return TTSProfileDraft(
        display_name=name,
        provider_id="openai",
        model_id="tts-1",
        voice_id="alloy",
        response_format="mp3",
        speed=1.0,
        options=cast(Any, {}),
    )


class _UUIDSequence:
    def __init__(self, values: Iterator[UUID]) -> None:
        self._values = values

    def __call__(self) -> UUID:
        return next(self._values)


@asynccontextmanager
async def _opened_repository(
    path: Path,
    *,
    reference_ids: tuple[UUID, ...] = (REFERENCE_A, REFERENCE_B, REFERENCE_C),
) -> AsyncIterator[profile_repository.TTSProfileRepository]:
    repository = profile_repository.TTSProfileRepository(
        path,
        _clock=lambda: NOW,
        _uuid_factory=_UUIDSequence(iter(reference_ids)),
    )
    await repository.open()
    try:
        yield repository
    finally:
        await repository.close()


async def _create(
    repository: profile_repository.TTSProfileRepository,
    profile_id: UUID,
    name: str,
) -> tuple[int, int]:
    created = await repository.create_profile(_draft(name), profile_id)
    return created.generation, created.value.revision


def _assert_error(error: ProfileRepositoryError, code: str) -> None:
    assert type(error) is ProfileRepositoryError
    assert error.code == code
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.asyncio
async def test_attach_stream_read_and_metadata_only_profile_surfaces(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.sqlite3"
    canonical = _canonical()
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        attached = await repository.set_reference(
            PROFILE_A,
            canonical,
            expected_revision=revision,
            expected_generation=generation,
        )

        assert attached.value.revision == 2
        assert attached.value.reference is not None
        assert attached.value.reference.reference_id == REFERENCE_A
        exact = await repository.get_reference(
            PROFILE_A,
            expected_revision=2,
            expected_generation=generation,
        )
        assert exact.value.wav_bytes == canonical.wav_bytes
        assert exact.value.reference_text == canonical.reference_text
        assert exact.value.sha256 == canonical.sha256

        loaded = await repository.get_profile(PROFILE_A)
        page = await repository.list_profiles()
        assert loaded.value.reference == attached.value.reference
        assert page.value.profiles[0].reference == attached.value.reference

        character = CharacterRef(
            source="local", authority_id="library", character_id="narrator"
        )
        await repository.set_assignment(
            character,
            PROFILE_A,
            expected_generation=generation,
            expected_profile_revision=2,
            expected_current_profile_id=None,
            expected_profile=attached.value,
        )
        assigned = await repository.get_assigned_profile(character)
        assert assigned.value is not None
        assert assigned.value.profile.reference == attached.value.reference


@pytest.mark.asyncio
async def test_replace_uses_new_identity_and_remove_recovers_reference_free_profile(
    tmp_path: Path,
) -> None:
    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        first = await repository.set_reference(
            PROFILE_A,
            _canonical(sample=1),
            expected_revision=revision,
            expected_generation=generation,
        )
        second = await repository.set_reference(
            PROFILE_A,
            _canonical(sample=2),
            expected_revision=first.value.revision,
            expected_generation=generation,
        )

        assert first.value.reference is not None
        assert second.value.reference is not None
        assert first.value.reference.reference_id == REFERENCE_A
        assert second.value.reference.reference_id == REFERENCE_B
        assert second.value.revision == 3

        removed = await repository.remove_reference(
            PROFILE_A,
            expected_revision=3,
            expected_generation=generation,
        )
        assert removed.value.revision == 4
        assert removed.value.reference is None
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.get_reference(
                PROFILE_A,
                expected_revision=4,
                expected_generation=generation,
            )
        _assert_error(caught.value, "missing")


@pytest.mark.asyncio
async def test_profile_update_preserves_attached_reference_summary(
    tmp_path: Path,
) -> None:
    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        attached = await repository.set_reference(
            PROFILE_A,
            _canonical(),
            expected_revision=revision,
            expected_generation=generation,
        )

        updated = await repository.update_profile(
            PROFILE_A,
            attached.value.revision,
            _draft("Renamed"),
            expected_generation=generation,
        )

        assert updated.value.display_name == "Renamed"
        assert updated.value.revision == 3
        assert updated.value.reference == attached.value.reference


@pytest.mark.asyncio
async def test_reference_mutations_enforce_generation_revision_and_parent_presence(
    tmp_path: Path,
) -> None:
    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        for expected_revision, expected_generation, code in (
            (revision + 1, generation, "conflict"),
            (revision, generation + 1, "stale"),
        ):
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.set_reference(
                    PROFILE_A,
                    _canonical(),
                    expected_revision=expected_revision,
                    expected_generation=expected_generation,
                )
            _assert_error(caught.value, code)

        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.set_reference(
                PROFILE_B,
                _canonical(),
                expected_revision=1,
                expected_generation=generation,
            )
        _assert_error(caught.value, "missing")


@pytest.mark.asyncio
async def test_profile_delete_cascades_reference_row(tmp_path: Path) -> None:
    path = tmp_path / "profiles.sqlite3"
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        await repository.set_reference(
            PROFILE_A,
            _canonical(),
            expected_revision=revision,
            expected_generation=generation,
        )
        await repository.delete_profile(PROFILE_A, expected_generation=generation)

    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute(
                "SELECT count(*) FROM tts_profile_clone_references"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_count_and_byte_quotas_include_replacement_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = _canonical(sample=1, frames=16)
    larger = _canonical(sample=2, frames=32)
    monkeypatch.setattr(profile_repository, "MAX_REFERENCE_COUNT", 2)
    monkeypatch.setattr(
        profile_repository, "MAX_REFERENCE_TOTAL_BYTES", larger.byte_length
    )
    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        generation, revision_a = await _create(repository, PROFILE_A, "A")
        _, revision_b = await _create(repository, PROFILE_B, "B")
        first = await repository.set_reference(
            PROFILE_A,
            small,
            expected_revision=revision_a,
            expected_generation=generation,
        )
        replacement = await repository.set_reference(
            PROFILE_A,
            larger,
            expected_revision=first.value.revision,
            expected_generation=generation,
        )
        assert replacement.value.reference is not None
        assert replacement.value.reference.byte_length == larger.byte_length

        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.set_reference(
                PROFILE_B,
                small,
                expected_revision=revision_b,
                expected_generation=generation,
            )
        _assert_error(caught.value, "reference_quota")

        monkeypatch.setattr(profile_repository, "MAX_REFERENCE_COUNT", 1)
        monkeypatch.setattr(
            profile_repository,
            "MAX_REFERENCE_TOTAL_BYTES",
            larger.byte_length + small.byte_length,
        )
        with pytest.raises(ProfileRepositoryError) as count_caught:
            await repository.set_reference(
                PROFILE_B,
                small,
                expected_revision=revision_b,
                expected_generation=generation,
            )
        _assert_error(count_caught.value, "reference_quota")


@pytest.mark.asyncio
async def test_concurrent_reference_mutations_admit_only_one_revision(
    tmp_path: Path,
) -> None:
    async with _opened_repository(tmp_path / "profiles.sqlite3") as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")

        async def attach(sample: int) -> object:
            try:
                return await repository.set_reference(
                    PROFILE_A,
                    _canonical(sample=sample),
                    expected_revision=revision,
                    expected_generation=generation,
                )
            except ProfileRepositoryError as error:
                return error

        outcomes = await asyncio.gather(attach(1), attach(2))
        successes = [value for value in outcomes if not isinstance(value, Exception)]
        failures = [value for value in outcomes if isinstance(value, Exception)]
        assert len(successes) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], ProfileRepositoryError)
        _assert_error(failures[0], "conflict")
        loaded = await repository.get_profile(PROFILE_A)
        assert loaded.value.revision == 2
        assert loaded.value.reference is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [RuntimeError("PRIVATE write detail"), _ControlFlow()]
)
async def test_reference_write_failure_rolls_back_payload_and_parent_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    path = tmp_path / "profiles.sqlite3"

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(profile_repository, "write_reference_blob", fail_write)
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        if isinstance(failure, Exception):
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.set_reference(
                    PROFILE_A,
                    _canonical(),
                    expected_revision=revision,
                    expected_generation=generation,
                )
            _assert_error(caught.value, "operation_failed")
        else:
            with pytest.raises(_ControlFlow):
                await repository.set_reference(
                    PROFILE_A,
                    _canonical(),
                    expected_revision=revision,
                    expected_generation=generation,
                )
        loaded = await repository.get_profile(PROFILE_A)
        assert loaded.value.revision == 1
        assert loaded.value.reference is None

    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute(
                "SELECT count(*) FROM tts_profile_clone_references"
            ).fetchone()[0]
            == 0
        )
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_later_parent_failure_rolls_back_completed_blob_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profiles.sqlite3"

    def fail_parent(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("PRIVATE parent failure")

    monkeypatch.setattr(
        profile_repository.TTSProfileRepository,
        "_worker_bump_reference_revision",
        fail_parent,
    )
    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.set_reference(
                PROFILE_A,
                _canonical(),
                expected_revision=revision,
                expected_generation=generation,
            )
        _assert_error(caught.value, "operation_failed")

    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute(
                "SELECT count(*) FROM tts_profile_clone_references"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT revision FROM tts_generation_profiles"
            ).fetchone()[0]
            == 1
        )
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_ordinary_reads_never_project_sensitive_reference_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profiles.sqlite3"
    observed_columns: list[set[str]] = []
    real_profile_decode = profile_repository._decode_profile_with_reference_row
    real_assigned_decode = profile_repository._decode_assigned_with_reference_row

    def inspect_profile(row: sqlite3.Row) -> object:
        observed_columns.append(set(row.keys()))
        return real_profile_decode(row)

    def inspect_assigned(row: sqlite3.Row) -> object:
        observed_columns.append(set(row.keys()))
        return real_assigned_decode(row)

    async with _opened_repository(path) as repository:
        generation, revision = await _create(repository, PROFILE_A, "Narrator")
        attached = await repository.set_reference(
            PROFILE_A,
            _canonical(),
            expected_revision=revision,
            expected_generation=generation,
        )
        character = CharacterRef("local", "library", "narrator")
        await repository.set_assignment(
            character,
            PROFILE_A,
            expected_generation=generation,
            expected_profile_revision=attached.value.revision,
            expected_current_profile_id=None,
        )
        monkeypatch.setattr(
            profile_repository, "_decode_profile_with_reference_row", inspect_profile
        )
        monkeypatch.setattr(
            profile_repository, "_decode_assigned_with_reference_row", inspect_assigned
        )
        monkeypatch.setattr(
            profile_repository,
            "read_reference_blob",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("ordinary read opened a reference BLOB")
            ),
        )

        await repository.get_profile(PROFILE_A)
        await repository.list_profiles()
        await repository.get_assigned_profile(character)

    assert observed_columns
    for columns in observed_columns:
        assert "wav_bytes" not in columns
        assert "reference_text" not in columns
        assert "sha256" not in columns


class _RecordingBlob:
    def __init__(
        self,
        blob: sqlite3.Blob,
        writes: list[int],
        reads: list[int],
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self._blob = blob
        self._writes = writes
        self._reads = reads
        self._close_error = close_error
        self.closed = False

    def __len__(self) -> int:
        return len(self._blob)

    def write(self, payload: bytes) -> None:
        self._writes.append(len(payload))
        self._blob.write(payload)

    def read(self, size: int = -1) -> bytes:
        self._reads.append(size)
        return self._blob.read(size)

    def tell(self) -> int:
        return self._blob.tell()

    def close(self) -> None:
        self._blob.close()
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _RecordingBlobConnection:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self.connection = connection
        self.close_error = close_error
        self.writes: list[int] = []
        self.reads: list[int] = []
        self.blobs: list[_RecordingBlob] = []

    def blobopen(self, *args: object, **kwargs: object) -> _RecordingBlob:
        blob = _RecordingBlob(
            self.connection.blobopen(*args, **kwargs),
            self.writes,
            self.reads,
            close_error=self.close_error,
        )
        self.blobs.append(blob)
        return blob


def test_blob_seam_streams_bounded_chunks_and_closes_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE tts_profile_clone_references(profile_id TEXT, wav_bytes BLOB)"
    )
    payload = b"0123456789abcdefghijklmnopqrstuvwxyz"
    cursor = connection.execute(
        "INSERT INTO tts_profile_clone_references VALUES ('p', zeroblob(?))",
        (len(payload),),
    )
    rowid = cursor.lastrowid
    assert type(rowid) is int
    recording = _RecordingBlobConnection(connection)
    monkeypatch.setattr(reference_storage, "REFERENCE_BLOB_CHUNK_BYTES", 7)

    reference_storage.write_reference_blob(cast(Any, recording), rowid, payload)
    restored = reference_storage.read_reference_blob(
        cast(Any, recording), rowid, len(payload)
    )

    assert restored == payload
    assert recording.writes == [7, 7, 7, 7, 7, 1]
    assert recording.reads == [7, 7, 7, 7, 7, 1, 1]
    assert all(blob.closed for blob in recording.blobs)
    connection.close()


@pytest.mark.parametrize("operation", ["read", "write"])
def test_blob_close_failure_is_safe_and_bounded(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE tts_profile_clone_references(profile_id TEXT, wav_bytes BLOB)"
    )
    payload = b"bounded payload"
    rowid = connection.execute(
        "INSERT INTO tts_profile_clone_references VALUES ('p', zeroblob(?))",
        (len(payload),),
    ).lastrowid
    assert type(rowid) is int
    recording = _RecordingBlobConnection(
        connection, close_error=RuntimeError("PRIVATE blob close detail")
    )
    monkeypatch.setattr(reference_storage, "REFERENCE_BLOB_CHUNK_BYTES", 7)

    with pytest.raises(ProfileRepositoryError) as caught:
        if operation == "write":
            reference_storage.write_reference_blob(cast(Any, recording), rowid, payload)
        else:
            reference_storage.read_reference_blob(
                cast(Any, recording), rowid, len(payload)
            )

    _assert_error(
        caught.value,
        "operation_failed" if operation == "write" else "reference_unavailable",
    )
    connection.close()
