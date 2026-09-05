"""Private, transactional prompt-cache snapshot storage.

All methods are synchronous: the app service must call them off-thread. The
catalog lock covers every owned handle lifetime, including hashing and staging.
HTTP completion is supplied by the service, never inferred from disk contents.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal
from uuid import uuid4

import portalocker
from pydantic import BaseModel, ConfigDict, ValidationError

from tldw_chatbook.LLM_Management.snapshot_models import (
    COMPATIBILITY_STATE_KEYS,
    SAFE_ID_PATTERN,
    CatalogPage,
    CompatibilityEvidence,
    SaveResult,
    SlotReceipt,
    SnapshotError,
    SnapshotRecord,
    WorkingFile,
)
from tldw_chatbook.Utils.private_paths import (
    _open_verified_parent,
    atomic_private_write_text,
    create_private_text,
    lexical_path,
    open_private_binary,
    open_private_text_append_stream,
    secure_private_directory,
)

MAX_METADATA_BYTES = 65536
MAX_SCAN_ENTRIES = 10000
CHUNK_BYTES = 1024 * 1024
_UUID = re.compile(r"[0-9a-f]{32}\Z")
_MEMBER = re.compile(r"slot-(\d+)-(\d{8}T\d{6}Z)-([0-9a-f]{32})\.bin\Z")
OperationState = Literal["reserved", "unknown", "acknowledged", "terminal"]


class _Reservation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    launch_id: str
    operation_id: str
    kind: Literal["save", "restore"]
    filename: str
    device: int
    inode: int
    state: OperationState


def _error(code: str) -> SnapshotError:
    return SnapshotError(code, submission_possible=False)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _supported_platform() -> bool:
    """V1 requires the descriptor-relative POSIX safety boundary."""
    return (
        os.name == "posix"
        and bool(getattr(os, "O_NOFOLLOW", 0))
        and bool(getattr(os, "O_DIRECTORY", 0))
        and {os.open, os.stat, os.rename, os.unlink}.issubset(os.supports_dir_fd)
        and os.scandir in os.supports_fd
        and os.stat in os.supports_follow_symlinks
    )


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _validate_id(value: str) -> None:
    if (
        not isinstance(value, str)
        or SAFE_ID_PATTERN.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise _error("invalid_identifier")


def _check_member(path: Path, expected: tuple[int, int]) -> os.stat_result:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or _identity(info) != expected
        or (os.name == "posix" and info.st_uid != os.geteuid())
    ):
        raise _error("member_identity_changed")
    return info


@contextmanager
def _directory(path: Path) -> Iterator[int]:
    secure_private_directory(path, create=False, application_owned=True)
    before = path.lstat()
    # Reuse ADR-029's descriptor walk rather than reopening through an ancestor
    # that could have been substituted after the directory privacy check.
    fd, _ = _open_verified_parent(path / ".directory-check", missing_leaf_allowed=True)
    try:
        if _identity(os.fstat(fd)) != _identity(before) or _identity(
            path.lstat()
        ) != _identity(before):
            raise _error("directory_identity_changed")
        yield fd
    finally:
        os.close(fd)


def _sync_directory(path: Path) -> None:
    if os.name == "posix":
        with _directory(path) as fd:
            os.fsync(fd)


def _unlink_verified(path: Path, expected: tuple[int, int]) -> None:
    with _directory(path.parent) as fd:
        _check_member(path, expected)
        current = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
        if _identity(current) != expected:
            raise _error("member_identity_changed")
        os.unlink(path.name, dir_fd=fd)


def _move_verified(source: Path, target: Path, expected: tuple[int, int]) -> None:
    with _directory(source.parent) as source_fd, _directory(target.parent) as target_fd:
        _check_member(source, expected)
        try:
            os.stat(target.name, dir_fd=target_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise _error("target_exists")
        if (
            _identity(os.stat(source.name, dir_fd=source_fd, follow_symlinks=False))
            != expected
        ):
            raise _error("member_identity_changed")
        os.rename(source.name, target.name, src_dir_fd=source_fd, dst_dir_fd=target_fd)
        _check_member(target, expected)


def _flush_binary(stream: BinaryIO) -> None:
    stream.flush()
    os.fsync(stream.fileno())


def _write_chunk(stream: BinaryIO, chunk: bytes) -> int:
    return os.write(stream.fileno(), chunk)


def _read_json(path: Path) -> bytes:
    info = _check_member(path, _identity(path.lstat()))
    if os.name == "posix" and stat.S_IMODE(info.st_mode) != 0o600:
        # Unknown metadata is not yet an owned private file eligible for hardening.
        raise _error("invalid_metadata")
    with open_private_binary(path) as opened:
        if os.fstat(opened.stream.fileno()).st_size > MAX_METADATA_BYTES:
            raise _error("metadata_too_large")
        payload = opened.stream.read(MAX_METADATA_BYTES + 1)
        if len(payload) > MAX_METADATA_BYTES:
            raise _error("metadata_too_large")
        return payload


def _write_json(path: Path, text: str) -> None:
    if len(text.encode("utf-8")) > MAX_METADATA_BYTES:
        raise _error("metadata_too_large")
    atomic_private_write_text(path, text)


def _validate_evidence(evidence: CompatibilityEvidence) -> CompatibilityEvidence:
    value = CompatibilityEvidence.model_validate_json(evidence.model_dump_json())
    if {key for key, _ in value.state_settings} != COMPATIBILITY_STATE_KEYS:
        raise _error("compatibility_incomplete")
    return value


def _validate_metadata(record: SnapshotRecord, snapshot_id: str) -> None:
    """Establish schema-v1 ownership before catalog use or tombstone deletion."""
    member = _MEMBER.fullmatch(record.filename)
    if (
        record.schema_version != 1
        or _UUID.fullmatch(record.snapshot_id) is None
        or snapshot_id != record.snapshot_id
        or member is None
        or member[3] != record.snapshot_id
        or int(member[1]) != record.source_slot
        or record.bytes <= 0
        or record.tokens <= 0
    ):
        raise _error("invalid_metadata")
    datetime.strptime(record.created_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    _validate_evidence(record.compatibility)


class SnapshotStore:
    """One profile's private retained catalog and isolated launch working files."""

    def __init__(self, root: Path) -> None:
        if not _supported_platform():
            raise _error("unsupported_platform")
        self.root = lexical_path(root)
        self.catalog = self.root / "catalog"
        self.working = self.root / "working"
        for directory in (self.root, self.catalog, self.working):
            secure_private_directory(directory, create=True, application_owned=True)
        # Existing primitives report Windows ACLs as unverified, not private.
        self.privacy_verified = os.name == "posix"
        try:
            create_private_text(self.root / "catalog.lock", "")
        except FileExistsError:
            pass

    @contextmanager
    def _locked(self) -> Iterator[None]:
        try:
            with open_private_text_append_stream(self.root / "catalog.lock") as stream:
                portalocker.lock(stream, portalocker.LOCK_EX)
                try:
                    _check_member(
                        self.root / "catalog.lock", _identity(os.fstat(stream.fileno()))
                    )
                    yield
                finally:
                    portalocker.unlock(stream)
        except (OSError, ValueError, ValidationError):
            raise _error("storage_failed") from None

    def _reserve(
        self, launch_id: str, slot_id: int, source: SnapshotRecord | None
    ) -> WorkingFile:
        _validate_id(launch_id)
        if type(slot_id) is not int or slot_id < 0:
            raise _error("invalid_slot")
        directory = self.working / launch_id
        secure_private_directory(directory, create=True, application_owned=True)
        operation = uuid4().hex
        stamp = (
            datetime.strptime(_utc_now(), "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=UTC)
            .strftime("%Y%m%dT%H%M%SZ")
        )
        path = directory / f"slot-{slot_id}-{stamp}-{operation}.bin"
        create_private_text(path, "")
        info = path.lstat()
        manifest = _Reservation(
            launch_id=launch_id,
            operation_id=operation,
            kind="restore" if source else "save",
            filename=path.name,
            device=info.st_dev,
            inode=info.st_ino,
            state="reserved",
        )
        try:
            _write_json(directory / f"{operation}.json", manifest.model_dump_json())
        except BaseException:
            _unlink_verified(path, _identity(info))
            raise
        return WorkingFile(launch_id, operation, path, source)

    def reserve_save(self, launch_id: str, slot_id: int) -> WorkingFile:
        """Precreate a private server write target and durable owned reservation."""
        with self._locked():
            return self._reserve(launch_id, slot_id, None)

    def prepare_launch_directory(self, launch_id: str) -> Path:
        """Prepare the sole directory to expose to this launch's llama-server."""
        _validate_id(launch_id)
        with self._locked():
            directory = self.working / launch_id
            secure_private_directory(directory, create=True, application_owned=True)
            return directory

    def _reservation(self, working: WorkingFile) -> _Reservation:
        _validate_id(working.launch_id)
        if _UUID.fullmatch(working.operation_id) is None:
            raise _error("invalid_reservation")
        directory = self.working / working.launch_id
        manifest = _Reservation.model_validate_json(
            _read_json(directory / f"{working.operation_id}.json")
        )
        member = _MEMBER.fullmatch(manifest.filename)
        if (
            manifest.launch_id != working.launch_id
            or manifest.operation_id != working.operation_id
            or member is None
            or member[3] != working.operation_id
            or working.path != directory / manifest.filename
        ):
            raise _error("invalid_reservation")
        return manifest

    def _set_state(
        self, working: WorkingFile, manifest: _Reservation, state: OperationState
    ) -> None:
        transitions = {
            "reserved": {"unknown", "acknowledged", "terminal"},
            "unknown": {"acknowledged", "terminal"},
            "acknowledged": {"terminal"},
            "terminal": set(),
        }
        if state != manifest.state and state not in transitions[manifest.state]:
            raise _error("invalid_operation_state")
        updated = _Reservation.model_validate({**manifest.model_dump(), "state": state})
        _write_json(
            working.path.parent / f"{working.operation_id}.json",
            updated.model_dump_json(),
        )

    def set_operation_state(
        self,
        working: WorkingFile,
        state: Literal["unknown", "acknowledged", "terminal"],
    ) -> None:
        """Persist service-supplied completion knowledge before subsequent work."""
        with self._locked():
            self._set_state(working, self._reservation(working), state)

    def _record(self, path: Path) -> SnapshotRecord:
        record = SnapshotRecord.model_validate_json(_read_json(path))
        if path.suffix != ".json":
            raise _error("invalid_metadata")
        _validate_metadata(record, path.stem)
        with open_private_binary(self.catalog / record.filename) as binary:
            info = os.fstat(binary.stream.fileno())
            _check_member(self.catalog / record.filename, _identity(info))
        return record

    def _entries(self, directory: Path) -> tuple[list[Path], bool]:
        entries = []
        with (
            _directory(directory) as fd,
            os.scandir(fd if os.name == "posix" else directory) as iterator,
        ):
            for entry in iterator:
                if len(entries) >= MAX_SCAN_ENTRIES:
                    return entries, False
                entries.append(directory / entry.name)
        return entries, True

    def _scan(self) -> tuple[list[SnapshotRecord], bool, int]:
        records: list[SnapshotRecord] = []
        total = 0
        try:
            entries, complete = self._entries(self.catalog)
        except (OSError, SnapshotError):
            return records, False, 0
        for path in entries:
            try:
                info = path.lstat()
                if (
                    stat.S_ISREG(info.st_mode)
                    and info.st_nlink == 1
                    and path.suffix == ".bin"
                ):
                    total += info.st_size
                if path.suffix == ".json" and _UUID.fullmatch(path.stem):
                    record = self._record(path)
                    records.append(record)
                    if (self.catalog / record.filename).lstat().st_size != record.bytes:
                        complete = False
            except OSError:
                complete = False
            except (ValueError, SnapshotError):
                # Foreign/malformed records do not establish ownership.
                continue
        records.sort(
            key=lambda item: (item.publication_sequence, item.snapshot_id), reverse=True
        )
        return records, complete, max(0, total - sum(item.bytes for item in records))

    def _sequence(self, records: list[SnapshotRecord], complete: bool) -> int:
        try:
            value = json.loads(_read_json(self.root / "publication.json"))
            if (
                set(value) != {"publication_sequence"}
                or type(value["publication_sequence"]) is not int
                or value["publication_sequence"] < 0
            ):
                raise ValueError("invalid counter")
            previous = value["publication_sequence"]
        except (OSError, ValueError, TypeError, SnapshotError):
            if not complete:
                raise _error("ordering_unavailable") from None
            previous = 0
        sequence = (
            max(
                previous,
                max((record.publication_sequence for record in records), default=0),
            )
            + 1
        )
        _write_json(
            self.root / "publication.json",
            json.dumps({"publication_sequence": sequence}),
        )
        return sequence

    def commit_save(
        self,
        working: WorkingFile,
        receipt: SlotReceipt,
        evidence: CompatibilityEvidence,
        model_label: str,
        keep_count: int,
        *,
        validate_publication: Callable[[], bool] | None = None,
    ) -> SaveResult:
        """Durably publish acknowledged complete bytes, then prune by publication order."""
        with self._locked():
            if type(keep_count) is not int or not 1 <= keep_count <= 1000:
                raise _error("invalid_keep_count")
            evidence = _validate_evidence(evidence)
            manifest = self._reservation(working)
            member = _MEMBER.fullmatch(manifest.filename)
            if (
                manifest.kind != "save"
                or manifest.state == "terminal"
                or receipt.filename != manifest.filename
                or receipt.slot_id != int(member[1])
                or receipt.tokens <= 0
                or receipt.bytes <= 0
            ):
                raise _error("invalid_receipt")
            self._set_state(working, manifest, "acknowledged")
            with open_private_binary(working.path) as opened:
                stream = opened.stream
                info = _check_member(working.path, (manifest.device, manifest.inode))
                if (
                    _identity(os.fstat(stream.fileno())) != _identity(info)
                    or info.st_size != receipt.bytes
                ):
                    raise _error("integrity_mismatch")
                _flush_binary(stream)
                digest = hashlib.sha256()
                while chunk := stream.read(CHUNK_BYTES):
                    digest.update(chunk)
                after = _check_member(working.path, _identity(info))
                if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                ):
                    raise _error("member_identity_changed")
                records, complete, _ = self._scan()
                sequence = self._sequence(records, complete)
                record = SnapshotRecord(
                    snapshot_id=working.operation_id,
                    filename=manifest.filename,
                    created_utc=datetime.strptime(member[2], "%Y%m%dT%H%M%SZ")
                    .replace(tzinfo=UTC)
                    .strftime("%Y-%m-%dT%H:%M:%SZ"),
                    publication_sequence=sequence,
                    source_slot=receipt.slot_id,
                    tokens=receipt.tokens,
                    bytes=receipt.bytes,
                    sha256=digest.hexdigest(),
                    model_label=model_label,
                    compatibility=evidence,
                )
                # Validate/size-bound metadata BEFORE moving the acknowledged binary.
                metadata = record.model_dump_json()
                if len(metadata.encode()) > MAX_METADATA_BYTES:
                    raise _error("metadata_too_large")
                if validate_publication is not None:
                    admitted = False
                    try:
                        admitted = validate_publication() is True
                    except Exception:  # noqa: BLE001 - failed validation cannot authorize publication
                        admitted = False
                    if not admitted:
                        raise _error("publication_invalidated")
                _move_verified(
                    working.path, self.catalog / record.filename, _identity(info)
                )
                _write_json(self.catalog / f"{record.snapshot_id}.json", metadata)
                _sync_directory(self.catalog)
                _sync_directory(working.path.parent)
            removed, failed = [], []
            # Commit-before-prune: no failure above may reach this block.
            if complete:
                for old in records[max(0, keep_count - 1) :]:
                    failures = self._delete_record(old)
                    (failed if failures else removed).append(old.snapshot_id)
            else:
                failed.append("cleanup_incomplete")
            try:
                self._set_state(working, self._reservation(working), "terminal")
                if self._cleanup(working):
                    failed.append("cleanup_failed")
            except (OSError, ValueError, SnapshotError):
                failed.append("cleanup_failed")
            return SaveResult(record, tuple(removed), tuple(failed))

    def stage_restore(self, snapshot_id: str, launch_id: str) -> WorkingFile:
        """Copy and verify retained bytes under lock before handing them to the service."""
        _validate_id(snapshot_id)
        with self._locked():
            record = self._record(self.catalog / f"{snapshot_id}.json")
            working = self._reserve(launch_id, record.source_slot, record)
            try:
                manifest = self._reservation(working)
                with open_private_binary(self.catalog / record.filename) as opened:
                    source = opened.stream
                    before = os.fstat(source.fileno())
                    digest, length = hashlib.sha256(), 0
                    with open_private_text_append_stream(working.path) as target:
                        if _identity(os.fstat(target.fileno())) != (
                            manifest.device,
                            manifest.inode,
                        ):
                            raise _error("member_identity_changed")
                        while chunk := source.read(
                            min(CHUNK_BYTES, record.bytes - length + 1)
                        ):
                            if _write_chunk(target, chunk) != len(chunk):
                                raise _error("short_write")
                            digest.update(chunk)
                            length += len(chunk)
                            if length > record.bytes:
                                raise _error("integrity_mismatch")
                        _flush_binary(target)
                        final = _check_member(
                            working.path, (manifest.device, manifest.inode)
                        )
                        if (
                            final.st_size != record.bytes
                            or length != record.bytes
                            or digest.hexdigest() != record.sha256
                        ):
                            raise _error("integrity_mismatch")
                    with open_private_binary(working.path) as staged:
                        staged_digest = hashlib.sha256()
                        while chunk := staged.stream.read(CHUNK_BYTES):
                            staged_digest.update(chunk)
                        final = _check_member(
                            working.path, (manifest.device, manifest.inode)
                        )
                        if (
                            final.st_size != record.bytes
                            or staged_digest.hexdigest() != record.sha256
                        ):
                            raise _error("integrity_mismatch")
                    after = _check_member(
                        self.catalog / record.filename, _identity(before)
                    )
                    if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    ):
                        raise _error("member_identity_changed")
                return working
            except BaseException:
                # Local handles have closed. Preserve any replacement, not ours.
                self._cleanup(working)
                raise

    def _delete_record(self, record: SnapshotRecord) -> tuple[str, ...]:
        metadata = self.catalog / f"{record.snapshot_id}.json"
        tombstone = self.catalog / f"{record.snapshot_id}.deleting"
        try:
            with open_private_binary(self.catalog / record.filename) as binary:
                info = os.fstat(binary.stream.fileno())
                _check_member(self.catalog / record.filename, _identity(info))
                # Tombstone carries the expected member identity for crash recovery.
                _write_json(
                    tombstone,
                    json.dumps(
                        {
                            "record": record.model_dump(mode="json"),
                            "device": info.st_dev,
                            "inode": info.st_ino,
                        }
                    ),
                )
                with open_private_binary(metadata) as opened:
                    _unlink_verified(
                        metadata, _identity(os.fstat(opened.stream.fileno()))
                    )
                _sync_directory(self.catalog)
                _unlink_verified(self.catalog / record.filename, _identity(info))
            with open_private_binary(tombstone) as opened:
                _unlink_verified(tombstone, _identity(os.fstat(opened.stream.fileno())))
            _sync_directory(self.catalog)
            return ()
        except (OSError, ValueError, SnapshotError):
            return ("cleanup_failed",)

    def delete(self, snapshot_id: str) -> tuple[str, ...]:
        """Delete only one verified committed pair, recording deletion intent first."""
        _validate_id(snapshot_id)
        with self._locked():
            try:
                record = self._record(self.catalog / f"{snapshot_id}.json")
            except FileNotFoundError:
                return ()
            return self._delete_record(record)

    def _cleanup(self, working: WorkingFile) -> tuple[str, ...]:
        try:
            try:
                manifest = self._reservation(working)
            except FileNotFoundError:
                return ("cleanup_failed",) if working.path.exists() else ()
            if manifest.state == "unknown":
                return ("operation_unknown",)
            try:
                _unlink_verified(working.path, (manifest.device, manifest.inode))
            except FileNotFoundError:
                pass
            if (
                manifest.kind == "save"
                and not (self.catalog / f"{working.operation_id}.json").exists()
            ):
                # Publication may have moved bytes before its commit marker failed.
                # The reservation's identity proves this orphan is ours to release.
                try:
                    _unlink_verified(
                        self.catalog / manifest.filename,
                        (manifest.device, manifest.inode),
                    )
                    _sync_directory(self.catalog)
                except FileNotFoundError:
                    pass
            path = working.path.parent / f"{working.operation_id}.json"
            with open_private_binary(path) as opened:
                _unlink_verified(path, _identity(os.fstat(opened.stream.fileno())))
            _sync_directory(path.parent)
            return ()
        except (OSError, ValueError, SnapshotError):
            return ("cleanup_failed",)

    def cleanup(self, working: WorkingFile) -> tuple[str, ...]:
        """Release owner-settled work; never release a possibly active server writer."""
        with self._locked():
            return self._cleanup(working)

    def _working_entries(self) -> tuple[list[Path], bool]:
        directories, complete = self._entries(self.working)
        entries = []
        for directory in directories:
            try:
                if not stat.S_ISDIR(directory.lstat().st_mode):
                    continue
                children, child_complete = self._entries(directory)
                remaining = MAX_SCAN_ENTRIES - len(entries)
                entries.extend(children[:remaining])
                complete &= child_complete and len(children) <= remaining
                if len(entries) >= MAX_SCAN_ENTRIES:
                    return entries, False
            except (OSError, SnapshotError):
                complete = False
        return entries, complete

    def list_records(self, offset: int = 0, limit: int = 50) -> CatalogPage:
        """Return at most 50 records; incomplete scans have no purported totals."""
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 50
        ):
            raise _error("invalid_page")
        with self._locked():
            records, complete, residual = self._scan()
            entries, working_complete = self._working_entries()
            for path in entries:
                try:
                    info = path.lstat()
                    if (
                        stat.S_ISREG(info.st_mode)
                        and info.st_nlink == 1
                        and path.suffix == ".bin"
                    ):
                        residual += info.st_size
                except OSError:
                    working_complete = False
            complete &= working_complete
            next_offset = offset + limit if offset + limit < len(records) else None
            return CatalogPage(
                tuple(records[offset : offset + limit]),
                next_offset,
                sum(item.bytes for item in records) if complete else None,
                residual if complete else None,
                complete,
            )

    def reconcile(self, terminated_launch_ids: frozenset[str]) -> tuple[str, ...]:
        """Reap terminal/confirmed-stopped work and interrupted deletion, never promote."""
        failures = []
        with self._locked():
            entries, complete = self._working_entries()
            for path in entries:
                if path.suffix != ".json" or not _UUID.fullmatch(path.stem):
                    continue
                try:
                    manifest = _Reservation.model_validate_json(_read_json(path))
                    working = WorkingFile(
                        path.parent.name,
                        path.stem,
                        path.parent / manifest.filename,
                        None,
                    )
                    manifest = self._reservation(working)
                except (OSError, ValueError, SnapshotError):
                    continue
                if (
                    manifest.state != "terminal"
                    and manifest.launch_id not in terminated_launch_ids
                ):
                    continue
                try:
                    self._set_state(working, manifest, "terminal")
                    failures.extend(self._cleanup(working))
                except (OSError, ValueError, SnapshotError):
                    failures.append("cleanup_failed")
            catalog, catalog_complete = self._entries(self.catalog)
            for path in catalog:
                if path.suffix != ".deleting" or not _UUID.fullmatch(path.stem):
                    continue
                try:
                    value = json.loads(_read_json(path))
                    if (
                        not isinstance(value, dict)
                        or set(value) != {"record", "device", "inode"}
                        or any(
                            type(value[key]) is not int for key in ("device", "inode")
                        )
                    ):
                        continue
                    record = SnapshotRecord.model_validate_json(
                        json.dumps(value["record"])
                    )
                    _validate_metadata(record, path.stem)
                    if (self.catalog / f"{path.stem}.json").exists():
                        continue  # Commit marker was not yet tombstoned: retain it.
                    try:
                        _unlink_verified(
                            self.catalog / record.filename,
                            (value["device"], value["inode"]),
                        )
                    except FileNotFoundError:
                        pass
                    with open_private_binary(path) as opened:
                        _unlink_verified(
                            path, _identity(os.fstat(opened.stream.fileno()))
                        )
                    _sync_directory(self.catalog)
                except (OSError, ValueError, SnapshotError):
                    failures.append("cleanup_failed")
            if not complete or not catalog_complete:
                failures.append("cleanup_incomplete")
        return tuple(dict.fromkeys(failures))
