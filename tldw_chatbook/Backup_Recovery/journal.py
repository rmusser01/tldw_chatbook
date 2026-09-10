"""Private durable restore intent; record labels never release admission (ADR-126)."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .admission import Admission, fcntl
from .bootstrap import _read
from .native_files import create_private_directory, flush_directory, pinned_directory
from .qualification import qualified_for

MAX_EVENTS = 100_000


class _Evidence(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class _Object(_Evidence):
    path: str
    device: int = Field(ge=0)
    inode: int = Field(ge=0)
    kind: Literal["file", "directory"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0, le=1024**4)

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value):
        if not Path(value).is_absolute() or ".." in Path(value).parts or "\0" in value:
            raise ValueError("artifact_path_invalid")
        return value


class _Artifact(_Evidence):
    logical_id: str = Field(min_length=1, max_length=1024)
    candidate: _Object
    target: str
    previous: _Object | None
    retained: str | None

    @field_validator("target", "retained")
    @classmethod
    def absolute_path(cls, value):
        return _Object.absolute_path(value) if value is not None else value


class _Prepared(_Evidence):
    generation: str = Field(min_length=1, max_length=256)
    mode: Literal["isolated", "replace"]
    artifacts: list[_Artifact] = Field(default_factory=list, max_length=MAX_EVENTS)


def observe_artifact(path: Path) -> dict:
    """Read bounded no-follow file/tree identity and bytes for local reconciliation.

    Modes are deliberately separate from content identity: native publication
    normalizes private modes before rename, and final metadata is validated later.
    Directory evidence includes every child and empty directory, never just names
    from an intended payload list.
    """
    count, total = 0, 0
    observed = {}

    def identity(info):
        return (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_mode,
        )

    def recheck(fd, relative):
        expected, names = observed[relative]
        if identity(os.fstat(fd)) != expected:
            raise ValueError("artifact_changed")
        if names is not None:
            if sorted(os.listdir(fd)) != names:
                raise ValueError("artifact_changed")
            for name in names:
                child = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
                )
                try:
                    recheck(child, relative + "/" + name)
                finally:
                    os.close(child)

    def read(fd, relative, device):
        nonlocal count, total
        before = os.fstat(fd)
        names = None
        count += 1
        if (
            count > MAX_EVENTS
            or before.st_dev != device
            or before.st_uid != os.geteuid()
        ):
            raise ValueError("artifact_unverified")
        if stat.S_ISREG(before.st_mode):
            if before.st_nlink != 1 or before.st_size > 256 * 1024**3:
                raise ValueError("artifact_unverified")
            digest = hashlib.sha256()
            size = 0
            while chunk := os.read(fd, 64 * 1024):
                size += len(chunk)
                total += len(chunk)
                if size > before.st_size or total > 1024**4:
                    raise ValueError("artifact_limit")
                digest.update(chunk)
            rows = [(relative, "file", size, digest.hexdigest())]
            kind = "file"
        elif stat.S_ISDIR(before.st_mode):
            names = sorted(os.listdir(fd))
            if len(names) > MAX_EVENTS - count:
                raise ValueError("artifact_limit")
            rows = [(relative, "directory")]
            kind = "directory"
            for name in names:
                child = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
                )
                try:
                    _, child_rows = read(child, relative + "/" + name, device)
                    rows.extend(child_rows)
                finally:
                    os.close(child)
            if names != sorted(os.listdir(fd)):
                raise ValueError("artifact_changed")
        else:
            raise ValueError("artifact_unverified")
        after = os.fstat(fd)
        if identity(before) != identity(after):
            raise ValueError("artifact_changed")
        observed[relative] = identity(after), names
        return kind, rows

    with pinned_directory(path.parent) as parent:
        fd = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        try:
            before = os.fstat(fd)
            kind, rows = read(fd, "", before.st_dev)
            recheck(fd, "")
            named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino):
                raise ValueError("artifact_changed")
        finally:
            os.close(fd)
    return _Object(
        path=str(path),
        device=before.st_dev,
        inode=before.st_ino,
        kind=kind,
        size=total,
        sha256=hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode()
        ).hexdigest(),
    ).model_dump()


def _matches(expected: _Object, path: str) -> bool:
    try:
        actual = observe_artifact(Path(path))
    except (OSError, ValueError, RecursionError):
        return False
    return {
        key: value for key, value in actual.items() if key != "path"
    } == expected.model_dump(exclude={"path"})


def _absent(path: str) -> bool:
    try:
        with pinned_directory(Path(path).parent) as parent:
            os.stat(Path(path).name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        # An absent parent could be a disconnected volume. It is not proof that
        # an artifact is absent under the reviewed mounted parent.
        try:
            with pinned_directory(Path(path).parent):
                return True
        except OSError:
            return False
    except OSError:
        return False
    return False


def _states(prepared: _Prepared) -> dict[str, str]:
    result = {}
    for item in prepared.artifacts:
        candidate_present = _matches(item.candidate, item.candidate.path)
        old_present = (
            _matches(item.previous, item.target)
            if item.previous
            else _absent(item.target)
        )
        if (
            candidate_present
            and old_present
            and (item.retained is None or _absent(item.retained))
        ):
            state = "staged"
        elif (
            _absent(item.candidate.path)
            and _matches(item.candidate, item.target)
            and (
                _matches(item.previous, item.retained)
                if item.previous and item.retained
                else item.previous is None
            )
        ):
            state = "published"
        else:
            state = "uncertain"
        result[item.logical_id] = state
    return result


class _Event(_Evidence):
    version: Literal[1] = 1
    operation_id: str
    sequence: int = Field(ge=0, lt=MAX_EVENTS)
    previous: str
    event: Literal["prepared", "publication_started"]
    evidence: dict


def _encoded(record: _Event) -> bytes:
    return json.dumps(
        record.model_dump(), sort_keys=True, separators=(",", ":")
    ).encode()


def _validate(event: str, evidence: Mapping[str, object], prior: list[_Event]) -> dict:
    allowed = (
        "prepared"
        if not prior
        else "publication_started"
        if prior[-1].event == "prepared"
        else None
    )
    if event != allowed:
        raise ValueError("journal_transition_invalid")
    if event == "publication_started" and prior[0].evidence["mode"] == "replace":
        raise ValueError("rollback_required")
    model = _Prepared if event == "prepared" else _Evidence
    try:
        validated = model.model_validate(dict(evidence))
        if isinstance(validated, _Prepared):
            artifacts = validated.artifacts
            if len({item.logical_id for item in artifacts}) != len(artifacts) or len(
                {item.target for item in artifacts}
            ) != len(artifacts):
                raise ValueError("duplicate_artifact")
            if any(
                item.candidate.path == item.target
                or item.previous
                and item.previous.path != item.target
                or (item.previous is None) != (item.retained is None)
                for item in artifacts
            ):
                raise ValueError("artifact_mapping_invalid")
        return validated.model_dump()
    except (ValidationError, TypeError, ValueError):
        raise ValueError("journal_evidence_invalid") from None


class Journal:
    """Serialize local records under a stable lock and retain incomplete writes.

    Reopening existing evidence never repairs files or clears bootstrap pointers.
    The caller supplies a locally chosen private control root, never an archive
    locator. A missing, corrupt, or interrupted sequence remains recovery-required.
    """

    def __init__(self, root: Path, operation_id: str):
        if (
            type(operation_id) is not str
            or not 0 < len(operation_id) <= 256
            or "\0" in operation_id
        ):
            raise ValueError("operation_id_invalid")
        allowed, reason = qualified_for("admission", root)
        if not allowed:
            raise OSError(reason)
        self.operation_id = operation_id
        self.root = root / (
            "operation-" + hashlib.sha256(operation_id.encode()).hexdigest()
        )
        try:
            create_private_directory(self.root)
        except FileExistsError:
            pass
        else:
            with pinned_directory(self.root) as parent:
                Admission._create_lock(parent, "journal.lock")
                flush_directory(parent)
        with pinned_directory(self.root) as parent:
            info = os.fstat(parent)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError("journal_not_private")
            self._identity = info.st_dev, info.st_ino

    @contextmanager
    def _locked(self, *, exclusive: bool):
        with pinned_directory(self.root) as parent:
            info = os.fstat(parent)
            if (info.st_dev, info.st_ino) != self._identity or info.st_mode & 0o077:
                raise ValueError("journal_identity_changed")
            lock = os.open(
                "journal.lock", os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
            try:
                info = os.fstat(lock)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077
                ):
                    raise ValueError("journal_lock_invalid")
                fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                named = os.stat("journal.lock", dir_fd=parent, follow_symlinks=False)
                if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino):
                    raise ValueError("journal_lock_changed")
                yield parent
            finally:
                os.close(lock)

    def _records(self, parent: int) -> list[_Event]:
        names = sorted(name for name in os.listdir(parent) if name != "journal.lock")
        if len(names) > MAX_EVENTS:
            raise ValueError("journal_limit")
        records = []
        previous = ""
        for index, name in enumerate(names):
            if name != f"{index:06d}.json":
                raise ValueError("journal_sequence_invalid")
            record = _Event.model_validate(_read(parent, name))
            if (
                record.operation_id != self.operation_id
                or record.sequence != index
                or record.previous != previous
            ):
                raise ValueError("journal_sequence_invalid")
            _validate(record.event, record.evidence, records)
            records.append(record)
            previous = hashlib.sha256(_encoded(record)).hexdigest()
        return records

    def record(self, event: str, evidence: Mapping[str, object]) -> None:
        """Flush one strictly typed exclusive record; failed writes remain evidence."""
        with self._locked(exclusive=True) as parent:
            records = self._records(parent)
            validated = _validate(event, evidence, records)
            record = _Event(
                operation_id=self.operation_id,
                sequence=len(records),
                previous=hashlib.sha256(_encoded(records[-1])).hexdigest()
                if records
                else "",
                event=event,
                evidence=validated,
            )
            encoded = _encoded(record)
            if len(encoded) > 1048576:
                raise ValueError("journal_record_limit")
            Admission._write_new_record(parent, f"{len(records):06d}.json", encoded)
            flush_directory(parent)

    def artifact_states(self) -> dict[str, str]:
        """Classify staged/published/uncertain objects without mutating either name."""
        with self._locked(exclusive=False) as parent:
            records = self._records(parent)
            if not records:
                raise ValueError("journal_preparation_missing")
            return _states(_Prepared.model_validate(records[0].evidence))

    def recover(self) -> str:
        """Read local evidence without cleanup, publication, or fence changes."""
        try:
            with self._locked(exclusive=False) as parent:
                records = self._records(parent)
                if len(records) == 1 and records[0].event == "prepared":
                    states = _states(_Prepared.model_validate(records[0].evidence))
                    if all(state == "staged" for state in states.values()):
                        return "prepared"
        except (OSError, ValueError, RuntimeError):
            return "recovery_required"
        return "recovery_required"
