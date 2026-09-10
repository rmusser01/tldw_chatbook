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


class _Directory(_Evidence):
    path: str
    device: int = Field(ge=0)
    inode: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value):
        return _Object.absolute_path(value)


class _Artifact(_Evidence):
    logical_id: str = Field(min_length=1, max_length=1024)
    candidate: _Object | None
    target: str
    previous: _Object | None
    retained: str | None
    previous_metadata: _Object | None = None
    action: Literal["publish", "retire", "container"] = "publish"
    rollback_requires_owner: bool = False
    parents: list[_Directory] = Field(default_factory=list, max_length=3)

    @field_validator("target", "retained")
    @classmethod
    def absolute_path(cls, value):
        return _Object.absolute_path(value) if value is not None else value


class _PublicationContext(_Evidence):
    bootstrap_root: str
    namespaces: list[str] = Field(min_length=1, max_length=4096)
    selectors: list[str] = Field(min_length=1, max_length=4096)
    archive_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    descriptor: _Object


class _CandidateReceipt(_Evidence):
    stage: _Object
    descriptor: _Object
    archive_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class _Rollback(_Evidence):
    ciphertext: _Object
    sealed_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage: dict[str, str]


class _Progress(_Evidence):
    logical_id: str
    observed: _Object


class _Prepared(_Evidence):
    generation: str = Field(min_length=1, max_length=256)
    mode: Literal["isolated", "replace"]
    artifacts: list[_Artifact] = Field(default_factory=list, max_length=MAX_EVENTS)
    publication: _PublicationContext | None = None


def observe_artifact(path: Path, *, metadata: bool = False) -> dict:
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
        if metadata:
            rows[0] += (stat.S_IMODE(before.st_mode), before.st_mtime_ns)
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


def _matches(expected: _Object, path: str, *, metadata: bool = False) -> bool:
    try:
        actual = observe_artifact(Path(path), metadata=metadata)
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


def _states(prepared: _Prepared, *, logical_id: str | None = None) -> dict[str, str]:
    result = {}
    for item in prepared.artifacts:
        if logical_id is not None and item.logical_id != logical_id:
            continue
        candidate_present = item.candidate is not None and _matches(
            item.candidate, item.candidate.path
        )
        previous = item.previous_metadata or item.previous
        container_present = False
        if item.action == "container" and item.candidate is not None:
            try:
                with pinned_directory(Path(item.target)) as fd:
                    info = os.fstat(fd)
                    children = {
                        Path(other.target).relative_to(item.target).parts[0]
                        for other in prepared.artifacts
                        if Path(item.target) in Path(other.target).parents
                    }
                    container_present = (info.st_dev, info.st_ino) == (
                        item.candidate.device,
                        item.candidate.inode,
                    ) and set(os.listdir(fd)) <= children
            except OSError:
                pass

        def old_at(
            path, previous=previous, metadata=item.previous_metadata is not None
        ):
            return previous is not None and _matches(previous, path, metadata=metadata)

        old_present = old_at(item.target) if item.previous else _absent(item.target)
        if (
            (candidate_present or item.action == "retire")
            and old_present
            and (item.retained is None or _absent(item.retained))
        ):
            state = "staged"
        elif (
            (candidate_present or item.action == "retire")
            and _absent(item.target)
            and item.retained
            and old_at(item.retained)
        ):
            state = "retired"
        elif (
            item.candidate is not None
            and _absent(item.candidate.path)
            and (
                container_present
                if item.action == "container"
                else _matches(item.candidate, item.target)
            )
            and (
                old_at(item.retained)
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
    event: Literal[
        "candidate_staged",
        "prepared",
        "rollback_verified",
        "publication_started",
        "artifact_retired",
        "artifact_published",
    ]
    evidence: dict


def _encoded(record: _Event) -> bytes:
    return json.dumps(
        record.model_dump(), sort_keys=True, separators=(",", ":")
    ).encode()


def _validate(event: str, evidence: Mapping[str, object], prior: list[_Event]) -> dict:
    events = [record.event for record in prior]
    lifecycle = [value for value in events if value != "candidate_staged"]
    prepared_record = next((row for row in prior if row.event == "prepared"), None)
    allowed = (
        {"candidate_staged", "prepared"}
        if not prior
        else (
            {"prepared"}
            if events == ["candidate_staged"]
            else {"rollback_verified", "publication_started"}
            if lifecycle == ["prepared"]
            else {"publication_started"}
            if lifecycle == ["prepared", "rollback_verified"]
            else {"artifact_retired", "artifact_published"}
            if "publication_started" in events
            else set()
        )
    )
    if event not in allowed:
        raise ValueError("journal_transition_invalid")
    if (
        event == "publication_started"
        and prepared_record.evidence["mode"] == "replace"
        and "rollback_verified" not in events
    ):
        raise ValueError("rollback_required")
    model = {
        "candidate_staged": _CandidateReceipt,
        "prepared": _Prepared,
        "rollback_verified": _Rollback,
        "artifact_retired": _Progress,
        "artifact_published": _Progress,
    }.get(event, _Evidence)
    try:
        validated = model.model_validate(dict(evidence))
        if isinstance(validated, _Prepared):
            artifacts = validated.artifacts
            if len({item.logical_id for item in artifacts}) != len(artifacts) or len(
                {item.target for item in artifacts}
            ) != len(artifacts):
                raise ValueError("duplicate_artifact")
            if any(
                (item.candidate is None) != (item.action == "retire")
                or item.candidate is not None
                and item.candidate.path == item.target
                or item.previous
                and item.previous.path != item.target
                or (item.previous is None) != (item.retained is None)
                for item in artifacts
            ):
                raise ValueError("artifact_mapping_invalid")
        elif isinstance(validated, _Progress):
            items = {
                item.logical_id: item
                for item in _Prepared.model_validate(prepared_record.evidence).artifacts
            }
            item = items.get(validated.logical_id)
            if item is None or any(
                row.event == event
                and row.evidence["logical_id"] == validated.logical_id
                for row in prior
                if row.event in {"artifact_retired", "artifact_published"}
            ):
                raise ValueError("artifact_progress_invalid")
            if (
                event == "artifact_retired"
                and item.previous is None
                or event == "artifact_published"
                and item.candidate is None
            ):
                raise ValueError("artifact_progress_invalid")
            expected = (
                item.previous_metadata
                if event == "artifact_retired"
                else item.candidate
            )
            destination = item.retained if event == "artifact_retired" else item.target
            if (
                expected is None
                or validated.observed.path != destination
                or validated.observed.model_dump(exclude={"path"})
                != expected.model_dump(exclude={"path"})
            ):
                raise ValueError("artifact_progress_invalid")
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
            self._append(parent, event, evidence)

    @staticmethod
    def _flush_record(parent: int, name: str, expected: dict) -> None:
        """Reestablish durability of a complete but possibly unflushed record."""

        def identity(info):
            return (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )

        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = os.fstat(fd)
            if _read(parent, name) != expected:
                raise ValueError("durable_record_changed")
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if identity(named) != identity(before):
                raise ValueError("durable_record_changed")
            os.fsync(fd)
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
            if identity(os.fstat(fd)) != identity(before) or identity(
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            ) != identity(before):
                raise ValueError("durable_record_changed")
        finally:
            os.close(fd)

    def _flush_records(self, parent: int) -> None:
        """Revalidate and flush accepted history under the held journal lock."""
        for row in self._records(parent):
            self._flush_record(parent, f"{row.sequence:06d}.json", row.model_dump())
        flush_directory(parent)

    def _append(self, parent: int, event: str, evidence: Mapping[str, object]) -> None:
        """Append while the caller retains this operation's exclusive lock."""
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

    def prepare_publication(
        self, candidate, plan, *, bootstrap_root, namespaces, selectors, generation
    ):
        """Bind an explicit local publication context; never infer its scope."""
        from .publication import _prepare

        _prepare(
            self, candidate, plan, bootstrap_root, namespaces, selectors, generation
        )

    def record_candidate(self, stage, plan, archive):
        """Durably bind successful staging to the independently verified archive."""
        from .archive_reader import verify_sealed
        from .publication import _descriptor, _plan_digest
        from .restore_plan import recheck_targets

        verify_sealed(archive)
        if plan.archive_digest != archive.digest:
            raise ValueError("archive_plan_mismatch")
        with self._locked(exclusive=True) as parent:
            recheck_targets(plan)
            _descriptor(stage, plan)
            self._append(
                parent,
                "candidate_staged",
                {
                    "stage": observe_artifact(stage),
                    "descriptor": observe_artifact(stage / "candidate.json"),
                    "archive_digest": archive.digest,
                    "manifest_digest": hashlib.sha256(
                        archive.manifest_bytes
                    ).hexdigest(),
                    "plan_digest": _plan_digest(plan),
                },
            )

    def verify_rollback(self, path, *, password, work_root, cancel, coverage):
        """Authenticate exact raw rollback coverage without persisting its secret."""
        from .publication import _verify_rollback

        _verify_rollback(self, path, password, work_root, cancel, coverage)

    def artifact_states(self) -> dict[str, str]:
        """Classify staged/published/uncertain objects without mutating either name."""
        with self._locked(exclusive=False) as parent:
            records = self._records(parent)
            if not records:
                raise ValueError("journal_preparation_missing")
            prepared = next((row for row in records if row.event == "prepared"), None)
            if prepared is None:
                raise ValueError("journal_preparation_missing")
            return _states(_Prepared.model_validate(prepared.evidence))

    def recover(self) -> str:
        """Read local evidence without cleanup, publication, or fence changes."""
        try:
            with self._locked(exclusive=False) as parent:
                records = self._records(parent)
                if records and records[-1].event == "prepared":
                    states = _states(_Prepared.model_validate(records[-1].evidence))
                    if all(state == "staged" for state in states.values()):
                        return "prepared"
        except (OSError, ValueError, RuntimeError):
            return "recovery_required"
        return "recovery_required"
