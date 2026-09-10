"""Local per-owner review requirements for restored generations (ADR-126)."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .admission import Admission, fcntl
from .bootstrap import _read
from .native_files import create_private_directory, flush_directory, pinned_directory
from .profile_paths import lexical_path
from .qualification import qualified_for


class _Requirement(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: int = Field(ge=1, le=1)
    generation: str
    owners: list[str] = Field(min_length=1, max_length=4096)


class _Approval(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: int = Field(ge=1, le=1)
    generation: str
    owner: str


def _identifier(value: str) -> str:
    if type(value) is not str or not 0 < len(value) <= 256 or "\0" in value:
        raise ValueError("activation_identifier_invalid")
    return value


def _key(value: str) -> str:
    return hashlib.sha256(_identifier(value).encode("utf-8")).hexdigest()


@contextmanager
def _private(root: Path):
    with pinned_directory(root) as parent:
        info = os.fstat(parent)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("activation_root_unsafe")
        yield parent


def _write(parent: int, name: str, record: BaseModel) -> None:
    data = record.model_dump_json().encode("utf-8")
    if len(data) > 1048576:
        raise ValueError("activation_record_too_large")
    Admission._write_new_record(parent, name, data)
    flush_directory(parent)


def _flush_existing(parent: int, name: str, expected: BaseModel) -> None:
    """A complete earlier write may still have failed its durability barrier."""
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if type(expected).model_validate(_read(parent, name)) != expected:
        raise ValueError("activation_record_changed")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        info = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) != (
            info.st_dev,
            info.st_ino,
            info.st_mtime_ns,
            info.st_size,
        ):
            raise ValueError("activation_record_changed")
        os.fsync(fd)
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        flush_directory(parent)
    finally:
        os.close(fd)


class ActivationStore:
    """Durable local requirements; construction and permission reads never write.

    The restore executor supplies a fresh generation and a private control root
    outside restored payloads. Existing local review owners call ``approve``;
    archive metadata and presentation reports must never call that method.
    Immutable requirements prevent later calls from dropping unreviewed owners.
    An interrupted initial write is retained as unavailable state, not repaired.
    """

    def __init__(self, root: Path):
        self.root = lexical_path(root)

    def _generation(self, generation: str) -> Path:
        return self.root / ("generation-" + _key(generation))

    def _required(self, parent: int, generation: str) -> _Requirement:
        record = _Requirement.model_validate(_read(parent, "required.json"))
        if record.generation != generation or record.owners != sorted(
            set(record.owners)
        ):
            raise ValueError("activation_requirements_invalid")
        for owner in record.owners:
            _identifier(owner)
        return record

    def require(self, generation: str, owners: tuple[str, ...]) -> None:
        """Durably require each owner, preserving reviews on an identical retry."""
        directory = self._generation(generation)
        if type(owners) is not tuple or not owners or len(owners) > 4096:
            raise ValueError("activation_owners_invalid")
        names = sorted({_identifier(owner) for owner in owners})
        record = _Requirement(version=1, generation=generation, owners=names)
        existing = self.root if self.root.exists() else self.root.parent
        allowed, reason = qualified_for("admission", existing)
        if not allowed:
            raise ValueError(reason)
        try:
            create_private_directory(self.root)
        except FileExistsError:
            pass
        with _private(self.root) as root:
            try:
                create_private_directory(directory)
                created = True
            except FileExistsError:
                created = False
            with _private(directory) as parent:
                if created:
                    _write(parent, "required.json", record)
                else:
                    previous = self._required(parent, generation)
                    if previous != record:
                        raise ValueError("activation_requirements_changed")
                    _flush_existing(parent, "required.json", record)
            flush_directory(root)
        with pinned_directory(self.root.parent) as parent:
            flush_directory(parent)

    def approve(self, generation: str, owner: str) -> None:
        """Record only the caller's explicitly reviewed owner for this generation."""
        name = "approved-" + _key(owner) + ".json"
        directory = self._generation(generation)
        try:
            with _private(self.root) as root, _private(directory) as parent:
                required = self._required(parent, generation)
                if owner not in required.owners:
                    raise ValueError("activation_owner_unknown")
                record = _Approval(version=1, generation=generation, owner=owner)
                try:
                    _write(parent, name, record)
                except FileExistsError:
                    previous = _Approval.model_validate(_read(parent, name))
                    if previous != record:
                        raise ValueError("activation_approval_invalid")
                    _flush_existing(parent, name, record)
                flush_directory(root)
        except (OSError, ValueError, RuntimeError):
            raise ValueError("activation_review_unavailable") from None

    def allowed(self, generation: str, owner: str) -> bool:
        """Return false for absent, malformed, unsafe or differently bound state."""
        try:
            name = "approved-" + _key(owner) + ".json"
            with _private(self.root), _private(self._generation(generation)) as parent:
                required = self._required(parent, generation)
                if owner not in required.owners:
                    return False
                record = _Approval.model_validate(_read(parent, name))
                return record.generation == generation and record.owner == owner
        except (OSError, ValueError, RuntimeError):
            return False
