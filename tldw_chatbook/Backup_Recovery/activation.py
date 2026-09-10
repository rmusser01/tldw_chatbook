"""Local per-owner review requirements for restored generations (ADR-126)."""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from . import bootstrap
from .admission import Admission, fcntl
from .bootstrap import _read
from .native_files import create_private_directory, flush_directory, pinned_directory
from .profile_paths import lexical_path
from .qualification import qualified_for


def bind_activation(
    bootstrap_root: Path,
    operation_id: str,
    config_selector: Path,
    generation: str,
    owners: tuple[str, ...],
    *,
    session=None,
) -> Path:
    """Associate local requirements under live matching maintenance and pending intent.

    Success proves paired durability only. The executor still owns installed
    validation, journal progression and fence completion.

    Initial profiles inherit the complete affected pending footprint, excluding
    guards held only by maintenance. For narrower multi-profile isolation, the
    executor must register separate affected scopes/pending records. Existing
    profiles retain their exact enrolled namespace/root mapping.
    """
    from .control_records import _bind_activation

    return _bind_activation(
        bootstrap_root, operation_id, config_selector, generation, owners, session
    )


def activation_permission(
    owner: str,
    *,
    config_selector: Path | None = None,
    bootstrap_root: Path | None = None,
    namespaces: tuple[str, ...] | None = None,
) -> bool:
    """Read paired history before config fingerprint fallback, without repair.

    Consumers supply their actual admitted namespaces from the retained storage
    admission hold, not UI or archive labels. With ``namespaces=None`` this checks
    only the selected profile and overlapping recorded roots; it cannot infer a
    separate shared store used by an unbound selector. Such consumers must pass
    the namespace group actually held by their storage admission.
    Intact ordinary authority with neither witness retains legacy behavior; loss
    of every independent local history record is outside this local guarantee.
    """
    try:
        _identifier(owner)
        selected = lexical_path(config_selector or bootstrap.effective_config_path())
        root = lexical_path(bootstrap_root or bootstrap.default_bootstrap_root())
        if not bootstrap.startup_permission(selected, root)[0]:
            return False
        _, profiles, associations = bootstrap._control_records(root)
        registry = bootstrap._registry(root)
        names = set()
        if namespaces is not None:
            names.update(Admission._names(namespaces))
            if registry is None or not names <= registry.keys():
                return False
        selected_profile = next(
            (p for p in profiles if p["selector"] == str(selected)), None
        )
        if selected_profile:
            names.update(selected_profile["namespaces"])
        candidates = {
            record["selector"]
            for record in profiles + associations
            if record["selector"] == str(selected)
            or names.intersection(record.get("activation", {}).get("namespaces", []))
            or any(
                bootstrap._overlap(selected, Path(p)) for p in record.get("roots", [])
            )
        }
        seen = {}
        for selector in candidates:
            profile = next((p for p in profiles if p["selector"] == selector), None)
            association = next(
                (a for a in associations if a["selector"] == selector), None
            )
            witness = profile.get("activation") if profile else None
            if witness is None and association is None:
                continue
            if association is None or witness != association["activation"]:
                return False
            if registry is None or any(
                n not in registry for n in witness["namespaces"]
            ):
                return False
            roots = sorted(
                {p for n in witness["namespaces"] for p in registry[n]["roots"]}
            )
            if roots != profile["roots"]:
                return False
            for namespace in witness["namespaces"]:
                if namespace in seen and seen[namespace] != witness:
                    return False
                seen[namespace] = witness
            store = ActivationStore(Path(witness["store_root"]))
            with (
                _private(store.root),
                _private(store._generation(witness["generation"])) as parent,
            ):
                if (
                    store._required(parent, witness["generation"]).owners
                    != witness["owners"]
                ):
                    return False
            if not store.allowed(witness["generation"], owner):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError):
        return False


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
