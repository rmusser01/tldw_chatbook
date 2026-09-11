"""Explicit access to retained local encrypted rollback copies."""

import hashlib
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import archive_reader, bootstrap
from .admission import fcntl
from .journal import Journal, _matches, _Prepared, _Rollback
from .native_files import flush_directory, pinned_directory

_OPERATION = re.compile(r"operation-[a-f0-9]{64}\Z")


@dataclass(frozen=True)
class RecoveryCopy:
    """Sanitized local evidence; verification does not imply an unlocked archive."""

    operation_id: str
    path: Path | None
    size: int
    coverage: tuple[tuple[str, str], ...]
    status: str
    pending_operation: bool


def deletion_allowed(
    *, pending_operation: bool, active_hold: bool, user_selected: bool
) -> bool:
    return user_selected and not pending_operation and not active_hold


def _journal(control_root: Path, operation_id: str) -> Journal:
    """Open an existing local operation without creating missing evidence."""
    if (
        type(operation_id) is not str
        or not 0 < len(operation_id) <= 256
        or "\0" in operation_id
    ):
        raise ValueError("operation_id_invalid")
    name = "operation-" + hashlib.sha256(operation_id.encode()).hexdigest()
    with pinned_directory(control_root / name):
        pass
    return Journal(control_root, operation_id)


def _copy_evidence(journal: Journal):
    with journal._locked(exclusive=False) as parent:
        records = journal._records(parent)
    rollback = next((row for row in records if row.event == "rollback_verified"), None)
    if rollback is None:
        return None
    proof = _Rollback.model_validate(rollback.evidence)
    prepared = _Prepared.model_validate(
        next(row.evidence for row in records if row.event == "prepared")
    )
    if prepared.mode != "replace" or prepared.publication is None:
        raise ValueError("recovery_copy_operation_invalid")
    pending, _ = bootstrap._records(Path(prepared.publication.bootstrap_root))
    unresolved = records[-1].event not in {"committed", "rolled_back"} or any(
        row["operation_id"] == journal.operation_id
        or set(row["namespaces"]).intersection(prepared.publication.namespaces)
        for row in pending
    )
    path = Path(proof.ciphertext.path)
    try:
        path.lstat()
    except FileNotFoundError:
        status = "missing"
    else:
        status = "verified" if _matches(proof.ciphertext, str(path)) else "changed"
    return (
        RecoveryCopy(
            journal.operation_id,
            path,
            proof.ciphertext.size,
            tuple(sorted(proof.coverage.items())),
            status,
            unresolved,
        ),
        proof,
    )


def list_recovery_copies(control_root: Path) -> tuple[RecoveryCopy, ...]:
    """Inspect journal-bound copies; unknown files are never adopted or cleaned."""
    if not control_root.exists():
        return ()
    with pinned_directory(control_root) as parent:
        names = []
        with os.scandir(parent) as entries:
            for index, entry in enumerate(entries):
                if index >= 100_000:
                    raise ValueError("recovery_copy_limit")
                if _OPERATION.fullmatch(entry.name):
                    names.append(entry.name)
    copies = []
    for name in sorted(names):
        operation_id = name
        try:
            with pinned_directory(control_root / name) as parent:
                candidate_id = bootstrap._read(parent, "000000.json")["operation_id"]
            journal = _journal(control_root, candidate_id)
            if journal.root.name != name:
                raise ValueError("recovery_copy_operation_invalid")
            operation_id = candidate_id
            evidence = _copy_evidence(journal)
            if evidence is not None:
                copies.append(evidence[0])
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            copies.append(
                RecoveryCopy(operation_id, None, 0, (), "recovery_required", True)
            )
    return tuple(copies)


@contextmanager
def _locked_copy(control_root, operation_id, *, exclusive):
    journal = _journal(control_root, operation_id)
    evidence = _copy_evidence(journal)
    if evidence is None:
        raise ValueError("recovery_copy_missing")
    entry, proof = evidence
    if entry.status != "verified":
        raise ValueError("recovery_copy_" + entry.status)
    with archive_reader._regular(entry.path) as stream:
        info = os.fstat(stream.fileno())
        if info.st_uid != os.geteuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise ValueError("recovery_copy_not_private")
        try:
            fcntl.flock(
                stream.fileno(),
                (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            raise ValueError("recovery_copy_held") from None
        # Bind the held native object to the journal and its still-named path.
        if (info.st_dev, info.st_ino) != (
            proof.ciphertext.device,
            proof.ciphertext.inode,
        ):
            raise ValueError("recovery_copy_changed")
        current = _copy_evidence(journal)
        if current != evidence:
            raise ValueError("recovery_copy_changed")
        yield entry, journal, proof, stream


@contextmanager
def hold_recovery_copy(control_root: Path, operation_id: str):
    """Retain the actual ciphertext inode while a service reads or unlocks it."""
    with _locked_copy(control_root, operation_id, exclusive=False) as (
        entry,
        _,
        _,
        _stream,
    ):
        yield entry


def delete_recovery_copy(
    control_root: Path, operation_id: str, *, user_selected: bool
) -> None:
    """Delete only the explicitly selected, unchanged and unheld completed copy."""
    if not user_selected:
        raise ValueError("recovery_copy_delete_not_selected")
    with _locked_copy(control_root, operation_id, exclusive=True) as (
        entry,
        journal,
        proof,
        stream,
    ):
        if not deletion_allowed(
            pending_operation=entry.pending_operation,
            active_hold=False,
            user_selected=user_selected,
        ):
            raise ValueError("recovery_copy_pending")
        with pinned_directory(entry.path.parent) as parent:
            current = _copy_evidence(journal)
            named = os.stat(entry.path.name, dir_fd=parent, follow_symlinks=False)
            held = os.fstat(stream.fileno())
            if current != (entry, proof) or (named.st_dev, named.st_ino) != (
                held.st_dev,
                held.st_ino,
            ):
                raise ValueError("recovery_copy_changed")
            os.unlink(entry.path.name, dir_fd=parent)
            flush_directory(parent)
