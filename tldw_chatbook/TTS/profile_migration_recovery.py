"""Pre-open recovery for interrupted profile-migration publication."""

from __future__ import annotations

import os
import sqlite3
import stat
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Final

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite_descriptor
from tldw_chatbook.TTS import profile_schema
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_migration_journal import (
    MAX_PROFILE_MIGRATION_ARTIFACT_BYTES,
    MAX_PROFILE_MIGRATION_JOURNAL_BYTES,
    ParsedProfileMigrationJournal,
    ProfileMigrationJournalSlot,
    ProfileMigrationPublicationSlot,
    parse_profile_migration_journal,
)
from tldw_chatbook.TTS.profile_migration_namespace import (
    MigrationTombstoneKey,
    ParentAuthority,
    move_exact_noreplace,
    remove_exact as remove_exact_namespace,
)
from tldw_chatbook.Utils import private_paths
from tldw_chatbook.Utils.private_paths import (
    PrivatePathStatus,
    lexical_path,
    secure_private_directory,
)


_SIDECARS: Final = ("-wal", "-shm", "-journal")
_RECOVERY_LOCK = Lock()
_CANDIDATE_TOMBSTONES: Final = {
    ProfileMigrationPublicationSlot.ACTIVE: MigrationTombstoneKey.ACTIVE_CANDIDATE,
    ProfileMigrationPublicationSlot.PRE_V3: MigrationTombstoneKey.PRE_V3_CANDIDATE,
    ProfileMigrationPublicationSlot.PRE_V4: MigrationTombstoneKey.PRE_V4_CANDIDATE,
}
_ROLLBACK_TOMBSTONES: Final = {
    ProfileMigrationPublicationSlot.ACTIVE: MigrationTombstoneKey.ACTIVE_ROLLBACK,
    ProfileMigrationPublicationSlot.PRE_V3: MigrationTombstoneKey.PRE_V3_ROLLBACK,
    ProfileMigrationPublicationSlot.PRE_V4: MigrationTombstoneKey.PRE_V4_ROLLBACK,
}


@dataclass(frozen=True, slots=True, repr=False)
class _ObservedArtifact:
    leaf: str
    identity: os.stat_result
    kind: str
    row: ProfileMigrationJournalSlot

    def __repr__(self) -> str:
        return "_ObservedArtifact(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class _ObservedRow:
    row: ProfileMigrationJournalSlot
    candidate: _ObservedArtifact | None
    target: _ObservedArtifact | None
    rollback: _ObservedArtifact | None

    def __repr__(self) -> str:
        return "_ObservedRow(<private>)"


@dataclass(frozen=True, slots=True, repr=False)
class _JournalSnapshot:
    raw: bytes
    identity: os.stat_result
    byte_length: int
    sha256_digest: bytes
    mtime_ns: int
    ctime_ns: int

    def __repr__(self) -> str:
        return "_JournalSnapshot(<private>)"


def _safe_failure(code: str = "migration_failed") -> ProfileRepositoryError:
    error = ProfileRepositoryError(code)
    error.__cause__ = None
    error.__context__ = None
    return error


def _valid_stat(value: os.stat_result, *, links: frozenset[int]) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and value.st_uid == os.geteuid()
        and stat.S_IMODE(value.st_mode) == 0o600
        and value.st_nlink in links
    )


def _valid_parent_stat(value: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(value.st_mode)
        and value.st_uid == os.geteuid()
        and stat.S_IMODE(value.st_mode) == 0o700
        and value.st_nlink >= 1
    )


def _same_parent_authority(
    current: os.stat_result,
    expected: os.stat_result,
) -> bool:
    return (
        private_paths._same_identity(current, expected)
        and stat.S_IFMT(current.st_mode) == stat.S_IFMT(expected.st_mode)
        and stat.S_IMODE(current.st_mode) == stat.S_IMODE(expected.st_mode)
        and current.st_uid == expected.st_uid
        and _valid_parent_stat(current)
    )


def _require_configured_parent(
    selected: Path,
    expected: os.stat_result,
    *,
    exact_links: bool = False,
) -> None:
    reopened_fd, _leaf = private_paths._open_verified_parent(
        selected,
        missing_leaf_allowed=True,
    )
    try:
        current = os.fstat(reopened_fd)
        if not _same_parent_authority(current, expected) or (
            exact_links and current.st_nlink != expected.st_nlink
        ):
            raise ValueError
    finally:
        os.close(reopened_fd)


def _artifact_size_allowed(byte_length: int) -> bool:
    return (
        type(byte_length) is int
        and 0 <= byte_length <= MAX_PROFILE_MIGRATION_ARTIFACT_BYTES
    )


def _sidecars_absent(parent_fd: int, leaf: str) -> bool:
    for suffix in _SIDECARS:
        try:
            os.stat(f"{leaf}{suffix}", dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        return False
    return True


def _read_stable(
    file_fd: int, *, maximum: int | None = None
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(file_fd)
    if maximum is not None and before.st_size > maximum:
        raise ValueError
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(file_fd, min(1024 * 1024, before.st_size - offset), offset)
        if not chunk:
            raise ValueError
        chunks.append(chunk)
        offset += len(chunk)
    after = os.fstat(file_fd)
    if (
        not private_paths._same_identity(before, after)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ValueError
    return b"".join(chunks), after


def _hash_sqlite(file_fd: int) -> tuple[int, bytes, int, os.stat_result]:
    before = os.fstat(file_fd)
    if not _artifact_size_allowed(before.st_size):
        raise ValueError
    digest = sha256()
    header = b""
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(file_fd, min(1024 * 1024, before.st_size - offset), offset)
        if not chunk:
            raise ValueError
        if len(header) < 64:
            header += chunk[: 64 - len(header)]
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(file_fd)
    if (
        len(header) < 64
        or not private_paths._same_identity(before, after)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ValueError
    return before.st_size, digest.digest(), int.from_bytes(header[60:64], "big"), after


def _open_leaf(parent_fd: int, leaf: str) -> int:
    return os.open(
        leaf,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOCTTY", 0),
        dir_fd=parent_fd,
    )


def _observe_artifact(
    parent_fd: int,
    leaf: str,
    row: ProfileMigrationJournalSlot,
) -> _ObservedArtifact | None:
    try:
        entry = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not _sidecars_absent(parent_fd, leaf):
            raise ValueError
        return None
    if not _valid_stat(entry, links=frozenset({1, 2})):
        raise ValueError
    file_fd = _open_leaf(parent_fd, leaf)
    try:
        byte_length, digest, schema_version, opened = _hash_sqlite(file_fd)
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not private_paths._same_identity(entry, opened)
            or not private_paths._same_identity(opened, current)
            or not _valid_stat(opened, links=frozenset({1, 2}))
            or not _valid_stat(current, links=frozenset({1, 2}))
            or not _sidecars_absent(parent_fd, leaf)
        ):
            raise ValueError
        kind = row.classify_artifact(
            opened,
            byte_length=byte_length,
            sha256_digest=digest,
            schema_version=schema_version,
        )
        if kind is None:
            raise ValueError
        return _ObservedArtifact(leaf, opened, kind, row)
    finally:
        os.close(file_fd)


def _observe_rows(
    parent_fd: int,
    parsed: ParsedProfileMigrationJournal,
) -> tuple[_ObservedRow, ...]:
    observed = tuple(
        _ObservedRow(
            row,
            _observe_artifact(parent_fd, row.candidate, row),
            _observe_artifact(parent_fd, row.target, row),
            _observe_artifact(parent_fd, row.rollback, row),
        )
        for row in parsed.recovery_rows
    )
    artifacts = tuple(
        artifact
        for item in observed
        for artifact in (item.candidate, item.target, item.rollback)
        if artifact is not None
    )
    counts = Counter(
        (artifact.identity.st_dev, artifact.identity.st_ino) for artifact in artifacts
    )
    if any(
        counts[(artifact.identity.st_dev, artifact.identity.st_ino)]
        != artifact.identity.st_nlink
        for artifact in artifacts
    ):
        raise ValueError
    for item in observed:
        candidate, target, rollback = item.candidate, item.target, item.rollback
        if (
            (candidate is not None and candidate.kind != "candidate")
            or (rollback is not None and rollback.kind != "prior")
            or (not item.row.had_prior and rollback is not None)
        ):
            raise ValueError
        for pair in ((candidate, target), (target, rollback)):
            left, right = pair
            if left is not None and right is not None and left.kind == right.kind:
                if not private_paths._same_identity(left.identity, right.identity):
                    raise ValueError
    return observed


def _read_journal(
    parent_fd: int,
    journal_leaf: str,
) -> _JournalSnapshot | None:
    try:
        entry = os.stat(journal_leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not _sidecars_absent(parent_fd, journal_leaf):
            raise ValueError
        return None
    if not _valid_stat(entry, links=frozenset({1})):
        raise ValueError
    file_fd = _open_leaf(parent_fd, journal_leaf)
    try:
        raw, opened = _read_stable(
            file_fd,
            maximum=MAX_PROFILE_MIGRATION_JOURNAL_BYTES,
        )
        current = os.stat(journal_leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not private_paths._same_identity(entry, opened)
            or not private_paths._same_identity(opened, current)
            or not _valid_stat(opened, links=frozenset({1}))
            or not _valid_stat(current, links=frozenset({1}))
            or not _sidecars_absent(parent_fd, journal_leaf)
        ):
            raise ValueError
        return _JournalSnapshot(
            raw,
            opened,
            len(raw),
            sha256(raw).digest(),
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
    finally:
        os.close(file_fd)


def _reobserve_exact(parent_fd: int, expected: _ObservedArtifact) -> os.stat_result:
    current = _observe_artifact(parent_fd, expected.leaf, expected.row)
    if (
        current is None
        or current.kind != expected.kind
        or not private_paths._same_identity(current.identity, expected.identity)
    ):
        raise ValueError
    return current.identity


def _fsync_parent(parent_fd: int, parent_identity: os.stat_result) -> None:
    os.fsync(parent_fd)
    if not private_paths._same_identity(os.fstat(parent_fd), parent_identity):
        raise ValueError


def _remove_exact(
    parent_fd: int,
    parent_identity: os.stat_result,
    parent_authority: ParentAuthority,
    selected: Path,
    expected: _ObservedArtifact,
) -> None:
    _require_configured_parent(selected, parent_identity)
    remove_exact_namespace(
        selected.parent / expected.leaf,
        parent_authority=parent_authority,
        file_identity=expected.identity,
        tombstone_key=(
            _CANDIDATE_TOMBSTONES[expected.row.slot]
            if expected.leaf == expected.row.candidate
            else _ROLLBACK_TOMBSTONES[expected.row.slot]
        ),
        allowed_links=frozenset({1, 2}),
    )


def _move_exact(
    parent_fd: int,
    parent_identity: os.stat_result,
    parent_authority: ParentAuthority,
    selected: Path,
    source: _ObservedArtifact,
    destination_leaf: str,
) -> None:
    _require_configured_parent(selected, parent_identity)
    identity = _reobserve_exact(parent_fd, source)
    moved = move_exact_noreplace(
        selected.parent / source.leaf,
        selected.parent / destination_leaf,
        parent_authority=parent_authority,
        file_identity=identity,
        allowed_links=frozenset({1, 2}),
    )
    file_fd = _open_leaf(parent_fd, destination_leaf)
    try:
        opened = os.fstat(file_fd)
        if not private_paths._same_identity(opened, moved) or not _valid_stat(
            opened, links=frozenset({1, 2})
        ):
            raise ValueError
        os.fsync(file_fd)
    finally:
        os.close(file_fd)
    _fsync_parent(parent_fd, parent_identity)


def _remove_journal(
    parent_fd: int,
    parent_identity: os.stat_result,
    parent_authority: ParentAuthority,
    selected: Path,
    journal_leaf: str,
    snapshot: _JournalSnapshot,
) -> None:
    _require_configured_parent(selected, parent_identity)
    current = _read_journal(parent_fd, journal_leaf)
    if (
        current is None
        or not private_paths._same_identity(current.identity, snapshot.identity)
        or current.byte_length != snapshot.byte_length
        or current.sha256_digest != snapshot.sha256_digest
        or current.mtime_ns != snapshot.mtime_ns
        or current.ctime_ns != snapshot.ctime_ns
        or current.raw != snapshot.raw
    ):
        raise ValueError
    parse_profile_migration_journal(current.raw)
    _require_configured_parent(selected, parent_identity)
    remove_exact_namespace(
        selected.parent / journal_leaf,
        parent_authority=parent_authority,
        file_identity=current.identity,
        tombstone_key=MigrationTombstoneKey.JOURNAL,
    )


def _rollback_possible(rows: Sequence[_ObservedRow]) -> bool:
    for item in rows:
        prior = tuple(
            artifact
            for artifact in (item.target, item.rollback)
            if artifact is not None and artifact.kind == "prior"
        )
        if item.row.had_prior and not prior:
            return False
        if not item.row.had_prior and prior:
            return False
    return True


def _completion_possible(rows: Sequence[_ObservedRow]) -> bool:
    return all(
        any(
            artifact is not None and artifact.kind == "candidate"
            for artifact in (item.candidate, item.target)
        )
        for item in rows
    )


def _only_completed_authority(rows: Sequence[_ObservedRow]) -> bool:
    return all(
        item.target is not None
        and item.target.kind == "candidate"
        and item.rollback is None
        for item in rows
    )


def _rollback(
    parent_fd: int,
    parent_identity: os.stat_result,
    parent_authority: ParentAuthority,
    selected: Path,
    rows: Sequence[_ObservedRow],
) -> None:
    for item in reversed(rows):
        candidate, target, rollback = item.candidate, item.target, item.rollback
        if target is not None and target.kind == "candidate":
            if candidate is None:
                _move_exact(
                    parent_fd,
                    parent_identity,
                    parent_authority,
                    selected,
                    target,
                    item.row.candidate,
                )
            else:
                _remove_exact(
                    parent_fd,
                    parent_identity,
                    parent_authority,
                    selected,
                    target,
                )
                if private_paths._same_identity(candidate.identity, target.identity):
                    _remove_exact(
                        parent_fd,
                        parent_identity,
                        parent_authority,
                        selected,
                        candidate,
                    )
                    candidate = None
            target = None
        if item.row.had_prior:
            if target is not None and target.kind == "prior":
                if rollback is not None:
                    _remove_exact(
                        parent_fd,
                        parent_identity,
                        parent_authority,
                        selected,
                        rollback,
                    )
            elif target is None and rollback is not None:
                _move_exact(
                    parent_fd,
                    parent_identity,
                    parent_authority,
                    selected,
                    rollback,
                    item.row.target,
                )
            else:
                raise ValueError
        elif target is not None or rollback is not None:
            raise ValueError
    for item in rows:
        candidate = _observe_artifact(parent_fd, item.row.candidate, item.row)
        if candidate is not None:
            _remove_exact(
                parent_fd,
                parent_identity,
                parent_authority,
                selected,
                candidate,
            )


def _complete(
    parent_fd: int,
    parent_identity: os.stat_result,
    parent_authority: ParentAuthority,
    selected: Path,
    rows: Sequence[_ObservedRow],
) -> None:
    for item in rows:
        candidate, target, rollback = item.candidate, item.target, item.rollback
        if target is not None and target.kind == "prior":
            if rollback is None:
                _move_exact(
                    parent_fd,
                    parent_identity,
                    parent_authority,
                    selected,
                    target,
                    item.row.rollback,
                )
            elif not private_paths._same_identity(target.identity, rollback.identity):
                raise ValueError
            else:
                _remove_exact(
                    parent_fd,
                    parent_identity,
                    parent_authority,
                    selected,
                    target,
                )
            target = None
        if target is None:
            if candidate is None:
                raise ValueError
            _move_exact(
                parent_fd,
                parent_identity,
                parent_authority,
                selected,
                candidate,
                item.row.target,
            )
            candidate = None
        elif target.kind != "candidate":
            raise ValueError
        if candidate is not None:
            _remove_exact(
                parent_fd,
                parent_identity,
                parent_authority,
                selected,
                candidate,
            )
    refreshed = _observe_rows(
        parent_fd,
        ParsedProfileMigrationJournal(
            2,
            "complete",
            tuple(item.row for item in rows),
            parent_identity.st_dev,
            parent_identity.st_ino,
        ),
    )
    for item in refreshed:
        if item.target is None or item.target.kind != "candidate":
            raise ValueError
        if item.rollback is not None:
            _remove_exact(
                parent_fd,
                parent_identity,
                parent_authority,
                selected,
                item.rollback,
            )


def _validate_authoritative_targets(
    parent_fd: int,
    parent_identity: os.stat_result,
    parent_authority: ParentAuthority,
    selected: Path,
    parsed: ParsedProfileMigrationJournal,
    *,
    kind: str,
) -> None:
    for row in parsed.recovery_rows:
        _require_configured_parent(
            selected, parent_authority.identity, exact_links=True
        )
        if kind == "prior" and not row.had_prior:
            try:
                os.stat(row.target, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                if not _sidecars_absent(parent_fd, row.target):
                    raise ValueError
                continue
            raise ValueError
        before = _observe_artifact(parent_fd, row.target, row)
        if before is None or before.kind != kind or before.identity.st_nlink != 1:
            raise ValueError
        file_fd = _open_leaf(parent_fd, row.target)
        try:
            opened_before = os.fstat(file_fd)
            if not private_paths._same_identity(opened_before, before.identity):
                raise ValueError
            connection = connect_private_sqlite_descriptor(
                "tts.profile_migration_recovery_descriptor",
                file_fd,
                isolation_level=None,
            )
            try:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA query_only = ON")
                version_row = connection.execute("PRAGMA user_version").fetchone()
                if version_row is None or type(version_row[0]) is not int:
                    raise ValueError
                profile_schema.validate_profile_store_version(
                    connection, version_row[0]
                )
            finally:
                connection.close()
            _hash_sqlite(file_fd)
            opened_after = os.fstat(file_fd)
            if not private_paths._same_identity(opened_before, opened_after):
                raise ValueError
        finally:
            os.close(file_fd)
        _require_configured_parent(
            selected, parent_authority.identity, exact_links=True
        )
        after = _observe_artifact(parent_fd, row.target, row)
        if (
            after is None
            or after.kind != kind
            or after.identity.st_nlink != 1
            or not private_paths._same_identity(before.identity, after.identity)
        ):
            raise ValueError
    _require_configured_parent(selected, parent_authority.identity, exact_links=True)


def _choose_action(
    phase: str,
    rows: Sequence[_ObservedRow],
) -> str:
    if phase == "prepared":
        if not _rollback_possible(rows):
            raise ValueError
        return "rollback"
    if phase == "complete":
        if not _completion_possible(rows):
            raise ValueError
        return "complete"
    if phase == "publishing" and _only_completed_authority(rows):
        return "complete"
    if _rollback_possible(rows):
        return "rollback"
    if phase == "publishing" and _completion_possible(rows):
        return "complete"
    raise ValueError


def recover_profile_migration_publication(
    active_store_path: str | os.PathLike[str],
    *,
    _stage_hook: Callable[[str], None] | None = None,
) -> bool:
    """Recover one recognized publication before any profile-store open."""

    selected = lexical_path(active_store_path)
    journal_leaf = f".{selected.name}.migration-publication.json"
    with _RECOVERY_LOCK:
        parent_fd = -1
        admitted = False
        deferred: BaseException | None = None
        body_error: BaseException | None = None
        try:
            result = secure_private_directory(
                selected.parent,
                create=False,
                application_owned=True,
            )
            if result.status is PrivatePathStatus.UNVERIFIED_PLATFORM:
                raise ValueError
            parent_fd, _leaf = private_paths._open_verified_parent(
                selected,
                missing_leaf_allowed=True,
            )
            parent_identity = os.fstat(parent_fd)
            parent_authority = ParentAuthority(parent_identity)
            if not _valid_parent_stat(parent_identity):
                raise ValueError
            journal_snapshot = _read_journal(parent_fd, journal_leaf)
            if journal_snapshot is None:
                os.close(parent_fd)
                parent_fd = -1
                return False
            admitted_snapshot: _JournalSnapshot = journal_snapshot
            parsed = parse_profile_migration_journal(admitted_snapshot.raw)
            if (
                not parsed.matches_parent(parent_identity)
                or not parsed.recovery_rows
                or parsed.recovery_rows[0].slot
                is not ProfileMigrationPublicationSlot.ACTIVE
                or not parsed.recovery_rows[0].had_prior
                or parsed.recovery_rows[0].target != selected.name
                or not all(
                    row.evidence_fits(MAX_PROFILE_MIGRATION_ARTIFACT_BYTES)
                    for row in parsed.recovery_rows
                )
            ):
                raise ValueError
            rows = _observe_rows(parent_fd, parsed)
            action = _choose_action(parsed.phase, rows)
            admitted = True
        except BaseException as error:
            body_error = error
        if body_error is not None:
            if parent_fd >= 0:
                os.close(parent_fd)
            if not isinstance(body_error, Exception):
                raise body_error
            raise _safe_failure("unavailable" if admitted else "migration_failed")

        # Admission pins the only authority this invocation may consume. From
        # here through settlement, control-flow signals are deferred and the
        # exact action is replayed from fresh observations until it converges.
        while True:
            attempt_error: BaseException | None = None
            settled = False
            try:
                _require_configured_parent(
                    selected, parent_authority.identity, exact_links=True
                )
                current_snapshot = _read_journal(parent_fd, journal_leaf)
                if current_snapshot is None:
                    # A deferred signal may have arrived after journal unlink
                    # but before its namespace fsync. Re-durably settle that
                    # exact absence before treating recovery as complete.
                    _fsync_parent(parent_fd, parent_identity)
                    _require_configured_parent(
                        selected, parent_authority.identity, exact_links=True
                    )
                    _validate_authoritative_targets(
                        parent_fd,
                        parent_identity,
                        parent_authority,
                        selected,
                        parsed,
                        kind="prior" if action == "rollback" else "candidate",
                    )
                    settled = True
                else:
                    if (
                        not private_paths._same_identity(
                            current_snapshot.identity,
                            admitted_snapshot.identity,
                        )
                        or current_snapshot.byte_length != admitted_snapshot.byte_length
                        or current_snapshot.sha256_digest
                        != admitted_snapshot.sha256_digest
                        or current_snapshot.mtime_ns != admitted_snapshot.mtime_ns
                        or current_snapshot.ctime_ns != admitted_snapshot.ctime_ns
                        or current_snapshot.raw != admitted_snapshot.raw
                    ):
                        raise ValueError
                    current_parsed = parse_profile_migration_journal(
                        current_snapshot.raw
                    )
                    if not current_parsed.matches_parent(parent_identity) or not all(
                        row.evidence_fits(MAX_PROFILE_MIGRATION_ARTIFACT_BYTES)
                        for row in current_parsed.recovery_rows
                    ):
                        raise ValueError
                    current_rows = _observe_rows(parent_fd, current_parsed)
                    if _stage_hook is not None:
                        _stage_hook("admitted")
                    _require_configured_parent(
                        selected,
                        parent_authority.identity,
                        exact_links=True,
                    )
                    if action == "rollback":
                        _rollback(
                            parent_fd,
                            parent_identity,
                            parent_authority,
                            selected,
                            current_rows,
                        )
                    else:
                        _complete(
                            parent_fd,
                            parent_identity,
                            parent_authority,
                            selected,
                            current_rows,
                        )
                    if _stage_hook is not None:
                        _stage_hook("repaired")
                    _validate_authoritative_targets(
                        parent_fd,
                        parent_identity,
                        parent_authority,
                        selected,
                        current_parsed,
                        kind="prior" if action == "rollback" else "candidate",
                    )
                    if _stage_hook is not None:
                        _stage_hook("validated")
                    _remove_journal(
                        parent_fd,
                        parent_identity,
                        parent_authority,
                        selected,
                        journal_leaf,
                        admitted_snapshot,
                    )
                    if _stage_hook is not None:
                        _stage_hook("settled")
                    _require_configured_parent(
                        selected, parent_authority.identity, exact_links=True
                    )
                    settled = True
            except BaseException as error:
                attempt_error = error

            if attempt_error is None and settled:
                break
            if attempt_error is not None and not isinstance(attempt_error, Exception):
                if deferred is None:
                    deferred = attempt_error
                continue
            body_error = attempt_error or ValueError()
            break

        os.close(parent_fd)
        if body_error is not None:
            raise _safe_failure("unavailable")
        if deferred is not None:
            raise deferred
        return True


__all__ = [
    "MAX_PROFILE_MIGRATION_ARTIFACT_BYTES",
    "recover_profile_migration_publication",
]
