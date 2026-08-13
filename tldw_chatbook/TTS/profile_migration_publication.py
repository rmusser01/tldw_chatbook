"""Recoverable publication of exact prepared profile-migration artifacts."""

from __future__ import annotations

import os
import sqlite3
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock, get_ident
from typing import Final

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite_descriptor
from tldw_chatbook.TTS import profile_schema
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_migration_journal import (
    MAX_PROFILE_MIGRATION_ARTIFACT_BYTES,
    MAX_PROFILE_MIGRATION_JOURNAL_BYTES,
    PROFILE_MIGRATION_CANDIDATE_LEAVES,
    PROFILE_MIGRATION_ROLLBACK_LEAVES,
    PROFILE_MIGRATION_SLOT_SEQUENCES,
    ParsedProfileMigrationJournal,
    ProfileMigrationJournalSlot,
    ProfileMigrationPublicationSlot,
    ProfileMigrationPublicationStage,
    _encode_initial_journal,
    _encode_later_journal,
    _JournalAuthority,
    _new_profile_migration_journal_authority,
    parse_profile_migration_journal,
)
from tldw_chatbook.TTS.profile_migration_namespace import (
    MigrationTombstoneKey,
    ParentAuthority,
    move_exact_noreplace,
    open_new_or_reused_private_file,
    remove_exact as remove_exact_namespace,
)
from tldw_chatbook.Utils import private_paths
from tldw_chatbook.Utils.private_paths import (
    PrivatePathStatus,
    lexical_path,
    secure_private_directory,
)


_SLOT_VERSION: Final = {
    ProfileMigrationPublicationSlot.ACTIVE: 4,
    ProfileMigrationPublicationSlot.PRE_V3: 2,
    ProfileMigrationPublicationSlot.PRE_V4: 3,
}
_SIDECARS: Final = ("-wal", "-shm", "-journal")
_PUBLICATION_LOCK = Lock()
_ACTIVE_PUBLICATIONS: set[tuple[int, int, str]] = set()
_IDENTITY_FACTORY_TOKEN = object()
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


class _OpaqueIdentity:
    __slots__ = (
        "_content_evidence",
        "_file_identity",
        "_parent_identity",
        "_path",
        "_schema_version",
        "_slot",
        "_state",
        "_thread_id",
    )

    def __init__(
        self,
        factory_token: object,
        *,
        path: Path,
        slot: ProfileMigrationPublicationSlot,
        parent_identity: os.stat_result,
        file_identity: os.stat_result | None,
        schema_version: int | None,
    ) -> None:
        if factory_token is not _IDENTITY_FACTORY_TOKEN:
            raise TypeError("opaque identity")
        self._path = path
        self._slot = slot
        self._parent_identity = parent_identity
        self._file_identity = file_identity
        self._schema_version = schema_version
        self._content_evidence: tuple[int, bytes] | None = None
        self._state = "ready"
        self._thread_id = get_ident()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<private>)"


class PreparedProfileMigrationArtifact(_OpaqueIdentity):
    """Single-use exact prepared-artifact authority with no public path API."""

    __slots__ = ()


class RetainedProfileMigrationDestination(_OpaqueIdentity):
    """Single-use retained target identity with no public path API."""

    __slots__ = ("_must_exist",)

    def __init__(self, *args: object, must_exist: bool, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._must_exist = must_exist


@dataclass(slots=True, repr=False)
class _PublicationSlotState:
    artifact: PreparedProfileMigrationArtifact
    destination: RetainedProfileMigrationDestination
    rollback_path: Path
    parent_authority: ParentAuthority
    prior_retained: bool = False
    candidate_published: bool = False

    def __repr__(self) -> str:
        return "_PublicationSlotState(<private>)"


@dataclass(frozen=True, slots=True)
class _JournalIdentity:
    file: os.stat_result
    parent: os.stat_result


def _safe_failure(code: str = "migration_failed") -> ProfileRepositoryError:
    error = ProfileRepositoryError(code)
    error.__cause__ = None
    error.__context__ = None
    return error


def _redeliver_control_flow(*errors: BaseException | None) -> None:
    for error in errors:
        if error is not None and not isinstance(error, Exception):
            raise error


def _valid_private_stat(value: os.stat_result) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and value.st_uid == os.geteuid()
        and value.st_nlink == 1
        and stat.S_IMODE(value.st_mode) == 0o600
    )


def _sidecars_absent(parent_fd: int, leaf: str) -> bool:
    for suffix in _SIDECARS:
        try:
            os.stat(f"{leaf}{suffix}", dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        return False
    return True


def _content_evidence(file_fd: int) -> tuple[int, bytes]:
    before = os.fstat(file_fd)
    if not 0 <= before.st_size <= MAX_PROFILE_MIGRATION_ARTIFACT_BYTES:
        raise ValueError
    digest = sha256()
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(file_fd, min(1024 * 1024, before.st_size - offset), offset)
        if not chunk:
            raise ValueError
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(file_fd)
    if (
        not private_paths._same_identity(before, after)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise ValueError
    return before.st_size, digest.digest()


def _open_exact(identity: _OpaqueIdentity) -> tuple[int, int, str]:
    parent_fd, leaf = private_paths._open_verified_parent(
        identity._path,
        missing_leaf_allowed=False,
    )
    file_fd = -1
    try:
        parent_stat = os.fstat(parent_fd)
        if not private_paths._same_identity(parent_stat, identity._parent_identity):
            raise ValueError
        file_fd = os.open(
            leaf,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOCTTY", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(file_fd)
        entry = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            identity._file_identity is None
            or not private_paths._same_identity(opened, identity._file_identity)
            or not private_paths._same_identity(entry, identity._file_identity)
            or not _valid_private_stat(opened)
            or not _valid_private_stat(entry)
            or not _sidecars_absent(parent_fd, leaf)
        ):
            raise ValueError
        if (
            identity._content_evidence is not None
            and _content_evidence(file_fd) != identity._content_evidence
        ):
            raise ValueError
        return parent_fd, file_fd, leaf
    except BaseException:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)
        raise


def _immutable_validate(identity: _OpaqueIdentity) -> None:
    parent_fd, file_fd, _leaf = _open_exact(identity)
    try:
        os.fsync(file_fd)
        os.fsync(parent_fd)
        connection = connect_private_sqlite_descriptor(
            "tts.profile_migration_publication_descriptor",
            file_fd,
            isolation_level=None,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA query_only = ON")
            schema_version = identity._schema_version
            if schema_version is None:
                row = connection.execute("PRAGMA user_version").fetchone()
                if (
                    row is None
                    or len(row) != 1
                    or type(row[0]) is not int
                    or row[0] not in (1, 2, 3, 4)
                ):
                    raise ValueError
                schema_version = row[0]
                identity._schema_version = schema_version
            profile_schema.validate_profile_store_version(connection, schema_version)
        finally:
            connection.close()
        if identity._content_evidence is not None:
            _pin = _content_evidence(file_fd)
            if _pin != identity._content_evidence:
                raise ValueError
        _parent_fd, reopened_fd, _reopened_leaf = _open_exact(identity)
        os.close(reopened_fd)
        os.close(_parent_fd)
    finally:
        os.close(file_fd)
        os.close(parent_fd)


def _pin_content(identity: _OpaqueIdentity) -> None:
    parent_fd, file_fd, _leaf = _open_exact(identity)
    try:
        identity._content_evidence = _content_evidence(file_fd)
    finally:
        os.close(file_fd)
        os.close(parent_fd)


def _prepare_parent(path: str | os.PathLike[str]) -> tuple[Path, os.stat_result]:
    selected = lexical_path(path)
    result = secure_private_directory(
        selected.parent,
        create=False,
        application_owned=True,
    )
    if result.status is PrivatePathStatus.UNVERIFIED_PLATFORM:
        raise ValueError
    parent = selected.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.geteuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise ValueError
    return selected, parent


def prepare_profile_migration_artifact(
    path: str | os.PathLike[str],
    *,
    slot: ProfileMigrationPublicationSlot,
) -> PreparedProfileMigrationArtifact:
    """Validate, fsync, and retain one exact private prepared artifact."""

    try:
        if type(slot) is not ProfileMigrationPublicationSlot:
            raise ValueError
        selected, parent = _prepare_parent(path)
        if selected.name != PROFILE_MIGRATION_CANDIDATE_LEAVES[slot]:
            raise ValueError
        file_identity = selected.lstat()
        artifact = PreparedProfileMigrationArtifact(
            _IDENTITY_FACTORY_TOKEN,
            path=selected,
            slot=slot,
            parent_identity=parent,
            file_identity=file_identity,
            schema_version=_SLOT_VERSION[slot],
        )
        _pin_content(artifact)
        _immutable_validate(artifact)
        return artifact
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        raise _safe_failure() from None


def retain_profile_migration_destination(
    path: str | os.PathLike[str],
    *,
    slot: ProfileMigrationPublicationSlot,
    must_exist: bool,
) -> RetainedProfileMigrationDestination:
    """Pin one existing or fresh publication destination without exposing it."""

    try:
        if (
            type(slot) is not ProfileMigrationPublicationSlot
            or type(must_exist) is not bool
        ):
            raise ValueError
        selected, parent = _prepare_parent(path)
        try:
            file_identity = selected.lstat()
        except FileNotFoundError:
            file_identity = None
        if must_exist and file_identity is None:
            raise ValueError
        destination = RetainedProfileMigrationDestination(
            _IDENTITY_FACTORY_TOKEN,
            path=selected,
            slot=slot,
            parent_identity=parent,
            file_identity=file_identity,
            schema_version=(
                None
                if slot is ProfileMigrationPublicationSlot.ACTIVE
                else _SLOT_VERSION[slot]
            ),
            must_exist=must_exist,
        )
        if file_identity is not None:
            _pin_content(destination)
            _immutable_validate(destination)
        else:
            parent_fd, leaf = private_paths._open_verified_parent(
                selected,
                missing_leaf_allowed=True,
            )
            try:
                if not _sidecars_absent(parent_fd, leaf):
                    raise ValueError
            finally:
                os.close(parent_fd)
        return destination
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        raise _safe_failure() from None


def _journal_authority(
    artifacts: Sequence[PreparedProfileMigrationArtifact],
    destinations: Sequence[RetainedProfileMigrationDestination],
) -> _JournalAuthority:
    rows: list[
        tuple[
            ProfileMigrationJournalSlot,
            os.stat_result,
            tuple[int, bytes],
            int,
            os.stat_result | None,
            tuple[int, bytes] | None,
            int | None,
        ]
    ] = []
    for artifact, destination in zip(artifacts, destinations, strict=True):
        if (
            artifact._file_identity is None
            or artifact._content_evidence is None
            or artifact._schema_version is None
            or (
                destination._file_identity is not None
                and (
                    destination._content_evidence is None
                    or destination._schema_version is None
                )
            )
        ):
            raise ValueError
        rows.append(
            (
                ProfileMigrationJournalSlot(
                    slot=artifact._slot,
                    candidate=PROFILE_MIGRATION_CANDIDATE_LEAVES[artifact._slot],
                    target=destination._path.name,
                    rollback=PROFILE_MIGRATION_ROLLBACK_LEAVES[artifact._slot],
                    had_prior=destination._file_identity is not None,
                ),
                artifact._file_identity,
                artifact._content_evidence,
                artifact._schema_version,
                destination._file_identity,
                destination._content_evidence,
                (
                    None
                    if destination._file_identity is None
                    else destination._schema_version
                ),
            )
        )
    return _new_profile_migration_journal_authority(
        parent_identity=artifacts[0]._parent_identity,
        rows=tuple(rows),
    )


def _identity_at(
    source: _OpaqueIdentity,
    path: Path,
    file_identity: os.stat_result,
) -> _OpaqueIdentity:
    identity = _OpaqueIdentity(
        _IDENTITY_FACTORY_TOKEN,
        path=path,
        slot=source._slot,
        parent_identity=source._parent_identity,
        file_identity=file_identity,
        schema_version=source._schema_version,
    )
    identity._content_evidence = source._content_evidence
    return identity


def _require_absent(path: Path, parent_identity: os.stat_result) -> None:
    parent_fd, leaf = private_paths._open_verified_parent(
        path,
        missing_leaf_allowed=True,
    )
    try:
        if not private_paths._same_identity(os.fstat(parent_fd), parent_identity):
            raise ValueError
        try:
            os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError
        if not _sidecars_absent(parent_fd, leaf):
            raise ValueError
    finally:
        os.close(parent_fd)


def _rename_exact(
    source: _OpaqueIdentity,
    destination_path: Path,
    parent_authority: ParentAuthority,
) -> _OpaqueIdentity:
    source_parent_fd, source_fd, source_leaf = _open_exact(source)
    try:
        assert source._file_identity is not None
        move_exact_noreplace(
            source._path,
            destination_path,
            parent_authority=parent_authority,
            file_identity=source._file_identity,
        )
        moved = _identity_at(
            source,
            destination_path,
            source._file_identity,
        )
        moved_parent_fd, moved_fd, _moved_leaf = _open_exact(moved)
        os.close(moved_fd)
        os.close(moved_parent_fd)
        try:
            os.stat(source_leaf, dir_fd=source_parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError
        return moved
    finally:
        os.close(source_fd)
        os.close(source_parent_fd)


def _fsync_exact(identity: _OpaqueIdentity) -> None:
    parent_fd, file_fd, _leaf = _open_exact(identity)
    try:
        os.fsync(file_fd)
        parent_before = os.fstat(parent_fd)
        if not private_paths._same_identity(parent_before, identity._parent_identity):
            raise ValueError
        os.fsync(parent_fd)
        opened = os.fstat(file_fd)
        entry = identity._path.lstat()
        if (
            not private_paths._same_identity(opened, identity._file_identity)
            or not private_paths._same_identity(entry, identity._file_identity)
            or not _valid_private_stat(opened)
            or not _valid_private_stat(entry)
            or not private_paths._same_identity(
                os.fstat(parent_fd), identity._parent_identity
            )
        ):
            raise ValueError
    finally:
        os.close(file_fd)
        os.close(parent_fd)


def _append_journal(
    path: Path,
    identity: _JournalIdentity,
    payload: bytes,
) -> None:
    parent_fd, leaf = private_paths._open_verified_parent(
        path,
        missing_leaf_allowed=False,
    )
    file_fd = -1
    try:
        file_fd = os.open(
            leaf,
            os.O_WRONLY
            | os.O_APPEND
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(file_fd)
        entry = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not private_paths._same_identity(os.fstat(parent_fd), identity.parent)
            or not private_paths._same_identity(opened, identity.file)
            or not private_paths._same_identity(entry, identity.file)
            or not _valid_private_stat(opened)
            or not _valid_private_stat(entry)
            or opened.st_size + len(payload) > MAX_PROFILE_MIGRATION_JOURNAL_BYTES
        ):
            raise ValueError
        offset = 0
        while offset < len(payload):
            offset += os.write(file_fd, payload[offset:])
        os.fsync(file_fd)
        opened = os.fstat(file_fd)
        entry = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not private_paths._same_identity(os.fstat(parent_fd), identity.parent)
            or not private_paths._same_identity(opened, identity.file)
            or not private_paths._same_identity(entry, identity.file)
            or not _valid_private_stat(opened)
            or not _valid_private_stat(entry)
        ):
            raise ValueError
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


def _post_ponr_stage(
    stage_hook: Callable[[ProfileMigrationPublicationStage], None] | None,
    stage: ProfileMigrationPublicationStage,
    deferred: list[BaseException],
) -> None:
    if stage_hook is None:
        return
    try:
        stage_hook(stage)
    except BaseException as error:
        if isinstance(error, Exception):
            raise
        if not deferred:
            deferred.append(error)


def _publish_slot(
    state: _PublicationSlotState,
    *,
    stage_hook: Callable[[ProfileMigrationPublicationStage], None] | None,
    deferred: list[BaseException],
) -> None:
    is_active = state.artifact._slot is ProfileMigrationPublicationSlot.ACTIVE
    if state.destination._file_identity is not None:
        _rename_exact(
            state.destination,
            state.rollback_path,
            state.parent_authority,
        )
        state.prior_retained = True
        _fsync_exact(
            _identity_at(
                state.destination,
                state.rollback_path,
                state.destination._file_identity,
            )
        )
        _post_ponr_stage(
            stage_hook,
            ProfileMigrationPublicationStage.ACTIVE_RETAINED
            if is_active
            else ProfileMigrationPublicationStage.BACKUP_RETAINED,
            deferred,
        )
    else:
        _require_absent(
            state.destination._path,
            state.destination._parent_identity,
        )

    published = _rename_exact(
        state.artifact,
        state.destination._path,
        state.parent_authority,
    )
    state.candidate_published = True
    _post_ponr_stage(
        stage_hook,
        ProfileMigrationPublicationStage.ACTIVE_REPLACED
        if is_active
        else ProfileMigrationPublicationStage.BACKUP_REPLACED,
        deferred,
    )
    _fsync_exact(published)
    _post_ponr_stage(
        stage_hook,
        ProfileMigrationPublicationStage.ACTIVE_FSYNCED
        if is_active
        else ProfileMigrationPublicationStage.BACKUP_FSYNCED,
        deferred,
    )
    _immutable_validate(published)
    _post_ponr_stage(
        stage_hook,
        ProfileMigrationPublicationStage.ACTIVE_REOPENED
        if is_active
        else ProfileMigrationPublicationStage.BACKUP_REOPENED,
        deferred,
    )


def _restore_slot(state: _PublicationSlotState) -> None:
    if state.candidate_published:
        assert state.artifact._file_identity is not None
        published = _identity_at(
            state.artifact,
            state.destination._path,
            state.artifact._file_identity,
        )
        restored_candidate = _rename_exact(
            published,
            state.artifact._path,
            state.parent_authority,
        )
        _fsync_exact(restored_candidate)
        state.candidate_published = False
    if state.prior_retained:
        assert state.destination._file_identity is not None
        rollback = _identity_at(
            state.destination,
            state.rollback_path,
            state.destination._file_identity,
        )
        restored_prior = _rename_exact(
            rollback,
            state.destination._path,
            state.parent_authority,
        )
        _fsync_exact(restored_prior)
        _immutable_validate(restored_prior)
        state.prior_retained = False
    elif state.destination._file_identity is not None:
        _immutable_validate(state.destination)
    else:
        _require_absent(
            state.destination._path,
            state.destination._parent_identity,
        )


def _restore_all(states: Sequence[_PublicationSlotState]) -> list[BaseException]:
    errors: list[BaseException] = []
    for state in reversed(states):
        try:
            _restore_slot(state)
        except BaseException as error:
            errors.append(error)
    return errors


def _claim(
    artifacts: tuple[PreparedProfileMigrationArtifact, ...],
    destinations: tuple[RetainedProfileMigrationDestination, ...],
) -> tuple[Path, tuple[int, int, str]]:
    if (
        not artifacts
        or len(artifacts) != len(destinations)
        or artifacts[0]._slot is not ProfileMigrationPublicationSlot.ACTIVE
        or destinations[0]._slot is not ProfileMigrationPublicationSlot.ACTIVE
    ):
        raise ValueError
    slots = tuple(artifact._slot for artifact in artifacts)
    if slots not in PROFILE_MIGRATION_SLOT_SEQUENCES:
        raise ValueError
    seen_slots: set[ProfileMigrationPublicationSlot] = set()
    seen_paths: set[Path] = set()
    parent = artifacts[0]._path.parent
    parent_identity = artifacts[0]._parent_identity
    for artifact, destination in zip(artifacts, destinations, strict=True):
        if (
            type(artifact) is not PreparedProfileMigrationArtifact
            or type(destination) is not RetainedProfileMigrationDestination
            or artifact._state != "ready"
            or destination._state != "ready"
            or artifact._thread_id != get_ident()
            or destination._thread_id != get_ident()
            or artifact._slot is not destination._slot
            or artifact._path.name != PROFILE_MIGRATION_CANDIDATE_LEAVES[artifact._slot]
            or artifact._slot in seen_slots
            or artifact._path in seen_paths
            or destination._path in seen_paths
            or artifact._path.parent != parent
            or destination._path.parent != parent
            or not private_paths._same_identity(
                artifact._parent_identity, parent_identity
            )
            or not private_paths._same_identity(
                destination._parent_identity, parent_identity
            )
        ):
            raise ValueError
        seen_slots.add(artifact._slot)
        seen_paths.update((artifact._path, destination._path))
    key = (parent_identity.st_dev, parent_identity.st_ino, destinations[0]._path.name)
    with _PUBLICATION_LOCK:
        if key in _ACTIVE_PUBLICATIONS:
            raise ValueError
        _ACTIVE_PUBLICATIONS.add(key)
    for item in (*artifacts, *destinations):
        item._state = "claimed"
    return parent / f".{destinations[0]._path.name}.migration-publication.json", key


def _write_new_journal(path: Path, payload: bytes) -> _JournalIdentity:
    selected, parent_identity = _prepare_parent(path)
    parent_authority = ParentAuthority(parent_identity)
    parent_fd = -1
    file_fd = -1
    identity: os.stat_result | None = None
    body_error: BaseException | None = None
    try:
        parent_fd, file_fd, identity, parent_authority = (
            open_new_or_reused_private_file(
                selected,
                parent_authority=parent_authority,
                tombstone_key=MigrationTombstoneKey.JOURNAL,
            )
        )
        parent_identity = parent_authority.identity
        offset = 0
        while offset < len(payload):
            offset += os.write(file_fd, payload[offset:])
        os.fsync(file_fd)
        identity = os.fstat(file_fd)
        entry = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not private_paths._same_identity(identity, entry)
            or not _valid_private_stat(identity)
            or not _valid_private_stat(entry)
        ):
            raise ValueError
        os.fsync(parent_fd)
        return _JournalIdentity(identity, parent_identity)
    except BaseException as error:
        body_error = error
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if body_error is not None and identity is not None:
            try:
                remove_exact_namespace(
                    selected,
                    parent_authority=parent_authority,
                    file_identity=identity,
                    tombstone_key=MigrationTombstoneKey.JOURNAL,
                )
            except BaseException:
                pass
        if parent_fd >= 0:
            os.close(parent_fd)
    assert body_error is not None
    raise body_error


def _unlink_exact(
    path: Path,
    identity: os.stat_result | None,
    parent_authority: ParentAuthority,
    tombstone_key: MigrationTombstoneKey,
) -> bool:
    if identity is None:
        return False
    try:
        remove_exact_namespace(
            path,
            parent_authority=parent_authority,
            file_identity=identity,
            tombstone_key=tombstone_key,
        )
        return True
    except (FileNotFoundError, ValueError):
        return False


def _finish_claim(
    artifacts: Sequence[PreparedProfileMigrationArtifact],
    destinations: Sequence[RetainedProfileMigrationDestination],
    key: tuple[int, int, str] | None,
) -> None:
    for item in (*artifacts, *destinations):
        item._state = "terminal"
    if key is not None:
        with _PUBLICATION_LOCK:
            _ACTIVE_PUBLICATIONS.discard(key)


def _cleanup_exact(
    path: Path,
    identity: os.stat_result | None,
    parent_authority: ParentAuthority,
    tombstone_key: MigrationTombstoneKey,
) -> None:
    if not _unlink_exact(path, identity, parent_authority, tombstone_key):
        raise OSError


def publish_profile_migration(
    *,
    active_candidate: PreparedProfileMigrationArtifact,
    backup_candidates: Sequence[PreparedProfileMigrationArtifact],
    active_destination: RetainedProfileMigrationDestination,
    backup_destinations: Sequence[RetainedProfileMigrationDestination],
    stage_hook: Callable[[ProfileMigrationPublicationStage], None] | None = None,
) -> None:
    """Publish one exact multi-file migration or restore every prior identity."""

    artifacts = (active_candidate, *tuple(backup_candidates))
    destinations = (active_destination, *tuple(backup_destinations))
    journal_path: Path | None = None
    journal_identity: _JournalIdentity | None = None
    authority_checksum: bytes | None = None
    key: tuple[int, int, str] | None = None
    body_error: BaseException | None = None
    deferred: list[BaseException] = []
    states: tuple[_PublicationSlotState, ...] = ()
    ponr = False
    completed = False
    parent_authority: ParentAuthority | None = None
    try:
        journal_path, key = _claim(artifacts, destinations)
        parent_authority = ParentAuthority(journal_path.parent.stat())
        for artifact in artifacts:
            _immutable_validate(artifact)
        for destination in destinations:
            if destination._file_identity is not None:
                _immutable_validate(destination)
        if stage_hook is not None:
            stage_hook(ProfileMigrationPublicationStage.PREFLIGHT)
        authority = _journal_authority(artifacts, destinations)
        payload, authority_checksum = _encode_initial_journal(authority)
        journal_identity = _write_new_journal(journal_path, payload)
        parent_authority = ParentAuthority(journal_path.parent.stat())
        if stage_hook is not None:
            stage_hook(ProfileMigrationPublicationStage.JOURNAL_DURABLE)
        states = tuple(
            _PublicationSlotState(
                artifact=artifact,
                destination=destination,
                rollback_path=destination._path.with_name(
                    PROFILE_MIGRATION_ROLLBACK_LEAVES[artifact._slot]
                ),
                parent_authority=parent_authority,
            )
            for artifact, destination in zip(artifacts, destinations, strict=True)
        )
        _append_journal(
            journal_path,
            journal_identity,
            _encode_later_journal(authority_checksum, phase="publishing"),
        )
        ponr = True
        _post_ponr_stage(
            stage_hook,
            ProfileMigrationPublicationStage.PONR,
            deferred,
        )
        for state in states:
            _publish_slot(state, stage_hook=stage_hook, deferred=deferred)
        _append_journal(
            journal_path,
            journal_identity,
            _encode_later_journal(authority_checksum, phase="complete"),
        )
        _post_ponr_stage(
            stage_hook,
            ProfileMigrationPublicationStage.FINAL_JOURNAL_DURABLE,
            deferred,
        )
        completed = True
    except BaseException as error:
        body_error = error

    if completed:
        complete_cleanup_errors: list[BaseException] = []
        for state in states:
            if state.prior_retained:
                try:
                    _cleanup_exact(
                        state.rollback_path,
                        state.destination._file_identity,
                        state.parent_authority,
                        _ROLLBACK_TOMBSTONES[state.artifact._slot],
                    )
                except BaseException as caught:
                    complete_cleanup_errors.append(caught)
        if (
            not complete_cleanup_errors
            and journal_path is not None
            and journal_identity is not None
            and parent_authority is not None
        ):
            try:
                _cleanup_exact(
                    journal_path,
                    journal_identity.file,
                    parent_authority,
                    MigrationTombstoneKey.JOURNAL,
                )
            except BaseException as caught:
                complete_cleanup_errors.append(caught)
        _finish_claim(artifacts, destinations, key)
        _redeliver_control_flow(*deferred, *complete_cleanup_errors)
        if complete_cleanup_errors:
            raise _safe_failure() from None
        return

    if ponr:
        assert parent_authority is not None
        journal_update_errors: list[BaseException] = []
        if (
            journal_path is not None
            and journal_identity is not None
            and authority_checksum is not None
        ):
            try:
                _append_journal(
                    journal_path,
                    journal_identity,
                    _encode_later_journal(authority_checksum, phase="restoring"),
                )
            except BaseException as caught:
                journal_update_errors.append(caught)
        restore_errors = _restore_all(states)
        if restore_errors:
            if (
                journal_path is not None
                and journal_identity is not None
                and authority_checksum is not None
            ):
                try:
                    _append_journal(
                        journal_path,
                        journal_identity,
                        _encode_later_journal(
                            authority_checksum,
                            phase="unavailable",
                        ),
                    )
                except BaseException as caught:
                    journal_update_errors.append(caught)
            _finish_claim(artifacts, destinations, key)
            # Authority is indeterminate: the bounded unavailable state must
            # dominate deferred control flow until Task D recovery completes.
            raise _safe_failure("unavailable") from None

        restore_cleanup_errors: list[BaseException] = []
        for artifact in artifacts:
            try:
                _cleanup_exact(
                    artifact._path,
                    artifact._file_identity,
                    parent_authority,
                    _CANDIDATE_TOMBSTONES[artifact._slot],
                )
            except BaseException as caught:
                restore_cleanup_errors.append(caught)
        if (
            not restore_cleanup_errors
            and journal_path is not None
            and journal_identity is not None
        ):
            try:
                _cleanup_exact(
                    journal_path,
                    journal_identity.file,
                    parent_authority,
                    MigrationTombstoneKey.JOURNAL,
                )
            except BaseException as caught:
                restore_cleanup_errors.append(caught)
        _finish_claim(artifacts, destinations, key)
        _redeliver_control_flow(
            body_error,
            *deferred,
            *journal_update_errors,
            *restore_cleanup_errors,
        )
        raise _safe_failure() from None

    if key is None:
        raise _safe_failure() from None

    prepublication_cleanup_errors: list[BaseException] = []
    assert parent_authority is not None
    for artifact in artifacts:
        try:
            _cleanup_exact(
                artifact._path,
                artifact._file_identity,
                parent_authority,
                _CANDIDATE_TOMBSTONES[artifact._slot],
            )
        except BaseException as caught:
            prepublication_cleanup_errors.append(caught)
    if (
        not prepublication_cleanup_errors
        and journal_path is not None
        and journal_identity is not None
    ):
        try:
            _cleanup_exact(
                journal_path,
                journal_identity.file,
                parent_authority,
                MigrationTombstoneKey.JOURNAL,
            )
        except BaseException as caught:
            prepublication_cleanup_errors.append(caught)
    _finish_claim(artifacts, destinations, key)

    if (
        prepublication_cleanup_errors
        and body_error is not None
        and not isinstance(body_error, Exception)
    ):
        raise _safe_failure("unavailable") from None
    _redeliver_control_flow(body_error)
    raise _safe_failure() from None


__all__ = [
    "MAX_PROFILE_MIGRATION_ARTIFACT_BYTES",
    "ParsedProfileMigrationJournal",
    "PreparedProfileMigrationArtifact",
    "ProfileMigrationPublicationSlot",
    "ProfileMigrationPublicationStage",
    "RetainedProfileMigrationDestination",
    "parse_profile_migration_journal",
    "prepare_profile_migration_artifact",
    "publish_profile_migration",
    "retain_profile_migration_destination",
]
