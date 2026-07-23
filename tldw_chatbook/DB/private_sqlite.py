"""Checked ownership and private connection boundary for SQLite targets."""

from __future__ import annotations

import errno
import os
import sqlite3
import stat
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from threading import Lock
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import quote

import tldw_chatbook.Utils.private_paths as private_paths
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
    lexical_path,
    verify_trusted_directory,
)


class SQLiteTargetKind(StrEnum):
    """Supported SQLite storage target classifications."""

    PRIVATE_FILE = "private_file"
    MEMORY = "memory"
    READ_ONLY_URI = "read_only_uri"


class SQLitePrivacyUnverifiedWarning(RuntimeWarning):
    """Warn that a successful SQLite file open lacks verified ACL privacy."""


@dataclass(frozen=True, slots=True)
class SQLiteOwnerPolicy:
    """Immutable storage policy for one registered production owner."""

    production_module: str
    allowed_target_kinds: frozenset[SQLiteTargetKind]
    reason: str
    centralized_backup_allowed: bool = False


_PRIVATE_FILE = frozenset({SQLiteTargetKind.PRIVATE_FILE})
_MEMORY = frozenset({SQLiteTargetKind.MEMORY})
_PRIVATE_OR_MEMORY = frozenset({SQLiteTargetKind.PRIVATE_FILE, SQLiteTargetKind.MEMORY})
_READ_ONLY_URI = frozenset({SQLiteTargetKind.READ_ONLY_URI})

_SQLITE_OWNER_POLICIES = {
    "app.prompts_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/app",
        _PRIVATE_FILE,
        "Prompts startup participates in configured database parent policy.",
    ),
    "config.server_sqlite_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/config",
        _PRIVATE_FILE,
        "Stale server-only SQLite directory creation has no connection consumer.",
    ),
    "config.server_user_db_base": SQLiteOwnerPolicy(
        "tldw_chatbook/config",
        _PRIVATE_FILE,
        "Stale server user-database creation has no connection consumer.",
    ),
    "config.user_data_directory": SQLiteOwnerPolicy(
        "tldw_chatbook/config",
        _PRIVATE_FILE,
        "The application-owned default data directory is the private root.",
    ),
    "cookies.chrome": SQLiteOwnerPolicy(
        "tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner",
        _READ_ONLY_URI,
        "Chrome cookie clones are validated read-only SQLite sources.",
    ),
    "cookies.edge": SQLiteOwnerPolicy(
        "tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner",
        _READ_ONLY_URI,
        "Edge cookie clones are validated read-only SQLite sources.",
    ),
    "cookies.firefox": SQLiteOwnerPolicy(
        "tldw_chatbook/Web_Scraping/cookie_scraping/cookie_cloner",
        _READ_ONLY_URI,
        "Firefox cookie clones are validated read-only SQLite sources.",
    ),
    "db.base": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/base_db",
        _PRIVATE_OR_MEMORY,
        "BaseDB is the shared file and memory connection owner for subclasses.",
    ),
    "db.chachanotes.backup": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/ChaChaNotes_DB",
        _PRIVATE_FILE,
        "ChaChaNotes backup targets require centralized private creation.",
        centralized_backup_allowed=True,
    ),
    "db.chachanotes.primary": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/ChaChaNotes_DB",
        _PRIVATE_OR_MEMORY,
        "ChaChaNotes owns private file and in-memory primary databases.",
    ),
    "db.evals": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Evals_DB",
        _PRIVATE_OR_MEMORY,
        "Evaluation storage supports private files and exact in-memory targets.",
    ),
    "db.library_ingest_jobs": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Library_Ingest_Jobs_DB",
        _PRIVATE_OR_MEMORY,
        "Library ingest jobs override the shared connection owner.",
    ),
    "db.media.backup": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Client_Media_DB_v2",
        _PRIVATE_FILE,
        "Media backup targets require centralized private creation.",
        centralized_backup_allowed=True,
    ),
    "db.media.integrity": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Client_Media_DB_v2",
        _READ_ONLY_URI,
        "Media integrity checks use a validated read-only SQLite URI.",
    ),
    "db.media.primary": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Client_Media_DB_v2",
        _PRIVATE_OR_MEMORY,
        "Media owns private file and in-memory primary databases.",
    ),
    "db.prompts.backup": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Prompts_DB",
        _PRIVATE_FILE,
        "Prompts backup targets require centralized private creation.",
        centralized_backup_allowed=True,
    ),
    "db.prompts.primary": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Prompts_DB",
        _PRIVATE_OR_MEMORY,
        "Prompts owns private file and in-memory primary databases.",
    ),
    "db.rag_indexing": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/RAG_Indexing_DB",
        _PRIVATE_OR_MEMORY,
        "RAG indexing supports private files and exact in-memory targets.",
    ),
    "db.search_history": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/search_history_db",
        _PRIVATE_OR_MEMORY,
        "Search history supports private files and exact in-memory targets.",
    ),
    "db.sync_client_example": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Sync_Client",
        _PRIVATE_FILE,
        "The executable sync example must not teach unsafe parent creation.",
    ),
    "eval.events_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/Event_Handlers/eval_events",
        _PRIVATE_FILE,
        "The evaluation event factory participates in default parent setup.",
    ),
    "eval.orchestrator_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/Evals/eval_orchestrator",
        _PRIVATE_FILE,
        "The evaluation orchestrator participates in default parent setup.",
    ),
    "kanban.local": SQLiteOwnerPolicy(
        "tldw_chatbook/Kanban_Interop/local_kanban_db",
        _PRIVATE_OR_MEMORY,
        "Local Kanban supports private files and exact in-memory targets.",
    ),
    "notes.library_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/Notes/Notes_Library",
        _PRIVATE_FILE,
        "The Notes library owns a per-user database parent.",
    ),
    "notifications.client": SQLiteOwnerPolicy(
        "tldw_chatbook/Notifications/client_notifications_db",
        _MEMORY,
        "Client notifications currently use only an in-memory database.",
    ),
    "notifications.event_state": SQLiteOwnerPolicy(
        "tldw_chatbook/Notifications/event_state_repository",
        _MEMORY,
        "Event state currently uses only an in-memory database.",
    ),
    "research.local": SQLiteOwnerPolicy(
        "tldw_chatbook/Research_Interop/local_research_service",
        _PRIVATE_OR_MEMORY,
        "Local research accepts private files and Path(':memory:').",
    ),
    "runtime.server_parity_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/runtime_policy/server_parity_state",
        _PRIVATE_FILE,
        "Server parity repositories use file-backed storage below this parent.",
    ),
    "settings.bulk_backup": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _PRIVATE_FILE,
        "The Settings bulk worker backs up all three Chatbook databases.",
        centralized_backup_allowed=True,
    ),
    "settings.integrity": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _READ_ONLY_URI,
        "Settings integrity checks require validated read-only access.",
    ),
    "settings.pre_restore_backup": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _PRIVATE_FILE,
        "Settings creates a private safety backup before restoring.",
        centralized_backup_allowed=True,
    ),
    "settings.restore": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _PRIVATE_FILE,
        "Settings restore uses verified source and destination identities.",
        centralized_backup_allowed=True,
    ),
    "settings.schema": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _READ_ONLY_URI,
        "Settings schema inspection requires validated read-only access.",
    ),
    "settings.single_backup": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _PRIVATE_FILE,
        "Settings single-database backups use centralized private creation.",
        centralized_backup_allowed=True,
    ),
    "settings.vacuum": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _PRIVATE_FILE,
        "Settings VACUUM requires a checked writable private database.",
    ),
    "sync.notes_mirror": SQLiteOwnerPolicy(
        "tldw_chatbook/Sync_Interop/notes_mirror",
        _PRIVATE_OR_MEMORY,
        "Notes mirror supports an optional private file or exact memory target.",
    ),
    "sync.state": SQLiteOwnerPolicy(
        "tldw_chatbook/Sync_Interop/sync_state_repository",
        _MEMORY,
        "Sync state currently uses only an in-memory database.",
    ),
    "tamagotchi.sqlite": SQLiteOwnerPolicy(
        "tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage",
        _PRIVATE_OR_MEMORY,
        "All SQLiteStorage methods share private file and Path(':memory:') support.",
    ),
    "utils.legacy_user_database_path": SQLiteOwnerPolicy(
        "tldw_chatbook/Utils/paths",
        _PRIVATE_FILE,
        "The unused legacy user database helper is an explicit exclusion.",
    ),
    "utils.project_databases_directory": SQLiteOwnerPolicy(
        "tldw_chatbook/Utils/paths",
        _PRIVATE_FILE,
        "Project template and demonstration databases are explicit exclusions.",
    ),
    "writing.local": SQLiteOwnerPolicy(
        "tldw_chatbook/Writing_Interop/local_writing_service",
        _PRIVATE_OR_MEMORY,
        "Local writing accepts private files and Path(':memory:').",
    ),
}

SQLITE_OWNER_REGISTRY: Mapping[str, SQLiteOwnerPolicy] = MappingProxyType(
    _SQLITE_OWNER_POLICIES
)

_WARNED_UNVERIFIED_OWNER_IDS: set[str] = set()
_UNVERIFIED_WARNING_LOCK = Lock()

_PRIVATE_FILE_MODE = 0o600
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _warn_unverified_platform(owner_id: str) -> None:
    with _UNVERIFIED_WARNING_LOCK:
        if owner_id in _WARNED_UNVERIFIED_OWNER_IDS:
            return
        warnings.warn(
            "SQLite file privacy is unverified on this platform",
            SQLitePrivacyUnverifiedWarning,
            stacklevel=3,
        )
        _WARNED_UNVERIFIED_OWNER_IDS.add(owner_id)


def _failure(
    selected: Path,
    status: PrivatePathStatus,
    reason: str,
) -> PrivatePathError:
    return PrivatePathError(PrivatePathResult(selected, status, reason=reason))


def _open_artifact_fd(
    parent_fd: int,
    leaf: str,
    *,
    writable: bool,
    create: bool,
) -> int:
    flags = os.O_RDWR if writable else os.O_RDONLY
    flags |= (
        getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOCTTY", 0)
    )
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    return os.open(
        leaf,
        flags,
        _PRIVATE_FILE_MODE,
        dir_fd=parent_fd,
    )


def _artifact_postcondition_holds(
    file_fd: int,
    parent_fd: int,
    leaf: str,
    *,
    expected_identity: os.stat_result,
    selected: Path,
) -> bool:
    del selected
    opened = os.fstat(file_fd)
    entry = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    return (
        private_paths._same_identity(opened, expected_identity)
        and private_paths._same_identity(entry, expected_identity)
        and stat.S_ISREG(opened.st_mode)
        and opened.st_nlink == 1
        and entry.st_nlink == 1
        and opened.st_uid == os.geteuid()
        and stat.S_IMODE(opened.st_mode) == _PRIVATE_FILE_MODE
    )


def _path_error_from_oserror(selected: Path, exc: OSError) -> PrivatePathError:
    status = (
        PrivatePathStatus.LINK_OR_NON_REGULAR
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}
        else PrivatePathStatus.OPERATION_FAILED
    )
    return _failure(selected, status, type(exc).__name__)


def _prepare_posix_artifact(
    selected: Path,
    *,
    writable: bool,
    create_if_missing: bool,
    optional: bool = False,
) -> bool:
    parent_fd, leaf = private_paths._open_verified_parent(
        selected,
        missing_leaf_allowed=create_if_missing,
    )
    file_fd = -1
    writable_fd = -1
    try:
        try:
            entry_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if optional:
                return False
            if not create_if_missing:
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "missing_sqlite_artifact",
                ) from None
            entry_stat = None
        except OSError as exc:
            raise _path_error_from_oserror(selected, exc) from None

        if entry_stat is not None and not stat.S_ISREG(entry_stat.st_mode):
            raise _failure(
                selected,
                PrivatePathStatus.LINK_OR_NON_REGULAR,
                "non_regular_sqlite_artifact",
            )

        created = entry_stat is None
        try:
            file_fd = _open_artifact_fd(
                parent_fd,
                leaf,
                writable=created,
                create=created,
            )
        except OSError as exc:
            raise _path_error_from_oserror(selected, exc) from None

        opened_stat = os.fstat(file_fd)
        rejected = private_paths._classify_private_file_stat(
            opened_stat,
            expected_uid=os.geteuid(),
        )
        if rejected is not None:
            raise _failure(selected, rejected, "unsafe_sqlite_artifact")
        if entry_stat is not None and not private_paths._same_identity(
            entry_stat,
            opened_stat,
        ):
            raise _failure(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                "private_sqlite_identity_changed",
            )

        if stat.S_IMODE(opened_stat.st_mode) != _PRIVATE_FILE_MODE:
            os.fchmod(file_fd, _PRIVATE_FILE_MODE)
        if not _artifact_postcondition_holds(
            file_fd,
            parent_fd,
            leaf,
            expected_identity=opened_stat,
            selected=selected,
        ):
            raise _failure(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                "private_sqlite_postcondition_failed",
            )

        if writable and not created:
            try:
                writable_fd = _open_artifact_fd(
                    parent_fd,
                    leaf,
                    writable=True,
                    create=False,
                )
            except OSError as exc:
                raise _path_error_from_oserror(selected, exc) from None

            writable_stat = os.fstat(writable_fd)
            rejected = private_paths._classify_private_file_stat(
                writable_stat,
                expected_uid=os.geteuid(),
            )
            if rejected is not None:
                raise _failure(selected, rejected, "unsafe_sqlite_artifact")
            if not private_paths._same_identity(opened_stat, writable_stat):
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "private_sqlite_identity_changed",
                )
            if not _artifact_postcondition_holds(
                writable_fd,
                parent_fd,
                leaf,
                expected_identity=opened_stat,
                selected=selected,
            ):
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "private_sqlite_postcondition_failed",
                )
        return True
    except PrivatePathError:
        raise
    except OSError as exc:
        raise _path_error_from_oserror(selected, exc) from None
    finally:
        if writable_fd >= 0:
            os.close(writable_fd)
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


def _prepare_windows_artifact(
    selected: Path,
    *,
    writable: bool,
    create_if_missing: bool,
    optional: bool = False,
) -> bool:
    del writable
    try:
        file_stat = selected.lstat()
    except FileNotFoundError:
        if optional:
            return False
        if not create_if_missing:
            raise _failure(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                "missing_sqlite_artifact",
            ) from None
        file_fd = os.open(
            selected,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            _PRIVATE_FILE_MODE,
        )
        os.close(file_fd)
        return True

    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise _failure(
            selected,
            PrivatePathStatus.LINK_OR_NON_REGULAR,
            "unsafe_sqlite_artifact",
        )
    return True


def _prepare_artifact(
    selected: Path,
    *,
    writable: bool,
    create_if_missing: bool,
    optional: bool = False,
) -> bool:
    if private_paths._posix_guards_available():
        return _prepare_posix_artifact(
            selected,
            writable=writable,
            create_if_missing=create_if_missing,
            optional=optional,
        )
    if private_paths._WINDOWS_PLATFORM:
        return _prepare_windows_artifact(
            selected,
            writable=writable,
            create_if_missing=create_if_missing,
            optional=optional,
        )
    raise _failure(
        selected,
        PrivatePathStatus.OPERATION_FAILED,
        "required_posix_guards_unavailable",
    )


def _build_read_only_uri(
    database: str | os.PathLike[str],
    *,
    windows: bool | None = None,
) -> str:
    raw = os.fspath(database)
    if not isinstance(raw, str):
        raise TypeError("SQLite paths must be text paths")
    if "\x00" in raw:
        raise ValueError("Path must not contain NUL")
    windows = os.name == "nt" if windows is None else windows
    if windows:
        path = PureWindowsPath(raw)
        posix_path = path.as_posix()
        if posix_path.startswith("//"):
            authority_and_path = posix_path[2:].split("/", 1)
            authority = authority_and_path[0]
            remainder = authority_and_path[1] if len(authority_and_path) == 2 else ""
            encoded = quote(remainder, safe="/")
            return f"file://{authority}/{encoded}?mode=ro"
        encoded = quote(posix_path, safe="/:")
        return f"file:///{encoded}?mode=ro"

    encoded = quote(raw, safe="/")
    return f"file:{encoded}?mode=ro"


def _validated_owner_policy(owner_id: str) -> SQLiteOwnerPolicy:
    try:
        return SQLITE_OWNER_REGISTRY[owner_id]
    except KeyError:
        raise ValueError(f"Unknown SQLite owner: {owner_id}") from None


def _classify_target(
    database: str | os.PathLike[str],
    *,
    read_only: bool,
) -> tuple[str, SQLiteTargetKind]:
    raw = os.fspath(database)
    if not isinstance(raw, str):
        raise TypeError("SQLite paths must be text paths")
    if "\x00" in raw:
        raise ValueError("Path must not contain NUL")
    if raw.startswith("file:"):
        raise ValueError("Caller-supplied file: SQLite URIs are not supported")
    if raw == ":memory:":
        if read_only:
            raise ValueError("A read-only memory database is not supported")
        return raw, SQLiteTargetKind.MEMORY
    return raw, (
        SQLiteTargetKind.READ_ONLY_URI if read_only else SQLiteTargetKind.PRIVATE_FILE
    )


def connect_private_sqlite(
    owner_id: str,
    database: str | os.PathLike[str],
    *,
    read_only: bool = False,
    **kwargs: Any,
) -> sqlite3.Connection:
    """Open SQLite only after enforcing the registered target policy."""

    if "uri" in kwargs:
        raise ValueError("The SQLite uri option is owned by the private seam")
    policy = _validated_owner_policy(owner_id)
    raw, target_kind = _classify_target(database, read_only=read_only)
    if target_kind not in policy.allowed_target_kinds:
        if not read_only and policy.allowed_target_kinds == _READ_ONLY_URI:
            raise ValueError(f"SQLite owner {owner_id} is read-only")
        raise ValueError(
            f"SQLite target kind {target_kind.value} is not allowed for {owner_id}"
        )

    connection_target = raw
    use_uri = False
    if target_kind is not SQLiteTargetKind.MEMORY:
        selected = lexical_path(raw)
        directory_result = verify_trusted_directory(
            selected.parent,
            allow_shared_sticky=False,
        )
        _prepare_artifact(
            selected,
            writable=not read_only,
            create_if_missing=not read_only,
        )
        for suffix in _SIDECAR_SUFFIXES:
            _prepare_artifact(
                Path(f"{selected}{suffix}"),
                writable=not read_only,
                create_if_missing=False,
                optional=True,
            )
        if read_only:
            connection_target = _build_read_only_uri(
                selected,
                windows=os.name == "nt",
            )
            use_uri = True
        if directory_result.status is PrivatePathStatus.UNVERIFIED_PLATFORM:
            _warn_unverified_platform(owner_id)

    return sqlite3.connect(connection_target, uri=use_uri, **kwargs)


__all__ = [
    "SQLITE_OWNER_REGISTRY",
    "SQLiteOwnerPolicy",
    "SQLitePrivacyUnverifiedWarning",
    "SQLiteTargetKind",
    "connect_private_sqlite",
]
