"""Checked ownership and private connection boundary for SQLite targets."""

from __future__ import annotations

import contextlib
import errno
import os
import sqlite3
import stat
import sys
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from threading import Lock, RLock, get_ident
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, cast
from urllib.parse import quote

import tldw_chatbook.Utils.private_paths as private_paths
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    PrivatePathResult,
    PrivatePathStatus,
    lexical_path,
    secure_private_directory,
    verify_trusted_directory,
)


_SQLITE_CONNECT = sqlite3.connect


class SQLiteTargetKind(StrEnum):
    """Supported SQLite storage target classifications."""

    PRIVATE_FILE = "private_file"
    MEMORY = "memory"
    READ_ONLY_URI = "read_only_uri"


class SQLitePrivacyUnverifiedWarning(RuntimeWarning):
    """Warn that a successful SQLite file open lacks verified ACL privacy."""


class SQLiteRestoreBusyError(sqlite3.OperationalError):
    """Report that a live database cannot be restored safely."""


class SQLiteRestoreIndeterminateError(sqlite3.OperationalError):
    """Report that a committed restore could not be rolled back reliably."""

    def __init__(
        self,
        destination_path: str | os.PathLike[str],
        pre_restore_path: str | os.PathLike[str],
    ) -> None:
        self.destination_path = Path(destination_path)
        self.pre_restore_path = Path(pre_restore_path)
        super().__init__(
            "The live database may already contain restored data because "
            "automatic recovery failed. Inspect "
            f"{self.destination_path} and the pre-restore snapshot at "
            f"{self.pre_restore_path}. Do not retry automatically."
        )


@dataclass(frozen=True, slots=True)
class SQLiteOwnerPolicy:
    """Immutable storage policy for one registered production owner."""

    production_module: str
    allowed_target_kinds: frozenset[SQLiteTargetKind]
    reason: str
    centralized_backup_allowed: bool = False
    preserve_read_only_source_mode: bool = False


_PRIVATE_FILE = frozenset({SQLiteTargetKind.PRIVATE_FILE})
_MEMORY = frozenset({SQLiteTargetKind.MEMORY})
_PRIVATE_OR_MEMORY = frozenset({SQLiteTargetKind.PRIVATE_FILE, SQLiteTargetKind.MEMORY})
_READ_ONLY_URI = frozenset({SQLiteTargetKind.READ_ONLY_URI})
_PRIVATE_AND_READ_ONLY = frozenset(
    {SQLiteTargetKind.PRIVATE_FILE, SQLiteTargetKind.READ_ONLY_URI}
)
_PRIVATE_MEMORY_AND_READ_ONLY = frozenset(
    {
        SQLiteTargetKind.PRIVATE_FILE,
        SQLiteTargetKind.MEMORY,
        SQLiteTargetKind.READ_ONLY_URI,
    }
)

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
    "db.subscriptions.agent_read": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Subscriptions_DB",
        _READ_ONLY_URI,
        "External agent tools read the existing Watchlists database without "
        "creating, migrating, or writing it.",
        preserve_read_only_source_mode=True,
    ),
    "db.subscriptions.site_configs": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Subscriptions_DB",
        _PRIVATE_FILE,
        "ensure_site_configs_schema declares site_configs on a caller-supplied "
        "path without opening the whole SubscriptionsDB.",
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
    "db.chunking_lab": SQLiteOwnerPolicy(
        "tldw_chatbook/DB/Chunking_Lab_DB",
        _PRIVATE_FILE,
        "Chunking Lab owns profile-local recovery checkpoints and immutable payloads.",
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
    "notes.file_notes_replica": SQLiteOwnerPolicy(
        "tldw_chatbook/Notes/file_notes_replica",
        _PRIVATE_OR_MEMORY,
        "The File Notes recovery replica contains private note bytes and "
        "supports private files or an exact in-memory test target.",
    ),
    "notes.library_parent": SQLiteOwnerPolicy(
        "tldw_chatbook/Notes/Notes_Library",
        _PRIVATE_FILE,
        "The Notes library owns a per-user database parent.",
    ),
    "notes.sync_state": SQLiteOwnerPolicy(
        "tldw_chatbook/Notes/notes_device_state_store",
        _PRIVATE_AND_READ_ONLY,
        "Device-private import receipts and lasting-sync state remain "
        "profile-local; planning may inspect an existing ledger read-only, and "
        "the owner remains outside centralized backup.",
    ),
    "notifications.client": SQLiteOwnerPolicy(
        "tldw_chatbook/Notifications/client_notifications_db",
        _MEMORY,
        "Client notifications currently use only an in-memory database.",
    ),
    "notifications.event_state": SQLiteOwnerPolicy(
        "tldw_chatbook/Notifications/event_state_repository",
        _PRIVATE_OR_MEMORY,
        "The durable server-event ledger. The app gives it a private file "
        "under the user data directory (`build_server_parity_state_"
        "repositories`) and falls back to an exact in-memory target when "
        "that directory cannot be trusted. TASK-21131: the file branch used "
        "to be opened by `BaseDB._get_connection` under `db.base`, so this "
        "entry described only half of it; the enforced target kinds are "
        "unchanged (`db.base` allows exactly the same two).",
    ),
    "personal_context.repository": SQLiteOwnerPolicy(
        "tldw_chatbook/Personal_Context/repository",
        _PRIVATE_AND_READ_ONLY,
        "The encrypted Personal Context repository owns one private local database.",
        preserve_read_only_source_mode=True,
    ),
    "personal_context.interview_drafts": SQLiteOwnerPolicy(
        "tldw_chatbook/Personal_Context/interview_draft_repository",
        _PRIVATE_FILE,
        "Encrypted interview drafts are local-only and excluded from centralized backup.",
    ),
    "rag.chachanotes_keyword_leg": SQLiteOwnerPolicy(
        "tldw_chatbook/RAG_Search/simplified/rag_service",
        _READ_ONLY_URI,
        "The RAG keyword leg reads notes and conversations out of the live "
        "ChaChaNotes database. Read-only URI only: search must never write "
        "to, create, or migrate the user's main database, and this owner "
        "preserves the source file's mode rather than reasserting private "
        "permissions on a file another owner already governs.",
        preserve_read_only_source_mode=True,
    ),
    "rag.prompts_keyword_leg": SQLiteOwnerPolicy(
        "tldw_chatbook/RAG_Search/simplified/rag_service",
        _READ_ONLY_URI,
        "The RAG keyword leg reads saved prompts out of the live Prompts "
        "database (TASK-15020/B2) -- the only retrieval path prompts have, "
        "since nothing indexes them semantically. Read-only URI only: "
        "search must never write to, create, or migrate the user's prompts "
        "database, and this owner preserves the source file's mode rather "
        "than reasserting private permissions on a file another owner "
        "already governs.",
        preserve_read_only_source_mode=True,
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
        _PRIVATE_AND_READ_ONLY,
        "Settings bulk backup reads a checked source into a private target.",
        centralized_backup_allowed=True,
    ),
    "settings.integrity": SQLiteOwnerPolicy(
        "tldw_chatbook/UI/Tools_Settings_Window",
        _READ_ONLY_URI,
        "Settings integrity checks require validated read-only access.",
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
    "tts.profile_store": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_schema",
        _PRIVATE_FILE,
        "TTS profile storage requires a checked writable private database.",
    ),
    "tts.profile_store_descriptor": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_schema",
        _READ_ONLY_URI,
        "TTS shared startup proves the exact current store through an immutable "
        "view bound to its retained descriptor before opening the live path.",
        preserve_read_only_source_mode=True,
    ),
    "tts.profile_candidate": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_schema",
        _READ_ONLY_URI,
        "TTS candidate validation reads an owner-only immutable snapshot.",
    ),
    "tts.profile_candidate_upgrade": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_schema",
        _PRIVATE_FILE,
        "TTS candidate validation briefly reopens its own disposable "
        "pre-copied snapshot read-write to run the same in-place schema "
        "upgrade the live open flow uses, before revalidating it through "
        "the existing immutable read-only handle. The caller-supplied "
        "candidate file itself is never opened for write.",
    ),
    "tts.profile_store_version_peek": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_schema",
        _READ_ONLY_URI,
        "TTS repository lease orchestration cheaply peeks the live store's "
        "on-disk schema version before deciding whether an upgrade needs an "
        "exclusive lease; never validates structure or migrates.",
    ),
    "tts.profile_backup": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_repository",
        _PRIVATE_FILE,
        "TTS profile backups use the centralized caller-connection seam.",
        centralized_backup_allowed=True,
    ),
    "tts.profile_migration_backup": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_repository",
        _PRIVATE_AND_READ_ONLY,
        "TTS profile v2 migration sources and retained backups require "
        "private exact-version validation and centralized online backup.",
        centralized_backup_allowed=True,
    ),
    "tts.profile_migration_boundary": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_migration_candidate",
        _PRIVATE_MEMORY_AND_READ_ONLY,
        "TTS migration boundaries use an isolated in-memory source and one "
        "exclusive owner-private future-artifact destination.",
        centralized_backup_allowed=True,
        preserve_read_only_source_mode=True,
    ),
    "tts.profile_migration_publication": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_migration_publication",
        _READ_ONLY_URI,
        "TTS migration publication immutably revalidates exact prepared and "
        "retained artifacts before and after durable namespace replacement.",
        preserve_read_only_source_mode=True,
    ),
    "tts.profile_migration_recovery": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_migration_recovery",
        _READ_ONLY_URI,
        "TTS startup recovery immutably validates journal-classified exact "
        "active and retained backup authority before profile-store open.",
        preserve_read_only_source_mode=True,
    ),
    "tts.profile_migration_publication_descriptor": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_migration_publication",
        _READ_ONLY_URI,
        "TTS migration publication validates an immutable "
        "read-only view bound to an already verified descriptor.",
        preserve_read_only_source_mode=True,
    ),
    "tts.profile_migration_recovery_descriptor": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_migration_recovery",
        _READ_ONLY_URI,
        "TTS startup recovery validates an immutable read-only view bound to "
        "an already verified descriptor.",
        preserve_read_only_source_mode=True,
    ),
    "tts.profile_restore_stage": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_repository",
        _PRIVATE_AND_READ_ONLY,
        "TTS restore staging copies a checked candidate into a private target.",
        centralized_backup_allowed=True,
        preserve_read_only_source_mode=True,
    ),
    "tts.profile_recovery": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_repository",
        _PRIVATE_FILE,
        "TTS pre-restore recovery uses the centralized caller-connection seam.",
        centralized_backup_allowed=True,
    ),
    "tts.profile_snapshot": SQLiteOwnerPolicy(
        "tldw_chatbook/TTS/profile_repository",
        _READ_ONLY_URI,
        "TTS standalone snapshots use immutable read-only access for incremental BLOB validation.",
    ),
    "tamagotchi.sqlite": SQLiteOwnerPolicy(
        "tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage",
        _PRIVATE_OR_MEMORY,
        "All SQLiteStorage methods share private file and Path(':memory:') support.",
    ),
    "utils.db_upgrade_notice": SQLiteOwnerPolicy(
        "tldw_chatbook/Utils/db_upgrade_notice",
        _READ_ONLY_URI,
        "The pre-boot upgrade notice reads only the schema-version row, "
        "read-only, before the app constructs (task-21100).",
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
_RESTORE_LOCK = RLock()

_PRIVATE_FILE_MODE = 0o600
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_PROFILE_MIGRATION_BOUNDARY_VERSIONS = frozenset({2, 3})
_PROFILE_MIGRATION_BOUNDARY_OWNER = "tts.profile_migration_boundary"
_PROFILE_DESTINATION_FACTORY_TOKEN = object()


class SQLitePrivateDestinationError(OSError):
    """Bounded failure for an opaque private SQLite destination."""

    def __init__(self, code: str = "destination_invalid") -> None:
        self.code = code
        super().__init__(f"SQLite private destination failed: {code}")

    def __repr__(self) -> str:
        return f"SQLitePrivateDestinationError({self.code!r})"


class ProfileMigrationBoundaryDestination:
    """Opaque, thread-confined owner for one future boundary artifact."""

    __slots__ = (
        "__connection",
        "__file_identity",
        "__file_fd",
        "__parent_identity",
        "__parent_fd",
        "__path",
        "__schema_version",
        "__state",
        "__thread_id",
    )

    def __init__(
        self,
        factory_token: object,
        *,
        connection: sqlite3.Connection,
        file_identity: os.stat_result,
        file_fd: int = -1,
        parent_identity: os.stat_result,
        parent_fd: int = -1,
        path: Path,
        schema_version: int,
    ) -> None:
        if factory_token is not _PROFILE_DESTINATION_FACTORY_TOKEN:
            raise SQLitePrivateDestinationError()
        self.__connection: sqlite3.Connection | None = connection
        self.__file_identity = file_identity
        self.__file_fd = file_fd
        self.__parent_identity = parent_identity
        self.__parent_fd = parent_fd
        self.__path = path
        self.__schema_version = schema_version
        self.__state = "open"
        self.__thread_id = get_ident()

    def __repr__(self) -> str:
        return "ProfileMigrationBoundaryDestination(<private>)"

    def __enter__(self) -> ProfileMigrationBoundaryDestination:
        if self.__state != "open" or self.__thread_id != get_ident():
            raise SQLitePrivateDestinationError()
        return self

    def __exit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        _close_profile_migration_destination(self)


def _warn_unverified_platform(owner_id: str) -> None:
    with _UNVERIFIED_WARNING_LOCK:
        if owner_id in _WARNED_UNVERIFIED_OWNER_IDS:
            return
        warnings.warn(
            "SQLite permission verification is unavailable on this platform; "
            "database operation continues with an unverified privacy posture",
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
    enforce_private_mode: bool = True,
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
        and (
            not enforce_private_mode
            or stat.S_IMODE(opened.st_mode) == _PRIVATE_FILE_MODE
        )
    )


def _path_error_from_oserror(selected: Path, exc: OSError) -> PrivatePathError:
    status = (
        PrivatePathStatus.LINK_OR_NON_REGULAR
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}
        else PrivatePathStatus.OPERATION_FAILED
    )
    return _failure(selected, status, type(exc).__name__)


class _OptionalSQLiteGenerationChanged(Exception):
    """Restart optional-sidecar validation against the current named inode."""


_OPTIONAL_SIDECAR_REVALIDATION_ATTEMPTS = 4


def _optional_sidecar_restart_or_absent(
    parent_fd: int,
    leaf: str,
    selected: Path,
) -> bool:
    try:
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise _path_error_from_oserror(selected, exc) from None

    rejected = private_paths._classify_private_file_stat(
        current,
        expected_uid=os.geteuid(),
    )
    if rejected is not None:
        raise _failure(selected, rejected, "unsafe_sqlite_artifact")
    raise _OptionalSQLiteGenerationChanged


def _prepare_posix_artifact_generation(
    selected: Path,
    *,
    writable: bool,
    create_if_missing: bool,
    optional: bool,
    enforce_private_mode: bool,
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
        if (
            optional
            and entry_stat is not None
            and entry_stat.st_nlink == 0
            and entry_stat.st_uid == os.geteuid()
        ):
            return _optional_sidecar_restart_or_absent(
                parent_fd,
                leaf,
                selected,
            )
        if entry_stat is not None:
            entry_rejected = private_paths._classify_private_file_stat(
                entry_stat,
                expected_uid=os.geteuid(),
            )
            if entry_rejected is not None:
                raise _failure(
                    selected,
                    entry_rejected,
                    "unsafe_sqlite_artifact",
                )

        created = entry_stat is None
        try:
            file_fd = _open_artifact_fd(
                parent_fd,
                leaf,
                writable=created,
                create=created,
            )
        except FileNotFoundError:
            if optional:
                return _optional_sidecar_restart_or_absent(
                    parent_fd,
                    leaf,
                    selected,
                )
            raise _failure(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                "missing_sqlite_artifact",
            ) from None
        except OSError as exc:
            raise _path_error_from_oserror(selected, exc) from None

        opened_stat = os.fstat(file_fd)
        if optional and opened_stat.st_nlink == 0:
            if not stat.S_ISREG(opened_stat.st_mode):
                raise _failure(
                    selected,
                    PrivatePathStatus.LINK_OR_NON_REGULAR,
                    "unsafe_sqlite_artifact",
                )
            if opened_stat.st_uid != os.geteuid():
                raise _failure(
                    selected,
                    PrivatePathStatus.WRONG_OWNER,
                    "unsafe_sqlite_artifact",
                )
            return _optional_sidecar_restart_or_absent(
                parent_fd,
                leaf,
                selected,
            )
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
            if optional:
                return _optional_sidecar_restart_or_absent(
                    parent_fd,
                    leaf,
                    selected,
                )
            raise _failure(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                "private_sqlite_identity_changed",
            )

        if (
            enforce_private_mode
            and stat.S_IMODE(opened_stat.st_mode) != _PRIVATE_FILE_MODE
        ):
            os.fchmod(file_fd, _PRIVATE_FILE_MODE)
        try:
            postcondition_holds = _artifact_postcondition_holds(
                file_fd,
                parent_fd,
                leaf,
                expected_identity=opened_stat,
                selected=selected,
                enforce_private_mode=enforce_private_mode,
            )
        except FileNotFoundError:
            if optional:
                return _optional_sidecar_restart_or_absent(
                    parent_fd,
                    leaf,
                    selected,
                )
            raise _failure(
                selected,
                PrivatePathStatus.OPERATION_FAILED,
                "private_sqlite_postcondition_failed",
            ) from None
        if not postcondition_holds:
            if optional:
                return _optional_sidecar_restart_or_absent(
                    parent_fd,
                    leaf,
                    selected,
                )
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
            except FileNotFoundError:
                if optional:
                    return _optional_sidecar_restart_or_absent(
                        parent_fd,
                        leaf,
                        selected,
                    )
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "missing_sqlite_artifact",
                ) from None
            except OSError as exc:
                raise _path_error_from_oserror(selected, exc) from None

            writable_stat = os.fstat(writable_fd)
            if optional and writable_stat.st_nlink == 0:
                if not stat.S_ISREG(writable_stat.st_mode):
                    raise _failure(
                        selected,
                        PrivatePathStatus.LINK_OR_NON_REGULAR,
                        "unsafe_sqlite_artifact",
                    )
                if writable_stat.st_uid != os.geteuid():
                    raise _failure(
                        selected,
                        PrivatePathStatus.WRONG_OWNER,
                        "unsafe_sqlite_artifact",
                    )
                return _optional_sidecar_restart_or_absent(
                    parent_fd,
                    leaf,
                    selected,
                )
            rejected = private_paths._classify_private_file_stat(
                writable_stat,
                expected_uid=os.geteuid(),
            )
            if rejected is not None:
                raise _failure(selected, rejected, "unsafe_sqlite_artifact")
            if not private_paths._same_identity(opened_stat, writable_stat):
                if optional:
                    return _optional_sidecar_restart_or_absent(
                        parent_fd,
                        leaf,
                        selected,
                    )
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "private_sqlite_identity_changed",
                )
            try:
                writable_postcondition_holds = _artifact_postcondition_holds(
                    writable_fd,
                    parent_fd,
                    leaf,
                    expected_identity=opened_stat,
                    selected=selected,
                    enforce_private_mode=enforce_private_mode,
                )
            except FileNotFoundError:
                if optional:
                    return _optional_sidecar_restart_or_absent(
                        parent_fd,
                        leaf,
                        selected,
                    )
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "private_sqlite_postcondition_failed",
                ) from None
            if not writable_postcondition_holds:
                if optional:
                    return _optional_sidecar_restart_or_absent(
                        parent_fd,
                        leaf,
                        selected,
                    )
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


def _prepare_posix_artifact(
    selected: Path,
    *,
    writable: bool,
    create_if_missing: bool,
    optional: bool = False,
    enforce_private_mode: bool = True,
) -> bool:
    attempts = _OPTIONAL_SIDECAR_REVALIDATION_ATTEMPTS if optional else 1
    for attempt in range(attempts):
        try:
            return _prepare_posix_artifact_generation(
                selected,
                writable=writable,
                create_if_missing=create_if_missing,
                optional=optional,
                enforce_private_mode=enforce_private_mode,
            )
        except _OptionalSQLiteGenerationChanged:
            if attempt + 1 == attempts:
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "optional_sqlite_generation_churn",
                ) from None
    raise AssertionError("unreachable optional SQLite revalidation state")


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
    enforce_private_mode: bool = True,
) -> bool:
    if private_paths._posix_guards_available():
        return _prepare_posix_artifact(
            selected,
            writable=writable,
            create_if_missing=create_if_missing,
            optional=optional,
            enforce_private_mode=enforce_private_mode,
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
    immutable: bool = False,
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
            uri = f"file://{authority}/{encoded}?mode=ro"
            return f"{uri}&immutable=1" if immutable else uri
        encoded = quote(posix_path, safe="/:")
        uri = f"file:///{encoded}?mode=ro"
        return f"{uri}&immutable=1" if immutable else uri

    uri = f"{Path(raw).as_uri()}?mode=ro"
    return f"{uri}&immutable=1" if immutable else uri


def _build_existing_writable_uri(
    database: str | os.PathLike[str],
    *,
    windows: bool | None = None,
) -> str:
    """Build a seam-owned ``mode=rw`` URI that cannot create a database."""

    return _build_read_only_uri(database, windows=windows).replace(
        "?mode=ro",
        "?mode=rw",
        1,
    )


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


def _connect_registered_sqlite(
    owner_id: str,
    database: str | os.PathLike[str],
    *,
    read_only: bool = False,
    must_exist: bool = False,
    immutable: bool = False,
    expected_identity: os.stat_result | None = None,
    _verified_descriptor_fd: int | None = None,
    **kwargs: Any,
) -> sqlite3.Connection:
    if "uri" in kwargs:
        raise ValueError("The SQLite uri option is owned by the private seam")
    if immutable and not read_only:
        raise ValueError("Immutable SQLite connections must be read-only")
    if expected_identity is not None and not isinstance(
        expected_identity,
        os.stat_result,
    ):
        raise TypeError("expected_identity must be an os.stat_result")
    policy = _validated_owner_policy(owner_id)
    if _verified_descriptor_fd is not None:
        if not read_only or not immutable or not must_exist:
            raise ValueError("Descriptor SQLite views are immutable read-only")
        if SQLiteTargetKind.READ_ONLY_URI not in policy.allowed_target_kinds:
            raise ValueError("SQLite owner does not allow descriptor reads")
        opened = os.fstat(_verified_descriptor_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
        ):
            raise ValueError("SQLite descriptor must be a regular file")
        return _SQLITE_CONNECT(
            f"file:/dev/fd/{_verified_descriptor_fd}?mode=ro&immutable=1",
            uri=True,
            **kwargs,
        )
    raw, target_kind = _classify_target(database, read_only=read_only)
    if target_kind is SQLiteTargetKind.MEMORY and must_exist:
        raise ValueError("must_exist is only valid for file-backed SQLite")
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
            create_if_missing=not read_only and not must_exist,
            enforce_private_mode=not (
                read_only and policy.preserve_read_only_source_mode
            ),
        )
        if expected_identity is not None:
            observed_identity = selected.lstat()
            if not private_paths._same_identity(
                observed_identity,
                expected_identity,
            ):
                raise _failure(
                    selected,
                    PrivatePathStatus.OPERATION_FAILED,
                    "private_sqlite_expected_identity_changed",
                )
        for suffix in _SIDECAR_SUFFIXES:
            _prepare_artifact(
                Path(f"{selected}{suffix}"),
                writable=not read_only,
                create_if_missing=False,
                optional=True,
                enforce_private_mode=not (
                    read_only and policy.preserve_read_only_source_mode
                ),
            )
        connection_target = os.fspath(selected)
        if read_only:
            connection_target = _build_read_only_uri(
                selected,
                windows=os.name == "nt",
                immutable=immutable,
            )
            use_uri = True
        elif must_exist:
            connection_target = _build_existing_writable_uri(
                selected,
                windows=os.name == "nt",
            )
            use_uri = True
        if directory_result.status is PrivatePathStatus.UNVERIFIED_PLATFORM:
            _warn_unverified_platform(owner_id)

    return sqlite3.connect(connection_target, uri=use_uri, **kwargs)


def connect_private_sqlite(
    owner_id: str,
    database: str | os.PathLike[str],
    *,
    read_only: bool = False,
    must_exist: bool = False,
    immutable: bool = False,
    expected_identity: os.stat_result | None = None,
    **kwargs: Any,
) -> sqlite3.Connection:
    """Open SQLite only after enforcing the registered target policy."""

    return _connect_registered_sqlite(
        owner_id,
        database,
        read_only=read_only,
        must_exist=must_exist,
        immutable=immutable,
        expected_identity=expected_identity,
        **kwargs,
    )


def connect_private_sqlite_descriptor(
    owner_id: str,
    file_fd: int,
    **kwargs: Any,
) -> sqlite3.Connection:
    """Open an immutable SQLite view bound to an already verified descriptor."""

    policy = _validated_owner_policy(owner_id)
    if SQLiteTargetKind.READ_ONLY_URI not in policy.allowed_target_kinds:
        raise ValueError("SQLite owner does not allow descriptor reads")
    opened = os.fstat(file_fd)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o600
        or opened.st_nlink != 1
    ):
        raise ValueError("SQLite descriptor must be a regular file")
    duplicate = os.dup(file_fd)
    try:
        return _connect_registered_sqlite(
            owner_id,
            ":memory:",
            read_only=True,
            must_exist=True,
            immutable=True,
            _verified_descriptor_fd=duplicate,
            **kwargs,
        )
    finally:
        os.close(duplicate)


@dataclass(slots=True)
class _PinnedSQLiteSource:
    selected: Path
    identity: os.stat_result
    parent_fd: int = -1
    file_fd: int = -1
    enforce_private_mode: bool = True

    def close(self) -> None:
        if self.file_fd >= 0:
            os.close(self.file_fd)
            self.file_fd = -1
        if self.parent_fd >= 0:
            os.close(self.parent_fd)
            self.parent_fd = -1


def _validate_backup_owner(
    owner_id: str,
    *,
    required_kinds: frozenset[SQLiteTargetKind],
) -> SQLiteOwnerPolicy:
    policy = _validated_owner_policy(owner_id)
    if not policy.centralized_backup_allowed:
        raise ValueError(f"SQLite owner {owner_id} does not allow centralized backup")
    missing = required_kinds - policy.allowed_target_kinds
    if missing:
        kinds = ", ".join(sorted(kind.value for kind in missing))
        raise ValueError(f"SQLite backup owner {owner_id} does not allow {kinds}")
    return policy


def _source_selection(
    database: str | os.PathLike[str],
    *,
    allow_memory: bool,
) -> tuple[str, Path | None]:
    raw = os.fspath(database)
    if not isinstance(raw, str):
        raise TypeError("SQLite paths must be text paths")
    if "\x00" in raw:
        raise ValueError("Path must not contain NUL")
    if raw.startswith("file:"):
        raise ValueError("Caller-supplied file: SQLite URIs are not supported")
    if raw == ":memory:":
        if not allow_memory:
            raise ValueError("A file-backed SQLite source is required")
        return raw, None
    return raw, lexical_path(raw)


def _prepare_source_artifacts(
    owner_id: str,
    selected: Path,
    *,
    enforce_private_mode: bool,
) -> None:
    directory_result = verify_trusted_directory(
        selected.parent,
        allow_shared_sticky=False,
    )
    _prepare_artifact(
        selected,
        writable=False,
        create_if_missing=False,
        enforce_private_mode=enforce_private_mode,
    )
    for suffix in _SIDECAR_SUFFIXES:
        _prepare_artifact(
            Path(f"{selected}{suffix}"),
            writable=False,
            create_if_missing=False,
            optional=True,
            enforce_private_mode=enforce_private_mode,
        )
    if directory_result.status is PrivatePathStatus.UNVERIFIED_PLATFORM:
        _warn_unverified_platform(owner_id)


def _source_postcondition_holds(source: _PinnedSQLiteSource) -> bool:
    if source.file_fd < 0:
        try:
            named = source.selected.lstat()
        except OSError:
            return False
        if private_paths._WINDOWS_PLATFORM:
            return (
                private_paths._same_identity(named, source.identity)
                and stat.S_ISREG(named.st_mode)
                and named.st_nlink == 1
            )
        return (
            private_paths._same_identity(named, source.identity)
            and private_paths._classify_private_file_stat(
                named,
                expected_uid=os.geteuid(),
            )
            is None
        )
    try:
        opened = os.fstat(source.file_fd)
        named = os.stat(
            source.selected.name,
            dir_fd=source.parent_fd,
            follow_symlinks=False,
        )
    except OSError:
        return False
    return (
        private_paths._same_identity(opened, source.identity)
        and private_paths._same_identity(named, source.identity)
        and private_paths._classify_private_file_stat(
            opened,
            expected_uid=os.geteuid(),
        )
        is None
        and private_paths._classify_private_file_stat(
            named,
            expected_uid=os.geteuid(),
        )
        is None
        and (
            not source.enforce_private_mode
            or stat.S_IMODE(opened.st_mode) == _PRIVATE_FILE_MODE
        )
    )


def _reverify_source(source: _PinnedSQLiteSource) -> None:
    if not _source_postcondition_holds(source):
        raise _failure(
            source.selected,
            PrivatePathStatus.OPERATION_FAILED,
            "private_sqlite_source_identity_changed",
        )


@contextlib.contextmanager
def _pin_sqlite_source(
    owner_id: str,
    database: str | os.PathLike[str],
    *,
    allow_memory: bool,
) -> Iterator[_PinnedSQLiteSource | None]:
    _raw, selected = _source_selection(database, allow_memory=allow_memory)
    if selected is None:
        yield None
        return

    policy = _validated_owner_policy(owner_id)
    enforce_private_mode = not policy.preserve_read_only_source_mode
    _prepare_source_artifacts(
        owner_id,
        selected,
        enforce_private_mode=enforce_private_mode,
    )
    if private_paths._posix_guards_available():
        parent_fd, leaf = private_paths._open_verified_parent(
            selected,
            missing_leaf_allowed=False,
        )
        file_fd = -1
        try:
            file_fd = _open_artifact_fd(
                parent_fd,
                leaf,
                writable=False,
                create=False,
            )
            identity = os.fstat(file_fd)
            source = _PinnedSQLiteSource(
                selected=selected,
                identity=identity,
                parent_fd=parent_fd,
                file_fd=file_fd,
                enforce_private_mode=enforce_private_mode,
            )
            parent_fd = -1
            file_fd = -1
            try:
                _reverify_source(source)
                yield source
            finally:
                source.close()
        finally:
            if file_fd >= 0:
                os.close(file_fd)
            if parent_fd >= 0:
                os.close(parent_fd)
        return

    identity = selected.lstat()
    source = _PinnedSQLiteSource(
        selected=selected,
        identity=identity,
        enforce_private_mode=enforce_private_mode,
    )
    _reverify_source(source)
    yield source


def _private_destination(database: str | os.PathLike[str]) -> Path:
    raw, kind = _classify_target(database, read_only=False)
    if kind is not SQLiteTargetKind.PRIVATE_FILE:
        raise ValueError("A file-backed private SQLite destination is required")
    return lexical_path(raw)


def _existing_entry_stat(selected: Path) -> os.stat_result | None:
    try:
        return selected.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _path_error_from_oserror(selected, exc) from None


def _reject_unsafe_or_aliased_destination(
    source: _PinnedSQLiteSource | None,
    destination: Path,
) -> None:
    if source is not None and source.selected == destination:
        raise ValueError("SQLite source and destination cannot be the same path")
    destination_stat = _existing_entry_stat(destination)
    if destination_stat is None:
        return
    rejected = (
        (
            None
            if stat.S_ISREG(destination_stat.st_mode) and destination_stat.st_nlink == 1
            else PrivatePathStatus.LINK_OR_NON_REGULAR
        )
        if private_paths._WINDOWS_PLATFORM
        else private_paths._classify_private_file_stat(
            destination_stat,
            expected_uid=os.geteuid(),
        )
    )
    if rejected is not None:
        raise _failure(destination, rejected, "unsafe_sqlite_backup_target")
    if source is not None and private_paths._same_identity(
        source.identity,
        destination_stat,
    ):
        raise ValueError("SQLite source and destination cannot be the same file")


def _reject_path_pair_alias(left: Path, right: Path) -> None:
    if left == right:
        raise ValueError("SQLite backup paths cannot be the same")
    left_stat = _existing_entry_stat(left)
    right_stat = _existing_entry_stat(right)
    if (
        left_stat is not None
        and right_stat is not None
        and private_paths._same_identity(left_stat, right_stat)
    ):
        raise ValueError("SQLite backup paths cannot reference the same file")


def _restore_busy(_exc: BaseException) -> SQLiteRestoreBusyError:
    return SQLiteRestoreBusyError(
        "Close database users and retry; if Chatbook already opened this "
        "database, live restore is unavailable in this session and requires "
        "offline maintenance."
    )


def _close_owned_connections(
    connections: tuple[tuple[str, sqlite3.Connection | None], ...],
) -> None:
    for label, connection in connections:
        if connection is None:
            continue
        try:
            connection.close()
        except Exception as exc:
            try:
                warnings.warn(
                    f"SQLite {label} close failed: {exc}",
                    RuntimeWarning,
                    stacklevel=3,
                )
            except Exception:
                pass


def _guard_destination(
    destination: sqlite3.Connection,
    *,
    restore: bool,
) -> str:
    journal_mode = ""
    changed_wal_to_delete = False
    try:
        destination.execute("PRAGMA busy_timeout = 0")
        row = destination.execute("PRAGMA journal_mode").fetchone()
        journal_mode = str(row[0]).lower() if row else ""
        if restore and journal_mode not in {"delete", "wal"}:
            raise ValueError(
                f"Live restore does not support {journal_mode or 'unknown'} "
                "journal mode"
            )
        locking_row = destination.execute("PRAGMA locking_mode = EXCLUSIVE").fetchone()
        if not locking_row or str(locking_row[0]).lower() != "exclusive":
            raise sqlite3.OperationalError("SQLite refused exclusive locking mode")
        if journal_mode == "wal":
            mode_row = destination.execute("PRAGMA journal_mode = DELETE").fetchone()
            if not mode_row or str(mode_row[0]).lower() != "delete":
                raise sqlite3.OperationalError(
                    "SQLite refused the DELETE journal quiescence probe"
                )
            changed_wal_to_delete = True
        destination.execute("BEGIN EXCLUSIVE")
        destination.rollback()
        return journal_mode
    except sqlite3.OperationalError as exc:
        if changed_wal_to_delete:
            try:
                mode_row = destination.execute("PRAGMA journal_mode = WAL").fetchone()
                if not mode_row or str(mode_row[0]).lower() != "wal":
                    raise sqlite3.OperationalError(
                        "SQLite refused to roll back the journal-mode probe"
                    )
            except sqlite3.OperationalError as recovery_exc:
                if restore:
                    raise _restore_busy(recovery_exc) from recovery_exc
                raise
        if restore:
            raise _restore_busy(exc) from exc
        raise


def _restore_destination_mode(
    destination: sqlite3.Connection,
    journal_mode: str,
    *,
    restore: bool,
) -> None:
    if journal_mode not in {"delete", "wal"}:
        raise ValueError(f"Cannot restore unsupported SQLite mode {journal_mode!r}")
    try:
        row = destination.execute(
            f"PRAGMA journal_mode = {journal_mode.upper()}"
        ).fetchone()
        if not row or str(row[0]).lower() != journal_mode:
            raise sqlite3.OperationalError(
                f"SQLite refused to restore {journal_mode.upper()} mode"
            )
    except sqlite3.OperationalError as exc:
        if restore:
            raise _restore_busy(exc) from exc
        raise


def _backup_pages(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    *,
    restore: bool,
    progress_guard: Callable[[], None] | None = None,
) -> None:
    def abort_busy(status: int, _remaining: int, _total: int) -> None:
        if status in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
            error = sqlite3.OperationalError(
                "SQLite backup could not acquire the required lock"
            )
            if restore:
                raise _restore_busy(error) from error
            raise error
        if progress_guard is not None:
            progress_guard()

    source.backup(
        destination,
        pages=1,
        progress=abort_busy,
        sleep=0.0,
    )


def _raise_private_destination_failure(error: BaseException) -> None:
    if not isinstance(error, Exception):
        raise error
    raise SQLitePrivateDestinationError() from None


def _profile_destination_connection(
    destination: ProfileMigrationBoundaryDestination,
) -> sqlite3.Connection:
    connection = object.__getattribute__(
        destination,
        "_ProfileMigrationBoundaryDestination__connection",
    )
    if connection is None:
        raise SQLitePrivateDestinationError()
    return connection


def _unattached_database_rows(
    rows: list[sqlite3.Row | tuple[object, ...]],
) -> tuple[object, ...] | None:
    main_rows = [row for row in rows if len(row) >= 3 and row[1] == "main"]
    if len(main_rows) != 1 or any(
        len(row) < 3
        or row[1] not in {"main", "temp"}
        or (row[1] == "temp" and row[2] not in {"", None})
        for row in rows
    ):
        return None
    return tuple(main_rows[0])


def _profile_destination_sidecars_absent(parent_fd: int, leaf: str) -> bool:
    for suffix in _SIDECAR_SUFFIXES:
        try:
            os.stat(
                f"{leaf}{suffix}",
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        return False
    return True


def _profile_destination_namespace_holds(
    destination: ProfileMigrationBoundaryDestination,
    parent_fd: int,
    file_fd: int,
    leaf: str,
) -> bool:
    return (
        private_paths._same_identity(
            os.fstat(parent_fd),
            object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__parent_identity",
            ),
        )
        and _artifact_postcondition_holds(
            file_fd,
            parent_fd,
            leaf,
            expected_identity=object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__file_identity",
            ),
            selected=cast(
                Path,
                object.__getattribute__(
                    destination,
                    "_ProfileMigrationBoundaryDestination__path",
                ),
            ),
        )
        and _profile_destination_sidecars_absent(parent_fd, leaf)
    )


def _verify_profile_migration_destination(
    destination: ProfileMigrationBoundaryDestination,
    *,
    require_empty: bool,
) -> sqlite3.Connection:
    if (
        type(destination) is not ProfileMigrationBoundaryDestination
        or object.__getattribute__(
            destination,
            "_ProfileMigrationBoundaryDestination__thread_id",
        )
        != get_ident()
        or object.__getattribute__(
            destination,
            "_ProfileMigrationBoundaryDestination__state",
        )
        != "open"
    ):
        raise SQLitePrivateDestinationError()
    selected = cast(
        Path,
        object.__getattribute__(
            destination,
            "_ProfileMigrationBoundaryDestination__path",
        ),
    )
    parent_result = verify_trusted_directory(
        selected.parent,
        allow_shared_sticky=False,
    )
    if parent_result.status is PrivatePathStatus.UNVERIFIED_PLATFORM:
        raise SQLitePrivateDestinationError()
    parent_stat = selected.parent.lstat()
    file_stat = selected.lstat()
    if (
        not private_paths._same_identity(
            parent_stat,
            object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__parent_identity",
            ),
        )
        or parent_stat.st_uid != os.geteuid()
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
        or not private_paths._same_identity(
            file_stat,
            object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__file_identity",
            ),
        )
        or private_paths._classify_private_file_stat(
            file_stat,
            expected_uid=os.geteuid(),
        )
        is not None
        or stat.S_IMODE(file_stat.st_mode) != _PRIVATE_FILE_MODE
    ):
        raise SQLitePrivateDestinationError()
    connection = _profile_destination_connection(destination)
    if connection.in_transaction:
        raise SQLitePrivateDestinationError()
    database_rows = list(connection.execute("PRAGMA database_list"))
    main_row = _unattached_database_rows(database_rows)
    if main_row is None or not main_row[2]:
        raise SQLitePrivateDestinationError()
    opened_path = lexical_path(os.fsdecode(cast(str, main_row[2])))
    verify_trusted_directory(opened_path.parent, allow_shared_sticky=False)
    if not private_paths._same_identity(opened_path.lstat(), file_stat):
        raise SQLitePrivateDestinationError()
    parent_fd, leaf = private_paths._open_verified_parent(
        selected,
        missing_leaf_allowed=False,
    )
    try:
        if not private_paths._same_identity(
            parent_stat, os.fstat(parent_fd)
        ) or not _profile_destination_sidecars_absent(parent_fd, leaf):
            raise SQLitePrivateDestinationError()
    finally:
        os.close(parent_fd)
    if require_empty:
        version_row = connection.execute("PRAGMA user_version").fetchone()
        objects_row = connection.execute(
            "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT GLOB 'sqlite_*'"
        ).fetchone()
        if (
            version_row is None
            or len(version_row) != 1
            or type(version_row[0]) is not int
            or version_row[0] != 0
            or objects_row is None
            or len(objects_row) != 1
            or type(objects_row[0]) is not int
            or objects_row[0] != 0
        ):
            raise SQLitePrivateDestinationError()
    return connection


def _close_profile_migration_destination(
    destination: ProfileMigrationBoundaryDestination,
) -> None:
    if type(destination) is not ProfileMigrationBoundaryDestination:
        raise SQLitePrivateDestinationError()
    connection = object.__getattribute__(
        destination,
        "_ProfileMigrationBoundaryDestination__connection",
    )
    errors: list[BaseException] = []
    if connection is not None:
        try:
            connection.close()
        except BaseException as close_error:
            errors.append(close_error)
        else:
            object.__setattr__(
                destination,
                "_ProfileMigrationBoundaryDestination__connection",
                None,
            )
    if errors:
        for pending_error in errors:
            if not isinstance(pending_error, Exception):
                raise pending_error
        raise errors[0]
    if (
        object.__getattribute__(
            destination,
            "_ProfileMigrationBoundaryDestination__state",
        )
        == "open"
    ):
        object.__setattr__(
            destination,
            "_ProfileMigrationBoundaryDestination__state",
            "closed",
        )
    for attribute in ("__file_fd", "__parent_fd"):
        private_name = f"_ProfileMigrationBoundaryDestination{attribute}"
        descriptor = cast(int, object.__getattribute__(destination, private_name))
        if descriptor >= 0:
            object.__setattr__(destination, private_name, -1)
            try:
                os.close(descriptor)
            except BaseException as close_error:
                errors.append(close_error)
    for pending_error in errors:
        if not isinstance(pending_error, Exception):
            raise pending_error
    if errors:
        raise errors[0]


def close_profile_migration_destination(
    destination: ProfileMigrationBoundaryDestination,
) -> None:
    """Close and revoke one opaque migration destination."""

    error: BaseException | None = None
    try:
        _close_profile_migration_destination(destination)
    except BaseException as caught:
        error = caught
    if error is not None:
        _raise_private_destination_failure(error)


def discard_profile_migration_destination(
    destination: ProfileMigrationBoundaryDestination,
    *,
    tombstone_key: object,
) -> None:
    """Descriptor-verify and quarantine one unpublished candidate."""

    body_error: BaseException | None = None
    try:
        from tldw_chatbook.TTS.profile_migration_namespace import (
            MigrationTombstoneKey,
            ParentAuthority,
            remove_exact,
        )

        if (
            type(destination) is not ProfileMigrationBoundaryDestination
            or type(tombstone_key) is not MigrationTombstoneKey
            or object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__thread_id",
            )
            != get_ident()
        ):
            raise ValueError
        selected = cast(
            Path,
            object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__path",
            ),
        )
        parent_fd = cast(
            int,
            object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__parent_fd",
            ),
        )
        file_fd = cast(
            int,
            object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__file_fd",
            ),
        )
        if (
            parent_fd < 0
            or file_fd < 0
            or not _profile_destination_namespace_holds(
                destination,
                parent_fd,
                file_fd,
                selected.name,
            )
        ):
            raise ValueError
        connection = object.__getattribute__(
            destination,
            "_ProfileMigrationBoundaryDestination__connection",
        )
        if connection is not None:
            connection.close()
            object.__setattr__(
                destination,
                "_ProfileMigrationBoundaryDestination__connection",
                None,
            )
        if not _profile_destination_namespace_holds(
            destination,
            parent_fd,
            file_fd,
            selected.name,
        ):
            raise ValueError
        identity = cast(
            os.stat_result,
            object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__file_identity",
            ),
        )
        parent = cast(
            os.stat_result,
            object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__parent_identity",
            ),
        )
        if file_fd < 0 or not private_paths._same_identity(os.fstat(file_fd), identity):
            raise ValueError
        remove_exact(
            selected,
            parent_authority=ParentAuthority(parent),
            file_identity=identity,
            tombstone_key=tombstone_key,
        )
        object.__setattr__(
            destination,
            "_ProfileMigrationBoundaryDestination__state",
            "discarded",
        )
    except BaseException as error:
        body_error = error
    close_error: BaseException | None = None
    try:
        _close_profile_migration_destination(destination)
    except BaseException as error:
        close_error = error
    for pending in (body_error, close_error):
        if pending is not None and not isinstance(pending, Exception):
            raise pending
    if body_error is not None or close_error is not None:
        raise SQLitePrivateDestinationError() from None


def open_canonical_profile_migration_destination(
    path: str | os.PathLike[str],
    *,
    schema_version: int,
    tombstone_key: object,
) -> ProfileMigrationBoundaryDestination:
    """Acquire one fixed migration leaf while retaining exact descriptors."""

    error: BaseException | None = None
    connection: sqlite3.Connection | None = None
    parent_fd = -1
    file_fd = -1
    destination: ProfileMigrationBoundaryDestination | None = None
    try:
        if type(schema_version) is not int or not 0 <= schema_version <= 4:
            raise ValueError
        selected = _private_destination(path)
        parent_result = secure_private_directory(
            selected.parent,
            create=False,
            application_owned=True,
        )
        if parent_result.status is PrivatePathStatus.UNVERIFIED_PLATFORM:
            raise ValueError
        from tldw_chatbook.TTS.profile_migration_namespace import (
            MigrationTombstoneKey,
            ParentAuthority,
            open_new_or_reused_private_file,
        )

        if type(tombstone_key) is not MigrationTombstoneKey:
            raise ValueError
        parent_fd, file_fd, file_identity, authority = open_new_or_reused_private_file(
            selected,
            parent_authority=ParentAuthority(selected.parent.lstat()),
            tombstone_key=tombstone_key,
        )
        connection = _connect_registered_sqlite(
            _PROFILE_MIGRATION_BOUNDARY_OWNER,
            selected,
            must_exist=True,
            expected_identity=file_identity,
            timeout=0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        destination = ProfileMigrationBoundaryDestination(
            _PROFILE_DESTINATION_FACTORY_TOKEN,
            connection=connection,
            file_identity=file_identity,
            file_fd=file_fd,
            parent_identity=authority.identity,
            parent_fd=parent_fd,
            path=selected,
            schema_version=schema_version,
        )
        _verify_profile_migration_destination(destination, require_empty=True)
        connection = None
        parent_fd = -1
        file_fd = -1
    except BaseException as caught:
        error = caught
    if connection is not None:
        try:
            connection.close()
        except BaseException as close_error:
            if error is None or not isinstance(close_error, Exception):
                error = close_error
    for descriptor in (file_fd, parent_fd):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as close_error:
                if error is None or not isinstance(close_error, Exception):
                    error = close_error
    if error is not None:
        _raise_private_destination_failure(error)
    assert destination is not None
    return destination


def open_profile_migration_boundary_destination(
    path: str | os.PathLike[str],
    *,
    schema_version: int,
) -> ProfileMigrationBoundaryDestination:
    """Exclusively create one opaque owner-private boundary destination."""

    error: BaseException | None = None
    connection: sqlite3.Connection | None = None
    destination: ProfileMigrationBoundaryDestination | None = None
    try:
        if (
            type(schema_version) is not int
            or schema_version not in _PROFILE_MIGRATION_BOUNDARY_VERSIONS
        ):
            raise ValueError
        selected = _private_destination(path)
        if _existing_entry_stat(selected) is not None:
            raise ValueError
        parent_result = secure_private_directory(
            selected.parent,
            create=False,
            application_owned=True,
        )
        if parent_result.status is PrivatePathStatus.UNVERIFIED_PLATFORM:
            raise ValueError
        parent_identity = selected.parent.lstat()
        parent_fd, leaf = private_paths._open_verified_parent(
            selected,
            missing_leaf_allowed=True,
        )
        try:
            if not private_paths._same_identity(
                os.fstat(parent_fd),
                parent_identity,
            ) or not _profile_destination_sidecars_absent(parent_fd, leaf):
                raise ValueError
        finally:
            os.close(parent_fd)
        connection = _connect_registered_sqlite(
            _PROFILE_MIGRATION_BOUNDARY_OWNER,
            selected,
            timeout=0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        file_identity = selected.lstat()
        destination = ProfileMigrationBoundaryDestination(
            _PROFILE_DESTINATION_FACTORY_TOKEN,
            connection=connection,
            file_identity=file_identity,
            parent_identity=parent_identity,
            path=selected,
            schema_version=schema_version,
        )
        _verify_profile_migration_destination(destination, require_empty=True)
        connection = None
    except BaseException as caught:
        error = caught
    if connection is not None:
        try:
            connection.close()
        except BaseException as close_error:
            if error is None or not isinstance(close_error, Exception):
                error = close_error
    if error is not None:
        _raise_private_destination_failure(error)
    assert destination is not None
    return destination


def _snapshot_connection_to_memory(
    source: sqlite3.Connection,
) -> sqlite3.Connection:
    """Create one isolated in-memory SQLite copy for a migration callback."""

    if not isinstance(source, sqlite3.Connection) or source.in_transaction:
        raise SQLitePrivateDestinationError()
    database_rows = list(source.execute("PRAGMA database_list"))
    if _unattached_database_rows(database_rows) is None:
        raise SQLitePrivateDestinationError()
    snapshot = _connect_registered_sqlite(
        _PROFILE_MIGRATION_BOUNDARY_OWNER,
        ":memory:",
        isolation_level=None,
    )
    try:
        snapshot.row_factory = sqlite3.Row
        snapshot.execute("PRAGMA foreign_keys = ON")
        _backup_pages(source, snapshot, restore=False)
        return snapshot
    except BaseException as primary_error:
        close_error: BaseException | None = None
        try:
            snapshot.close()
        except BaseException as error:
            close_error = error
        for pending_error in (primary_error, close_error):
            if pending_error is not None and not isinstance(pending_error, Exception):
                raise pending_error
        raise primary_error


def _validate_closed_profile_migration_destination(
    destination: ProfileMigrationBoundaryDestination,
    validate: Callable[[sqlite3.Connection], None],
) -> None:
    selected = cast(
        Path,
        object.__getattribute__(
            destination,
            "_ProfileMigrationBoundaryDestination__path",
        ),
    )
    connection: sqlite3.Connection | None = None
    body_error: BaseException | None = None
    close_error: BaseException | None = None
    try:
        connection = _connect_registered_sqlite(
            _PROFILE_MIGRATION_BOUNDARY_OWNER,
            selected,
            read_only=True,
            must_exist=True,
            immutable=True,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA query_only = ON")
        validate(connection)
    except BaseException as error:
        body_error = error
    if connection is not None:
        try:
            connection.close()
        except BaseException as error:
            close_error = error
    for pending_error in (body_error, close_error):
        if pending_error is not None and not isinstance(pending_error, Exception):
            raise pending_error
    if body_error is not None:
        raise body_error
    if close_error is not None:
        raise close_error


def backup_profile_migration_boundary(
    source: sqlite3.Connection,
    destination: ProfileMigrationBoundaryDestination,
    *,
    schema_version: int,
    validate: Callable[[sqlite3.Connection], None],
) -> None:
    """Back up one isolated boundary into its exact private destination."""

    body_error: BaseException | None = None
    close_error: BaseException | None = None
    connection: sqlite3.Connection | None = None
    try:
        if (
            type(destination) is not ProfileMigrationBoundaryDestination
            or type(schema_version) is not int
            or schema_version
            != object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__schema_version",
            )
            or not callable(validate)
        ):
            raise ValueError
        connection = _verify_profile_migration_destination(
            destination,
            require_empty=True,
        )
        _backup_pages(source, connection, restore=False)
        _verify_profile_migration_destination(destination, require_empty=False)
        validate(connection)
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or len(checkpoint) != 3 or checkpoint[0] != 0:
            raise sqlite3.OperationalError
        journal_mode = connection.execute("PRAGMA journal_mode = DELETE").fetchone()
        if journal_mode is None or str(journal_mode[0]).lower() != "delete":
            raise sqlite3.OperationalError
        _verify_profile_migration_destination(destination, require_empty=False)
    except BaseException as caught:
        body_error = caught
    if connection is not None:
        try:
            connection.close()
            object.__setattr__(
                destination,
                "_ProfileMigrationBoundaryDestination__connection",
                None,
            )
        except BaseException as caught:
            close_error = caught
    if body_error is None and close_error is None:
        try:
            selected = cast(
                Path,
                object.__getattribute__(
                    destination,
                    "_ProfileMigrationBoundaryDestination__path",
                ),
            )
            parent_fd, leaf = private_paths._open_verified_parent(
                selected,
                missing_leaf_allowed=False,
            )
            file_fd = -1
            try:
                file_fd = _open_artifact_fd(
                    parent_fd,
                    leaf,
                    writable=False,
                    create=False,
                )
                if not _profile_destination_namespace_holds(
                    destination,
                    parent_fd,
                    file_fd,
                    leaf,
                ):
                    raise SQLitePrivateDestinationError()
                os.fsync(file_fd)
                os.fsync(parent_fd)
                if not _profile_destination_namespace_holds(
                    destination,
                    parent_fd,
                    file_fd,
                    leaf,
                ):
                    raise SQLitePrivateDestinationError()
                _validate_closed_profile_migration_destination(destination, validate)
                if not _profile_destination_namespace_holds(
                    destination,
                    parent_fd,
                    file_fd,
                    leaf,
                ):
                    raise SQLitePrivateDestinationError()
                object.__setattr__(
                    destination,
                    "_ProfileMigrationBoundaryDestination__state",
                    "ready",
                )
                return
            finally:
                if file_fd >= 0:
                    os.close(file_fd)
                os.close(parent_fd)
        except BaseException as caught:
            body_error = caught
    object.__setattr__(
        destination,
        "_ProfileMigrationBoundaryDestination__state",
        "failed",
    )
    for pending in (body_error, close_error):
        if pending is not None and not isinstance(pending, Exception):
            raise pending
    raise SQLitePrivateDestinationError() from None


def migrate_profile_store_to_candidate(
    source: sqlite3.Connection,
    destination: ProfileMigrationBoundaryDestination,
    *,
    migrate: Callable[[sqlite3.Connection], Any],
    validate: Callable[[sqlite3.Connection], None],
    progress_guard: Callable[[], None] | None = None,
) -> Any:
    """Copy and migrate one candidate while its exact descriptors stay pinned."""

    connection: sqlite3.Connection | None = None
    result: Any = None
    body_error: BaseException | None = None
    try:
        if (
            not callable(migrate)
            or not callable(validate)
            or (progress_guard is not None and not callable(progress_guard))
            or source.in_transaction
        ):
            raise ValueError
        if progress_guard is not None:
            progress_guard()
        connection = _verify_profile_migration_destination(
            destination,
            require_empty=True,
        )
        _backup_pages(
            source,
            connection,
            restore=False,
            progress_guard=progress_guard,
        )
        _verify_profile_migration_destination(destination, require_empty=False)
        object.__setattr__(
            destination,
            "_ProfileMigrationBoundaryDestination__connection",
            None,
        )
        object.__setattr__(
            destination,
            "_ProfileMigrationBoundaryDestination__state",
            "migrating",
        )
        result = migrate(connection)
        connection = None  # The migration callback owns and closes it.
        if progress_guard is not None:
            progress_guard()
        file_fd = cast(
            int,
            object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__file_fd",
            ),
        )
        parent_fd = cast(
            int,
            object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__parent_fd",
            ),
        )
        if file_fd < 0 or parent_fd < 0:
            raise ValueError
        os.fsync(file_fd)
        os.fsync(parent_fd)
        immutable = connect_private_sqlite_descriptor(
            "tts.profile_migration_publication_descriptor",
            file_fd,
            isolation_level=None,
        )
        try:
            immutable.row_factory = sqlite3.Row
            immutable.execute("PRAGMA foreign_keys = ON")
            immutable.execute("PRAGMA query_only = ON")
            validate(immutable)
            if progress_guard is not None:
                progress_guard()
        finally:
            immutable.close()
        selected = cast(
            Path,
            object.__getattribute__(
                destination,
                "_ProfileMigrationBoundaryDestination__path",
            ),
        )
        parent_check, leaf = private_paths._open_verified_parent(
            selected,
            missing_leaf_allowed=False,
        )
        try:
            if not _profile_destination_namespace_holds(
                destination,
                parent_check,
                file_fd,
                leaf,
            ):
                raise ValueError
        finally:
            os.close(parent_check)
        object.__setattr__(
            destination,
            "_ProfileMigrationBoundaryDestination__state",
            "ready",
        )
        return result
    except BaseException as error:
        body_error = error
    object.__setattr__(
        destination,
        "_ProfileMigrationBoundaryDestination__state",
        "failed",
    )
    if connection is not None:
        try:
            connection.close()
        except BaseException as close_error:
            if body_error is None or not isinstance(close_error, Exception):
                body_error = close_error
    assert body_error is not None
    if not isinstance(body_error, Exception):
        raise body_error
    raise SQLitePrivateDestinationError() from None


def backup_connection_to_private(
    owner_id: str,
    source_connection: sqlite3.Connection,
    source_database: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    progress_guard: Callable[[], None] | None = None,
) -> None:
    """Back up a caller-owned connection to a checked private target."""

    _validate_backup_owner(owner_id, required_kinds=_PRIVATE_FILE)
    if getattr(source_connection, "in_transaction", False):
        raise sqlite3.OperationalError(
            "Finish the source transaction before starting a backup"
        )
    destination_path = _private_destination(target)
    with _pin_sqlite_source(
        owner_id,
        source_database,
        allow_memory=True,
    ) as source_pin:
        _reject_unsafe_or_aliased_destination(source_pin, destination_path)
        if source_pin is not None:
            _reverify_source(source_pin)
        destination = _connect_registered_sqlite(
            owner_id,
            destination_path,
            timeout=0,
        )
        try:
            journal_mode = _guard_destination(destination, restore=False)
            _restore_destination_mode(
                destination,
                journal_mode,
                restore=False,
            )
            if source_pin is not None:
                _reverify_source(source_pin)
            _backup_pages(
                source_connection,
                destination,
                restore=False,
                progress_guard=progress_guard,
            )
            if source_pin is not None:
                _reverify_source(source_pin)
        finally:
            _close_owned_connections((("backup destination", destination),))


def _connection_main_database(connection: sqlite3.Connection) -> Path:
    rows = connection.execute("PRAGMA database_list").fetchall()
    main_rows = [row for row in rows if len(row) >= 3 and row[1] == "main"]
    if len(main_rows) != 1 or not main_rows[0][2]:
        raise ValueError("SQLite connection must have one file-backed main database")
    return Path(os.fsdecode(main_rows[0][2]))


def backup_open_connections_to_private(
    owner_id: str,
    source_connection: sqlite3.Connection,
    destination_connection: sqlite3.Connection,
    *,
    progress_guard: Callable[[], None] | None = None,
) -> None:
    """Back up two caller-owned file connections through the checked boundary."""

    _validate_backup_owner(owner_id, required_kinds=_PRIVATE_FILE)
    if getattr(source_connection, "in_transaction", False):
        raise sqlite3.OperationalError(
            "Finish the source transaction before starting a backup"
        )
    source_database = _connection_main_database(source_connection)
    destination_path = _private_destination(
        _connection_main_database(destination_connection)
    )
    with _pin_sqlite_source(
        owner_id,
        source_database,
        allow_memory=False,
    ) as source_pin:
        assert source_pin is not None
        _reject_unsafe_or_aliased_destination(source_pin, destination_path)
        _reverify_source(source_pin)
        journal_mode = _guard_destination(destination_connection, restore=False)
        _restore_destination_mode(
            destination_connection,
            journal_mode,
            restore=False,
        )
        _reverify_source(source_pin)
        _backup_pages(
            source_connection,
            destination_connection,
            restore=False,
            progress_guard=progress_guard,
        )
        _reverify_source(source_pin)


def copy_private_sqlite(
    owner_id: str,
    source_path: str | os.PathLike[str],
    target_path: str | os.PathLike[str],
    *,
    progress_guard: Callable[[], None] | None = None,
) -> None:
    """Copy a checked file source to a checked private target via SQLite."""

    _validate_backup_owner(
        owner_id,
        required_kinds=_PRIVATE_AND_READ_ONLY,
    )
    destination_path = _private_destination(target_path)
    with _pin_sqlite_source(
        owner_id,
        source_path,
        allow_memory=False,
    ) as source_pin:
        assert source_pin is not None
        _reject_unsafe_or_aliased_destination(source_pin, destination_path)
        _reverify_source(source_pin)
        source = _connect_registered_sqlite(
            owner_id,
            source_pin.selected,
            read_only=True,
            timeout=0,
        )
        try:
            _reverify_source(source_pin)
            destination = _connect_registered_sqlite(
                owner_id,
                destination_path,
                timeout=0,
            )
            try:
                journal_mode = _guard_destination(destination, restore=False)
                _restore_destination_mode(
                    destination,
                    journal_mode,
                    restore=False,
                )
                _reverify_source(source_pin)
                _backup_pages(
                    source,
                    destination,
                    restore=False,
                    progress_guard=progress_guard,
                )
                _reverify_source(source_pin)
            finally:
                _close_owned_connections((("copy destination", destination),))
        finally:
            _close_owned_connections((("copy source", source),))


def restore_private_sqlite(
    owner_id: str,
    pre_restore_owner_id: str,
    source_path: str | os.PathLike[str],
    destination_path: str | os.PathLike[str],
    pre_restore_path: str | os.PathLike[str],
) -> None:
    """Restore a live database after a private safety snapshot and quiescence."""

    _validate_backup_owner(
        owner_id,
        required_kinds=_PRIVATE_AND_READ_ONLY,
    )
    _validate_backup_owner(
        pre_restore_owner_id,
        required_kinds=_PRIVATE_AND_READ_ONLY,
    )
    selected_destination = _private_destination(destination_path)
    selected_pre_restore = _private_destination(pre_restore_path)
    with (
        _RESTORE_LOCK,
        _pin_sqlite_source(
            owner_id,
            source_path,
            allow_memory=False,
        ) as source_pin,
    ):
        assert source_pin is not None
        _reject_unsafe_or_aliased_destination(
            source_pin,
            selected_destination,
        )
        if _existing_entry_stat(selected_destination) is None:
            raise FileNotFoundError("Live SQLite destination must exist before restore")
        _reject_unsafe_or_aliased_destination(
            source_pin,
            selected_pre_restore,
        )
        _reject_path_pair_alias(
            selected_destination,
            selected_pre_restore,
        )
        _reverify_source(source_pin)
        source = _connect_registered_sqlite(
            owner_id,
            source_pin.selected,
            read_only=True,
            timeout=0,
        )
        destination: sqlite3.Connection | None = None
        pre_restore: sqlite3.Connection | None = None
        original_mode = ""
        mode_restored = False
        final_backup_completed = False
        try:
            _reverify_source(source_pin)
            destination = _connect_registered_sqlite(
                owner_id,
                selected_destination,
                timeout=0,
            )
            original_mode = _guard_destination(destination, restore=True)
            mode_restored = original_mode != "wal"

            pre_restore = _connect_registered_sqlite(
                pre_restore_owner_id,
                selected_pre_restore,
                timeout=0,
            )
            pre_mode = _guard_destination(pre_restore, restore=False)
            _restore_destination_mode(
                pre_restore,
                pre_mode,
                restore=False,
            )
            _backup_pages(
                destination,
                pre_restore,
                restore=False,
            )

            _restore_destination_mode(
                destination,
                original_mode,
                restore=True,
            )
            mode_restored = True
            _reverify_source(source_pin)
            try:
                mode_restored = False
                _backup_pages(source, destination, restore=True)
                final_backup_completed = True
                _restore_destination_mode(
                    destination,
                    original_mode,
                    restore=True,
                )
                mode_restored = True
                _reverify_source(source_pin)
            except BaseException as restore_exc:
                if final_backup_completed:
                    try:
                        mode_restored = False
                        _backup_pages(pre_restore, destination, restore=True)
                        _restore_destination_mode(
                            destination,
                            original_mode,
                            restore=True,
                        )
                        mode_restored = True
                    except BaseException as recovery_exc:
                        indeterminate = SQLiteRestoreIndeterminateError(
                            selected_destination,
                            selected_pre_restore,
                        )
                        indeterminate.add_note(
                            f"Restore validation failure: {restore_exc!r}"
                        )
                        raise indeterminate from recovery_exc
                raise
        finally:
            active_error = sys.exception()
            cleanup_error: BaseException | None = None
            try:
                if destination is not None and original_mode and not mode_restored:
                    _restore_destination_mode(
                        destination,
                        original_mode,
                        restore=True,
                    )
            except BaseException as exc:
                cleanup_error = exc
            finally:
                _close_owned_connections(
                    (
                        ("destination", destination),
                        ("pre-restore backup", pre_restore),
                        ("source", source),
                    )
                )
            if cleanup_error is not None:
                if active_error is None:
                    raise cleanup_error
                active_error.add_note(
                    f"SQLite restore cleanup also failed: {cleanup_error!r}"
                )


__all__ = [
    "ProfileMigrationBoundaryDestination",
    "SQLITE_OWNER_REGISTRY",
    "SQLiteOwnerPolicy",
    "SQLitePrivacyUnverifiedWarning",
    "SQLitePrivateDestinationError",
    "SQLiteRestoreBusyError",
    "SQLiteRestoreIndeterminateError",
    "SQLiteTargetKind",
    "backup_connection_to_private",
    "backup_open_connections_to_private",
    "backup_profile_migration_boundary",
    "close_profile_migration_destination",
    "connect_private_sqlite",
    "connect_private_sqlite_descriptor",
    "copy_private_sqlite",
    "discard_profile_migration_destination",
    "migrate_profile_store_to_candidate",
    "open_canonical_profile_migration_destination",
    "open_profile_migration_boundary_destination",
    "restore_private_sqlite",
]
