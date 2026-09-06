"""Read-only recovery and coherent export for legacy generic Collections."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import stat
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
from tldw_chatbook.Utils.path_validation import validate_path_simple


LEGACY_EXPORT_FORMAT = "tldw-chatbook-legacy-collections"
LEGACY_EXPORT_VERSION = 1
MAX_LEGACY_RECOVERY_PAGE_SIZE = 100
MAX_LEGACY_EXPORT_BATCH_SIZE = 100
_MAX_SQLITE_OFFSET = 2**63 - 1
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_PRIVATE_CREATE_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
)

_REQUIRED_COLUMNS = {
    "library_collections": frozenset(
        {
            "collection_id",
            "name",
            "description",
            "created_at",
            "updated_at",
            "deleted_at",
        }
    ),
    "library_collection_items": frozenset(
        {
            "membership_id",
            "collection_id",
            "source_type",
            "source_id",
            "title",
            "created_at",
        }
    ),
}


def _traditional_mode_write_authority_complete(
    platform: str | None = None,
) -> bool:
    """Return whether mode bits include effective directory ACL write masks."""
    return (sys.platform if platform is None else platform).startswith("linux")


def _secure_dirfd_publication_available() -> bool:
    required = {os.open, os.stat, os.link, os.rename, os.unlink}
    return (
        os.name == "posix"
        and _traditional_mode_write_authority_complete()
        and getattr(os, "O_DIRECTORY", 0) != 0
        and getattr(os, "O_NOFOLLOW", 0) != 0
        and required.issubset(os.supports_dir_fd)
        and os.stat in os.supports_follow_symlinks
        and os.link in os.supports_follow_symlinks
        and hasattr(os, "fstat")
        and hasattr(os, "fsync")
    )


def _posix_private_mode_verifiable() -> bool:
    return os.name == "posix"


class LegacyCollectionsRecoveryError(RuntimeError):
    """Stable, path- and content-free recovery failure."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class LegacyCollectionRecord:
    collection_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    deleted_at: str | None


@dataclass(frozen=True)
class LegacyMembershipRecord:
    membership_id: str
    collection_id: str
    source_type: str
    source_id: str
    title: str
    created_at: str


@dataclass(frozen=True)
class LegacyRecoveryPage:
    items: tuple[LegacyCollectionRecord | LegacyMembershipRecord, ...]
    total: int
    page: int
    size: int


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


class LegacyCollectionsRecovery:
    """Bounded v1 inspection that never initializes capture schema."""

    def __init__(
        self,
        database: LibraryCollectionsDB | str | Path,
        *,
        clock: Callable[[], str] = _utc_now,
        export_batch_size: int = 20,
    ) -> None:
        if isinstance(database, LibraryCollectionsDB):
            self._database = database
            self._database_path: Path | None = None
        elif isinstance(database, (str, Path)):
            path = Path(database)
            try:
                if not path.is_absolute() or not path.is_file():
                    raise LegacyCollectionsRecoveryError("legacy_database_unavailable")
            except LegacyCollectionsRecoveryError:
                raise
            except (OSError, ValueError):
                raise LegacyCollectionsRecoveryError(
                    "legacy_database_unavailable"
                ) from None
            self._database = None
            # Preserve aliases for the private seam's no-follow validation.
            self._database_path = path
        else:
            raise LegacyCollectionsRecoveryError("legacy_database_unavailable")
        if not callable(clock):
            raise LegacyCollectionsRecoveryError("invalid_legacy_recovery_clock")
        self._validate_bound(export_batch_size, "invalid_legacy_export_batch")
        if export_batch_size > MAX_LEGACY_EXPORT_BATCH_SIZE:
            raise LegacyCollectionsRecoveryError("invalid_legacy_export_batch")
        self._clock = clock
        self._export_batch_size = export_batch_size

    @property
    def export_publication_posture(self) -> str:
        """Describe the publication guarantees available on this platform.

        The verified path is limited to platforms where effective ACL write
        masks are represented by traditional mode bits. It requires a parent
        owned by this effective user with no group/other write bits, so another
        filesystem principal cannot replace the randomized temporary sibling
        in the final check/use window. Processes running as this same user are
        the same file authority and can also replace the completed export
        afterward. Other platforms use the explicit unverified fallback.
        """
        return (
            "verified_private_parent_dirfd"
            if _secure_dirfd_publication_available()
            else "unverified_platform"
        )

    def list_collections(self, *, page: int, size: int = 20) -> LegacyRecoveryPage:
        """Return one stable page including active and deleted v1 rows."""
        offset = self._page_offset(page, size)
        try:
            with self._read_transaction() as connection:
                self._require_legacy_schema(connection)
                total = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM library_collections"
                    ).fetchone()[0]
                )
                rows = connection.execute(
                    "SELECT collection_id, name, description, created_at, "
                    "updated_at, deleted_at FROM library_collections "
                    "ORDER BY collection_id LIMIT ? OFFSET ?",
                    (size, offset),
                ).fetchall()
        except LegacyCollectionsRecoveryError:
            raise
        except sqlite3.Error:
            raise LegacyCollectionsRecoveryError("legacy_recovery_failed") from None
        return LegacyRecoveryPage(
            tuple(self._collection(row) for row in rows),
            total,
            page,
            size,
        )

    def list_memberships(self, *, page: int, size: int = 20) -> LegacyRecoveryPage:
        """Return one stable page of untouched v1 membership records."""
        offset = self._page_offset(page, size)
        try:
            with self._read_transaction() as connection:
                self._require_legacy_schema(connection)
                total = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM library_collection_items"
                    ).fetchone()[0]
                )
                rows = connection.execute(
                    "SELECT membership_id, collection_id, source_type, source_id, "
                    "title, created_at FROM library_collection_items "
                    "ORDER BY collection_id, membership_id LIMIT ? OFFSET ?",
                    (size, offset),
                ).fetchall()
        except LegacyCollectionsRecoveryError:
            raise
        except sqlite3.Error:
            raise LegacyCollectionsRecoveryError("legacy_recovery_failed") from None
        return LegacyRecoveryPage(
            tuple(self._membership(row) for row in rows),
            total,
            page,
            size,
        )

    def export_json(
        self,
        destination: str | os.PathLike[str],
        *,
        overwrite_identity: tuple[int, int] | None,
    ) -> Path:
        """Publish one coherent, private JSON recovery snapshot."""
        secure_publication = _secure_dirfd_publication_available()
        target, initial_identity, parent_identity = self._export_target(
            destination,
            overwrite_identity=overwrite_identity,
            require_private_parent=secure_publication,
        )
        if not secure_publication:
            return self._export_json_unverified(
                target,
                initial_identity,
                parent_identity,
            )
        self._before_parent_open()
        parent_fd = self._open_export_parent(target.parent, parent_identity)
        temporary_leaf: str | None = None
        temporary_exists = False
        descriptor = -1
        try:
            current_identity = self._existing_identity_at(parent_fd, target.name)
            if current_identity != initial_identity:
                raise LegacyCollectionsRecoveryError("legacy_export_target_changed")
            descriptor, temporary_leaf = self._create_temporary(
                parent_fd,
                target.name,
            )
            temporary_exists = True
            temporary_metadata = os.fstat(descriptor)
            temporary_identity = self._private_temporary_identity(
                temporary_metadata,
                require_private_mode=True,
            )
            active_descriptor = descriptor
            descriptor = -1
            self._write_export_descriptor(active_descriptor)

            self._before_publish()
            self._require_parent_identity(target.parent, parent_identity)
            self._publish_export(
                parent_fd,
                temporary_leaf,
                target.name,
                initial_identity,
                temporary_identity,
            )
            temporary_exists = False
            os.fsync(parent_fd)
            self._require_parent_identity(target.parent, parent_identity)
            return target
        except LegacyCollectionsRecoveryError:
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError):
            raise LegacyCollectionsRecoveryError("legacy_export_failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_exists and temporary_leaf is not None:
                try:
                    os.unlink(temporary_leaf, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            os.close(parent_fd)

    def _export_json_unverified(
        self,
        target: Path,
        initial_identity: tuple[int, int] | None,
        parent_identity: tuple[int, int, int, int],
    ) -> Path:
        """Use an explicitly unverified path fallback on non-dirfd platforms."""
        temporary: Path | None = None
        descriptor = -1
        require_private_mode = _posix_private_mode_verifiable()
        try:
            self._before_parent_open()
            self._require_parent_identity(target.parent, parent_identity)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            temporary_identity = self._private_temporary_identity(
                os.fstat(descriptor),
                require_private_mode=require_private_mode,
            )
            active_descriptor = descriptor
            descriptor = -1
            self._write_export_descriptor(active_descriptor)

            self._before_publish()
            self._require_parent_identity(target.parent, parent_identity)
            self._publish_export_path(
                temporary,
                target,
                initial_identity,
                temporary_identity,
                require_private_mode=require_private_mode,
            )
            temporary = None
            self._require_parent_identity(target.parent, parent_identity)
            return target
        except LegacyCollectionsRecoveryError:
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError):
            raise LegacyCollectionsRecoveryError("legacy_export_failed") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def _write_export_descriptor(self, descriptor: int) -> None:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
            closefd=True,
        ) as stream:
            with self._read_transaction() as connection:
                self._require_legacy_schema(connection)
                self._write_export(stream, connection)
            stream.flush()
            os.fsync(stream.fileno())

    def _write_export(self, stream, connection: sqlite3.Connection) -> None:
        stream.write('{"format":')
        json.dump(LEGACY_EXPORT_FORMAT, stream, ensure_ascii=False)
        stream.write(',"version":')
        json.dump(LEGACY_EXPORT_VERSION, stream)
        stream.write(',"exported_at":')
        json.dump(self._clock(), stream, ensure_ascii=False)
        stream.write(',"collections":[')
        self._write_query_rows(
            stream,
            connection,
            "collections",
            "SELECT collection_id, name, description, created_at, updated_at, "
            "deleted_at FROM library_collections ORDER BY collection_id",
            self._collection,
        )
        stream.write('],"memberships":[')
        self._after_export_collections()
        self._write_query_rows(
            stream,
            connection,
            "memberships",
            "SELECT membership_id, collection_id, source_type, source_id, title, "
            "created_at FROM library_collection_items "
            "ORDER BY collection_id, membership_id",
            self._membership,
        )
        stream.write("]}")

    def _write_query_rows(
        self,
        stream,
        connection: sqlite3.Connection,
        kind: str,
        query: str,
        convert,
    ) -> None:
        cursor = connection.execute(query)
        first = True
        while True:
            rows = cursor.fetchmany(self._export_batch_size)
            if not rows:
                return
            self._on_export_batch(kind, len(rows))
            for row in rows:
                if not first:
                    stream.write(",")
                json.dump(
                    asdict(convert(row)),
                    stream,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                first = False

    def _publish_export(
        self,
        parent_fd: int,
        temporary_leaf: str,
        target_leaf: str,
        initial_identity: tuple[int, int] | None,
        temporary_identity: tuple[int, int],
    ) -> None:
        self._require_entry_identity_at(
            parent_fd,
            temporary_leaf,
            temporary_identity,
            reason="legacy_export_temporary_changed",
        )
        current_identity = self._existing_identity_at(parent_fd, target_leaf)
        if current_identity != initial_identity:
            raise LegacyCollectionsRecoveryError("legacy_export_target_changed")
        if initial_identity is None:
            try:
                os.link(
                    temporary_leaf,
                    target_leaf,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                raise LegacyCollectionsRecoveryError(
                    "legacy_export_target_changed"
                ) from None
            os.unlink(temporary_leaf, dir_fd=parent_fd)
            self._require_entry_identity_at(
                parent_fd,
                target_leaf,
                temporary_identity,
                reason="legacy_export_temporary_changed",
            )
            return
        os.rename(
            temporary_leaf,
            target_leaf,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        self._require_entry_identity_at(
            parent_fd,
            target_leaf,
            temporary_identity,
            reason="legacy_export_temporary_changed",
        )

    def _publish_export_path(
        self,
        temporary: Path,
        target: Path,
        initial_identity: tuple[int, int] | None,
        temporary_identity: tuple[int, int],
        *,
        require_private_mode: bool,
    ) -> None:
        self._require_path_identity(
            temporary,
            temporary_identity,
            reason="legacy_export_temporary_changed",
            require_private_mode=require_private_mode,
        )
        current_identity = self._existing_identity(target)
        if current_identity != initial_identity:
            raise LegacyCollectionsRecoveryError("legacy_export_target_changed")
        if initial_identity is None:
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError:
                raise LegacyCollectionsRecoveryError(
                    "legacy_export_target_changed"
                ) from None
            os.unlink(temporary)
        else:
            os.replace(temporary, target)
        self._require_path_identity(
            target,
            temporary_identity,
            reason="legacy_export_temporary_changed",
            require_private_mode=require_private_mode,
        )

    def _export_target(
        self,
        destination: str | os.PathLike[str],
        *,
        overwrite_identity: tuple[int, int] | None,
        require_private_parent: bool,
    ) -> tuple[Path, tuple[int, int] | None, tuple[int, int, int, int]]:
        try:
            target = Path(destination)
            if not target.is_absolute() or target.name in {"", ".", ".."}:
                raise ValueError
            validated = validate_path_simple(target, require_exists=False)
            if validated != target:
                raise ValueError
            parent = target.parent
            if not parent.is_dir() or parent.resolve(strict=True) != parent:
                raise ValueError
            parent_metadata = os.stat(parent, follow_symlinks=False)
            if not stat.S_ISDIR(parent_metadata.st_mode):
                raise ValueError
            parent_identity = self._parent_identity(parent_metadata)
            # Descriptor-relative publication still names the temporary leaf.
            # Limiting directory mutation to this effective user closes that
            # final name-resolution window against other OS principals.
            if require_private_parent and (
                parent_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(parent_metadata.st_mode) & 0o022
            ):
                raise ValueError
            existing = self._existing_identity(target)
            self._validate_overwrite_identity(overwrite_identity)
            if existing is None:
                if overwrite_identity is not None:
                    raise LegacyCollectionsRecoveryError("legacy_export_target_changed")
            elif overwrite_identity is None:
                raise LegacyCollectionsRecoveryError("legacy_export_target_exists")
            elif overwrite_identity != existing:
                raise LegacyCollectionsRecoveryError("legacy_export_target_changed")
            return target, existing, parent_identity
        except LegacyCollectionsRecoveryError:
            raise
        except (OSError, TypeError, ValueError, UnicodeError):
            raise LegacyCollectionsRecoveryError(
                "invalid_legacy_export_destination"
            ) from None

    @staticmethod
    def _existing_identity(target: Path) -> tuple[int, int] | None:
        try:
            metadata = os.lstat(target)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise LegacyCollectionsRecoveryError("invalid_legacy_export_destination")
        return metadata.st_dev, metadata.st_ino

    @staticmethod
    def _existing_identity_at(
        parent_fd: int,
        target_leaf: str,
    ) -> tuple[int, int] | None:
        try:
            metadata = os.stat(
                target_leaf,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise LegacyCollectionsRecoveryError("invalid_legacy_export_destination")
        return metadata.st_dev, metadata.st_ino

    @staticmethod
    def _private_temporary_identity(
        metadata: os.stat_result,
        *,
        require_private_mode: bool,
    ) -> tuple[int, int]:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (require_private_mode and stat.S_IMODE(metadata.st_mode) != 0o600)
        ):
            raise LegacyCollectionsRecoveryError("legacy_export_temporary_changed")
        return metadata.st_dev, metadata.st_ino

    @classmethod
    def _require_entry_identity_at(
        cls,
        parent_fd: int,
        leaf: str,
        expected_identity: tuple[int, int],
        *,
        reason: str,
    ) -> None:
        try:
            metadata = os.stat(
                leaf,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except OSError:
            raise LegacyCollectionsRecoveryError(reason) from None
        if (
            cls._private_temporary_identity(
                metadata,
                require_private_mode=True,
            )
            != expected_identity
        ):
            raise LegacyCollectionsRecoveryError(reason)

    @classmethod
    def _require_path_identity(
        cls,
        path: Path,
        expected_identity: tuple[int, int],
        *,
        reason: str,
        require_private_mode: bool,
    ) -> None:
        try:
            metadata = os.lstat(path)
        except OSError:
            raise LegacyCollectionsRecoveryError(reason) from None
        if (
            cls._private_temporary_identity(
                metadata,
                require_private_mode=require_private_mode,
            )
            != expected_identity
        ):
            raise LegacyCollectionsRecoveryError(reason)

    @staticmethod
    def _parent_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            stat.S_IMODE(metadata.st_mode),
        )

    @staticmethod
    def _open_export_parent(
        parent: Path,
        expected_identity: tuple[int, int, int, int],
    ) -> int:
        descriptor = -1
        try:
            descriptor = os.open(parent, _DIRECTORY_OPEN_FLAGS)
            opened = os.fstat(descriptor)
            current = os.stat(parent, follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or LegacyCollectionsRecovery._parent_identity(opened)
                != expected_identity
                or LegacyCollectionsRecovery._parent_identity(current)
                != expected_identity
            ):
                raise OSError
            return descriptor
        except OSError:
            if descriptor >= 0:
                os.close(descriptor)
            raise LegacyCollectionsRecoveryError(
                "legacy_export_parent_changed"
            ) from None

    @staticmethod
    def _require_parent_identity(
        parent: Path,
        expected_identity: tuple[int, int, int, int],
    ) -> None:
        try:
            current = os.stat(parent, follow_symlinks=False)
        except OSError:
            raise LegacyCollectionsRecoveryError(
                "legacy_export_parent_changed"
            ) from None
        if (
            not stat.S_ISDIR(current.st_mode)
            or LegacyCollectionsRecovery._parent_identity(current) != expected_identity
        ):
            raise LegacyCollectionsRecoveryError("legacy_export_parent_changed")

    @staticmethod
    def _create_temporary(parent_fd: int, target_leaf: str) -> tuple[int, str]:
        for _attempt in range(32):
            temporary_leaf = f".{target_leaf}.{secrets.token_hex(8)}.tmp"
            try:
                descriptor = os.open(
                    temporary_leaf,
                    _PRIVATE_CREATE_FLAGS,
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            return descriptor, temporary_leaf
        raise LegacyCollectionsRecoveryError("legacy_export_failed")

    @staticmethod
    def _validate_overwrite_identity(value: tuple[int, int] | None) -> None:
        if value is None:
            return
        if (
            type(value) is not tuple
            or len(value) != 2
            or any(type(part) is not int or part < 0 for part in value)
        ):
            raise LegacyCollectionsRecoveryError("invalid_overwrite_identity")

    @contextmanager
    def _read_transaction(self) -> Iterator[sqlite3.Connection]:
        if self._database is not None:
            with self._database.read_transaction() as connection:
                yield connection
            return
        database_path = self._database_path
        if database_path is None:
            raise LegacyCollectionsRecoveryError("legacy_database_unavailable")
        try:
            connection = connect_private_sqlite(
                "library.legacy_recovery",
                database_path,
                read_only=True,
                must_exist=True,
            )
        except (OSError, ValueError, sqlite3.Error):
            raise LegacyCollectionsRecoveryError(
                "legacy_database_unavailable"
            ) from None
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            connection.execute("BEGIN DEFERRED")
            yield connection
        finally:
            try:
                connection.rollback()
            finally:
                connection.close()

    @staticmethod
    def _require_legacy_schema(connection: sqlite3.Connection) -> None:
        for table, expected in _REQUIRED_COLUMNS.items():
            columns = {
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if not expected <= columns:
                raise LegacyCollectionsRecoveryError("legacy_schema_unavailable")

    @staticmethod
    def _collection(row: sqlite3.Row) -> LegacyCollectionRecord:
        return LegacyCollectionRecord(
            str(row["collection_id"]),
            str(row["name"]),
            str(row["description"]),
            str(row["created_at"]),
            str(row["updated_at"]),
            None if row["deleted_at"] is None else str(row["deleted_at"]),
        )

    @staticmethod
    def _membership(row: sqlite3.Row) -> LegacyMembershipRecord:
        return LegacyMembershipRecord(
            str(row["membership_id"]),
            str(row["collection_id"]),
            str(row["source_type"]),
            str(row["source_id"]),
            str(row["title"]),
            str(row["created_at"]),
        )

    @classmethod
    def _page_offset(cls, page: int, size: int) -> int:
        cls._validate_bound(page, "invalid_legacy_recovery_page")
        cls._validate_bound(size, "invalid_legacy_recovery_page")
        if size > MAX_LEGACY_RECOVERY_PAGE_SIZE:
            raise LegacyCollectionsRecoveryError("invalid_legacy_recovery_page")
        offset = (page - 1) * size
        if offset > _MAX_SQLITE_OFFSET:
            raise LegacyCollectionsRecoveryError("invalid_legacy_recovery_page")
        return offset

    @staticmethod
    def _validate_bound(value: int, reason: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise LegacyCollectionsRecoveryError(reason)

    def _after_export_collections(self) -> None:
        """Test seam after collections while the read snapshot remains pinned."""

    def _before_publish(self) -> None:
        """Test seam after fsync and before target identity revalidation."""

    def _before_parent_open(self) -> None:
        """Test seam after path validation and before parent pinning."""

    def _on_export_batch(self, kind: str, count: int) -> None:
        """Test seam proving bounded cursor fetches."""
