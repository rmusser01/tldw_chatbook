"""Durable recovered-media payloads and references (ADR-126).

Each operation holds its native catalog connection through file publication or
retirement. A committed journal row survives interruption; no age-based cleanup
or temporary-media resolver is allowed to reinterpret a known reference.
"""

import hashlib
import json
import os
import re
import sqlite3
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
from tldw_chatbook.DB.recovery_sqlite import _validate_sqlite
from tldw_chatbook.Utils.private_paths import (
    _open_verified_parent,
    atomic_private_write_bytes,
    lexical_path,
    open_private_binary,
    secure_private_directory,
)

from .participants import _core_access, _register_core_connection
from .recovered_media_schema import SCHEMAS, migrate, schema_policy

MAX_PAYLOAD_BYTES = 512 * 1024 * 1024
_ASSET_ID = re.compile(r"^[0-9a-f]{32}$")


def _identity(profile, message, slug, media_type):
    values = (profile, message, slug, media_type)
    if any(
        type(value) is not str or not value or len(value) > 4096 for value in values
    ):
        raise ValueError("invalid_recovered_reference")
    if not media_type.startswith(("image/", "video/")):
        raise ValueError("invalid_recovered_media_type")
    return values


def _read(path):
    with open_private_binary(path) as opened:
        if not opened.result.verified_private:
            raise ValueError("recovered_media_privacy_unverified")
        payload = opened.stream.read(MAX_PAYLOAD_BYTES + 1)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("recovered_media_too_large")
    return payload


class RecoveredMedia:
    """A profile-scoped catalog; source identities never select file names."""

    is_memory_db = False

    def __init__(self, root: Path):
        self.db_path = lexical_path(root) / "catalog.sqlite3"
        result = secure_private_directory(
            self.root, create=True, application_owned=True
        )
        if not result.verified_private:
            raise ValueError("recovered_media_privacy_unverified")
        with self._connection() as connection:
            migrate(connection)
            issues = _validate_sqlite(connection, (1,), SCHEMAS)
            if issues:
                raise ValueError(issues[0])
        self.recover()

    @property
    def root(self):
        return self.db_path.parent

    @contextmanager
    def _connection(self):
        _core_access(self)
        with closing(
            connect_private_sqlite("recovered.media", self.db_path)
        ) as connection:
            _register_core_connection(self, connection)
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection

    def _path(self, asset_id):
        if type(asset_id) is not str or not _ASSET_ID.fullmatch(asset_id):
            raise ValueError("invalid_recovered_asset_id")
        return self.root / (asset_id + ".payload")

    def retain(
        self, source: Path, *, profile: str, message: str, slug: str, media_type: str
    ) -> str:
        identity = _identity(profile, message, slug, media_type)
        with self._connection() as connection:
            payload = _read(source)
            digest = hashlib.sha256(payload).hexdigest()
            row = connection.execute(
                "SELECT a.asset_id,a.digest,a.size,a.state FROM refs r JOIN assets a USING(asset_id) WHERE profile=? AND message=? AND slug=? AND r.media_type=?",
                identity,
            ).fetchone()
            if row:
                if row[1:3] != (digest, len(payload)) or row[3] == "deleted":
                    raise ValueError("recovered_reference_collision")
                asset_id = row[0]
                if row[3] == "ready":
                    if self._resolve(connection, asset_id)[0] != "ready":
                        raise ValueError("recovered_payload_missing")
                    return asset_id
            else:
                asset_id = uuid.uuid4().hex
                if self._path(asset_id).exists():
                    raise FileExistsError("recovered_asset_id_collision")
                with connection:
                    connection.execute(
                        "INSERT INTO assets VALUES (?,?,?,?, 'pending')",
                        (asset_id, digest, len(payload), media_type),
                    )
                    connection.execute(
                        "INSERT INTO refs VALUES (?,?,?,?,?)", identity + (asset_id,)
                    )
                    connection.execute(
                        "INSERT INTO operations VALUES (?, 'retain')", (asset_id,)
                    )
            connection.execute("BEGIN IMMEDIATE")
            try:
                if connection.execute(
                    "SELECT state FROM assets WHERE asset_id=?", (asset_id,)
                ).fetchone() != ("pending",):
                    raise ValueError("recovered_operation_changed")
                destination = self._path(asset_id)
                if destination.exists() or destination.is_symlink():
                    try:
                        matches = _read(destination) == payload
                    except (OSError, ValueError):
                        matches = False
                    if not matches:
                        raise ValueError("recovered_payload_collision")
                else:
                    result = atomic_private_write_bytes(destination, payload)
                    if not result.verified_private:
                        raise ValueError("recovered_media_privacy_unverified")
                self._finish_retain(connection, asset_id)
            finally:
                connection.rollback()
            return asset_id

    def _finish_retain(self, connection, asset_id):
        row = connection.execute(
            "SELECT digest,size,state FROM assets WHERE asset_id=?", (asset_id,)
        ).fetchone()
        if row is None or row[2] != "pending":
            raise ValueError("recovered_operation_changed")
        payload = _read(self._path(asset_id))
        if (hashlib.sha256(payload).hexdigest(), len(payload)) != row[:2]:
            raise ValueError("recovered_payload_missing")
        with connection:
            changed = connection.execute(
                "UPDATE assets SET state='ready' WHERE asset_id=? AND state='pending'",
                (asset_id,),
            ).rowcount
            if changed != 1:
                raise ValueError("recovered_operation_changed")
            connection.execute(
                "DELETE FROM operations WHERE asset_id=? AND kind='retain'", (asset_id,)
            )

    @staticmethod
    def _valid_tombstone(connection, asset_id):
        if connection.execute(
            "SELECT state FROM assets WHERE asset_id=?", (asset_id,)
        ).fetchone() != ("deleted",):
            return False
        row = connection.execute(
            "SELECT version,references_json FROM tombstones WHERE asset_id=?",
            (asset_id,),
        ).fetchone()
        if row is None or row[0] != 1:
            return False
        try:
            references = json.loads(row[1])
            current = [
                list(row)
                for row in connection.execute(
                    "SELECT profile,message,slug,media_type FROM refs WHERE asset_id=? ORDER BY profile,message,slug,media_type",
                    (asset_id,),
                )
            ]
            return type(references) is list and references == current
        except (TypeError, ValueError):
            return False

    def _resolve(self, connection, asset_id):
        path = self._path(asset_id)
        row = connection.execute(
            "SELECT digest,size,state FROM assets WHERE asset_id=?", (asset_id,)
        ).fetchone()
        if row is None:
            return "unknown", None
        if row[2] == "deleted":
            return (
                ("deleted", None)
                if self._valid_tombstone(connection, asset_id)
                else ("missing", None)
            )
        if row[2] != "ready":
            return "missing", None
        try:
            payload = _read(path)
        except (OSError, ValueError):
            return "missing", None
        if (hashlib.sha256(payload).hexdigest(), len(payload)) != row[:2]:
            return "missing", None
        return "ready", path

    def resolve(self, asset_id: str) -> tuple[str, Path | None]:
        with self._connection() as connection:
            return self._resolve(connection, asset_id)

    def resolve_reference(
        self, *, profile: str, message: str, slug: str, media_type: str
    ):
        identity = _identity(profile, message, slug, media_type)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT asset_id FROM refs WHERE profile=? AND message=? AND slug=? AND media_type=?",
                identity,
            ).fetchone()
            return self._resolve(connection, row[0]) if row else ("unknown", None)

    def add_reference(self, asset_id, *, profile, message, slug, media_type):
        identity = _identity(profile, message, slug, media_type)
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT media_type,state FROM assets WHERE asset_id=?", (asset_id,)
            ).fetchone()
            if row != (media_type, "ready"):
                raise ValueError("recovered_asset_not_ready")
            connection.execute(
                "INSERT INTO refs VALUES (?,?,?,?,?)", identity + (asset_id,)
            )

    def delete(self, asset_id: str) -> None:
        self._path(asset_id)
        with self._connection() as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM recovery_holds WHERE asset_id=?", (asset_id,)
                ).fetchone():
                    raise ValueError("recovered_asset_held")
                row = connection.execute(
                    "SELECT state FROM assets WHERE asset_id=?", (asset_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(asset_id)
                if row[0] != "deleted":
                    references = list(
                        connection.execute(
                            "SELECT profile,message,slug,media_type FROM refs WHERE asset_id=? ORDER BY profile,message,slug,media_type",
                            (asset_id,),
                        )
                    )
                    connection.execute(
                        "INSERT INTO tombstones VALUES (?,1,?)",
                        (asset_id, json.dumps(references)),
                    )
                    connection.execute(
                        "UPDATE assets SET state='deleted' WHERE asset_id=?",
                        (asset_id,),
                    )
                    connection.execute(
                        "INSERT OR REPLACE INTO operations VALUES (?,'delete')",
                        (asset_id,),
                    )
            self._finish_delete(connection, asset_id)

    def _finish_delete(self, connection, asset_id):
        if not self._valid_tombstone(connection, asset_id):
            raise ValueError("invalid_recovered_tombstone")
        operation = connection.execute(
            "SELECT kind FROM operations WHERE asset_id=?", (asset_id,)
        ).fetchone()
        if operation is None:
            return  # Completed, validated deletion is idempotent.
        if operation != ("delete",):
            raise ValueError("invalid_recovered_tombstone")
        path = self._path(asset_id)
        parent, leaf = _open_verified_parent(path, missing_leaf_allowed=True)
        try:
            try:
                before = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                expected = connection.execute(
                    "SELECT digest,size FROM assets WHERE asset_id=?", (asset_id,)
                ).fetchone()
                payload = _read(path)
                after = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
                if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or (
                    hashlib.sha256(payload).hexdigest(),
                    len(payload),
                ) != expected:
                    raise ValueError("recovered_payload_replacement")
                os.unlink(leaf, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.fsync(parent)
        finally:
            os.close(parent)
        with connection:
            connection.execute(
                "DELETE FROM operations WHERE asset_id=? AND kind='delete'", (asset_id,)
            )

    def recover(self):
        """Finish only catalog-journaled operations; never sweep by age/name."""
        with self._connection() as connection:
            for asset_id, kind in connection.execute(
                "SELECT asset_id,kind FROM operations"
            ).fetchall():
                if kind == "delete":
                    self._finish_delete(connection, asset_id)
                else:
                    try:
                        self._finish_retain(connection, asset_id)
                    except (OSError, ValueError):
                        # Required unpublished bytes remain an actionable missing
                        # operation; they are never silently reclassified deleted.
                        continue

    def recovery_adapter(self):
        return _RecoveredAdapter(self.root)

    def hold(self, asset_id, hold_id):
        if type(hold_id) is not str or not hold_id:
            raise ValueError("invalid_recovery_hold")
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._resolve(connection, asset_id)[0] != "ready":
                raise ValueError("recovered_asset_not_ready")
            connection.execute(
                "INSERT INTO recovery_holds VALUES (?,?)", (asset_id, hold_id)
            )

    def release_hold(self, asset_id, hold_id):
        with self._connection() as connection, connection:
            connection.execute(
                "DELETE FROM recovery_holds WHERE asset_id=? AND hold_id=?",
                (asset_id, hold_id),
            )

    def release_reference(self, *, profile, message, slug, media_type):
        identity = _identity(profile, message, slug, media_type)
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT asset_id FROM refs WHERE profile=? AND message=? AND slug=? AND media_type=?",
                identity,
            ).fetchone()
            if row is None:
                return
            asset_id = row[0]
            connection.execute(
                "DELETE FROM refs WHERE profile=? AND message=? AND slug=? AND media_type=?",
                identity,
            )
            if connection.execute(
                "SELECT 1 FROM tombstones WHERE asset_id=?", (asset_id,)
            ).fetchone():
                remaining = list(
                    connection.execute(
                        "SELECT profile,message,slug,media_type FROM refs WHERE asset_id=? ORDER BY profile,message,slug,media_type",
                        (asset_id,),
                    )
                )
                connection.execute(
                    "UPDATE tombstones SET references_json=? WHERE asset_id=?",
                    (json.dumps(remaining), asset_id),
                )

    def cleanup_orphan(self, asset_id):
        """Explicit cleanup only, with a fresh transactional reference/hold check."""
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if connection.execute(
                    "SELECT 1 FROM refs WHERE asset_id=? UNION ALL SELECT 1 FROM recovery_holds WHERE asset_id=?",
                    (asset_id, asset_id),
                ).fetchone():
                    return False
                if not connection.execute(
                    "SELECT 1 FROM assets WHERE asset_id=?", (asset_id,)
                ).fetchone():
                    return False
                connection.execute(
                    "INSERT OR IGNORE INTO tombstones VALUES (?,1,'[]')", (asset_id,)
                )
                connection.execute(
                    "UPDATE assets SET state='deleted' WHERE asset_id=?", (asset_id,)
                )
                connection.execute(
                    "INSERT OR REPLACE INTO operations VALUES (?,'delete')", (asset_id,)
                )
                connection.commit()
                self._finish_delete(connection, asset_id)
                return True
            finally:
                connection.rollback()


class _RecoveredAdapter:
    """Installed catalog policy plus its explicit generated-name dependencies."""

    owner_id = "recovered.media"
    activation_required = True

    def __init__(self, root=None):
        self.root = root

    def schema_policy(self):
        return schema_policy()

    def restore_role(self, item):
        """Classify only installed catalog/generated payload archive topology."""
        meta = item.metadata
        if item.owner != self.owner_id or meta is None:
            raise ValueError("invalid_dependency_context")
        if (
            meta.kind == "directory"
            and meta.parent_id is None
            and meta.relative_path in {"", "."}
        ):
            return "directory"
        if meta.kind == "file" and meta.relative_path == "catalog.sqlite3":
            return "sqlite"
        name = meta.relative_path
        if (
            meta.kind == "file"
            and name.endswith(".payload")
            and _ASSET_ID.fullmatch(name[:-8])
        ):
            return "file"
        raise ValueError("invalid_recovered_role")

    def validate_restore(self, item, candidate):
        """Keep opaque payload bytes out of the SQLite inspection route."""
        from .recovery_files import _RawDeclaration

        role = self.restore_role(item)
        if role == "sqlite":
            return self.validate(candidate)
        if role == "file":
            return _RawDeclaration(self.owner_id, max_bytes=MAX_PAYLOAD_BYTES).validate(
                candidate
            )
        return ()

    def relocate_restore(self, item, candidate, mapping):
        """Generated relative locators need no rewrite; validate the exact role."""
        issues = self.validate_restore(item, candidate)
        if issues:
            raise ValueError(issues[0])

    def validate_restore_dependencies(self, item, candidate, candidates, *, topology):
        """Match ready catalog entries to declared archive-local payload IDs."""
        from .storage_admission import _digest_recovery_file

        meta = item.metadata
        if meta is None or topology.get(item.logical_id) != (
            meta.root_id,
            meta.parent_id,
            meta.relative_path,
            meta.kind,
        ):
            return ("invalid_dependency_context",)
        try:
            if self.restore_role(item) != "sqlite":
                return ()
            issues = self.validate(candidate)
            if issues:
                return issues
            located = {}
            for key in set(item.dependencies) & candidates.keys() & topology.keys():
                root, _parent, relative, kind = topology[key]
                if root == meta.root_id and kind == "file":
                    located.setdefault(relative, []).append(key)
            with closing(
                connect_private_sqlite(
                    "recovery.recovered_media", candidate, read_only=True
                )
            ) as connection:
                for asset_id, digest, size in connection.execute(
                    "SELECT asset_id,digest,size FROM assets WHERE state='ready'"
                ):
                    keys = located.get(asset_id + ".payload", ())
                    if len(keys) != 1 or _digest_recovery_file(
                        self.owner_id, candidates[keys[0]], max_bytes=MAX_PAYLOAD_BYTES
                    ) != (size, digest):
                        return ("recovered_payload_missing",)
            return ()
        except (OSError, ValueError, RuntimeError, sqlite3.Error):
            return ("recovered_payload_missing",)

    def validate(self, candidate):
        try:
            with closing(
                connect_private_sqlite(
                    "recovery.recovered_media", candidate, read_only=True
                )
            ) as connection:
                return self._validate_connection(connection)
        except (OSError, ValueError, TypeError, sqlite3.Error):
            return ("recovered_validation_unavailable",)

    @staticmethod
    def _validate_connection(connection):
        """Validate installed schema/domain on the caller-owned connection."""
        issues = _validate_sqlite(connection, (1,), SCHEMAS)
        if issues:
            return issues
        if connection.execute("SELECT 1 FROM operations").fetchone():
            return ("recovered_operation_pending",)
        if connection.execute(
            "SELECT 1 FROM refs r JOIN assets a USING(asset_id) WHERE r.media_type != a.media_type"
        ).fetchone():
            return ("invalid_recovered_reference",)
        if connection.execute(
            "SELECT 1 FROM tombstones t JOIN assets a USING(asset_id) WHERE a.state != 'deleted'"
        ).fetchone():
            return ("invalid_recovered_tombstone",)
        for asset_id, digest, size, media_type, state in connection.execute(
            "SELECT * FROM assets"
        ):
            if (
                not _ASSET_ID.fullmatch(asset_id)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or type(size) is not int
                or not 0 <= size <= MAX_PAYLOAD_BYTES
                or not media_type.startswith(("image/", "video/"))
            ):
                return ("invalid_recovered_asset",)
            if state == "pending":
                return ("recovered_operation_pending",)
            if state == "deleted" and not RecoveredMedia._valid_tombstone(
                connection, asset_id
            ):
                return ("invalid_recovered_tombstone",)
        for profile, message, slug, media_type in connection.execute(
            "SELECT profile,message,slug,media_type FROM refs"
        ):
            _identity(profile, message, slug, media_type)
        return ()

    def discover(self, config):
        from dataclasses import replace

        from .models import StorageItem, discovery_context, storage_logical_id
        from .profile_paths import user_data_dir
        from .recovery_files import _RawDeclaration

        root = self.root or user_data_dir(config) / "recovered_media"
        context = discovery_context(config)
        catalog = root / "catalog.sqlite3"
        entries = _RawDeclaration(self.owner_id)._tree(config, root)
        if not root.exists():
            return entries
        issues = self.validate(catalog)
        if issues:
            return (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id),
                    catalog,
                    "unavailable",
                    (),
                ),
            )
        with closing(
            connect_private_sqlite("recovery.recovered_media", catalog, read_only=True)
        ) as connection:
            assets = {
                asset_id + ".payload": (digest, size)
                for asset_id, digest, size in connection.execute(
                    "SELECT asset_id,digest,size FROM assets WHERE state='ready'"
                )
            }
        wanted = {"catalog.sqlite3", *assets}
        selected = []
        found = set()
        for item in entries:
            if item.path == root:
                selected.append(item)
            elif item.path.name in wanted and item.path.parent == root:
                found.add(item.path.name)
                if item.path.name in assets:
                    try:
                        from .storage_admission import _consume_recovery_file

                        size, digest = _consume_recovery_file(
                            self.owner_id,
                            item.path,
                            max_bytes=MAX_PAYLOAD_BYTES,
                            collect=False,
                            digest=True,
                            private=True,
                        )
                        valid = (digest, size) == assets[item.path.name]
                    except (OSError, ValueError):
                        valid = False
                    if not valid:
                        item = replace(item, status="missing_required")
                selected.append(item)
            elif item.path.name not in {
                "catalog.sqlite3-wal",
                "catalog.sqlite3-shm",
                "catalog.sqlite3-journal",
            }:
                # Unknown files are evidence, never inferred deletions or TTL junk.
                selected.append(replace(item, status="unsupported"))
        for missing in wanted - found:
            selected.append(
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id, missing),
                    root / missing,
                    "missing_required",
                    (),
                )
            )
        dependencies = tuple(
            item.logical_id for item in selected if item.path != catalog
        )
        return tuple(
            replace(
                item,
                dependencies=tuple(dict.fromkeys(item.dependencies + dependencies)),
            )
            if item.path == catalog
            else item
            for item in selected
        )

    def capture(self, item, destination, cancel):
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite
        from tldw_chatbook.DB.recovery_sqlite import _checked_capture

        from .recovery_files import _RawDeclaration

        if item.path is not None and item.path.name == "catalog.sqlite3":
            with _checked_capture(
                self.owner_id, item, destination, cancel, self.validate
            ) as guard:
                copy_private_sqlite(
                    "recovery.recovered_media",
                    item.path,
                    destination,
                    progress_guard=guard,
                )
        else:
            _RawDeclaration(self.owner_id, max_bytes=MAX_PAYLOAD_BYTES).capture(
                item, destination, cancel
            )

    def relocate(self, candidate, mapping):
        # Payload locators are generated relative names. Stable source identities,
        # references, tombstones and recovery holds remain evidence after a move.
        issues = self.validate(candidate)
        if issues:
            raise ValueError(issues[0])

    def validate_dependencies(self, item, candidate, candidates):
        """Require every registered ready payload, independently of its name."""
        from .models import DiscoveryContext
        from .recovery_files import _tree_member_id
        from .storage_admission import _digest_recovery_file

        issues = self.validate(candidate)
        if issues:
            return issues
        parts = item.logical_id.split(":")
        if (
            item.owner != self.owner_id
            or item.path is None
            or len(parts) < 3
            or parts[0] != "profile"
        ):
            return ("invalid_dependency_context",)
        context = DiscoveryContext(Path("/historical-selector-not-opened"), parts[1])
        try:
            with closing(
                connect_private_sqlite(
                    "recovery.recovered_media", candidate, read_only=True
                )
            ) as connection:
                for asset_id, digest, size in connection.execute(
                    "SELECT asset_id,digest,size FROM assets WHERE state='ready'"
                ):
                    key = _tree_member_id(
                        context,
                        self.owner_id,
                        item.path.parent,
                        item.path.parent / (asset_id + ".payload"),
                    )
                    if key not in item.dependencies or key not in candidates:
                        return ("recovered_payload_missing",)
                    if _digest_recovery_file(
                        self.owner_id, candidates[key], max_bytes=MAX_PAYLOAD_BYTES
                    ) != (size, digest):
                        return ("recovered_payload_missing",)
            return ()
        except (OSError, ValueError, sqlite3.Error):
            return ("recovered_payload_missing",)


def recovery_adapters():
    """Declare the baseline owner without opening any runtime store."""
    return (_RecoveredAdapter(),)


def current_profile_id():
    """Use the same selector identity as recovery inventory discovery."""
    from .profile_paths import effective_config_path

    return hashlib.sha256(str(effective_config_path()).encode()).hexdigest()[:24]


def message_image_metadata(root, profile, messages):
    """Read existing reference metadata in bounded batches, without payload IO.

    Readers do not migrate or replay publication journals. The ordinary owner
    connection still binds catalog admission and retires on this worker thread.
    """
    path = lexical_path(root) / "catalog.sqlite3"
    if not path.exists():
        return {}
    store = object.__new__(RecoveredMedia)
    store.db_path = path
    found = {}
    with store._connection() as connection:
        issues = _validate_sqlite(connection, (1,), SCHEMAS)
        if issues:
            raise ValueError(issues[0])
        for start in range(0, len(messages), 200):
            batch = messages[start : start + 200]
            slots = ",".join("?" for _ in batch)
            rows = connection.execute(
                "SELECT r.message,a.asset_id,a.digest,a.size,a.state,r.media_type "
                "FROM refs r JOIN assets a ON a.asset_id=r.asset_id "
                f"WHERE r.profile=? AND r.message IN ({slots}) "  # nosec B608: placeholders only
                "AND r.media_type LIKE 'image/%'",
                (profile, *batch),
            ).fetchall()
            for message, asset, digest, size, state, media_type in rows:
                status = state if state in {"ready", "deleted"} else "missing"
                if status == "deleted" and not store._valid_tombstone(
                    connection, asset
                ):
                    status = "missing"
                found[message] = (
                    ("missing", None, None, None, None)
                    if message in found
                    else (status, asset, digest, size, media_type)
                )
    return found


def read_message_image_payload(root, metadata):
    """Read and verify the selected ready payload exactly once on a worker."""
    status, asset, digest, size, _media_type = metadata
    if status != "ready" or not _ASSET_ID.fullmatch(asset):
        raise ValueError("invalid_recovered_image")
    payload = _read(lexical_path(root) / (asset + ".payload"))
    if (hashlib.sha256(payload).hexdigest(), len(payload)) != (digest, size):
        raise ValueError("recovered_payload_mismatch")
    return payload


def resolve_message_image(message, *, root=None, profile=None):
    """Resolve the enhanced widget's single-image slot from persisted identity.

    Multiple recovered image keys cannot be inferred from a single-image widget;
    render missing rather than silently substituting one or using transient bytes.
    """
    if root is None:
        from tldw_chatbook.Utils.paths import get_user_data_dir

        root = get_user_data_dir() / "recovered_media"
    root = Path(root)
    if not (root / "catalog.sqlite3").exists():
        return "unknown", None, None
    store = RecoveredMedia(root)
    with store._connection() as connection:
        rows = connection.execute(
            "SELECT asset_id,media_type FROM refs WHERE profile=? AND message=? AND media_type LIKE 'image/%'",
            (profile or current_profile_id(), message),
        ).fetchall()
        if not rows:
            return "unknown", None, None
        if len(rows) != 1:
            return "missing", None, None
        asset_id, media_type = rows[0]
        status, path = store._resolve(connection, asset_id)
        return status, _read(path) if path else None, media_type
