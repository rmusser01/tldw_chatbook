"""Installed backup owner for the plugin root (TASK-40132); importing opens no stores.

One tree owner, shaped like ``recovered.media``: ``registry.sqlite3`` is the one
SQLite member, captured through the SQLite backup route (committed WAL frames
fold in) and validated against the exact frozen catalog below. Every other file
is raw bytes, including staging snapshots and ``acquisition-custody/``: plugin
recovery re-verifies git-installed packages from them. Left out: the runtime
owner's ``runtime.lock``, and inside each ``staging/<name>/`` everything but
``snapshot/`` (owner decision D1, 2026-11-10, as an allow-list): the fetch-time
host git/SSH configuration copies (they can hold credential helpers and SSH
config), ``repository.git``, ``content/`` and whatever a later fetch leaves there.
Each is one excluded row whose contents are never listed.

While a plugin process may be running, or a live runtime owner holds the lock,
the preview shows one unavailable plugin root and the issue
``plugin_writers_active``, so a partial backup omits plugins; capture never waits.
The app's plugin service holds the lock from its first use until it quits, so a
complete in-app backup after plugin use needs a restart (accepted ceiling, owner
decision 2026-11-10).
"""

from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event

from tldw_chatbook.Backup_Recovery.config_adapter import _Definition
from tldw_chatbook.Backup_Recovery.models import (
    OwnerAdapter,
    SchemaPolicy,
    StorageItem,
    discovery_context,
    storage_logical_id,
)

OWNER = "plugins"
REGISTRY = "registry.sqlite3"
LOCK = "runtime.lock"
WRITERS_ACTIVE = "plugin_writers_active"
FETCH_ADDED = "plugin_fetch_added_since_backup"

# Exact installed v5 catalog, measured from PluginRegistry._reference_schema()
# (migrations 001-005). A new migration must add its catalog here; the pinning
# test compares this text with the installed migrations.
REGISTRY_SCHEMAS = (
    (
        5,
        (
            'CREATE UNIQUE INDEX installation_alias_unique ON installations(alias) WHERE alias IS NOT NULL',
            'CREATE INDEX process_recovery ON processes(installation_id, state, owner_session)',
            'CREATE INDEX process_revision ON processes(installation_id, revision_digest, kind, state)',
            'CREATE INDEX root_user_owner ON root_users(owner_token)',
            'CREATE INDEX root_user_recovery ON root_users(root_id, state, root_generation)',
            "CREATE TABLE activation (\n    installation_id TEXT NOT NULL REFERENCES installations(installation_id),\n    workspace_id TEXT NOT NULL,\n    intent TEXT NOT NULL CHECK (intent IN ('inherit', 'enabled', 'disabled')),\n    PRIMARY KEY (installation_id, workspace_id)\n)",
            "CREATE TABLE authority_generations (\n    installation_id TEXT NOT NULL REFERENCES installations(installation_id),\n    scope_kind TEXT NOT NULL CHECK (scope_kind IN ('installation', 'global_default', 'workspace')),\n    workspace_id TEXT NOT NULL DEFAULT '',\n    generation INTEGER NOT NULL CHECK (generation >= 0),\n    revoked INTEGER NOT NULL DEFAULT 0 CHECK (revoked IN (0, 1)),\n    PRIMARY KEY (installation_id, scope_kind, workspace_id),\n    CHECK ((scope_kind = 'workspace' AND workspace_id != '') OR\n           (scope_kind != 'workspace' AND workspace_id = ''))\n)",
            'CREATE TABLE components (\n    installation_id TEXT NOT NULL,\n    revision_digest TEXT NOT NULL,\n    component_id TEXT NOT NULL,\n    definition_json TEXT NOT NULL,\n    PRIMARY KEY (installation_id, revision_digest, component_id),\n    FOREIGN KEY (installation_id, revision_digest) REFERENCES revisions\n)',
            'CREATE TABLE data_roots (\n    root_id TEXT PRIMARY KEY NOT NULL,\n    installation_id TEXT NOT NULL,\n    workspace_id TEXT,\n    path TEXT NOT NULL UNIQUE,\n    generation INTEGER NOT NULL CHECK (generation >= 0),\n    deletion_fenced INTEGER NOT NULL DEFAULT 0 CHECK (deletion_fenced IN (0, 1))\n, custody_json TEXT, cleanup_json TEXT)',
            'CREATE TABLE installations (\n    installation_id TEXT PRIMARY KEY NOT NULL,\n    revision_digest TEXT,\n    activation_default INTEGER NOT NULL DEFAULT 0 CHECK (activation_default IN (0, 1))\n, alias TEXT)',
            'CREATE TABLE mappings (\n    installation_id TEXT NOT NULL REFERENCES installations(installation_id),\n    mapping_id TEXT NOT NULL,\n    mapping_json TEXT NOT NULL,\n    PRIMARY KEY (installation_id, mapping_id)\n)',
            'CREATE TABLE marketplace_sources (\n    source_id TEXT PRIMARY KEY NOT NULL,\n    binding_json TEXT NOT NULL,\n    binding_mac TEXT NOT NULL,\n    snapshot_json TEXT,\n    refresh_error TEXT\n)',
            'CREATE TABLE operations (\n    operation_id TEXT PRIMARY KEY NOT NULL,\n    installation_id TEXT NOT NULL,\n    phase TEXT NOT NULL,\n    intent_json TEXT NOT NULL\n)',
            "CREATE TABLE processes (\n    token TEXT PRIMARY KEY NOT NULL,\n    operation_id TEXT NOT NULL,\n    installation_id TEXT NOT NULL,\n    workspace_id TEXT,\n    revision_digest TEXT NOT NULL,\n    owner_session TEXT NOT NULL,\n    state TEXT NOT NULL CHECK (state IN ('pending', 'published', 'unresolved', 'settled')),\n    kind TEXT NOT NULL CHECK (kind IN ('pending_launch', 'active_run', 'idle_connection', 'archived_history')),\n    provenance_json TEXT\n, root_coverage TEXT NOT NULL DEFAULT 'unknown' CHECK (root_coverage IN ('unknown', 'qualified_none', 'known')), root_grants_json TEXT)",
            'CREATE TABLE receipts (\n    receipt_id TEXT PRIMARY KEY NOT NULL,\n    operation_id TEXT NOT NULL,\n    receipt_json TEXT NOT NULL\n)',
            'CREATE TABLE revision_trust (\n    installation_id TEXT NOT NULL,\n    revision_digest TEXT NOT NULL,\n    reviewed INTEGER NOT NULL CHECK (reviewed IN (0, 1)),\n    PRIMARY KEY (installation_id, revision_digest),\n    FOREIGN KEY (installation_id, revision_digest) REFERENCES revisions\n)',
            'CREATE TABLE revisions (\n    installation_id TEXT NOT NULL REFERENCES installations(installation_id),\n    revision_digest TEXT NOT NULL,\n    inspection_json TEXT NOT NULL,\n    PRIMARY KEY (installation_id, revision_digest)\n)',
            "CREATE TABLE root_users (\n    usage_token TEXT PRIMARY KEY NOT NULL,\n    root_id TEXT NOT NULL,\n    root_generation INTEGER NOT NULL CHECK (root_generation >= 0),\n    binding_digest TEXT NOT NULL,\n    owner_token TEXT NOT NULL REFERENCES processes(token),\n    access TEXT NOT NULL CHECK (access IN ('read', 'write', 'read_write')),\n    state TEXT NOT NULL CHECK (state IN ('held', 'unresolved', 'released')),\n    UNIQUE (owner_token, root_id, root_generation)\n)",
            'CREATE TABLE selections (\n    installation_id TEXT NOT NULL,\n    revision_digest TEXT NOT NULL,\n    component_id TEXT NOT NULL,\n    selected INTEGER NOT NULL CHECK (selected IN (0, 1)),\n    PRIMARY KEY (installation_id, revision_digest, component_id),\n    FOREIGN KEY (installation_id, revision_digest, component_id) REFERENCES components\n)',
            'CREATE TABLE sources (\n    source_id TEXT PRIMARY KEY NOT NULL,\n    installation_id TEXT NOT NULL REFERENCES installations(installation_id),\n    source_json TEXT NOT NULL\n)',
            'CREATE TABLE tombstones (\n    installation_id TEXT PRIMARY KEY NOT NULL,\n    generation INTEGER NOT NULL CHECK (generation >= 0),\n    operation_id TEXT NOT NULL\n)',
            "CREATE TRIGGER immutable_components BEFORE UPDATE ON components BEGIN\n    SELECT RAISE(ABORT, 'immutable component');\nEND",
            "CREATE TRIGGER immutable_revisions BEFORE UPDATE ON revisions BEGIN\n    SELECT RAISE(ABORT, 'immutable revision');\nEND",
        ),
    ),
)


def _staging_excluded(relative: str) -> bool:
    """Whether a root-relative path is a fetch folder member other than snapshot/."""
    parts = Path(relative).parts
    return len(parts) > 2 and parts[0] == "staging" and parts[2] != "snapshot"


def refuse_added_fetch_folders(items, archived) -> None:
    """Refuse a replace restore over a fetch folder the archive lacks that holds
    excluded members: restore would retire the folder but must keep what it never
    captured, so Start could not finish (owner decision 2026-11-12: refuse early
    with a reason that says what to do; automatic handling is a follow-up).

    ``items`` is the target inventory, ``archived`` the archive's producer rows
    by logical id. A folder holding only ``snapshot/`` restores normally.
    """
    paths = {item.path: item for item in items if item.path is not None}
    roots = [
        item.path
        for item in items
        if item.owner == OWNER
        and item.metadata is not None
        and item.metadata.relative_path == ""
    ]
    for item in items:
        if item.owner != OWNER or item.status != "intentionally_excluded":
            continue
        for root in roots:
            if item.path is None or root not in item.path.parents:
                continue
            parts = item.path.relative_to(root).parts
            if not _staging_excluded("/".join(parts)):
                continue
            folder = paths.get(root.joinpath(*parts[:2]))
            if folder is None or folder.logical_id not in archived:
                raise ValueError(FETCH_ADDED)


@contextmanager
def _runtime_lock(root: Path, *, create: bool = False):
    """Hold the runtime owner's lock so no owner starts while plugin bytes are read.

    Preview never creates the file (no lock means no owner ever ran). Capture
    creates it, as the owner does, so a restored root without one is guarded too.
    """
    import os

    import portalocker

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(root / LOCK, flags | (os.O_CREAT if create else 0), 0o600)
    except FileNotFoundError:
        yield
        return
    with os.fdopen(descriptor, "rb") as handle:
        try:
            portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.exceptions.LockException:
            raise ValueError(WRITERS_ACTIVE) from None
        try:
            yield
        finally:
            portalocker.unlock(handle)


def _root_of(item: StorageItem) -> Path:
    if item.owner != OWNER or item.metadata is None or item.path is None:
        raise ValueError("invalid_dependency_context")
    root = item.path
    for _ in Path(item.metadata.relative_path).parts:
        root = root.parent
    return root


class _Plugins(_Definition):
    def schema_policy(self) -> SchemaPolicy:
        return SchemaPolicy(self.owner_id, (5,), REGISTRY_SCHEMAS, ())

    def restore_role(self, item: StorageItem) -> str:
        _root_of(item)
        if item.metadata.kind == "directory":
            return "directory"
        return "sqlite" if item.metadata.relative_path == REGISTRY else "file"

    def _inspect(self, candidate: Path) -> tuple[tuple[str, ...], bool]:
        """Validate one registry view; report whether a plugin process may run."""
        import sqlite3

        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with closing(
                connect_private_sqlite(
                    "recovery.plugins.registry", candidate, read_only=True
                )
            ) as connection:
                return self._registry_state(connection)
        except (OSError, ValueError, sqlite3.Error):
            return ("plugin_registry_validation_unavailable",), False

    @staticmethod
    def _registry_state(connection):
        from tldw_chatbook.DB.recovery_sqlite import _validate_sqlite

        issues = _validate_sqlite(connection, (5,), REGISTRY_SCHEMAS)
        if issues:
            return issues, False
        # Unresolved rows survive owner death: no lease is no proof of a stop.
        running = connection.execute(
            "SELECT 1 FROM processes WHERE state != 'settled' LIMIT 1"
        ).fetchone()
        return (), running is not None

    def validate(self, candidate: Path) -> tuple[str, ...]:
        return self._inspect(candidate)[0]

    def validate_restore(self, item, candidate):
        """Staged copies open only through the restricted validation authority."""
        import sqlite3

        from tldw_chatbook.DB.private_sqlite import open_recovery_validation

        role = self.restore_role(item)
        if role == "file":
            return super().validate(candidate)
        if role != "sqlite":
            return ()
        try:
            with open_recovery_validation(
                self.owner_id, candidate, writable=False
            ) as connection:
                return self._registry_state(connection)[0]
        except (OSError, ValueError, TypeError, sqlite3.Error):
            return ("plugin_registry_validation_unavailable",)

    def relocate_restore(self, item, candidate, mapping):
        """Restore in place only: registry rows bind absolute plugin-root paths."""
        from tldw_chatbook.DB.private_sqlite import open_recovery_validation

        issues = self.validate_restore(item, candidate)
        if issues:
            raise ValueError(issues[0])
        if self.restore_role(item) != "sqlite":
            return
        import json

        def materialized(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "materialized_identity" and isinstance(child, str):
                        yield child
                    else:
                        yield from materialized(child)
            elif isinstance(value, list):
                for child in value:
                    yield from materialized(child)

        from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

        # The registry names the root as the runtime owner resolves it (M8).
        root = PluginRuntimeOwner(item.path.parent).root
        with open_recovery_validation(
            self.owner_id, candidate, writable=False
        ) as connection:
            for root_id, path in connection.execute(
                "SELECT root_id, path FROM data_roots"
            ):
                if path != str(root / "data" / root_id):
                    raise ValueError("owner_relocation_unverified:" + self.owner_id)
            # Installed and fetched package paths bind the root as data roots do.
            for (raw,) in connection.execute("SELECT inspection_json FROM revisions"):
                if any(
                    not Path(path).is_relative_to(root)
                    for path in materialized(json.loads(raw))
                ):
                    raise ValueError("owner_relocation_unverified:" + self.owner_id)

    def relocate(self, candidate, mapping):
        raise ValueError("owner_relocation_unverified:" + self.owner_id)

    def _tree(self, config, root, **kwargs):
        return super()._tree(config, root, prune=_staging_excluded, **kwargs)

    def discover(self, config):
        context = discovery_context(config)
        root = self._definition_path(config)
        registry, journal = root / REGISTRY, root / (REGISTRY + "-journal")
        sidecars = {root / (REGISTRY + "-wal"), root / (REGISTRY + "-shm")}
        entries = []
        for item in super().discover(config):
            if item.path in sidecars:
                continue  # Declared as transient SQLite sidecars by inventory.
            if item.status == "included" and item.path == root / LOCK:
                item = replace(item, status="intentionally_excluded")
            elif root in item.path.parents and _staging_excluded(
                item.path.relative_to(root).as_posix()
            ):
                # Whatever it is (even a link), it is never copied: never blocks.
                item = replace(item, status="intentionally_excluded")
            elif item.status == "included" and item.path == journal:
                # A stray rollback journal could undo a restored registry copy.
                item = replace(item, status="unsupported")
            entries.append(item)
        if not any(item.status == "included_directory" for item in entries):
            return tuple(entries)
        active = False
        try:
            with _runtime_lock(root):
                for index, item in enumerate(entries):
                    if item.path == registry and item.status == "included":
                        issues, active = self._inspect(registry)
                        if issues:
                            entries[index] = replace(item, status="unsupported")
        except ValueError:
            active = True
        except OSError:
            active = None
        if active is False:
            return tuple(entries)
        # One unavailable root, no children: the tree stays topologically valid,
        # an unselected Tools group excludes it, and partial capture omits it.
        unavailable = (
            replace(next(item for item in entries if item.path == root), status="unavailable"),
        )
        if active is None:
            return unavailable
        return unavailable + (
            StorageItem(
                self.owner_id,
                storage_logical_id(context, self.owner_id, "writers_active"),
                None,
                "unavailable",
                (storage_logical_id(context, "config"),),
            ),
        )

    def capture(self, item: StorageItem, destination: Path, cancel: Event) -> None:
        root = _root_of(item)
        if _staging_excluded(item.metadata.relative_path):
            raise ValueError("plugin_staging_member_excluded")
        with _runtime_lock(root, create=True):
            if self.restore_role(item) != "sqlite":
                super().capture(item, destination, cancel)
                return
            from tldw_chatbook.DB.private_sqlite import copy_private_sqlite
            from tldw_chatbook.DB.recovery_sqlite import _checked_capture

            with _checked_capture(
                self.owner_id, item, destination, cancel, self.validate
            ) as guard:
                copy_private_sqlite(
                    "recovery.plugins.registry",
                    item.path,
                    destination,
                    progress_guard=guard,
                )


def _restored_generations(plugins_root: Path, *, approve: bool = False):
    """Restored generations still requiring plugin review; optionally approve them.

    Only generations that list this owner count: a profile restored before the
    owner existed has nothing for plugin recovery to approve.
    """
    from tldw_chatbook.Backup_Recovery.activation import ActivationStore
    from tldw_chatbook.Backup_Recovery.generation_witnesses import _witnesses
    from tldw_chatbook.Backup_Recovery.storage_admission import acquire_storage

    pending = []
    with acquire_storage(plugins_root) as lease:
        for witness in _witnesses(plugins_root, lease):
            store = ActivationStore(Path(witness["store_root"]))
            if OWNER not in witness["owners"] or store.allowed(
                witness["generation"], OWNER
            ):
                continue
            if approve:
                store.approve(witness["generation"], OWNER)
            else:
                pending.append(witness["generation"])
    return tuple(pending)


def restore_review_required(plugins_root: Path) -> bool:
    """Whether restored plugins must wait for authority recovery; fails closed."""
    try:
        return bool(_restored_generations(plugins_root))
    except (OSError, ValueError, RuntimeError):
        return True


def approve_restore(plugins_root: Path) -> None:
    """Record completed plugin authority recovery for the restored generation."""
    _restored_generations(plugins_root, approve=True)


def recovery_adapters() -> tuple[OwnerAdapter, ...]:
    return (_Plugins("plugins", leaf="plugins", tree=True),)
