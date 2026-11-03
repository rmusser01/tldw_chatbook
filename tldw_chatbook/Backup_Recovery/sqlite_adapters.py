"""Installed operational recovery discovery, capture and restore adapters.

Literal catalogs remain DB-owned. Compatibility lookups retain the original
schema module's live values; no archive supplies policy or SQLite ownership.
"""

from tldw_chatbook.DB import recovery_operations as _schema

from .profile_paths import database_path, user_data_dir

__all__ = ("database_path", "recovery_adapters", "user_data_dir")


def _sqlite_inventory_status(
    config, path, *, owner, setting_name=None, optional_default=False
):
    """Recognize observable absence only for an explicitly optional default.

    This cannot prove historical never-use after all deletion evidence is gone.
    Existing files, explicit selectors and residual SQLite state are never unused.
    """
    from tldw_chatbook.Backup_Recovery.file_inventory import _inventory_root
    from tldw_chatbook.Backup_Recovery.profile_paths import (
        DATABASE_PATHS,
        custom_database_input,
        setting,
    )

    item = _inventory_root(path, owner=owner, external=False)
    if item.status != "unused":
        return "unsupported" if item.status == "included_directory" else item.status
    if not optional_default:
        return "missing_required"
    if setting_name:
        row = next(row for row in DATABASE_PATHS if row[1] == setting_name)
        legacy = "~/.local/share/tldw_cli/" + row[3] if row[3] else None
        if (
            custom_database_input(setting(config, "database", setting_name), legacy)
            is not None
        ):
            return "missing_required"
    suffixes = ("-wal", "-shm", "-journal")
    if owner in {"tts.profile_store", "tts.references"}:
        suffixes += (".lock",)
    for suffix in suffixes:
        sidecar = _inventory_root(
            _schema.Path(str(path) + suffix), owner=owner, external=False
        )
        if sidecar.status != "unused":
            return "missing_required" if sidecar.status == "included" else "unavailable"
    return "unused"


@_schema.dataclass(frozen=True)
class _SQLiteDeclaration:
    owner_id: str
    setting_name: str | None
    leaf: str
    versions: tuple[int, ...]
    schemas: tuple[tuple[int, tuple[str, ...]], ...]
    dependencies: tuple[str, ...] = ()
    activation_required: bool = True
    optional_default: bool = False

    def discover(
        self, config: _schema.Mapping[str, object]
    ) -> tuple[_schema.StorageItem, ...]:
        context = _schema.discovery_context(config)
        if self.setting_name:
            path = _schema.database_path(config, self.setting_name)
        elif self.owner_id == "db.agent_runs":
            path = (
                _schema.database_path(config, "chachanotes_db_path").parent / self.leaf
            )
        else:
            path = _schema.user_data_dir(config) / self.leaf
        status = _schema._sqlite_inventory_status(
            config,
            path,
            owner=self.owner_id,
            setting_name=self.setting_name,
            optional_default=self.optional_default,
        )
        return (
            _schema.StorageItem(
                self.owner_id,
                _schema.storage_logical_id(context, self.owner_id),
                path,
                status,
                tuple(
                    _schema.storage_logical_id(context, owner)
                    for owner in ("config",) + self.dependencies
                ),
            ),
        )

    def schema_policy(self) -> _schema.SchemaPolicy:
        return _schema.SchemaPolicy(self.owner_id, self.versions, self.schemas, ())

    def relocate(
        self, candidate: _schema.Path, mapping: _schema.Mapping[str, _schema.Path]
    ) -> None:
        """Validate immutable historical bytes, granting no execution authority.

        Historical IDs, leases, pending states and paths are evidence, not local
        claims. Staged restore and the durable activation ledger (tasks17/20/21)
        own fresh local identities and reviewed bindings; this method never
        starts a runtime constructor or rewrites history to look locally approved.
        """
        issues = self.validate(candidate)
        if issues:
            raise ValueError(issues[0])


class _WorkspacesAdapter(_SQLiteDeclaration):
    def schema_policy(self) -> _schema.SchemaPolicy:
        return _schema.SchemaPolicy(
            self.owner_id,
            self.versions,
            self.schemas,
            (
                (9, 10, _schema._WORKSPACES_ARCHIVE_V9_TO_V10_SQL),
                (10, 11, _schema.WORKSPACE_MIGRATION),
            ),
        )

    def validate(self, candidate: _schema.Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with _schema.closing(
                connect_private_sqlite(
                    "recovery.operations.workspaces", candidate, read_only=True
                )
            ) as connection:
                return _schema._validate_sqlite(
                    connection,
                    self.versions,
                    self.schemas,
                    version_query="SELECT MAX(version) FROM schema_version",
                )
        except (OSError, ValueError, _schema.sqlite3.Error):
            return ("operational_validation_unavailable",)

    def capture(
        self,
        item: _schema.StorageItem,
        destination: _schema.Path,
        cancel: _schema.Event,
    ) -> None:
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        with _schema._checked_capture(
            self.owner_id, item, destination, cancel, self.validate
        ) as guard:
            copy_private_sqlite(
                "recovery.operations.workspaces",
                item.path,
                destination,
                progress_guard=guard,
            )


class _AgentRunsAdapter(_SQLiteDeclaration):
    def _validate_connection(self, connection):
        from .sqlite_validation import validate_goal_rows

        return validate_goal_rows(connection)

    def schema_policy(self) -> _schema.SchemaPolicy:
        return _schema.SchemaPolicy(
            self.owner_id,
            self.versions,
            self.schemas,
            (
                (18, 21, _schema._AGENT_RUNS_MIGRATION_18_21),
                (21, 22, _schema._AGENT_RUNS_MIGRATION_21_22),
                (22, 23, _schema._AGENT_RUNS_MIGRATION_22_23),
                (23, 24, _schema.GOAL_MIGRATION),
                (24, 25, _schema.REVIEW_BOUNDARY_MIGRATION),
                (25, 26, _schema.COMMAND_RECEIPT_MIGRATION),
                (26, 27, _schema.AGENT_RUNS_MIGRATION),
            ),
        )

    def validate(self, candidate: _schema.Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with _schema.closing(
                connect_private_sqlite(
                    "recovery.operations.agent_runs", candidate, read_only=True
                )
            ) as connection:
                return _schema._validate_sqlite(
                    connection,
                    self.versions,
                    self.schemas,
                    version_query="SELECT MAX(version) FROM schema_version",
                )
        except (OSError, ValueError, _schema.sqlite3.Error):
            return ("operational_validation_unavailable",)

    def capture(
        self,
        item: _schema.StorageItem,
        destination: _schema.Path,
        cancel: _schema.Event,
    ) -> None:
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        with _schema._checked_capture(
            self.owner_id, item, destination, cancel, self.validate
        ) as guard:
            copy_private_sqlite(
                "recovery.operations.agent_runs",
                item.path,
                destination,
                progress_guard=guard,
            )


def _briefing_relocation_authorizer(read_authorizer):
    """Permit only the installed locator update in a disposable candidate."""

    def authorize(action, first, second, database, source):
        if source is None and (
            action == _schema.sqlite3.SQLITE_TRANSACTION
            and database is None
            and first in {"BEGIN", "COMMIT", "ROLLBACK"}
            or action == _schema.sqlite3.SQLITE_UPDATE
            and database == "main"
            and (first, second) == ("briefing_audio", "file_path")
        ):
            return _schema.sqlite3.SQLITE_OK
        return read_authorizer(action, first, second, database, source)

    return authorize


class _SubscriptionsAdapter(_SQLiteDeclaration):
    def schema_policy(self) -> _schema.SchemaPolicy:
        migration = (
            _schema.Path(_schema.__file__).with_name("migrations")
            / "subscriptions_v2_to_v3_url_provenance.sql"
        )
        statements = []
        pending = ""
        for line in migration.read_text(encoding="utf-8").splitlines(keepends=True):
            pending += line
            if _schema.sqlite3.complete_statement(pending):
                statements.append(pending)
                pending = ""
        if pending.strip():
            raise ValueError("incomplete_provenance_migration")
        return _schema.SchemaPolicy(
            self.owner_id, self.versions, self.schemas, ((2, 3, tuple(statements)),)
        )

    def relocate_restore(
        self,
        item: _schema.StorageItem,
        candidate: _schema.Path,
        mapping: _schema.Mapping[str, _schema.Path],
    ) -> None:
        """Relocate only completed rows through their original declared asset IDs."""
        from tldw_chatbook.Backup_Recovery.models import DiscoveryContext
        from tldw_chatbook.Backup_Recovery.recovery_files import _tree_member_id
        from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions
        from tldw_chatbook.DB.private_sqlite import open_recovery_validation

        parts = item.logical_id.split(":")
        if (
            item.owner != self.owner_id
            or len(parts) != 3
            or parts[0] != "profile"
            or parts[2] != self.owner_id
            or not parts[1]
        ):
            raise ValueError("briefing_audio_mapping_required")
        context = DiscoveryContext(
            _schema.Path("/historical-selector-not-opened"), parts[1]
        )
        with (
            open_recovery_validation(
                self.owner_id, candidate, writable=True
            ) as connection,
            connection,
        ):
            restrictions = _Restrictions(connection)
            connection.set_authorizer(
                _schema._briefing_relocation_authorizer(restrictions.authorize)
            )
            connection.execute("BEGIN IMMEDIATE")
            issues = self._validate_catalog(connection)
            if issues:
                raise ValueError(issues[0])
            updates = []
            for row_id, value in connection.execute(
                "SELECT id,file_path FROM briefing_audio WHERE status='complete'"
            ):
                if (
                    not isinstance(value, str)
                    or not _schema.Path(value).is_absolute()
                    or ".." in _schema.Path(value).parts
                ):
                    raise ValueError("briefing_audio_mapping_required")
                original = _schema.Path(value)
                key = _tree_member_id(
                    context, "subscriptions.assets", original.parent, original
                )
                selected = mapping.get(key)
                if (
                    key not in item.dependencies
                    or not isinstance(selected, _schema.Path)
                    or not selected.is_absolute()
                    or ".." in selected.parts
                    or selected.name != original.name
                ):
                    raise ValueError("briefing_audio_mapping_required")
                if str(selected) != value:
                    updates.append((str(selected), row_id))
            connection.executemany(
                "UPDATE briefing_audio SET file_path=? WHERE id=?", updates
            )

    def validate_restore_dependencies(
        self,
        item: _schema.StorageItem,
        candidate: _schema.Path,
        candidates: _schema.Mapping[str, _schema.Path],
        *,
        topology: _schema.Mapping[str, tuple[str, str | None, str, str]],
    ) -> tuple[str, ...]:
        """Bind relocated rows to the selected config and original finite topology.

        Archive and publication fingerprints check bytes; this hook checks the
        declared relationship and regular candidate, without opening source paths.
        """
        from tldw_chatbook.Backup_Recovery.mobile_preview_budget import table_cost
        from tldw_chatbook.Backup_Recovery.mobile_preview_budget import (
            toml_loads as _budgeted_toml_loads,
        )
        from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration
        from tldw_chatbook.Backup_Recovery.storage_admission import _read_recovery_file
        from tldw_chatbook.DB.private_sqlite import open_recovery_validation

        parts = item.logical_id.split(":")
        if (
            item.owner != self.owner_id
            or len(parts) != 3
            or parts[0] != "profile"
            or parts[2] != self.owner_id
            or not parts[1]
        ):
            return ("invalid_dependency_context",)
        issues = self.validate_restore(item, candidate)
        if issues:
            return issues
        try:
            with open_recovery_validation(
                self.owner_id, candidate, writable=False
            ) as connection:
                table_cost(connection, "briefing_audio", ("file_path",))
                paths = tuple(
                    row[0]
                    for row in connection.execute(
                        "SELECT file_path FROM briefing_audio WHERE status='complete'"
                    )
                )
            if not paths:
                return ()
            config_key = f"profile:{parts[1]}:config"
            if config_key not in item.dependencies or config_key not in candidates:
                return ("dependency_unavailable",)
            config = _budgeted_toml_loads(
                _read_recovery_file(
                    "config", candidates[config_key], max_bytes=16 * 1024**2
                )
            )
            root = _schema.user_data_dir(config) / "briefing_audio"
            prefix = f"profile:{parts[1]}:subscriptions.assets:"
            for value in paths:
                if (
                    not isinstance(value, str)
                    or not _schema.Path(value).is_absolute()
                    or ".." in _schema.Path(value).parts
                    or _schema.Path(value).parent != root
                ):
                    return ("dependency_unavailable",)
                matches = []
                for key in item.dependencies:
                    if not key.startswith(prefix) or key not in candidates:
                        continue
                    record = topology.get(key)
                    if record is None:
                        continue
                    root_id, parent_id, relative, kind = record
                    if (
                        kind == "file"
                        and relative == _schema.Path(value).name
                        and parent_id == root_id
                        and root_id.startswith(prefix)
                        and topology.get(root_id) == (root_id, None, "", "directory")
                    ):
                        matches.append(key)
                if len(matches) != 1 or _RawDeclaration(
                    "subscriptions.assets"
                ).validate(candidates[matches[0]]):
                    return ("dependency_unavailable",)
            return ()
        except (OSError, ValueError, _schema.sqlite3.Error):
            return ("dependency_unavailable",)

    def discover(
        self, config: _schema.Mapping[str, object]
    ) -> tuple[_schema.StorageItem, ...]:
        from dataclasses import replace

        from tldw_chatbook.Backup_Recovery.mobile_preview_budget import table_cost
        from tldw_chatbook.Backup_Recovery.recovery_files import _tree_member_id
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        item = super().discover(config)[0]
        if item.status != "included":
            return (item,)
        if self.validate(item.path):
            return (replace(item, status="unsupported"),)
        root = _schema.user_data_dir(config) / "briefing_audio"
        try:
            with _schema.closing(
                connect_private_sqlite(
                    "recovery.operations.subscriptions", item.path, read_only=True
                )
            ) as connection:
                table_cost(connection, "briefing_audio", ("file_path",))
                paths = tuple(
                    row[0]
                    for row in connection.execute(
                        "SELECT file_path FROM briefing_audio WHERE status='complete'"
                    )
                )
            dependencies = list(item.dependencies)
            for value in paths:
                if (
                    not isinstance(value, str)
                    or _schema.Path(value).parent != root
                    or _schema.Path(value).name in ("", ".", "..")
                ):
                    return (replace(item, status="unsupported"),)
                dependencies.append(
                    _tree_member_id(
                        _schema.discovery_context(config),
                        "subscriptions.assets",
                        root,
                        _schema.Path(value),
                    )
                )
            return (replace(item, dependencies=tuple(dict.fromkeys(dependencies))),)
        except (OSError, ValueError, _schema.sqlite3.Error):
            return (replace(item, status="unavailable"),)

    def validate_dependencies(
        self,
        item: _schema.StorageItem,
        candidate: _schema.Path,
        candidates: _schema.Mapping[str, _schema.Path],
    ) -> tuple[str, ...]:
        from tldw_chatbook.Backup_Recovery.models import DiscoveryContext
        from tldw_chatbook.Backup_Recovery.recovery_files import (
            _RawDeclaration,
            _tree_member_id,
        )
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        parts = item.logical_id.split(":")
        if (
            item.owner != self.owner_id
            or len(parts) != 3
            or parts[0] != "profile"
            or parts[2] != self.owner_id
            or not parts[1]
        ):
            return ("invalid_dependency_context",)
        issues = self.validate(candidate)
        if issues:
            return issues
        context = DiscoveryContext(
            _schema.Path("/historical-selector-not-opened"), parts[1]
        )
        try:
            with _schema.closing(
                connect_private_sqlite(
                    "recovery.operations.subscriptions", candidate, read_only=True
                )
            ) as connection:
                for (value,) in connection.execute(
                    "SELECT file_path FROM briefing_audio WHERE status='complete'"
                ):
                    if (
                        not isinstance(value, str)
                        or not _schema.Path(value).is_absolute()
                        or ".." in _schema.Path(value).parts
                    ):
                        return ("dependency_unavailable",)
                    path = _schema.Path(value)
                    key = _tree_member_id(
                        context, "subscriptions.assets", path.parent, path
                    )
                    if (
                        key not in item.dependencies
                        or key not in candidates
                        or _RawDeclaration("subscriptions.assets").validate(
                            candidates[key]
                        )
                    ):
                        return ("dependency_unavailable",)
            return ()
        except (OSError, ValueError, _schema.sqlite3.Error):
            return ("dependency_unavailable",)

    def _validate_catalog(
        self, connection: _schema.sqlite3.Connection
    ) -> tuple[str, ...]:
        issues = _schema._validate_sqlite(
            connection,
            self.versions,
            self.schemas,
            version_query="SELECT MAX(version) FROM schema_version",
        )
        if issues:
            return issues
        from tldw_chatbook.Backup_Recovery.mobile_preview_budget import (
            catalog_cost,
            scalar_cost,
        )

        scalar_cost(connection, 2)
        if connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE name='db_schema_version'"
        ).fetchone():
            stamp = connection.execute(
                "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
            ).fetchone()
            catalog_cost(connection)
            actual = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
                )
            )
            if stamp not in _schema._subscriptions_chachanotes_stamps(actual):
                return ("unsupported_schema_version",)
        return ()

    def validate_restore(
        self, item: _schema.StorageItem, candidate: _schema.Path
    ) -> tuple[str, ...]:
        """Inspect disposable bytes without caching them as immutable live sources."""
        from tldw_chatbook.DB.private_sqlite import open_recovery_validation

        if item.owner != self.owner_id:
            return ("invalid_dependency_context",)
        try:
            with open_recovery_validation(
                self.owner_id, candidate, writable=False
            ) as connection:
                return self._validate_catalog(connection)
        except (OSError, ValueError, _schema.sqlite3.Error):
            return ("operational_validation_unavailable",)

    def validate(self, candidate: _schema.Path) -> tuple[str, ...]:
        from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

        try:
            with _schema.closing(
                connect_private_sqlite(
                    "recovery.operations.subscriptions", candidate, read_only=True
                )
            ) as connection:
                return self._validate_catalog(connection)
        except (OSError, ValueError, _schema.sqlite3.Error):
            return ("operational_validation_unavailable",)

    def capture(
        self,
        item: _schema.StorageItem,
        destination: _schema.Path,
        cancel: _schema.Event,
    ) -> None:
        from tldw_chatbook.DB.private_sqlite import copy_private_sqlite

        with _schema._checked_capture(
            self.owner_id, item, destination, cancel, self.validate
        ) as guard:
            copy_private_sqlite(
                "recovery.operations.subscriptions",
                item.path,
                destination,
                progress_guard=guard,
            )


def recovery_adapters() -> tuple[_schema.OwnerAdapter, ...]:
    return (
        _WorkspacesAdapter(
            "db.workspaces",
            "workspaces_db_path",
            "tldw_chatbook_workspaces.db",
            (11, 10, 9, 8),
            _schema._WORKSPACES_SCHEMA,
            (),
        ),
        _AgentRunsAdapter(
            "db.agent_runs",
            None,
            "agent_runs.db",
            (18, 21, 22, 23, 24, 25, 26, 27),
            _schema._AGENT_RUNS_SCHEMA,
            ("db.chachanotes.primary",),
            optional_default=True,
        ),
        _SubscriptionsAdapter(
            "db.subscriptions",
            "subscriptions_db_path",
            "tldw_chatbook_subscriptions.db",
            (3, 2),
            _schema._SUBSCRIPTIONS_SCHEMA,
            (),
        ),
    )
