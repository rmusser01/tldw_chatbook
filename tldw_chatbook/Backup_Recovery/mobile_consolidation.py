"""Read-only legacy mobile inventory; observations never grant execution authority."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from tldw_chatbook.DB.sql_validation import escape_identifier

from .file_inventory import _inventory_root
from .profile_paths import lexical_path


# Physical names are installed declarations, never inferred from a glob.
class MobileConsolidationRefused(ValueError):
    """A fixed consolidation refusal with an optional existing local operation."""

    def __init__(self, reason, *, operation_id=None, control_root=None):
        super().__init__(reason)
        self.operation_id = operation_id
        self.control_root = control_root


LEGACY_STORES = (
    ("chat.db", "db.chachanotes.primary"),
    ("workspaces.db", "db.workspaces"),
    ("runs.db", "db.agent_runs"),
)


def _resource_identity(path: Path) -> tuple:
    path = lexical_path(path)
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError("mobile_unsafe_alias")
    try:
        info = path.stat()
    except FileNotFoundError:
        return (str(path), "absent")
    if info.st_nlink != 1 and path.is_file():
        raise ValueError("mobile_unsafe_alias")
    return (str(path), info.st_dev, info.st_ino)


def _source_group_identity(path: Path) -> tuple:
    """Group admitted existing objects without weakening path-bound fingerprints."""
    identity = _resource_identity(path)
    return identity if identity[-1] == "absent" else identity[1:]


def _refuse_witness_aliases(witnesses) -> None:
    """Different spellings of an observed object cannot grant separate destinations."""
    representations = {}
    for _, chain in witnesses:
        for path, device, inode, _ in chain:
            if device is None:
                continue
            previous = representations.setdefault((device, inode), path)
            if previous != path:
                raise ValueError("mobile_unsafe_alias")


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def _sqlite_row_digest(values) -> str:
    """Bind SQLite storage types as well as values in reviewed source content."""
    encoded = []
    for value in values:
        if value is None:
            encoded.append(("null", None))
        elif type(value) is int:
            encoded.append(("integer", value))
        elif type(value) is float:
            encoded.append(("real", value.hex()))
        elif type(value) is str:
            encoded.append(("text", value))
        elif type(value) is bytes:
            encoded.append(("blob", value.hex()))
        else:
            raise ValueError("mobile_unsupported_sqlite_value")
    return _digest(encoded)


@dataclass(frozen=True)
class CanonicalConsoleStores:
    """Explicit physical destinations; identity is metadata, never permission."""

    chat_path: Path
    workspace_path: Path
    run_path: Path

    def __post_init__(self) -> None:
        for field in ("chat_path", "workspace_path", "run_path"):
            object.__setattr__(self, field, lexical_path(getattr(self, field)))

    @property
    def identity(self) -> str:
        normalized = tuple(lexical_path(path) for path in self.paths)
        if len(set(normalized)) != len(normalized):
            raise ValueError("mobile_unsafe_alias")
        for path in normalized:
            _resource_identity(path)
        return _digest(tuple(str(lexical_path(path)) for path in self.paths))

    @property
    def paths(self) -> tuple[Path, Path, Path]:
        return self.chat_path, self.workspace_path, self.run_path


@dataclass(frozen=True)
class MobileSelectorProvenance:
    """Checked locator/content evidence; no configuration bodies or permission."""

    selector: Path
    fingerprint: str | None
    witnesses: tuple
    destination: CanonicalConsoleStores | None = None
    destination_witnesses: tuple = ()
    destination_sidecars: tuple = ()


@dataclass(frozen=True)
class MobileConsolidationPreview:
    source_dir: Path
    source_fingerprint: str
    destination: CanonicalConsoleStores
    has_data: bool
    needs_destination_review: bool
    issues: tuple[str, ...]
    schema_identities: tuple[tuple[str, int, str], ...] = ()
    counts: tuple[tuple[str, str, int], ...] = ()
    mappings: tuple[tuple[str, str, tuple], ...] = ()
    dependencies: tuple[tuple[str, str, str, str, str], ...] = ()
    derived_tables: tuple[tuple[str, str], ...] = ()
    capture_scope: tuple[Path, ...] = ()
    rollback_scope: tuple[Path, ...] = ()
    candidate_destinations: tuple[CanonicalConsoleStores, ...] = ()
    destination_witnesses: tuple[tuple[str, tuple], ...] = ()
    destination_sidecars: tuple[tuple[str, tuple], ...] = ()
    identity_conflicts: tuple[tuple[str, str, tuple], ...] = ()
    source_selector: Path | None = None
    target_selector: Path | None = None
    selector_provenance: tuple[MobileSelectorProvenance, ...] = ()
    source_witnesses: tuple = ()
    credential_acknowledgements: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _adapters():
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.DB.recovery_operations import recovery_adapters

    return {a.owner_id: a for a in (*core_adapters(), *recovery_adapters())}


def _inspect_store(path, adapter, *, external_references=True):
    """Inspect only an exact installed catalog, retaining IDs and counts only."""
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    from .sqlite_validation import _catalog, _check, _restrict_connection, _version

    policy = adapter.schema_policy()
    with closing(
        connect_private_sqlite("recovery.mobile.preview", path, read_only=True)
    ) as connection:
        restrictions = _restrict_connection(connection)
        catalog = _catalog(connection)
        sql = tuple(row[3] for row in catalog if row[3] is not None)
        matches = tuple(
            version for version, schema in policy.schema_sql if schema == sql
        )
        if not matches:
            raise ValueError("unsupported_schema")
        version = _version(connection, adapter.owner_id)
        if version not in matches:
            raise ValueError("unsupported_schema_version")
        issues, version = _check(connection, adapter, policy, restrictions)
        if issues:
            raise ValueError(issues[0])
        counts, identities, edges, derived = [], {}, [], []
        content_digest = hashlib.sha256()
        # Names are admitted by exact catalog equality before identifier quoting.
        tables = {name: ddl for kind, name, _, ddl in catalog if kind == "table"}
        fts = {
            name
            for name, ddl in tables.items()
            if ddl and ddl.startswith("CREATE VIRTUAL TABLE")
        }
        for name in tables:
            if name == "sqlite_sequence" or any(
                name == root or name.startswith(root + "_") for root in fts
            ):
                derived.append(name)
                continue
            quoted = escape_identifier(name)
            count = connection.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]
            if count > 100_000:
                raise ValueError("mobile_inventory_limit")
            counts.append((name, count))
            columns = connection.execute(f"PRAGMA table_xinfo({quoted})").fetchall()
            keys = [row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5]]
            if keys:
                selection = ",".join(escape_identifier(key) for key in keys)
                values = connection.execute(
                    f"SELECT {selection} FROM {quoted} ORDER BY {selection}"
                ).fetchall()
                if len(values) > 100_000:
                    raise ValueError("mobile_inventory_limit")
                identities[name] = tuple(tuple(row) for row in values)
            order = (
                " ORDER BY " + ",".join(escape_identifier(key) for key in keys)
                if keys
                else ""
            )
            content_digest.update(name.encode())
            for values in connection.execute(f"SELECT * FROM {quoted}" + order):
                content_digest.update(_sqlite_row_digest(values).encode())
            for row in connection.execute(f"PRAGMA foreign_key_list({quoted})"):
                edges.append((name, row[3], row[2], row[4]))
        if path.name == "runs.db":
            identities["agent_definitions.name"] = tuple(
                tuple(row)
                for row in connection.execute("SELECT name FROM agent_definitions")
            )
        # External owner bytes cannot be silently dropped or guessed from locators.
        if external_references and path.name == "chat.db":
            for query in (
                "SELECT 1 FROM notes WHERE file_path_on_disk IS NOT NULL OR sync_root_folder IS NOT NULL LIMIT 1",
                "SELECT 1 FROM chat_dictionaries WHERE file_path IS NOT NULL LIMIT 1",
                "SELECT 1 FROM persona_visual_assets LIMIT 1",
                "SELECT 1 FROM visual_identity_assets LIMIT 1",
            ):
                if connection.execute(query).fetchone() is not None:
                    raise ValueError("mobile_missing_file_dependency")
        # Cross-store/soft references do not appear in SQLite foreign_key_check.
        refs = {}
        for name, columns in _SOFT_REFERENCES.get(path.name, {}).items():
            if name not in tables:
                continue
            present = {
                row[1]
                for row in connection.execute(
                    f"PRAGMA table_xinfo({escape_identifier(name)})"
                )
            }
            for column, target_store, target_table, target_column in columns:
                if column not in present:
                    continue
                rows = connection.execute(
                    f"SELECT DISTINCT {escape_identifier(column)} FROM {escape_identifier(name)} "
                    f"WHERE {escape_identifier(column)} IS NOT NULL"
                ).fetchall()
                refs[(name, column, target_store, target_table, target_column)] = tuple(
                    row[0] for row in rows if row[0] != ""
                )
        if restrictions.expired():
            raise ValueError("mobile_inventory_limit")
        return (
            version,
            _digest(sql),
            counts,
            identities,
            edges,
            derived,
            refs,
            content_digest.hexdigest(),
        )


_SOFT_REFERENCES = {
    "chat.db": {
        "conversations": (
            ("workspace_id", "workspaces.db", "workspace_records", "workspace_id"),
        ),
        "messages": (("parent_message_id", "chat.db", "messages", "id"),),
        "console_trace_owners": (("run_id", "runs.db", "agent_runs", "id"),),
    },
    "runs.db": {
        "agent_runs": (
            ("conversation_id", "chat.db", "conversations", "id"),
            ("parent_run_id", "runs.db", "agent_runs", "id"),
            ("resumed_from_run_id", "runs.db", "agent_runs", "id"),
            ("assistant_message_id", "chat.db", "messages", "id"),
            ("spawn_event_id", "chat.db", "console_trace_events", "event_id"),
            ("agent_definition", "runs.db", "agent_definitions", "name"),
        ),
        "automatic_work_chains": (
            ("conversation_id", "chat.db", "conversations", "id"),
        ),
        "goal_runs": (("conversation_id", "chat.db", "conversations", "id"),),
    },
}


def _path_witnesses(paths):
    """Strict ancestor/object evidence; creation or replacement needs a journal transition."""
    result = []
    for path in paths:
        path = lexical_path(path)
        chain = []
        for ancestor in reversed((path, *path.parents)):
            try:
                info = ancestor.lstat()
                witness = (info.st_dev, info.st_ino, info.st_mode)
            except FileNotFoundError:
                witness = (None, None, None)
            chain.append((str(ancestor), *witness))
        result.append((str(path), tuple(chain)))
    return tuple(result)


def _destination_witnesses(destination):
    return _path_witnesses(destination.paths)


def _selector_observation(path):
    from .bootstrap import _fingerprint

    _resource_identity(path)
    before = _path_witnesses((path,))[0][1]
    fingerprint = _fingerprint(path) if before[-1][2] is not None else None
    if _path_witnesses((path,))[0][1] != before:
        raise ValueError("mobile_selector_changed")
    return fingerprint, before


def _sqlite_sidecars(path):
    """Observe only the installed fixed cohort under a qualified native parent."""
    import stat

    from tldw_chatbook.Utils.platform_files import os

    from .native_files import pinned_directory

    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    with pinned_directory(parent):
        pass
    result = []
    for suffix in ("-wal", "-shm", "-journal"):
        selected = Path(f"{path}{suffix}")
        _resource_identity(selected)
        try:
            info = selected.lstat()
        except FileNotFoundError:
            result.append((str(selected), None, None, None))
            continue
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
        ):
            raise ValueError("mobile_unsafe_alias")
        result.append((str(selected), info.st_dev, info.st_ino, info.st_mode))
    return tuple(result)


def _destination_sidecars(destination):
    paths = tuple(
        path
        for main in destination.paths
        for path in (
            main,
            *(Path(f"{main}{suffix}") for suffix in ("-wal", "-shm", "-journal")),
        )
    )
    if len(set(paths)) != len(paths):
        raise ValueError("mobile_unsafe_alias")
    physical = [_resource_identity(path)[1:] for path in paths if path.exists()]
    if len(set(physical)) != len(physical):
        raise ValueError("mobile_unsafe_alias")
    return tuple((str(path), _sqlite_sidecars(path)) for path in destination.paths)


def _preview(source_dir, destination, config=None, *, review_destination=True):
    source_dir = lexical_path(source_dir)
    issues, schemas, counts, mappings, dependencies, derived, scope = (
        set(),
        [],
        [],
        [],
        [],
        [],
        [],
    )
    identities, references, evidence = {}, {}, []
    conflicts = []
    witnesses = _destination_witnesses(destination)
    sidecars = ()
    try:
        sidecars = _destination_sidecars(destination)
        _ = destination.identity  # Validate the explicit namespace before observation.
        _refuse_witness_aliases(witnesses)
        resources = tuple(
            _resource_identity(path)
            for path in (
                *destination.paths,
                *(source_dir / leaf for leaf, _ in LEGACY_STORES),
            )
        )
        physical = [row[1:] for row in resources if row[-1] != "absent"]
        if len(set(physical)) != len(physical) or any(
            path in destination.paths
            for path in (source_dir / leaf for leaf, _ in LEGACY_STORES)
        ):
            raise ValueError("mobile_unsafe_alias")
        evidence.append(_resource_identity(source_dir))
        if not source_dir.exists():
            return MobileConsolidationPreview(
                source_dir,
                _digest((str(source_dir), "absent")),
                destination,
                False,
                False,
                (),
                rollback_scope=destination.paths,
                destination_witnesses=witnesses,
                destination_sidecars=sidecars,
            )
        root = _inventory_root(source_dir, owner="db.agent_runs", external=False)
        if root.status not in {"unused", "included_directory"}:
            raise ValueError("mobile_unsafe_source")
        if root.status == "unused":
            return MobileConsolidationPreview(
                source_dir,
                _digest((str(source_dir), "absent")),
                destination,
                False,
                False,
                (),
                rollback_scope=destination.paths,
                destination_witnesses=witnesses,
                destination_sidecars=sidecars,
            )
        adapters = _adapters()
        known = {
            leaf + suffix
            for leaf, _ in LEGACY_STORES
            for suffix in ("", "-wal", "-shm")
        }
        known.add(".runs.db.execution.lock")
        lock = _inventory_root(
            source_dir / ".runs.db.execution.lock",
            owner="db.agent_runs",
            external=False,
        )
        if lock.status not in {"unused", "included"}:
            issues.add("mobile_unsafe_source")
        for path in source_dir.iterdir():
            if path.name not in known:
                issues.add("mobile_unknown_source_content")
                break
        for leaf, owner in LEGACY_STORES:
            path = source_dir / leaf
            item = _inventory_root(path, owner=owner, external=False)
            if item.status == "unused":
                if any(
                    Path(str(path) + suffix).exists()
                    for suffix in ("-wal", "-shm", "-journal")
                ):
                    issues.add("mobile_missing_store")
                continue
            scope.append(path)
            if item.status != "included":
                issues.add("mobile_unsafe_source")
                continue
            try:
                version, schema, rows, keys, edges, indexes, refs, content_hash = (
                    _inspect_store(path, adapters[owner])
                )
                schemas.append((leaf, version, schema))
                counts.extend((leaf, table, count) for table, count in rows)
                mappings.extend(
                    (leaf, table, ids) for table, ids in keys.items() if ids
                )
                identities[leaf] = keys
                references[leaf] = refs
                dependencies.extend(
                    (leaf, table, column, leaf, target + "." + str(key))
                    for table, column, target, key in edges
                )
                derived.extend((leaf, table) for table in indexes)
                evidence.append(
                    (
                        _resource_identity(path),
                        path.stat().st_mtime_ns,
                        path.stat().st_size,
                        schema,
                        rows,
                        keys,
                        content_hash,
                    )
                )
                wal = Path(str(path) + "-wal")
                if wal.exists():
                    info = wal.stat()
                    evidence.append(
                        (
                            "wal",
                            info.st_dev,
                            info.st_ino,
                            info.st_mtime_ns,
                            info.st_size,
                        )
                    )
            except (OSError, ValueError, sqlite3.Error) as error:
                code = (
                    str(error)
                    if isinstance(error, ValueError)
                    else "mobile_inspection_unavailable"
                )
                issues.add(code)
        if not issues and len(identities) == len(LEGACY_STORES):
            from tldw_chatbook.DB.recovery_operations import legacy_mobile_declarations

            from .models import (
                DISCOVERY_CONTEXT_KEY,
                DiscoveryContext,
                DiscoverySelections,
            )

            file_config = (
                dict(config)
                if config is not None
                else {
                    "paths": {"data_dir": str(source_dir.parent.parent)},
                    "general": {"users_name": source_dir.parent.name},
                }
            )
            if DISCOVERY_CONTEXT_KEY not in file_config:
                file_config[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(
                    source_dir.parent / "config.toml",
                    "mobile-preview",
                    DiscoverySelections(),
                )
            for item in _referenced_files(
                file_config,
                legacy_mobile_declarations(file_config, source_dir=source_dir),
            ):
                if item.status in {"unsupported", "unavailable", "missing_required"}:
                    issues.add("mobile_missing_file_dependency")
                if item.path is not None and item.status == "included":
                    scope.append(item.path)
                    info = item.path.stat()
                    evidence.append(
                        (_resource_identity(item.path), info.st_mtime_ns, info.st_size)
                    )
                    from .storage_admission import _digest_recovery_file

                    evidence.append(
                        _digest_recovery_file(
                            item.owner, item.path, max_bytes=256 * 1024**3
                        )
                    )
        elif scope and len(identities) != len(LEGACY_STORES):
            issues.add("mobile_missing_store")
        for leaf, refs in references.items():
            for (
                table,
                column,
                target_store,
                target_table,
                target_column,
            ), values in refs.items():
                dependencies.append(
                    (
                        leaf,
                        table,
                        column,
                        target_store,
                        target_table + "." + target_column,
                    )
                )
                target = identities.get(target_store, {})
                target_keys = target.get(
                    target_table + "." + target_column, target.get(target_table, ())
                )
                available = set(target_keys)
                if any((value,) not in available for value in values):
                    issues.add(
                        "mobile_missing_dependency:" + leaf + "." + table + "." + column
                    )
        if review_destination and identities and not issues:
            for (leaf, owner), path in zip(
                LEGACY_STORES, destination.paths, strict=True
            ):
                if not path.exists():
                    continue
                try:
                    target_keys = _inspect_store(
                        path, adapters[owner], external_references=False
                    )[3]
                except (OSError, ValueError, sqlite3.Error):
                    issues.add("mobile_destination_unavailable")
                    continue
                for table, keys in identities.get(leaf, {}).items():
                    available = set(target_keys.get(table, ()))
                    overlap = tuple(row for row in keys if row in available)
                    if overlap:
                        conflicts.append((leaf, table, overlap))
    except (OSError, ValueError) as error:
        issues.add(
            str(error)
            if isinstance(error, ValueError)
            else "mobile_inspection_unavailable"
        )
    return MobileConsolidationPreview(
        source_dir,
        _digest((str(source_dir), evidence)),
        destination,
        bool(scope) or "mobile_unknown_source_content" in issues,
        False,
        tuple(sorted(issues)),
        tuple(schemas),
        tuple(counts),
        tuple(mappings),
        tuple(dependencies),
        tuple(derived),
        tuple(scope),
        destination.paths,
        destination_witnesses=witnesses,
        destination_sidecars=sidecars,
        identity_conflicts=tuple(conflicts),
    )


def preview_mobile_consolidation(
    source_dir: Path, destination: CanonicalConsoleStores
) -> MobileConsolidationPreview:
    """Observe exact physical source snapshots without creating live stores."""
    from .storage_admission import _preview_reads

    if type(destination) is not CanonicalConsoleStores:
        raise TypeError("invalid_mobile_destination")
    with _preview_reads():
        return _preview(source_dir, destination)


def mobile_inventory(config):
    """Add legacy files to normal coherent capture with existing native owners."""
    from dataclasses import replace

    from tldw_chatbook.DB.recovery_operations import legacy_mobile_declarations

    from .models import StorageItem, discovery_context, storage_logical_id
    from .profile_paths import canonical_console_paths, legacy_mobile_dir

    context = discovery_context(config)
    stores = legacy_mobile_declarations(config)
    controls = _source_claim_inventory(config)
    if not legacy_mobile_dir(config).exists():
        return (*stores, *controls)
    preview = _preview(
        legacy_mobile_dir(config),
        CanonicalConsoleStores(*canonical_console_paths(config)),
        config,
        review_destination=False,
    )
    items = [*stores, *controls]
    for issue in preview.issues:
        items.append(
            StorageItem(
                "db.agent_runs",
                storage_logical_id(
                    context, "db.agent_runs", "legacy-mobile-issue-" + _digest(issue)
                ),
                None,
                "unsupported",
                (),
            )
        )
    if preview.has_data:
        items.extend(_referenced_files(config, stores))
    lock = legacy_mobile_dir(config) / ".runs.db.execution.lock"
    lock_item = _inventory_root(lock, owner="db.agent_runs", external=False)
    items.append(
        replace(
            lock_item,
            logical_id=storage_logical_id(
                context, "db.agent_runs", "legacy-mobile-execution-lock"
            ),
            status="intentionally_excluded"
            if lock_item.status in {"included", "unused"}
            else "unsupported",
        )
    )
    return tuple(items)


def _referenced_files(config, stores):
    from tldw_chatbook.Agents.recovery import _RunLogs

    return _RunLogs("agents.history").discover_for_stores(
        config, {item.owner: item for item in stores}
    )


def _preview_selected_mobile(
    source_config: Path, target_config: Path, *, review_destination=True
) -> MobileConsolidationPreview:
    """Review admitted local config selectors and all known shared-source candidates."""
    from dataclasses import replace

    from .capture_service import _selectors
    from .inventory import read_profile_config
    from .profile_paths import canonical_console_paths, legacy_mobile_dir

    source_config, target_config = map(lexical_path, (source_config, target_config))
    selectors = _selectors((source_config, target_config), include_known_profiles=True)
    observations = {path: _selector_observation(path) for path in selectors}
    provenance = []
    for path in (source_config, target_config):
        _resource_identity(path)
    source = read_profile_config(source_config)
    target = read_profile_config(target_config)
    source_dir = legacy_mobile_dir(source)
    destination = CanonicalConsoleStores(*canonical_console_paths(target))
    candidates = {destination.identity: destination}
    issues = set()
    for selected in selectors:
        fingerprint, chain = observations[selected]
        candidate = None
        if fingerprint is None:
            provenance.append(MobileSelectorProvenance(selected, None, chain))
            continue
        try:
            config = read_profile_config(selected)
            if _source_group_identity(
                legacy_mobile_dir(config)
            ) == _source_group_identity(source_dir):
                candidate = CanonicalConsoleStores(*canonical_console_paths(config))
                candidates[candidate.identity] = candidate
        except (OSError, ValueError):
            issues.add("mobile_candidate_unavailable")
        provenance.append(
            MobileSelectorProvenance(
                selected,
                fingerprint,
                chain,
                candidate,
                _destination_witnesses(candidate) if candidate else (),
                _destination_sidecars(candidate) if candidate else (),
            )
        )
    try:
        _refuse_witness_aliases(
            witness
            for candidate in candidates.values()
            for witness in _destination_witnesses(candidate)
        )
    except ValueError:
        issues.add("mobile_unsafe_alias")
    preview = _preview(
        source_dir, destination, source, review_destination=review_destination
    )
    if _selectors(
        (source_config, target_config), include_known_profiles=True
    ) != selectors or any(
        _selector_observation(path) != row for path, row in observations.items()
    ):
        raise ValueError("mobile_selector_changed")
    needs_review = preview.has_data and len(candidates) > 1
    if needs_review:
        issues.add("mobile_destination_review_required")
    return replace(
        preview,
        needs_destination_review=needs_review,
        issues=tuple(sorted(set(preview.issues) | issues)),
        candidate_destinations=tuple(candidates[key] for key in sorted(candidates)),
        source_selector=source_config,
        target_selector=target_config,
        selector_provenance=tuple(provenance),
        source_witnesses=_path_witnesses((source_dir,))[0][1],
    )


def preview_selected_mobile(
    source_config: Path, target_config: Path
) -> MobileConsolidationPreview:
    """Review checked selectors without ordinary runtime admission or mutation."""
    from .storage_admission import _preview_reads

    with _preview_reads():
        return _preview_selected_mobile(source_config, target_config)


def _selector_evidence(preview):
    """Encode only fixed locator and witness fields for the native typed journal."""

    def witness(value):
        return dict(zip(("path", "device", "inode", "mode"), value, strict=True))

    return [
        {
            "selector": str(row.selector),
            "fingerprint": row.fingerprint,
            "witnesses": [witness(item) for item in row.witnesses],
            "destination_identity": row.destination.identity
            if row.destination
            else None,
            "targets": [
                {
                    "owner_id": owner,
                    "witnesses": [witness(item) for item in chain],
                    "sidecars": [
                        witness(item) for item in dict(row.destination_sidecars)[path]
                    ],
                }
                for owner, (path, chain) in zip(
                    (owner for _, owner in LEGACY_STORES),
                    row.destination_witnesses,
                    strict=True,
                )
            ]
            if row.destination
            else [],
        }
        for row in preview.selector_provenance
    ]


def _source_claim_locations(source_dir):
    """Fixed source-addressed names, including safe absent-source ancestors."""
    stem = ".mobile-consolidation-" + _digest(str(lexical_path(source_dir)))
    return tuple(
        (parent / (stem + ".lock"), parent / (stem + ".json"))
        for parent in source_dir.parents
    )


def _source_claim_inventory(config):
    from dataclasses import replace

    from tldw_chatbook.Utils.platform_files import os

    from . import bootstrap
    from .journal import _MobileSourceClaim
    from .models import discovery_context, storage_logical_id
    from .native_files import pinned_directory
    from .profile_paths import legacy_mobile_dir

    source = legacy_mobile_dir(config)
    context = discovery_context(config)
    result = []
    for index, pair in enumerate(_source_claim_locations(source)):
        for path in pair:
            item = _inventory_root(path, owner="recovery.control", external=False)
            safe = item.status in {"unused", "included"}
            if item.status == "included":
                try:
                    with pinned_directory(path.parent) as parent:
                        if path.suffix == ".json":
                            record = _MobileSourceClaim.model_validate(
                                bootstrap._read(parent, path.name)
                            )
                            safe = record.source_dir == str(
                                source
                            ) and record.claim_path == str(path)
                        else:
                            from .admission import Admission

                            descriptor = Admission._open(parent, path.name, 0)
                            os.close(descriptor)
                except (OSError, ValueError):
                    safe = False
            result.append(
                replace(
                    item,
                    logical_id=storage_logical_id(
                        context,
                        "recovery.control",
                        f"legacy-mobile-control-{index}-{path.suffix[1:]}",
                    ),
                    status="intentionally_excluded" if safe else "unsupported",
                    metadata=None,
                )
            )
    return tuple(result)


def _credential_review_evidence(preview):
    """Validate separately reviewed manual-only issue metadata; never infer consent."""
    from .journal import _MobileCredentialReview

    rows = preview.credential_acknowledgements
    if type(rows) is not tuple:
        raise ValueError("mobile_credential_review_invalid")
    try:
        if any(
            type(row) is not tuple or len(row) != 2 or type(row[1]) is not tuple
            for row in rows
        ):
            raise ValueError()
        result = [
            _MobileCredentialReview(purpose=purpose, issues=list(issues)).model_dump()
            for purpose, issues in rows
        ]
        if len({row["purpose"] for row in result}) != len(result):
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("mobile_credential_review_invalid") from None
    return result
