"""Read-only source discovery and conservative coverage classification (ADR-126).

This is local source metadata, not archive validation or capture authorization.
Discovery never imports config bootstrap or opens a service-owned database.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import tomllib

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    DISCOVERY_CONTEXT_KEY,
    DiscoveryContext,
    DiscoverySelections,
    Inventory,
    StorageItem,
    storage_logical_id,
)
from .owner_registry import registered
from . import profile_paths


class _ProfileSelectors(BaseModel):
    """Strict TOML table boundary; owner-private data remains uninterpreted."""

    model_config = ConfigDict(strict=True, extra="allow")
    general: dict[str, object] = Field(default_factory=dict)
    paths: dict[str, object] = Field(default_factory=dict)
    Paths: dict[str, object] = Field(default_factory=dict)
    database: dict[str, object] = Field(default_factory=dict)
    notes: dict[str, object] = Field(default_factory=dict)
    console: dict[str, object] = Field(default_factory=dict)
    llm_management: dict[str, object] = Field(default_factory=dict)


STATUSES = frozenset(
    {
        "included",
        "included_directory",
        "intentionally_excluded",
        "unused",
        "intentionally_deleted",
        "unavailable",
        "unsupported",
        "missing_required",
    }
)
BLOCKING = frozenset({"unsupported", "unavailable", "missing_required"})

# These source cohorts contain additional configured/DB-referenced stores. Their
# exact producer symbols and unresolved locators are in the checked owner census.
UNRESOLVED_OWNERS = (
    "runtime.event_state",
    "runtime.sync_state",
    "notes.sync_state",
    "kanban.local",
    "chat.dictionary_history",
    "personas",
    "chat.rag_context",
    "chat.grammars",
    "feedback",
    "audio.history",
    "mcp.local",
    "mcp.targets",
    "mcp.context",
    "skills",
    "workspaces.change_tracking",
    "chat.dictionaries",
    "agents.history",
    "notes.file_notes",
    "notes.sync_bindings",
    "tts.references",
    "tts.voices",
    "rag.projections",
    "rag.definitions",
    "models.artifacts",
    "persona.assets",
    "mcp.permissions",
    "config.history",
    "subscriptions.assets",
    "eval.definitions",
    "chat.attachments",
    "chat.prompts",
    "chunking.templates",
    "tokenizers.custom",
    "tamagotchi.config",
)

# Managed credentials are a staged cross-owner policy, not a missing filesystem
# store. Capture applies the installed credential policies to these owners and
# refuses unsupported formats or unacknowledged encrypted recovery omissions.


def _identity(path: Path) -> tuple[int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino


def _sqlite_sidecars(declared, adapters):
    """Exclude exact SQLite transient files without hiding unknown lookalikes.

    Both paths are declared even while absent so SQLite's ordinary WAL lifecycle
    cannot expand reviewed scope. Unsafe aliases remain explicit blocking items.
    """
    sqlite_owners = {
        adapter.owner_id
        for adapter in adapters
        if (policy := adapter.schema_policy()) is not None and policy.schema_sql
    }
    sidecars = []
    declared_paths = {item.path for item in declared if item.path is not None}
    for item in declared:
        if (
            item.owner not in sqlite_owners
            or item.status != "included"
            or item.path is None
        ):
            continue
        # This installed adapter owns both its SQLite catalog and raw media.
        # Mirror its capture dispatch; payload names never confer SQLite scope.
        if item.owner == "recovered.media" and item.path.name != "catalog.sqlite3":
            continue
        main = item.path.lstat()
        for suffix in ("-wal", "-shm"):
            path = item.path.with_name(item.path.name + suffix)
            safe = stat.S_ISREG(main.st_mode) and path not in declared_paths
            try:
                info = path.lstat()
            except FileNotFoundError:
                pass
            else:
                safe = (
                    safe
                    and stat.S_ISREG(info.st_mode)
                    and info.st_uid == os.geteuid()
                    and info.st_nlink == 1
                )
            sidecars.append(
                StorageItem(
                    "sqlite.transient",
                    item.logical_id + ":sqlite" + suffix,
                    path,
                    "intentionally_excluded" if safe else "unsupported",
                    (item.logical_id,),
                )
            )
    return tuple(sidecars)


def classify_entries(items: tuple[StorageItem, ...]) -> Inventory:
    """Classify trusted installed-owner outputs; never trust archive evidence bits.

    deletion_validated is only local owner-validation output. Raw serialized
    records cannot become StorageItem authority; readers must obtain fresh owner
    validation. Task 3 supplies no tombstone validator and emits no such evidence.
    """
    issues: set[str] = set()
    complete = all(item.status not in BLOCKING for item in items)
    if any(item.owner == "unknown" for item in items):
        complete = False
        issues.add("unsupported_owner")
    by_id: dict[str, StorageItem] = {}
    physical: dict[tuple[int, int], list[StorageItem]] = {}
    shared: dict[str, set[tuple[int, int]]] = {}
    roots: set[Path] = set()
    resolved_paths: dict[str, str] = {}
    for item in items:
        if item.status not in STATUSES:
            issues.add("invalid_status")
        if item.status in BLOCKING:
            issues.add(item.status)
        if item.logical_id in by_id:
            issues.add("duplicate_logical_id")
        by_id[item.logical_id] = item
        if item.status == "intentionally_deleted" and not item.deletion_validated:
            issues.add("unvalidated_deletion")
        if item.path is None:
            if item.status in {"included", "included_directory"} or item.shared_group:
                issues.add("missing_identity")
            continue
        if item.status in {"unused", "intentionally_excluded", "intentionally_deleted"}:
            continue
        try:
            identity = _identity(item.path)
            physical.setdefault(identity, []).append(item)
            if item.shared_group:
                shared.setdefault(item.shared_group, set()).add(identity)
            mode = item.path.stat().st_mode
            if (item.status == "included" and not stat.S_ISREG(mode)) or (
                item.status == "included_directory" and not stat.S_ISDIR(mode)
            ):
                issues.add("unsupported_path_kind")
            resolved = item.path.resolve(strict=True)
            roots.add(resolved)
            resolved_paths[item.logical_id] = str(resolved)
        except (OSError, RuntimeError):
            if item.status in {"included", "included_directory"} or item.shared_group:
                issues.add("missing_identity")
    for values in physical.values():
        if len(values) > 1 and (
            not values[0].shared_group
            or any(v.shared_group != values[0].shared_group for v in values)
        ):
            issues.add("undeclared_alias")
    if any(len(values) > 1 for values in shared.values()):
        issues.add("shared_identity_mismatch")
    # A tree's explicit same-owner/profile parent edges describe topology, not
    # competing root ownership. Any missing edge or cross-owner overlap refuses.
    by_path = {}
    for item in items:
        if item.logical_id in resolved_paths:
            by_path.setdefault(Path(resolved_paths[item.logical_id]), []).append(item)
    for child_path, children in by_path.items():
        for parent_path in child_path.parents:
            if parent_path not in by_path:
                continue
            for child in children:
                current = child
                prefix = current.logical_id.split(":")[:3]
                valid = (
                    len(prefix) == 3
                    and prefix[0] == "profile"
                    and prefix[2] == child.owner
                )
                visited = set()
                while valid and Path(resolved_paths[current.logical_id]) != parent_path:
                    if current.logical_id in visited:
                        valid = False
                        break
                    visited.add(current.logical_id)
                    expected_parent = Path(resolved_paths[current.logical_id]).parent
                    matches = [
                        by_id[key]
                        for key in current.dependencies
                        if key in by_id
                        and by_id[key].status == "included_directory"
                        and by_id[key].owner == child.owner
                        and by_id[key].logical_id.split(":")[:3] == prefix
                        and resolved_paths.get(key) == str(expected_parent)
                    ]
                    if len(matches) != 1:
                        valid = False
                        break
                    current = matches[0]
                if not valid or any(
                    parent.owner != child.owner for parent in by_path[parent_path]
                ):
                    issues.add("overlapping_owner_roots")
    for item in items:
        if item.status in {"unused", "intentionally_excluded"}:
            continue
        for dependency in item.dependencies:
            target = by_id.get(dependency)
            if (
                target is None
                or target.status in BLOCKING | {"unused", "intentionally_excluded"}
                or (
                    target.status == "intentionally_deleted"
                    and not target.deletion_validated
                )
            ):
                issues.add("dependency_unavailable")
    # Approved tree boundaries cover ordinary included descendants. Keep every
    # blocking/excluded/deleted record and explicit physical alias in scope; a
    # new owner, root, coverage choice or shared identity still requires review.
    def scope_id(item):
        metadata = item.metadata
        root = by_id.get(metadata.root_id) if metadata else None
        if (
            item.status in {"included", "included_directory"}
            and not item.shared_group
            and root is not None
            and root.owner == item.owner
            and root.status == "included_directory"
            and root.metadata is not None
            and root.metadata.root_id == root.logical_id
            and root.metadata.relative_path == ""
        ):
            return root.logical_id
        return item.logical_id

    # Stable scope, not payload or inode fingerprint: ordinary record growth and
    # atomic replacement inside an approved logical owner do not alter its scope.
    payload = sorted(
        (
            (
                item.owner,
                item.logical_id,
                str(item.path) if item.path else None,
                resolved_paths.get(item.logical_id),
                item.status,
                tuple(sorted({scope_id(by_id[key]) if key in by_id else key for key in item.dependencies})),
                item.shared_group,
                item.deletion_validated,
                (
                    item.metadata.version,
                    item.metadata.root_id,
                    item.metadata.relative_path,
                    item.metadata.parent_id,
                    item.metadata.kind,
                    item.metadata.policy,
                )
                if item.metadata
                else None,
            )
            for item in items
            if scope_id(item) == item.logical_id
        ),
        key=lambda row: json.dumps(row),
    )
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    return Inventory(
        tuple(items), complete and not issues, digest, tuple(sorted(issues))
    )


def _source_item(
    owner: str,
    logical_id: str,
    path: Path,
    dependencies: tuple[str, ...] = (),
    *,
    required: bool = False,
) -> StorageItem:
    try:
        value = path.lstat()
        status = (
            "unsupported"
            if stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)
            else "unavailable"
        )
    except FileNotFoundError:
        # Absence alone never proves an owner unused; optional owner adapters have
        # to establish that separately, including references and journals.
        status = "missing_required" if required else "unsupported"
    except OSError:
        status = "unavailable"
    return StorageItem(owner, logical_id, path, status, dependencies)


def _unknown_children(
    root: Path, known: set[Path], prefix: str
) -> tuple[StorageItem, ...]:
    try:
        if any(path.is_symlink() for path in (root, *root.parents)):
            return (
                StorageItem(
                    "unknown", prefix + ":linked_root", root, "unavailable", ()
                ),
            )
        root.lstat()
        return tuple(
            _source_item("unknown", prefix + ":unknown:" + child.name, child)
            for child in sorted(root.iterdir())
            if child not in known
        )
    except FileNotFoundError:
        return ()
    except OSError:
        return (
            StorageItem(
                "unknown", prefix + ":unreadable_root", root, "unavailable", ()
            ),
        )


def _planned_output_exclusion(context, declared, data_root):
    """An absent output is a plan, never permission to omit existing bytes."""
    selected = context.selections.planned_output_root
    if selected is None:
        return ()
    from .bootstrap import pinned_directory

    output = profile_paths.lexical_path(selected)
    with pinned_directory(output.parent) as parent:
        try:
            os.stat(output.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("output_root_not_new")
        for path in (
            data_root,
            *(
                item.path
                for item in declared
                if item.path is not None
                and item.status != "intentionally_excluded"
                and not item.owner.startswith("external.")
            ),
        ):
            if output == path or output in path.parents or path in output.parents:
                raise ValueError("output_overlaps_baseline")
    return (
        StorageItem(
            "recovery.output",
            storage_logical_id(context, "recovery.output"),
            output,
            "intentionally_excluded",
            (),
        ),
    )


def _fixed_control_exclusion() -> tuple[StorageItem, ...]:
    """Exclude only the installed fixed authority, never a directory lookalike.

    Discovery does not initialize admission, repair evidence, or follow control
    locators from configuration/archive data. Damaged local authority remains an
    explicit blocking item. Actual-session output protection is added separately
    by capture, with its own logical ID.
    """
    from .bootstrap import _control_records, _registry, default_bootstrap_root
    from .control_records import UNBOUND_NAMESPACE

    root = default_bootstrap_root()
    try:
        before = root.lstat()
    except FileNotFoundError:
        return ()
    except OSError:
        status = "unavailable"
    else:
        try:
            _control_records(root)  # Strict private/no-follow fixed record reader.
            registry = _registry(root)
            marker = root / "unbound-owner"
            after = root.lstat()
            if (
                registry is None
                or UNBOUND_NAMESPACE not in registry
                or registry[UNBOUND_NAMESPACE]["roots"] != [str(marker)]
                # Directory reads may update access time; identity/metadata
                # changes still invalidate this observation.
                or (after.st_dev, after.st_ino, after.st_ctime_ns)
                != (before.st_dev, before.st_ino, before.st_ctime_ns)
            ):
                raise ValueError("recovery_control_unverified")
            status = "intentionally_excluded"
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            status = "unavailable"
    return (
        StorageItem(
            "recovery.control", "recovery.control:fixed-bootstrap", root, status, ()
        ),
    )


def _service_control_exclusion() -> tuple[StorageItem, ...]:
    """Exclude recognized local service work, keeping unknown/damaged roots visible."""
    from .service_storage import default_control_root, verify_default_storage

    root = default_control_root().parent
    try:
        root.lstat()
    except FileNotFoundError:
        return ()
    try:
        verify_default_storage()
        status = "intentionally_excluded"
    except (OSError, ValueError, TypeError, RuntimeError):
        status = "unavailable"
    return (
        StorageItem("recovery.control", "recovery.control:service", root, status, ()),
    )


def discover(
    config_paths: tuple[Path, ...], *, selections: DiscoverySelections | None = None
) -> Inventory:
    """Read only explicitly selected TOML sources and canonical app-owned roots.

    Empty selection means the canonical effective config, never fallback creation.
    Config parse failures produce no guessed profile/database targets. Installed
    adapters may extend declarations without importing optional engines here.
    """
    if selections is None:
        selections = DiscoverySelections()
    if type(selections) is not DiscoverySelections:
        raise ValueError("invalid_discovery_selections")
    items: list[StorageItem] = []
    from .recovered_media import recovery_adapters

    adapters = registered()
    if not any(adapter.owner_id == "recovered.media" for adapter in adapters):
        adapters += recovery_adapters()
    selected_roots: set[Path] = set()
    for selected in config_paths or (profile_paths.effective_config_path(),):
        selected = profile_paths.lexical_path(selected)
        profile_id = hashlib.sha256(str(selected).encode()).hexdigest()[:24]
        prefix = "profile:" + profile_id
        context = DiscoveryContext(selected, profile_id, selections)
        config_id = storage_logical_id(context, "config")
        profile_start = len(items)
        item = _source_item("config", config_id, selected, required=True)
        items.append(item)
        try:
            # Refuse non-regular sources before open, including FIFOs. O_NONBLOCK
            # avoids hanging if a selected path is replaced during discovery.
            flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
            with os.fdopen(os.open(selected, flags), "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("config_unavailable")
                raw_config = tomllib.load(stream)
                if DISCOVERY_CONTEXT_KEY in raw_config:
                    raise ValueError("reserved_context_key")
                config = _ProfileSelectors.model_validate(raw_config).model_dump()
                config[DISCOVERY_CONTEXT_KEY] = context
            root = profile_paths.user_data_dir(config)
            selected_roots.add(root)
            declared = [item]
            for owner, key, _, _ in profile_paths.DATABASE_PATHS:
                path = profile_paths.database_path(config, key)
                declared.append(
                    _source_item(
                        owner,
                        storage_logical_id(context, owner),
                        path,
                        (config_id,),
                        required=True,
                    )
                )
            # Resolvers/reference discovery not yet qualified for this source.
            declared.extend(
                StorageItem(
                    owner,
                    storage_logical_id(context, owner, "unresolved"),
                    None,
                    "unsupported",
                    (config_id,),
                )
                for owner in UNRESOLVED_OWNERS
            )
            for section, key, owner in (
                ("notes", "sync_directory", "external.notes"),
                ("console", "workspace_root", "external.workspace"),
                ("llm_management", "model_download_dir", "external.models"),
            ):
                raw = profile_paths.setting(config, section, key)
                if raw:
                    if not isinstance(raw, str):
                        raise ValueError("invalid_config_path")
                    declared.append(
                        StorageItem(
                            owner,
                            storage_logical_id(context, owner),
                            profile_paths.lexical_path(raw),
                            "intentionally_excluded",
                            (),
                        )
                    )
            declared.append(
                StorageItem(
                    "server.data",
                    storage_logical_id(context, "server.data"),
                    None,
                    "intentionally_excluded",
                    (),
                )
            )
            for adapter in adapters:
                extras = adapter.discover(dict(config))
                for extra in extras:
                    if extra.owner != adapter.owner_id or extra.deletion_validated:
                        # Until a qualified tombstone validator is installed there
                        # is no validated-deletion producer in this subsystem.
                        raise ValueError("invalid_owner_evidence")
                declared = [
                    entry for entry in declared if entry.owner != adapter.owner_id
                ]
                declared.extend(extras)
            if selections.external_roots:
                from .recovery_files import _RawDeclaration

                for external_root in selections.external_roots:
                    declared.extend(
                        _RawDeclaration("external.files")._tree(
                            config,
                            profile_paths.lexical_path(external_root),
                            external=True,
                        )
                    )
            declared.extend(_sqlite_sidecars(declared, adapters))
            declared.extend(_planned_output_exclusion(context, declared, root))
            items[profile_start:] = declared
            known = {entry.path for entry in declared if entry.path is not None}
            from tldw_chatbook.MCP.recovery_activation import inventory_container

            known.update(inventory_container(context, root))
            items.extend(_unknown_children(root, known, prefix))
        except (tomllib.TOMLDecodeError, UnicodeError):
            items.append(
                StorageItem(
                    "config",
                    storage_logical_id(context, "config", "parse_failure"),
                    None,
                    "unavailable",
                    (config_id,),
                )
            )
        except (OSError, ValueError, TypeError, RuntimeError):
            items.append(
                StorageItem(
                    "config",
                    storage_logical_id(context, "config", "discovery_failure"),
                    None,
                    "unavailable",
                    (config_id,),
                )
            )
    # Known default namespaces only, never custom config parents or the home
    # tree. Unselected historical directories require a config locator/review.
    items.extend(
        _unknown_children(
            profile_paths.default_base_data_dir(), selected_roots, "profiles"
        )
    )
    items.extend(_fixed_control_exclusion())
    items.extend(_service_control_exclusion())
    items.extend(
        _unknown_children(
            profile_paths.default_config_path().parent,
            {entry.path for entry in items if entry.path is not None},
            "shared_config",
        )
    )
    # Same logical owner selected in multiple profiles can share verified bytes.
    # Cross-owner aliases require an installed explicit shared-group declaration.
    aliases: dict[tuple[str, tuple[int, int]], list[int]] = {}
    for index, entry in enumerate(items):
        if (
            entry.path is not None
            and entry.owner != "sqlite.transient"
            and entry.shared_group is None
            and entry.status
            not in {
                "unavailable",
                "missing_required",
            }
        ):
            try:
                aliases.setdefault((entry.owner, _identity(entry.path)), []).append(
                    index
                )
            except (OSError, RuntimeError):
                pass
    for (owner, _), indexes in aliases.items():
        if len(indexes) > 1 and owner != "unknown":
            group = (
                "shared:"
                + hashlib.sha256(
                    "\0".join(sorted(items[i].logical_id for i in indexes)).encode()
                ).hexdigest()[:24]
            )
            for index in indexes:
                items[index] = replace(items[index], shared_group=group)
    items, cohort_issues = _merge_chachanotes_cohort(tuple(items))
    items, tts_issues = _merge_chachanotes_cohort(items, cohort="tts")
    cohort_issues = (*cohort_issues, *tts_issues)
    result = classify_entries(tuple(items))
    issues = set(result.issues) | set(cohort_issues)
    if any(item.logical_id.endswith(":parse_failure") for item in items):
        issues.add("config_parse_failure")
    if any(item.logical_id.endswith(":discovery_failure") for item in items):
        issues.add("config_discovery_failure")
    return replace(
        result,
        complete=result.complete and not issues,
        issues=tuple(sorted(issues)),
        scope_digest=hashlib.sha256(
            json.dumps(
                (
                    result.scope_digest,
                    tuple(
                        str(profile_paths.lexical_path(p))
                        for p in selections.external_roots
                    ),
                    selections.model_ids,
                    selections.temporary_media,
                    selections.diagnostics,
                    str(profile_paths.lexical_path(selections.planned_output_root))
                    if selections.planned_output_root
                    else None,
                ),
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )


def _merge_chachanotes_cohort(
    items: tuple[StorageItem, ...],
    *,
    cohort: str = "chachanotes",
) -> tuple[tuple[StorageItem, ...], tuple[str, ...]]:
    """Merge only installed shared declarations after checking original groups.

    Stable final logical IDs name each group. Physical identity is fresh proof,
    never part of the durable scope label. A mismatching original declaration is
    refused before any rewrite can hide it by splitting into different groups.
    """
    cohorts = {
        "chachanotes": {
            "db.chachanotes.primary",
            "study.local",
            "quiz.local",
            "notes.sync_bindings",
            "chat.attachments",
        },
        "tts": {"tts.profile_store", "tts.references"},
    }
    owners = cohorts[cohort]
    original = {}
    physical = {}
    try:
        for item in items:
            if item.shared_group and item.path is not None:
                original.setdefault(item.shared_group, set()).add(_identity(item.path))
        if any(len(identities) != 1 for identities in original.values()):
            return items, ("shared_identity_mismatch",)
        for index, item in enumerate(items):
            if item.owner not in owners or item.status != "included":
                continue
            if item.owner == "db.chachanotes.primary" and item.shared_group is None:
                # Unqualified default census rows are not installed declarations.
                continue
            parts = item.logical_id.split(":")
            if (
                len(parts) != 3
                or parts[0] != "profile"
                or parts[2] != item.owner
                or not parts[1]
            ):
                return items, ("invalid_shared_declaration",)
            expected = "shared:" + cohort + ":profile:" + parts[1]
            if item.shared_group != expected or item.path is None:
                return items, ("invalid_shared_declaration",)
            physical.setdefault(_identity(item.path), []).append(index)
    except (OSError, RuntimeError):
        return items, ("shared_identity_unavailable",)
    result = list(items)
    for indexes in physical.values():
        label = (
            "shared:"
            + cohort
            + ":"
            + hashlib.sha256(
                "\0".join(sorted(items[i].logical_id for i in indexes)).encode()
            ).hexdigest()[:24]
        )
        for index in indexes:
            result[index] = replace(items[index], shared_group=label)
    return tuple(result), ()
