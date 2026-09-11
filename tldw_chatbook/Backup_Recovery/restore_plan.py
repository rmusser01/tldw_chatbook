"""Explicit local restore planning (ADR-126); no publication authority."""

import hashlib
import json
import os
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Literal

from .archive_models import Metadata, SealedArchive
from .archive_reader import _hash, _manifest, verify_sealed
from .limits import ArchiveLimits
from .models import Inventory
from .native_files import pinned_directory
from .owner_registry import install_adapters


@dataclass(frozen=True)
class RestorePlan:
    """Reviewed artifact paths and independently observed local target state."""

    archive_digest: str
    mode: str
    restore: tuple[tuple[str, Path], ...]
    retire: tuple[tuple[str, Path], ...]
    preserve: tuple[tuple[str, Path], ...]
    target_fingerprint: str
    destinations: tuple[tuple[str, Path], ...] = ()
    target: Inventory | None = None
    selectors: tuple[tuple[str, Path], ...] = ()
    profile_names: tuple[tuple[str, str], ...] = ()
    metadata: tuple[tuple[str, Metadata | None, Metadata], ...] = ()
    issues: tuple[str, ...] = ()
    containers: tuple[tuple[str, Path], ...] = ()
    safety_scope: tuple[str, ...] = ()
    acknowledged_credential_issues: tuple[str, ...] = ()


def _document(archive):
    verify_sealed(archive)
    return _manifest(archive.manifest_bytes, ArchiveLimits(), encrypted=True)


def _ancestor(path):
    """Check existing path components without accepting a symlink spelling."""
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("invalid_destination")
    selected = path
    while not selected.exists() and not selected.is_symlink():
        selected = selected.parent
    if selected.is_symlink():
        raise ValueError("destination_alias")
    parent = selected if selected.is_dir() else selected.parent
    try:
        with pinned_directory(parent):
            pass
    except OSError:
        raise ValueError("destination_alias") from None
    return parent


def _observed(path):
    ancestor = _ancestor(path)
    ancestors = []
    current = ancestor
    while True:
        info = current.lstat()
        ancestors.append((str(current), info.st_dev, info.st_ino))
        if current == current.parent:
            break
        current = current.parent
    try:
        info = path.lstat()
    except FileNotFoundError:
        return (str(path), ancestors, None)
    if stat.S_ISLNK(info.st_mode) or info.st_nlink != 1 and stat.S_ISREG(info.st_mode):
        raise ValueError("destination_alias")
    if stat.S_ISREG(info.st_mode):
        content = _hash(path, Event())
    elif stat.S_ISDIR(info.st_mode):
        with pinned_directory(path) as fd:
            content = sorted(os.listdir(fd))
    else:
        raise ValueError("unsupported_target_kind")
    return (
        str(path),
        ancestors,
        (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
            info.st_mode,
            content,
        ),
    )


def _fingerprint(paths, target):
    document = {
        "paths": [_observed(path) for path in sorted(set(paths))],
        "inventory": [
            (
                item.owner,
                item.logical_id,
                str(item.path),
                item.status,
                item.dependencies,
                item.shared_group,
            )
            for item in target.items
        ]
        if target is not None
        else [],
    }
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _paths(plan):
    return tuple(
        path
        for _, path in (
            *plan.restore,
            *plan.retire,
            *plan.preserve,
            *plan.selectors,
            *plan.destinations,
            *plan.containers,
            *(
                (item.logical_id, item.path)
                for item in (plan.target.items if plan.target else ())
                if item.logical_id in plan.safety_scope
            ),
        )
    )


def recheck_targets(plan: RestorePlan) -> None:
    """Invalidate reviewed paths; never silently recalculate destination mappings."""
    try:
        actual = _fingerprint(_paths(plan), plan.target)
    except (OSError, ValueError):
        raise ValueError("target_changed") from None
    if actual != plan.target_fingerprint:
        raise ValueError("target_changed")


def plan_restore(
    archive: SealedArchive,
    *,
    mode: Literal["isolated", "replace"],
    destinations: Mapping[str, Path],
    target: Inventory | None,
    profile_names: Mapping[str, str] | None = None,
    safety_scope: tuple[str, ...] = (),
    acknowledged_credential_issues: tuple[str, ...] = (),
) -> RestorePlan:
    """Plan from explicit local owner selections; never load current configuration.

    Destination keys are archive root IDs. Values are locally selected roots;
    original relocation strings are not destination defaults or authority.
    """
    if mode == "replace" and target is None:
        raise ValueError("target_unverified")
    if mode not in {"isolated", "replace"}:
        raise ValueError("invalid_restore_mode")
    if (
        type(safety_scope) is not tuple
        or type(acknowledged_credential_issues) is not tuple
        or any(
            type(key) is not str
            for key in (*safety_scope, *acknowledged_credential_issues)
        )
        or len(set(safety_scope)) != len(safety_scope)
        or len(set(acknowledged_credential_issues))
        != len(acknowledged_credential_issues)
        or any(
            not issue.startswith("credential_") or len(issue) > 256
            for issue in acknowledged_credential_issues
        )
        or mode != "replace"
        and (safety_scope or acknowledged_credential_issues)
    ):
        raise ValueError("invalid_replacement_review")
    if safety_scope:
        by_id = {item.logical_id: item for item in target.items}
        if set(safety_scope) - by_id.keys() or any(
            by_id[key].path is None
            or by_id[key].status not in {"included", "included_directory"}
            for key in safety_scope
        ):
            raise ValueError("safety_scope_unverified")
    doc = _document(archive)
    if mode == "replace" and doc.consistency != "coherent":
        raise ValueError("partial_replacement")
    names = dict(profile_names or {})
    if set(names) - set(doc.profile_ids) or any(
        type(name) is not str or not name.strip() for name in names.values()
    ):
        raise ValueError("invalid_profile_identity")
    owners = {adapter.owner_id: adapter for adapter in install_adapters()}
    from tldw_chatbook.Agents.recovery import _RunLogs
    from tldw_chatbook.Evals.recovery import _DefinitionsAdapter
    from tldw_chatbook.Persona_Visual.recovery import _Assets
    from tldw_chatbook.TTS.recovery import _Voices

    from .config_adapter import CONFIG_LOCATION_KEYS

    deferred_classes = {
        "agents.history": _RunLogs,
        "eval.definitions": _DefinitionsAdapter,
        "tts.voices": _Voices,
        "persona.visual_identity_builtin": _Assets,
    }
    deferred = {
        name for name, cls in deferred_classes.items() if type(owners.get(name)) is cls
    }
    directories = {row.logical_id: row for row in doc.directories}
    roots = {key: row for key, row in directories.items() if row.parent_id is None}
    accepted = {
        f"profile:{profile}:{section}.{key}"
        for profile in doc.profile_ids
        for section, key in CONFIG_LOCATION_KEYS
    }
    selectors = {key: value for key, value in destinations.items() if key in accepted}
    destinations = {
        key: value for key, value in destinations.items() if key not in selectors
    }
    if not destinations or set(destinations) - roots.keys():
        raise ValueError("explicit_destination_required")
    for path in selectors.values():
        _ancestor(path)
        if mode == "isolated" and path.exists():
            raise ValueError("destination_exists")
    for path in destinations.values():
        _ancestor(path)
        if mode == "isolated" and (path.exists() or path.is_symlink()):
            raise ValueError("destination_exists")
        if (
            path == archive.path
            or path in archive.path.parents
            or archive.path.parent in path.parents
        ):
            raise ValueError("archive_destination_alias")
    normalized = [
        unicodedata.normalize("NFC", str(path)).casefold()
        for path in destinations.values()
    ]
    if len(set(normalized)) != len(normalized):
        for key, path in destinations.items():
            for other, other_path in destinations.items():
                if key == other:
                    continue
                if (
                    unicodedata.normalize("NFC", str(path)).casefold()
                    == unicodedata.normalize("NFC", str(other_path)).casefold()
                ) and (
                    path != other_path
                    or not (roots[key].synthetic and roots[other].synthetic)
                ):
                    raise ValueError("destination_collision")
    for left in destinations.values():
        if any(
            left in right.parents for right in destinations.values() if right != left
        ):
            raise ValueError("destination_overlap")
    records = (*doc.directories, *doc.files)
    selected = {row.logical_id for row in records if row.root_id in destinations}
    selected_owners = {row.owner_id for row in doc.files if row.logical_id in selected}
    if selected_owners - owners.keys():
        raise ValueError("unsupported_owner")
    if doc.required_capabilities:
        raise ValueError("unsupported_restore_capability")
    for owner in doc.owners:
        if owner.owner_id not in selected_owners:
            continue
        policy = owners[owner.owner_id].schema_policy()
        if owner.capabilities:
            raise ValueError("unsupported_restore_capability")
        if policy is None or owner.schema_version not in policy.versions:
            raise ValueError("unsupported_schema_version")
    for group in doc.dependency_groups:
        if selected.intersection(group.members) and (
            not group.complete or not set(group.members) <= selected
        ):
            raise ValueError("dependency_group_incomplete")
    restore = tuple(
        sorted(
            (row.logical_id, destinations[row.root_id] / row.relative_path)
            for row in records
            if row.logical_id in selected
        )
    )
    producer = {row.logical_id: row for row in doc.producer_inventory}
    payloads = {row.logical_id: row for row in doc.files}
    projection_roots = {
        row.root_id
        for row in doc.files
        if row.logical_id in selected and row.owner_id == "rag.projections"
    }
    for key in projection_roots:
        owner = producer.get(key)
        if (
            owner is None
            or owner.owner_id != "rag.projections"
            or roots[key].synthetic
            or any(
                row.owner_id != "rag.projections"
                for row in doc.files
                if row.root_id == key
            )
        ):
            raise ValueError("projection_root_incomplete")
    deferred_selected = {
        row.owner_id
        for row in doc.files
        if row.logical_id in selected and row.owner_id in deferred
    }
    deferred_selected.update(
        row.owner_id
        for row in producer.values()
        if row.logical_id in selected and row.owner_id in deferred
    )
    for record in records:
        item = producer.get(record.logical_id)
        owner = item.owner_id if item else getattr(record, "owner_id", None)
        if (
            record.logical_id in selected
            and owner in deferred_selected
            and destinations[record.root_id].exists()
        ):
            raise ValueError("owner_setup_destination_exists")
    for root_id, destination in destinations.items():
        item = producer.get(root_id)
        if item is None and not any(row.root_id == root_id for row in doc.files):
            raise ValueError("producer_inventory_required")
        external = (
            item is not None
            and item.owner_id == "external.files"
            or any(
                row.root_id == root_id and row.owner_id == "external.files"
                for row in doc.files
            )
        )
        if external and destination.exists():
            raise ValueError("external_destination_exists")
    seen_paths = {}
    trimmed = []
    for key, path in restore:
        directory = directories.get(key)
        if directory is not None and directory.synthetic and path.exists():
            continue  # Container metadata is not ownership of an existing parent.
        norm = unicodedata.normalize("NFC", str(path)).casefold()
        if norm in seen_paths:
            previous = seen_paths[norm]
            if (
                key in payloads
                and previous in payloads
                and key in producer
                and previous in producer
            ):
                group = producer[key].shared_group
                if (
                    group
                    and group == producer[previous].shared_group
                    and path == dict(restore)[previous]
                ):
                    if payloads[key].metadata != payloads[previous].metadata:
                        raise ValueError("shared_metadata_mismatch")
                    trimmed.append((key, path))
                    continue
            if (
                directory is None
                or not directory.synthetic
                or previous not in roots
                or not roots[previous].synthetic
            ):
                raise ValueError("destination_collision")
            continue
        seen_paths[norm] = key
        trimmed.append((key, path))
    restore = tuple(trimmed)
    producer = {row.logical_id: row for row in getattr(doc, "producer_inventory", ())}
    for key in selected:
        item = producer.get(key)
        if item is not None:
            if item.owner_id not in owners:
                raise ValueError("unsupported_owner")
            if item.shared_group and any(
                row.logical_id not in selected
                for row in producer.values()
                if row.shared_group == item.shared_group
            ):
                raise ValueError("shared_scope_expansion_required")
    mapped = dict(restore)
    shared_destinations = {}
    for key in selected:
        item = producer.get(key)
        if item and item.shared_group and key in payloads:
            shared_destinations.setdefault(item.shared_group, set()).add(mapped[key])
    if any(len(paths) != 1 for paths in shared_destinations.values()):
        raise ValueError("shared_target_split")
    pending = list(selected)
    visited = set()
    while pending:
        key = pending.pop()
        if key in visited or key not in producer:
            continue
        visited.add(key)
        for dependency in producer[key].dependencies:
            if dependency not in selected:
                raise ValueError("dependency_group_incomplete")
            pending.append(dependency)
    # Config selectors are installed typed fields. No original value is a key.
    from .profile_paths import DATABASE_PATHS

    for payload in doc.files:
        if payload.logical_id not in selected or payload.owner_id != "config":
            continue
        parts = payload.logical_id.split(":")
        if len(parts) != 3 or parts[0] != "profile" or parts[1] not in doc.profile_ids:
            raise ValueError("config_profile_unverified")
        profile = parts[1]
        if profile not in names:
            raise ValueError("profile_identity_required")
        if mode == "isolated" and f"profile:{profile}:paths.data_dir" not in selectors:
            raise ValueError("config_data_destination_required")
        for owner_id, setting, _, _ in DATABASE_PATHS:
            matches = [
                path for key, path in restore if key == f"profile:{profile}:{owner_id}"
            ]
            if len(matches) == 1:
                key = f"profile:{profile}:database.{setting}"
                if key in selectors and selectors[key] != matches[0]:
                    raise ValueError("ambiguous_relocation_mapping")
                selectors[key] = matches[0]
    if mode == "replace" and not producer:
        raise ValueError("producer_inventory_required")
    retire, preserve = [], []
    if target is not None:
        if type(target) is not Inventory or len(
            {item.logical_id for item in target.items}
        ) != len(target.items):
            raise ValueError("target_unverified")
        replacements = dict(restore)
        selected_owners = {
            producer[key].owner_id for key in selected if key in producer
        }
        by_target_id = {item.logical_id: item for item in target.items}
        for item in target.items:
            if item.path is None:
                if item.status not in {
                    "unused",
                    "intentionally_excluded",
                    "intentionally_deleted",
                }:
                    raise ValueError("target_unverified")
                continue
            _ancestor(item.path)
            if (
                item.status == "included"
                and not item.path.is_file()
                or item.status == "included_directory"
                and not item.path.is_dir()
            ):
                raise ValueError("target_unverified")
            if mode == "replace" and item.owner == "sqlite.transient":
                if len(item.dependencies) != 1:
                    raise ValueError("target_owner_unclassified")
                main = by_target_id.get(item.dependencies[0])
                adapter = owners.get(main.owner) if main else None
                policy = adapter.schema_policy() if adapter else None
                if (
                    main is None
                    or main.path is None
                    or policy is None
                    or not policy.schema_sql
                    or item.path
                    not in {
                        main.path.with_name(main.path.name + suffix)
                        for suffix in ("-wal", "-shm")
                    }
                    or item.status != "intentionally_excluded"
                ):
                    raise ValueError("target_owner_unclassified")
                if main.path in replacements.values() and item.path.exists():
                    retire.append((item.logical_id, item.path))
                else:
                    preserve.append((item.logical_id, item.path))
                continue
            affected = item.path in replacements.values() or any(
                not roots[key].synthetic
                and (item.path == root or root in item.path.parents)
                for key, root in destinations.items()
            )
            if mode == "isolated":
                if affected or any(
                    item.path in root.parents for root in destinations.values()
                ):
                    raise ValueError("source_destination_alias")
                preserve.append((item.logical_id, item.path))
                continue
            if not affected:
                preserve.append((item.logical_id, item.path))
                continue
            if item.owner not in owners or item.status in {
                "unsupported",
                "unavailable",
                "missing_required",
            }:
                raise ValueError("target_owner_unclassified")
            archived_item = producer.get(item.logical_id)
            if item.owner != "rag.projections" and (
                item.status == "intentionally_excluded"
                or archived_item is not None
                and archived_item.status == "intentionally_excluded"
            ):
                if item.path in replacements.values():
                    raise ValueError("preserved_target_collision")
                preserve.append((item.logical_id, item.path))
                continue
            if item.shared_group:
                aliases = [
                    entry
                    for entry in target.items
                    if entry.shared_group == item.shared_group
                ]
                if any(
                    entry.path is not None
                    and not any(
                        entry.path == root or root in entry.path.parents
                        for root in destinations.values()
                    )
                    for entry in aliases
                ):
                    raise ValueError("shared_scope_expansion_required")
            if item.path in replacements.values():
                matching = [key for key, path in restore if path == item.path]
                if not any(
                    key in producer and producer[key].owner_id == item.owner
                    for key in matching
                ):
                    raise ValueError("target_owner_mismatch")
            elif item.status in {"included", "included_directory"}:
                if item.owner not in selected_owners:
                    raise ValueError("target_owner_unclassified")
                # An explicit producer tree describes the selected owner's full
                # generation; a standalone file never authorizes sibling deletion.
                if not any(
                    key in producer
                    and producer[key].owner_id == item.owner
                    and not getattr(roots[key], "synthetic", False)
                    and roots[key].logical_id in selected
                    and (
                        item.path == destinations[key]
                        or destinations[key] in item.path.parents
                    )
                    for key in destinations
                ):
                    raise ValueError("retirement_unclassified")
                retire.append((item.logical_id, item.path))
        declared = {item.path for item in target.items if item.path is not None}
        for _, path in restore:
            if path.exists() and path not in declared:
                raise ValueError("target_unverified")
        # Unknown children are never inferred garbage inside a managed directory.
        for item in target.items:
            if (
                item.path
                and item.path.is_dir()
                and any(
                    not roots[key].synthetic
                    and (item.path == root or root in item.path.parents)
                    for key, root in destinations.items()
                )
                and any(child not in declared for child in item.path.iterdir())
            ):
                raise ValueError("target_owner_unclassified")
    projection_issues = ()
    if mode == "replace" and target is not None:
        from .projection_publication import dependent_retirements

        retire, preserve, projection_issues = dependent_retirements(
            target, restore, retire, preserve, safety_scope=safety_scope
        )
    projection_issues += tuple(
        "projection_reconciliation_required:" + row.logical_id
        for row in doc.directories
        if row.logical_id in selected
        and row.logical_id in producer
        and producer[row.logical_id].owner_id == "rag.projections"
        and row.parent_id is None
    )
    plan = RestorePlan(
        archive.digest,
        mode,
        restore,
        tuple(sorted(retire)),
        tuple(sorted(preserve)),
        "",
        tuple(sorted(destinations.items())),
        target,
        tuple(sorted(selectors.items())),
        tuple(sorted(names.items())),
        safety_scope=tuple(sorted(safety_scope)),
        acknowledged_credential_issues=tuple(sorted(acknowledged_credential_issues)),
    )
    from dataclasses import replace

    metadata, issues = (
        [],
        ["owner_setup_required:" + owner for owner in sorted(deferred_selected)]
        + sorted(set(projection_issues)),
    )
    restored_ids = {key for key, _ in restore}
    for record in records:
        if record.logical_id not in restored_ids:
            continue
        desired = record.metadata
        private_mode = 0o700 if record.logical_id in directories else 0o600
        if desired is None:
            applied = Metadata(version=1, mode=private_mode, mtime_ns=0)
            issues.append("metadata_unavailable:" + record.logical_id)
        else:
            applied = Metadata(
                version=1,
                mode=(desired.mode & 0o700) | private_mode,
                mtime_ns=desired.mtime_ns,
            )
            if applied != desired:
                issues.append("metadata_normalized:" + record.logical_id)
        metadata.append((record.logical_id, desired, applied))
    parents = set()
    root_paths = set(destinations.values())
    if mode == "isolated":
        supplied_directories = {path for key, path in restore if key in directories}
        for key, path in selectors.items():
            if key.endswith(":paths.data_dir") and path not in supplied_directories:
                if any(root in path.parents for root in root_paths):
                    raise ValueError("isolated_data_container_overlap")
                parents.add(path)
    for root in root_paths | parents.copy():
        parent = root.parent
        while not parent.exists():
            if parent not in root_paths:
                parents.add(parent)
            parent = parent.parent
    plan = replace(
        plan,
        containers=tuple(
            ("local:" + hashlib.sha256(str(path).encode()).hexdigest(), path)
            for path in sorted(parents, key=lambda path: (len(path.parts), str(path)))
        ),
    )
    return replace(
        plan,
        target_fingerprint=_fingerprint(_paths(plan), target),
        metadata=tuple(metadata),
        issues=tuple(issues),
    )
