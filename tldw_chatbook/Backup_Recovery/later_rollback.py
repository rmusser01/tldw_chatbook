"""New replacements from authenticated local stored-data recovery snapshots."""

import hashlib
import shutil
import stat
import tomllib
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from threading import Event
from uuid import uuid4

from tldw_chatbook.Utils.platform_files import os

from . import archive_reader, bootstrap
from .journal import _evidence_digest, _Prepared, _Rollback
from .limits import ArchiveLimits
from .native_files import create_private_directory, pinned_directory
from .plan_records import load_plan
from .recovery_copies import _journal, _locked_copy
from .restore_plan import (
    LocalSnapshotSource,
    RestorePlan,
    RetainedConfig,
    _retained_config_items,
    plan_restore,
    recheck_retained_configs,
    recheck_targets,
    retained_config_names,
    retained_config_observation,
)
from .service_storage import ensure_storage


def verify_snapshot_source(plan, archive=None):
    """Match local terminal evidence, never a caller's archive-policy assertion."""
    return _verify_snapshot_source(
        plan.local_snapshot, plan.mode, plan.archive_digest, archive
    )


def _verify_snapshot_source(source, mode, archive_digest, archive=None):
    if type(source) is not LocalSnapshotSource or mode != "replace":
        raise ValueError("local_snapshot_source_required")
    journal = _journal(source.control_root, source.operation_id)
    with journal._locked(exclusive=False) as parent:
        rows = journal._records(parent)
    if rows[-1].event not in {"committed", "rolled_back"}:
        raise ValueError("recovery_copy_pending")
    record = next(row for row in rows if row.event == "rollback_verified")
    proof = _Rollback.model_validate(record.evidence)
    if (
        _evidence_digest(record.evidence) != source.rollback_digest
        or proof.sealed_digest != archive_digest
    ):
        raise ValueError("local_snapshot_source_changed")
    if archive is not None:
        doc = archive_reader.verify_sealed(archive)
        if (
            archive.digest != proof.sealed_digest
            or hashlib.sha256(archive.manifest_bytes).hexdigest()
            != proof.manifest_digest
            or doc.credential_policy != "rollback"
        ):
            raise ValueError("local_snapshot_source_changed")
    return journal, proof


def _snapshot_target_available(original, target, *, session=None):
    """Require the reviewed footprint, allowing only proved absent SQL stores."""
    from .data_groups import resolve_inventory_groups
    from .replacement import _first_binding_scope_available
    from .restore_groups import required_target_groups

    if target is None:
        return False
    if not original.effective_groups:
        return target.complete
    try:
        scope = resolve_inventory_groups(target.items, original.effective_groups)
    except ValueError:
        return False
    if scope.effective_groups != original.effective_groups or required_target_groups(
        target, original.effective_groups
    ):
        return False
    if session is not None:
        from .control_records import UNBOUND_NAMESPACE
        from .storage_admission import MaintenanceSession, _contains_owned_path

        if type(session) is not MaintenanceSession:
            return False
        session._check()
        if (
            session._control != bootstrap.default_bootstrap_root() / "admission"
            or UNBOUND_NAMESPACE not in session._names
            or any(
                not any(
                    _contains_owned_path(root, item.path) for root in session._roots
                )
                for item in target.items
                if item.status == "missing_required"
            )
        ):
            return False
    return _first_binding_scope_available(target)


def _historical_snapshot_target_available(original, prepared, proof, document):
    """Use recorded SQL absence and original receipts, never live historical paths."""
    from .data_groups import resolve_inventory_groups
    from .inventory import BLOCKING
    from .owner_registry import install_adapters
    from .replacement import _absent_sqlite_dependency_scope
    from .restore_groups import required_target_groups

    target = original.target
    if target is None:
        return False
    if not original.effective_groups:
        return target.complete
    try:
        scope = resolve_inventory_groups(target.items, original.effective_groups)
    except ValueError:
        return False
    if scope.effective_groups != original.effective_groups or required_target_groups(
        target, original.effective_groups
    ):
        return False
    if target.complete:
        return True
    missing = [item for item in target.items if item.status in BLOCKING]
    if proof is None or not _absent_sqlite_dependency_scope(target, missing):
        return False
    owners = {owner.owner_id: owner for owner in install_adapters()}
    selected = set(scope.member_ids) | set(scope.support_ids)
    records = {row.logical_id: row for row in document.files} if document else {}
    for item in missing:
        owner = owners.get(item.owner)
        policy = owner.schema_policy() if owner is not None else None
        if (
            item.status != "missing_required"
            or policy is None
            or not policy.schema_sql
            or item.path is None
            or not item.path.is_absolute()
            or ".." in item.path.parts
            or item.metadata is not None
            or item.logical_id in original.safety_scope
            or any(row.logical_id == item.logical_id for row in proof.safety_sources)
        ):
            return False
        overlapping = [
            row
            for row in prepared.artifacts
            if item.path == Path(row.target)
            or item.path in Path(row.target).parents
            or Path(row.target) in item.path.parents
        ]
        if any(
            item.path == Path(row.previous.path)
            or item.path in Path(row.previous.path).parents
            or Path(row.previous.path) in item.path.parents
            for row in prepared.directory_metadata
        ):
            return False
        if item.logical_id not in selected:
            if (item.logical_id, item.path) not in original.preserve or overlapping:
                return False
            continue
        if len(overlapping) != 1:
            return False
        artifact = overlapping[0]
        record = records.get(artifact.logical_id)
        if (
            artifact.action != "publish"
            or artifact.previous is not None
            or artifact.previous_metadata is not None
            or artifact.candidate is None
            or artifact.candidate.kind != "file"
            or Path(artifact.target) != item.path
            or dict(original.restore).get(artifact.logical_id) != item.path
            or record is None
            or record.owner_id != item.owner
            or artifact.logical_id in proof.coverage
            or item.logical_id in proof.coverage.values()
        ):
            return False
    return bool(missing)


def _builtin_snapshot_target(original, proof, document, target, *, session=None):
    """Reobserve authenticated builtin/legacy persona trees for this operation."""
    from pydantic import TypeAdapter

    from tldw_chatbook.Persona_Visual.recovery import _Assets

    from .generation_witnesses import _witnesses
    from .models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
    from .owner_registry import install_adapters
    from .storage_admission import _digest_recovery_file, acquire_storage

    if original.target is None or not _snapshot_target_available(
        original, target, session=session
    ):
        raise ValueError("local_snapshot_preservation_unverified")
    originals = {item.logical_id: item for item in original.target.items}
    selected = {
        key: originals[key]
        for key in original.safety_scope
        if key in originals
        and originals[key].owner
        in {"persona.visual_identity_builtin", "persona.assets"}
    }
    if not selected:
        return target
    saved = {item.logical_id: item for item in proof.safety_sources}
    producers = {item.logical_id: item for item in document.producer_inventory}
    records = {
        item.logical_id: item for item in (*document.files, *document.directories)
    }
    for key, item in selected.items():
        record, producer, source = records.get(key), producers.get(key), saved.get(key)
        meta = item.metadata
        if (
            meta is None
            or record is None
            or producer is None
            or source is None
            or source.owner_id != item.owner
            or producer.owner_id != item.owner
            or producer.status != item.status
            or set(producer.dependencies) != set(item.dependencies)
            or Path(source.source.path) != item.path
            or record.root_id != meta.root_id
            or record.parent_id != meta.parent_id
            or record.relative_path != meta.relative_path
            or meta.kind
            != ("directory" if item.status == "included_directory" else "file")
        ):
            raise ValueError("local_snapshot_preservation_unverified")
    roots = [item for item in selected.values() if item.metadata.parent_id is None]
    if not roots or any(
        item.metadata.root_id not in {root.logical_id for root in roots}
        for item in selected.values()
    ):
        raise ValueError("local_snapshot_preservation_unverified")
    updated = list(target.items)
    observations = []
    for root in roots:
        owner_id = root.owner
        owner = next((a for a in install_adapters() if a.owner_id == owner_id), None)
        if type(owner) is not _Assets:
            raise ValueError("local_snapshot_preservation_unverified")
        if (
            owner_id == "persona.visual_identity_builtin"
            and owner._root({}) != root.path
            or root.status != "included_directory"
        ):
            raise ValueError("local_snapshot_builtin_root_changed")
        members = [
            item
            for item in selected.values()
            if item.metadata.root_id == root.logical_id
        ]
        configs = [
            originals[key]
            for key in root.dependencies
            if key in originals and originals[key].owner == "config"
        ]
        if len(configs) != 1:
            raise ValueError("local_snapshot_preservation_unverified")
        configs = [
            item
            for item in target.items
            if item.path == configs[0].path
            and item.owner == "config"
            and item.status == "included"
        ]
        if len(configs) != 1:
            raise ValueError("local_snapshot_preservation_unverified")
        config = configs[0]
        context = DiscoveryContext(config.path, config.logical_id.split(":", 2)[1])
        current_roots = [item for item in updated if item.path == root.path]
        if (
            len(current_roots) != 1
            or current_roots[0].owner != owner_id
            or current_roots[0].status not in {"unused", "included_directory"}
            or owner_id == "persona.assets"
            and (
                current_roots[0].status != "included_directory"
                or current_roots[0].metadata is None
                or current_roots[0].metadata.parent_id is not None
                or current_roots[0].metadata.relative_path != ""
                or current_roots[0].metadata.root_id != current_roots[0].logical_id
            )
        ):
            raise ValueError("local_snapshot_builtin_conflict")
        lease = acquire_storage(root.path) if session is None else None
        try:
            before = _witnesses(root.path, lease) if lease is not None else None
            _, profiles, associations = bootstrap._control_records(
                bootstrap.default_bootstrap_root()
            )
            binding = next(
                (row for row in profiles if row["selector"] == str(config.path)), None
            )
            if (
                binding is None
                or (before is not None and binding.get("activation") not in before)
                or not any(
                    root.path == Path(path) or Path(path) in root.path.parents
                    for path in binding["roots"]
                )
            ):
                raise ValueError("local_snapshot_builtin_scope_unverified")
            generation = binding.get("activation")
            if session is not None:
                from .activation import ActivationStore, _private
                from .storage_admission import _contains_owned_path

                session._check()
                if (
                    session._control != bootstrap.default_bootstrap_root() / "admission"
                    or not set(binding["namespaces"]) <= set(session._names)
                    or not any(
                        _contains_owned_path(path, root.path) for path in session._roots
                    )
                    or generation is None
                    or [
                        row["activation"]
                        for row in associations
                        if row["selector"] == str(config.path)
                    ]
                    != [generation]
                ):
                    raise ValueError("local_snapshot_builtin_scope_unverified")
                store = ActivationStore(Path(generation["store_root"]))
                with _private(store._generation(generation["generation"])) as parent:
                    if (
                        store._required(parent, generation["generation"]).owners
                        != generation["owners"]
                    ):
                        raise ValueError("local_snapshot_builtin_generation_changed")
            entries = owner._tree(
                {DISCOVERY_CONTEXT_KEY: context},
                root.path,
                selected_paths=frozenset(
                    item.metadata.relative_path
                    for item in members
                    if item.status == "included"
                )
                if owner_id == "persona.visual_identity_builtin"
                else None,
            )
            if {item.path for item in entries} != {
                item.path for item in members
            } or any(
                item.status not in {"included", "included_directory"}
                for item in entries
            ):
                raise ValueError("local_snapshot_builtin_members_changed")
            by_path = {item.path: item for item in entries}
            mapped = {}
            for item in members:
                actual = by_path[item.path]
                if (
                    actual.status != item.status
                    or actual.metadata.relative_path != item.metadata.relative_path
                ):
                    raise ValueError("local_snapshot_builtin_members_changed")
                mapped[item.logical_id] = actual
            for item in members:
                dependencies = []
                for key in item.dependencies:
                    prior = originals.get(key)
                    matches = (
                        [mapped[key]]
                        if key in mapped
                        else [
                            entry
                            for entry in target.items
                            if prior is not None
                            and entry.path == prior.path
                            and entry.owner == prior.owner
                        ]
                    )
                    if len(matches) != 1:
                        raise ValueError("local_snapshot_preservation_unverified")
                    dependencies.append(matches[0].logical_id)
                actual = replace(
                    mapped[item.logical_id],
                    dependencies=tuple(dependencies),
                    shared_group=item.shared_group,
                )
                current = [
                    entry
                    for entry in updated
                    if entry.path == item.path or entry.logical_id == actual.logical_id
                ]
                # A replacement core may no longer reference retained legacy
                # files. Reattach only authenticated old edges to this exact
                # reobserved tree; config/core edges and new dependencies must
                # still agree. Staging validates the old core against held bytes.
                if current and (
                    len(current) != 1
                    or current[0].owner != owner_id
                    or current[0].logical_id != actual.logical_id
                    or current[0].status != "unused"
                    and (
                        current[0].status != actual.status
                        or (
                            set(current[0].dependencies) != set(actual.dependencies)
                            and not (
                                owner_id == "persona.assets"
                                and item is root
                                and set(current[0].dependencies)
                                <= set(actual.dependencies)
                                and set(actual.dependencies)
                                - set(current[0].dependencies)
                                <= {
                                    row.logical_id
                                    for row in mapped.values()
                                    if row.status == "included"
                                }
                            )
                        )
                    )
                ):
                    raise ValueError("local_snapshot_builtin_conflict")
                updated = [entry for entry in updated if entry not in current]
                updated.append(actual)
                if actual.status == "included":
                    observations.append(
                        (
                            str(actual.path),
                            _digest_recovery_file(
                                owner_id, actual.path, max_bytes=owner.max_bytes
                            ),
                        )
                    )
            _, after_profiles, after_associations = bootstrap._control_records(
                bootstrap.default_bootstrap_root()
            )
            after_binding = next(
                (row for row in after_profiles if row["selector"] == str(config.path)),
                None,
            )
            if (
                after_binding != binding
                or [
                    row["activation"]
                    for row in after_associations
                    if row["selector"] == str(config.path)
                ]
                != [generation]
                or lease is not None
                and _witnesses(root.path, lease) != before
            ):
                raise ValueError("local_snapshot_builtin_generation_changed")
            observations.append(generation)
        finally:
            if lease is not None:
                lease.close()
    result = replace(
        target,
        items=tuple(sorted(updated, key=lambda item: item.logical_id)),
        scope_digest="",
    )
    return replace(
        result,
        scope_digest=_evidence_digest(
            {
                "target": TypeAdapter(type(result)).dump_python(result, mode="json"),
                "observed": observations,
            }
        ),
    )


def _preserved_snapshot_members(source, archive, target):
    """Match authenticated original safety sources to observed current owners."""
    from .restore_plan import _ancestor

    journal, proof = _verify_snapshot_source(source, "replace", archive.digest, archive)
    original = load_plan(journal)
    with journal._locked(exclusive=False) as parent:
        rows = journal._records(parent)
    prepared = _Prepared.model_validate(
        next(row.evidence for row in rows if row.event == "prepared")
    )
    incoming = None
    if original.effective_groups and original.target is not None:
        from .data_groups import group_for_owner

        if any(
            item.status == "missing_required"
            and group_for_owner(item.owner) in original.effective_groups
            for item in original.target.items
        ):
            incoming, _ = _created_manifest(journal, original, prepared, rows)
    if (
        not _snapshot_target_available(original, target)
        or not _historical_snapshot_target_available(
            original, prepared, proof, incoming
        )
        or prepared.publication is None
        or prepared.publication.bootstrap_root
        != str(bootstrap.default_bootstrap_root())
        or proof.safety_sources != prepared.safety_sources
    ):
        raise ValueError("local_snapshot_preservation_unverified")
    document = archive_reader.verify_sealed(archive)
    producers = {item.logical_id: item for item in document.producer_inventory}
    records = {item.logical_id for item in (*document.files, *document.directories)}
    originals = {item.logical_id: item for item in original.target.items}
    safety = {item.logical_id: item for item in proof.safety_sources}
    preserved = {}
    for key in original.safety_scope:
        item, saved, producer = originals.get(key), safety.get(key), producers.get(key)
        if (
            item is None
            or saved is None
            or producer is None
            or key not in records
            or item.path is None
            or item.status not in {"included", "included_directory"}
            or saved.owner_id != item.owner
            or producer.owner_id != item.owner
            or producer.status != item.status
            or set(producer.dependencies) != set(item.dependencies)
            or Path(saved.source.path) != item.path
        ):
            raise ValueError("local_snapshot_preservation_unverified")
        matches = [current for current in target.items if current.path == item.path]
        if (
            len(matches) != 1
            or matches[0].owner != item.owner
            or matches[0].status != item.status
        ):
            raise ValueError("local_snapshot_preservation_unverified")
        dependencies = set()
        for dependency in item.dependencies:
            prior = originals.get(dependency)
            current = [
                entry
                for entry in target.items
                if prior is not None
                and entry.path == prior.path
                and entry.owner == prior.owner
            ]
            if len(current) != 1:
                raise ValueError("local_snapshot_preservation_unverified")
            dependencies.add(current[0].logical_id)
        if set(matches[0].dependencies) != dependencies:
            raise ValueError("local_snapshot_preservation_unverified")
        _ancestor(item.path)
        info = os.stat(item.path, follow_symlinks=False)
        if (
            item.status == "included"
            and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1)
            or item.status == "included_directory"
            and not stat.S_ISDIR(info.st_mode)
        ):
            raise ValueError("local_snapshot_preservation_unverified")
        preserved[key] = matches[0]
    return preserved


def _builtin_sources(plan):
    return tuple(
        item
        for item in plan.target.items
        if item.owner in {"persona.visual_identity_builtin", "persona.assets"}
        and item.logical_id in plan.safety_scope
    )


def _stage_read_sources(plan):
    """Read preserved persona safety trees and reviewed created-tree retirements."""
    sources = {item.logical_id: item for item in _builtin_sources(plan)}
    retired = dict(plan.retire)
    roots = {
        item.logical_id: item
        for item in plan.target.items
        if item.owner in {"persona.visual_identity_builtin", "eval.definitions"}
        and item.status == "included_directory"
        and item.metadata is not None
        and item.metadata.parent_id is None
        and item.metadata.root_id == item.logical_id
        and retired.get(item.logical_id) == item.path
    }
    for item in plan.target.items:
        root = roots.get(item.metadata.root_id) if item.metadata else None
        if (
            root is not None
            and item.owner == root.owner
            and item.status in {"included", "included_directory"}
            and retired.get(item.logical_id) == item.path
            and item.path == root.path / item.metadata.relative_path
        ):
            sources[item.logical_id] = item
    active_roots = {
        item.metadata.root_id
        for item in plan.target.items
        if item.owner == "persona.assets"
        and item.status == "included_directory"
        and item.metadata is not None
        and item.metadata.parent_id is not None
        and retired.get(item.logical_id) == item.path
    }
    for key in active_roots:
        semantic = [row for row in plan.target.items if row.logical_id == key]
        parts = key.split(":")
        if (
            len(semantic) != 1
            or len(parts) != 3
            or parts[0] != "profile"
            or parts[2] != "persona.assets"
            or semantic[0].owner != "persona.assets"
            or semantic[0].status != "included_directory"
            or semantic[0].metadata is None
            or semantic[0].metadata.kind != "directory"
            or semantic[0].metadata.parent_id is not None
            or semantic[0].metadata.root_id != key
        ):
            raise ValueError("local_snapshot_created_scope_unverified")
        root = semantic[0]
        core_key = f"profile:{parts[1]}:db.chachanotes.primary"
        config_key = f"profile:{parts[1]}:config"
        core = [row for row in plan.target.items if row.logical_id == core_key]
        configs = [row for row in plan.target.items if row.logical_id == config_key]
        if (
            len(core) != 1
            or core[0].owner != "db.chachanotes.primary"
            or core[0].status != "included"
            or core_key not in root.dependencies
            or len(configs) != 1
            or configs[0].owner != "config"
            or configs[0].status != "included"
            or config_key not in root.dependencies
            or config_key not in core[0].dependencies
        ):
            raise ValueError("local_snapshot_created_scope_unverified")
        sources[core_key] = core[0]
        sources[config_key] = configs[0]
        for item in plan.target.items:
            if item.owner != root.owner or item.status != "included":
                continue
            if item.metadata is not None and item.metadata.root_id == key:
                if (
                    item.metadata.kind != "file"
                    or item.path != root.path / item.metadata.relative_path
                ):
                    raise ValueError("local_snapshot_created_scope_unverified")
                sources[item.logical_id] = item
    return tuple(sources.values())


def _builtin_stage_names(plan):
    """Use only current native bindings containing authenticated safety sources."""
    sources = _stage_read_sources(plan)
    if not sources:
        return ()
    root = bootstrap.default_bootstrap_root()
    _, profiles, _ = bootstrap._control_records(root)
    registry = bootstrap._registry(root)
    names = set()
    for item in sources:
        bindings = [
            row
            for row in profiles
            if any(
                item.path == Path(path) or Path(path) in item.path.parents
                for path in row["roots"]
            )
        ]
        if not bindings:
            raise ValueError("local_snapshot_builtin_scope_unverified")
        for row in bindings:
            binding = bootstrap._binding(Path(row["selector"]), profiles, registry)
            if binding is None:
                raise ValueError("local_snapshot_builtin_scope_unverified")
            names.update(binding["namespaces"])
    return tuple(sorted(names))


def _preserved_builtin_validation(
    plan, document, manifest_digest, work, session, cancel
):
    """Copy current authenticated preserved members only for private validation."""
    from .journal import _Object
    from .models import FileMetadata
    from .publication import _observe_safety_source, _safety_source_matches
    from .storage_admission import copy_capture_file

    if plan.local_snapshot is None or not _builtin_sources(plan):
        return {}, {}
    journal, proof = verify_snapshot_source(plan)
    original = load_plan(journal)
    if manifest_digest != proof.manifest_digest:
        raise ValueError("local_snapshot_source_changed")
    selected = set(original.safety_scope)
    originals = {item.logical_id: item for item in original.target.items}
    source_evidence = {item.logical_id: item for item in proof.safety_sources}
    records = {
        item.logical_id: item for item in (*document.files, *document.directories)
    }
    producers = {item.logical_id: item for item in document.producer_inventory}
    if session is None:
        raise ValueError("local_snapshot_builtin_native_session_required")
    session._check()
    from .control_records import UNBOUND_NAMESPACE
    from .storage_admission import _contains_owned_path

    if (
        session._control != bootstrap.default_bootstrap_root() / "admission"
        or UNBOUND_NAMESPACE not in session._names
    ):
        raise ValueError("local_snapshot_builtin_scope_unverified")
    items, candidates, observed = {}, {}, {}
    for key in selected:
        old = originals[key]
        if old.owner not in {"persona.visual_identity_builtin", "persona.assets"}:
            continue
        current = [
            item
            for item in _builtin_sources(plan)
            if item.path == old.path and item.owner == old.owner
        ]
        producer, record, saved = (
            producers.get(key),
            records.get(key),
            source_evidence.get(key),
        )
        if (
            len(current) != 1
            or saved is None
            or producer is None
            or record is None
            or saved.owner_id != old.owner
            or Path(saved.source.path) != old.path
            or producer.owner_id != old.owner
            or producer.status != old.status
            or set(producer.dependencies) != set(old.dependencies)
            or current[0].status != old.status
        ):
            raise ValueError("local_snapshot_preservation_unverified")
        item = current[0]
        if not any(_contains_owned_path(root, item.path) for root in session._roots):
            raise ValueError("local_snapshot_builtin_scope_unverified")
        observed[key] = _Object.model_validate(_observe_safety_source(item.path))
        meta = item.metadata
        items[key] = replace(
            item,
            logical_id=key,
            dependencies=producer.dependencies,
            metadata=FileMetadata(
                1,
                record.root_id,
                record.relative_path,
                record.parent_id,
                meta.kind,
                meta.mode,
                meta.mtime_ns,
                meta.policy,
            ),
        )
    files = tuple(
        (item.path.resolve(strict=True), observed[key].device, observed[key].inode)
        for key, item in items.items()
        if item.status == "included"
    )
    limits = ArchiveLimits()
    # The current selector may already contain the authenticated historical bytes.
    # Exact saved sources plus held native scope remain authority after publication.
    with session._capture_bound_sources(files, work, limits, limits.expanded_bytes):
        for key, item in items.items():
            destination = work / ("builtin-" + hashlib.sha256(key.encode()).hexdigest())
            if item.status == "included_directory":
                create_private_directory(destination)
            else:
                copy_capture_file(
                    item.owner,
                    item.path,
                    destination,
                    cancel,
                    max_bytes=limits.member_bytes,
                )
            candidates[key] = destination
        if any(not _safety_source_matches(value) for value in observed.values()):
            raise ValueError("local_snapshot_builtin_members_changed")
    return items, candidates


def _acquire(entry, proof, work, password, cancel):
    if type(password) is not bytes or not password:
        raise ValueError("rollback_password_required")
    archive = archive_reader.acquire(
        entry.path, work / "acquired", ArchiveLimits(), password, cancel
    )
    document = archive_reader.verify_sealed(archive, cancel)
    if (
        archive.digest != proof.sealed_digest
        or hashlib.sha256(archive.manifest_bytes).hexdigest() != proof.manifest_digest
        or document.credential_policy != "rollback"
    ):
        raise ValueError("local_snapshot_source_changed")
    return archive


def validate_snapshot_config(payload, candidate):
    """Validate retained config as exact bounded bytes, including malformed TOML."""
    from .storage_admission import _read_recovery_file

    content = _read_recovery_file("config", candidate, max_bytes=16 * 1024**2)
    if (
        payload.owner_id != "config"
        or len(content) != payload.size
        or hashlib.sha256(content).hexdigest() != payload.sha256
    ):
        raise ValueError("local_snapshot_config_changed")
    return ()


def _current_config_scope(plan, document):
    """Reobserve current locators without expanding reviewed rollback scope."""
    from .destinations import _config_tables
    from .owner_registry import install_adapters
    from .staging import _config_targets, _retained_config_targets

    owners = {owner.owner_id: owner for owner in install_adapters()}
    retained = _retained_config_items(plan, document)
    recheck_retained_configs(plan)
    configs = {
        item.logical_id: item
        for item in plan.target.items
        if item.owner == "config" and item.path in dict(plan.restore).values()
    }
    configs.update(retained)
    for key, item in configs.items():
        with archive_reader._regular(item.path) as stream:
            before = archive_reader._identity(os.fstat(stream.fileno()))
            content = stream.read(16 * 1024**2 + 1)
            if len(content) > 16 * 1024**2 or before != archive_reader._identity(
                os.fstat(stream.fileno())
            ):
                raise ValueError("target_changed")
        try:
            data = tomllib.loads(content.decode("utf-8"))
            if key in retained:
                data = _config_tables(data)
        except (ValueError, UnicodeError, RecursionError):
            if key in retained:
                raise ValueError("retained_config_invalid") from None
            # Exact typed local mapping, target fingerprint and held capture
            # remain authoritative when historical config cannot be parsed.
            continue
        validator = _retained_config_targets if key in retained else _config_targets
        validator(
            data, key.split(":", 2)[1], item.path, document, plan, owners
        )
    recheck_retained_configs(plan)


def _discard(work, identity):
    with pinned_directory(work) as descriptor:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != identity:
            raise ValueError("recovery_workspace_changed")
    shutil.rmtree(work)


def _created_manifest(journal, original, prepared, rows):
    """Read the installed incoming manifest from its original local receipt."""
    from .journal import _CandidateReceipt

    receipt = _CandidateReceipt.model_validate(rows[0].evidence)
    installed = next(row.evidence for row in rows if row.event == "installed_validated")
    if (
        rows[-1].event != "committed"
        or receipt.plan_digest != prepared.publication.plan_digest
        or receipt.archive_digest != original.archive_digest
        or installed["plan_digest"] != receipt.plan_digest
        or installed["manifest_digest"] != receipt.manifest_digest
    ):
        raise ValueError("local_snapshot_created_source_changed")
    limits = ArchiveLimits()
    with archive_reader._regular(journal.root / "verified-manifest.json") as stream:
        info = os.fstat(stream.fileno())
        if (
            info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("verified_manifest_changed")
        raw = stream.read(limits.manifest_bytes + 1)
        if archive_reader._identity(info) != archive_reader._identity(
            os.fstat(stream.fileno())
        ):
            raise ValueError("verified_manifest_changed")
    if (
        len(raw) > limits.manifest_bytes
        or hashlib.sha256(raw).hexdigest() != receipt.manifest_digest
    ):
        raise ValueError("verified_manifest_changed")
    return archive_reader._manifest(
        raw, limits, encrypted=True
    ), receipt.manifest_digest


def _config_file_relations(original, prepared, document, target, *, proof=None):
    """Join authenticated incoming IDs to exact original/local owner identities."""
    from .staging import _items
    from .storage_admission import _CONFIG_CAPTURE_LEAVES

    if not _snapshot_target_available(
        original, target
    ) or not _historical_snapshot_target_available(original, prepared, proof, document):
        raise ValueError("local_snapshot_created_scope_unverified")
    incoming = _items(document, original)
    incoming.update(_retained_config_items(original, document))
    files = {row.logical_id: row for row in document.files}
    result = {}
    for artifact in prepared.artifacts:
        source = incoming.get(artifact.logical_id)
        if (
            artifact.action != "publish"
            or artifact.previous is not None
            or source is None
            or source.owner not in _CONFIG_CAPTURE_LEAVES
        ):
            continue
        record = files.get(artifact.logical_id)
        configs = [
            incoming[key]
            for key in source.dependencies
            if key in incoming
            and incoming[key].owner == "config"
            and incoming[key].status == "included"
            and key in files
        ]
        if (
            record is None
            or record.owner_id != source.owner
            or artifact.candidate is None
            or artifact.candidate.kind != "file"
            or source.path != Path(artifact.target)
            or len(configs) != 1
            or source.dependencies != (configs[0].logical_id,)
            or source.path
            != configs[0].path.parent / _CONFIG_CAPTURE_LEAVES[source.owner]
        ):
            raise ValueError("local_snapshot_created_scope_unverified")
        selector = configs[0].path
        matched = []
        for inventory, status in ((original.target, "unused"), (target, "included")):
            local_configs = [
                row
                for row in inventory.items
                if row.owner == "config"
                and row.path == selector
                and row.status == "included"
            ]
            local_files = [
                row
                for row in inventory.items
                if row.path == source.path and row.owner == source.owner
            ]
            if len(local_configs) != 1 or len(local_files) != 1:
                raise ValueError("local_snapshot_created_scope_unverified")
            config, item = local_configs[0], local_files[0]
            identity = config.logical_id.split(":")
            if (
                len(identity) != 3
                or identity[0] != "profile"
                or identity[2] != "config"
                or item.logical_id != f"profile:{identity[1]}:{source.owner}"
                or item.status != status
                or item.dependencies != (config.logical_id,)
                or status == "unused"
                and item.metadata is not None
                or status == "included"
                and (
                    item.metadata is None
                    or item.metadata.kind != "file"
                    or item.metadata.policy != "private"
                    or item.metadata.root_id != item.logical_id
                    or item.metadata.parent_id is not None
                )
            ):
                raise ValueError("local_snapshot_created_scope_unverified")
            matched.append(item)
        if matched[0].logical_id in original.safety_scope:
            raise ValueError("local_snapshot_absence_overlap")
        result[artifact.logical_id] = (matched[1], selector)
    return result


def _config_file_retirement_scopes(plan):
    """Return receipt-proven retirements; callers still prove their live phase."""
    from .storage_admission import _CONFIG_CAPTURE_LEAVES

    if plan.local_snapshot is None:
        return ()
    if plan.target is None or not plan.effective_groups and not plan.target.complete:
        raise ValueError("local_snapshot_created_scope_unverified")
    retirement_keys, retirement_paths = (
        set(dict(plan.retire)),
        set(dict(plan.retire).values()),
    )
    if not any(
        item.owner in _CONFIG_CAPTURE_LEAVES
        and (item.logical_id in retirement_keys or item.path in retirement_paths)
        for item in plan.target.items
    ):
        return ()
    journal, proof = verify_snapshot_source(plan)
    original = load_plan(journal)
    if not _snapshot_target_available(original, plan.target):
        raise ValueError("local_snapshot_created_scope_unverified")
    if original.local_snapshot == plan.local_snapshot:
        raise ValueError("local_snapshot_created_history_unverified")
    with journal._locked(exclusive=False) as parent:
        rows = journal._records(parent)
    prepared = _Prepared.model_validate(
        next(row.evidence for row in rows if row.event == "prepared")
    )
    document, _ = _created_manifest(journal, original, prepared, rows)
    relations = _config_file_relations(
        original, prepared, document, plan.target, proof=proof
    )
    proved = {
        (item.logical_id, item.path): selector for item, selector in relations.values()
    }
    current = {item.logical_id: item for item in plan.target.items}
    paths = {path for _, path in proved}
    result = []
    for key, path in plan.retire:
        item = current.get(key)
        if path not in paths and (
            item is None or item.owner not in _CONFIG_CAPTURE_LEAVES
        ):
            continue
        if (key, path) not in proved:
            raise ValueError("local_snapshot_created_history_unverified")
        result.append((key, path, proved[key, path]))
    return tuple(result)


def _created_config_file_targets(
    journal, original, prepared, rows, target, *, session=None
):
    """Reobserve finite config files while their originating activation is live."""
    from ..runtime_policy.recovery import recovery_adapters as runtime_adapters
    from .config_adapter import recovery_adapters
    from .generation_witnesses import _witnesses
    from .models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
    from .storage_admission import (
        _CONFIG_CAPTURE_LEAVES,
        _contains_owned_path,
        acquire_storage,
    )

    config_paths = {
        item.path for item in target.items if item.owner in _CONFIG_CAPTURE_LEAVES
    }
    if not any(
        row.action == "publish"
        and row.previous is None
        and row.candidate is not None
        and row.candidate.kind == "file"
        and Path(row.target) in config_paths
        for row in prepared.artifacts
    ):
        return {}

    document, _ = _created_manifest(journal, original, prepared, rows)
    proof = _Rollback.model_validate(
        next(row.evidence for row in rows if row.event == "rollback_verified")
    )
    relations = _config_file_relations(
        original, prepared, document, target, proof=proof
    )
    owners = {
        owner.owner_id: owner for owner in (*recovery_adapters(), *runtime_adapters())
    }
    result = {}
    for key, (item, selector) in relations.items():
        lease = acquire_storage(selector) if session is None else None
        try:
            root = bootstrap.default_bootstrap_root()
            before = bootstrap._control_records(root)
            pending, profiles, associations = before
            binding = next(
                (row for row in profiles if row["selector"] == str(selector)), None
            )
            generation = binding.get("activation") if binding else None
            registry = bootstrap._registry(root)
            if (
                pending
                or generation is None
                or generation["operation_id"] != journal.operation_id
                or generation["generation"] != prepared.generation
                or Path(generation["store_root"]) != journal.root.parent / "activation"
                or [
                    row["activation"]
                    for row in associations
                    if row["selector"] == str(selector)
                ]
                != [generation]
                or any(
                    Path(row["selector"]).parent == selector.parent
                    and row is not binding
                    for row in profiles
                )
            ):
                raise ValueError("local_snapshot_created_scope_unverified")
            if lease is not None:
                if generation not in _witnesses(selector, lease):
                    raise ValueError("local_snapshot_created_scope_unverified")
            else:
                session._check()
                if (
                    session._control != root / "admission"
                    or not set(binding["namespaces"]) <= set(session._names)
                    or not any(
                        _contains_owned_path(path, selector) for path in session._roots
                    )
                ):
                    raise ValueError("local_snapshot_created_scope_unverified")
            info = os.stat(item.path, follow_symlinks=False)
            tokens = {
                "path:" + str(item.path),
                "path:" + str(item.path.resolve()),
                f"inode:{info.st_dev}:{info.st_ino}",
            }
            with pinned_directory(selector.parent) as parent:
                posture = os.fstat(parent)
                if posture.st_uid != os.geteuid() or posture.st_mode & 0o077:
                    raise ValueError("local_snapshot_created_scope_unverified")
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077
                ):
                    raise ValueError("local_snapshot_created_scope_unverified")
                if any(
                    tokens.intersection(entry["historical"])
                    or any(
                        bootstrap._overlap(item.path, Path(path))
                        for path in entry["roots"]
                    )
                    or any(
                        bootstrap._overlap(item.path, Path(token[5:]))
                        for token in entry["historical"]
                        if token.startswith("path:")
                    )
                    for name, entry in registry.items()
                    if name not in binding["namespaces"]
                ):
                    raise ValueError("local_snapshot_created_scope_unverified")
                context = DiscoveryContext(selector, item.logical_id.split(":", 2)[1])
                observed = owners[item.owner].discover({DISCOVERY_CONTEXT_KEY: context})
                if (
                    observed != (item,)
                    or before != bootstrap._control_records(root)
                    or registry != bootstrap._registry(root)
                ):
                    raise ValueError("local_snapshot_created_scope_unverified")
            result[key] = item.logical_id
        finally:
            if lease is not None:
                lease.close()
    return result


def _created_persona_subtree(
    item, saved, target, config, owner, prepared, *, session=None
):
    """Map an authenticated nested publication to its unchanged native graph."""
    from .models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
    from .profile_paths import database_path
    from .storage_admission import _preview_reads, _read_recovery_file

    root = saved.get(item.metadata.root_id)
    identity = root.logical_id.split(":") if root is not None else ()
    if (
        root is None
        or len(identity) != 3
        or identity[0] != "profile"
        or identity[2] != "persona.assets"
        or root.owner != item.owner
        or root.status != "included_directory"
        or root.metadata is None
        or root.metadata.kind != "directory"
        or root.metadata.parent_id is not None
        or root.metadata.root_id != root.logical_id
        or root.metadata.relative_path != ""
    ):
        raise ValueError("local_snapshot_created_scope_unverified")
    current, seen, ancestors = item, set(), []
    while current is not root:
        if current.logical_id in seen or current.metadata is None:
            raise ValueError("local_snapshot_created_scope_unverified")
        seen.add(current.logical_id)
        parent = saved.get(current.metadata.parent_id)
        if (
            parent is None
            or parent.owner != root.owner
            or parent.status != "included_directory"
            or parent.metadata is None
            or parent.metadata.kind != "directory"
            or current.metadata.kind != "directory"
            or parent.metadata.root_id != root.logical_id
            or current.metadata.root_id != root.logical_id
            or current.path != root.path / current.metadata.relative_path
            or parent.path != current.path.parent
        ):
            raise ValueError("local_snapshot_created_scope_unverified")
        ancestors.append(parent)
        current = parent
    source_config = saved.get(f"profile:{identity[1]}:config")
    source_core = saved.get(f"profile:{identity[1]}:db.chachanotes.primary")
    if (
        source_config is None
        or source_config.owner != "config"
        or source_config.status != "included"
        or source_config.path != config.path
        or source_core is None
        or source_core.owner != "db.chachanotes.primary"
        or source_core.status != "included"
        or not {source_config.logical_id, source_core.logical_id}
        <= set(root.dependencies)
    ):
        raise ValueError("local_snapshot_created_scope_unverified")
    data = tomllib.loads(
        _read_recovery_file("config", config.path, max_bytes=16 * 1024**2).decode(
            "utf-8"
        )
    )
    profile = config.logical_id.split(":", 2)[1]
    data[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(config.path, profile)
    local_core = [
        row
        for row in target.items
        if row.logical_id == f"profile:{profile}:db.chachanotes.primary"
    ]
    if (
        owner._root(data) != root.path
        or len(local_core) != 1
        or local_core[0].owner != source_core.owner
        or local_core[0].status != "included"
        or local_core[0].path != source_core.path
        or database_path(data, "chachanotes_db_path") != source_core.path
    ):
        raise ValueError("local_snapshot_created_scope_unverified")
    # Native references must inspect a verified main/WAL copy, not live SHM.
    # The held caller already supplies the existing finite capture scope.
    with _preview_reads() if session is None else nullcontext():
        observed = owner.discover(data)
    current = {row.logical_id: row for row in target.items}
    paths = {row.path for row in observed}
    keys = {row.logical_id for row in observed}
    matched = [
        row for row in target.items if row.path in paths or row.logical_id in keys
    ]
    if (
        len(current) != len(target.items)
        or len(paths) != len(observed)
        or len(keys) != len(observed)
        or len(matched) != len(observed)
        or any(
            row.status not in {"included", "included_directory"}
            or current.get(row.logical_id) != row
            for row in observed
        )
    ):
        raise ValueError("local_snapshot_created_owner_conflict")
    matching = [row for row in observed if row.path == item.path]
    if (
        len(matching) != 1
        or matching[0].metadata is None
        or matching[0].metadata.relative_path != item.metadata.relative_path
        or matching[0].metadata.root_id != f"profile:{profile}:persona.assets"
        or matching[0].metadata.parent_id is None
    ):
        raise ValueError("local_snapshot_created_scope_unverified")
    recorded = {
        row.previous.path: row.previous
        for row in prepared.directory_metadata
        if row.owner_id == "persona.assets"
    }
    native = {row.logical_id: row for row in observed}
    current, proved = matching[0], []
    while current.metadata.parent_id is not None:
        current = native[current.metadata.parent_id]
        prior = recorded.get(str(current.path))
        if (
            prior is None
            or not any(row.path == current.path for row in ancestors)
            or current.status != "included_directory"
            or current.metadata.kind != "directory"
        ):
            raise ValueError("local_snapshot_created_scope_unverified")
        with pinned_directory(current.path) as descriptor:
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) != (prior.device, prior.inode):
                raise ValueError("local_snapshot_created_root_changed")
        proved.append((current.logical_id, current.path, info.st_dev, info.st_ino))
    if len(proved) != len(ancestors):
        raise ValueError("local_snapshot_created_scope_unverified")
    return matching[0].logical_id, tuple(proved)


def _created_destination_target(
    journal, original, prepared, rows, target, *, session=None, config_files=()
):
    """Reobserve only committed inactive builtin/eval roots created by this copy."""
    from pydantic import TypeAdapter

    from tldw_chatbook.Evals.recovery import _DefinitionsAdapter
    from tldw_chatbook.Persona_Visual.recovery import _Assets

    from .activation import ActivationStore, _private
    from .generation_witnesses import _witnesses
    from .models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
    from .owner_registry import install_adapters
    from .recovery_files import _RawDeclaration
    from .staging import _items
    from .storage_admission import (
        _contains_owned_path,
        _digest_recovery_file,
        acquire_storage,
    )

    originals = {item.logical_id: item for item in original.target.items}
    artifacts = [
        row
        for row in prepared.artifacts
        if row.action == "publish"
        and row.previous is None
        and row.logical_id not in config_files
        and (
            row.logical_id not in originals
            or originals[row.logical_id].path != Path(row.target)
        )
    ]
    if not artifacts or rows[-1].event != "committed":
        return target, {}, {}
    document, _ = _created_manifest(journal, original, prepared, rows)
    limits = ArchiveLimits()
    saved = _items(document, original)
    saved.update(_retained_config_items(original, document))
    owners = {owner.owner_id: owner for owner in install_adapters()}
    updated = list(target.items)
    mapped_roots, observations, active_ancestors = {}, [], {}
    for artifact in artifacts:
        item = saved.get(artifact.logical_id)
        active_persona = (
            item is not None
            and item.owner == "persona.assets"
            and item.status == "included_directory"
            and item.metadata is not None
            and item.metadata.kind == "directory"
            and item.metadata.parent_id is not None
        )
        if (
            item is None
            or item.metadata is None
            or item.metadata.parent_id is not None
            and not active_persona
            or artifact.candidate is None
            or artifact.candidate.kind != "directory"
            or item.path != Path(artifact.target)
        ):
            raise ValueError("local_snapshot_absence_unclassified")
        owner = owners.get(item.owner)
        if not (
            item.owner == "persona.visual_identity_builtin"
            and type(owner) is _Assets
            or item.owner == "eval.definitions"
            and type(owner) is _DefinitionsAdapter
            or active_persona
            and type(owner) is _Assets
        ):
            raise ValueError("local_snapshot_absence_unclassified")
        members = [
            row
            for row in saved.values()
            if row.metadata.root_id == item.logical_id
            or active_persona
            and (row.path == item.path or item.path in row.path.parents)
        ]
        if any(row.owner != item.owner for row in members) or not any(
            row.status == "included" for row in members
        ):
            raise ValueError("local_snapshot_created_members_changed")
        pending = [row.logical_id for row in members]
        seen, configs = set(), set()
        while pending:
            key = pending.pop()
            if key in seen or key not in saved:
                continue
            seen.add(key)
            source = saved[key]
            if source.owner == "config":
                configs.add(source.path)
            else:
                pending.extend(source.dependencies)
        selected = [
            row
            for row in target.items
            if row.owner == "config"
            and row.status == "included"
            and row.path in configs
        ]
        if len(configs) != 1 or len(selected) != 1:
            raise ValueError("local_snapshot_created_scope_unverified")
        config = selected[0]
        if any(
            row.path is not None
            and (
                row.path == item.path
                or item.path in row.path.parents
                or row.path in item.path.parents
            )
            for row in original.target.items
            if row.logical_id in original.safety_scope
        ):
            raise ValueError("local_snapshot_absence_overlap")
        lease = acquire_storage(item.path) if session is None else None
        try:
            before = bootstrap._control_records(bootstrap.default_bootstrap_root())
            _, profiles, associations = before
            binding = next(
                (row for row in profiles if row["selector"] == str(config.path)), None
            )
            generation = binding.get("activation") if binding else None
            from .effective_roots import effective_roots

            registry = bootstrap._registry(bootstrap.default_bootstrap_root())
            physical = effective_roots(
                binding["roots"],
                (registry[name] for name in binding["namespaces"]),
            ) if binding is not None else ()
            if (
                generation is None
                or generation["operation_id"] != journal.operation_id
                or generation["generation"] != prepared.generation
                or Path(generation["store_root"]) != journal.root.parent / "activation"
                or [
                    row["activation"]
                    for row in associations
                    if row["selector"] == str(config.path)
                ]
                != [generation]
                or not any(
                    _contains_owned_path(path, item.path)
                    for path in physical
                )
            ):
                raise ValueError("local_snapshot_created_scope_unverified")
            if lease is not None:
                if generation not in _witnesses(item.path, lease):
                    raise ValueError("local_snapshot_created_scope_unverified")
            else:
                session._check()
                if (
                    session._control != bootstrap.default_bootstrap_root() / "admission"
                    or not set(binding["namespaces"]) <= set(session._names)
                    or not any(
                        _contains_owned_path(path, item.path) for path in session._roots
                    )
                ):
                    raise ValueError("local_snapshot_created_scope_unverified")
                store = ActivationStore(Path(generation["store_root"]))
                with _private(store._generation(prepared.generation)) as parent:
                    if (
                        store._required(parent, prepared.generation).owners
                        != generation["owners"]
                    ):
                        raise ValueError("local_snapshot_created_scope_unverified")
            info = os.stat(item.path, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (
                artifact.candidate.device,
                artifact.candidate.inode,
            ):
                raise ValueError("local_snapshot_created_root_changed")
            if active_persona:
                (
                    mapped_roots[artifact.logical_id],
                    active_ancestors[artifact.logical_id],
                ) = _created_persona_subtree(
                    item, saved, target, config, owner, prepared, session=session
                )
                if (
                    bootstrap._control_records(bootstrap.default_bootstrap_root())
                    != before
                ):
                    raise ValueError("local_snapshot_created_scope_unverified")
                if lease is not None and generation not in _witnesses(item.path, lease):
                    raise ValueError("local_snapshot_created_scope_unverified")
                continue
            context = DiscoveryContext(config.path, config.logical_id.split(":", 2)[1])
            entries = _RawDeclaration(item.owner)._tree(
                {DISCOVERY_CONTEXT_KEY: context}, item.path
            )
            if {(row.path, row.status) for row in entries} != {
                (row.path, row.status) for row in members
            } or any(
                row.status not in {"included", "included_directory"} for row in entries
            ):
                raise ValueError("local_snapshot_created_members_changed")
            for row in entries:
                if row.status == "included":
                    if owner.validate(row.path):
                        raise ValueError("local_snapshot_created_member_invalid")
                    observations.append(
                        (
                            str(row.path),
                            _digest_recovery_file(
                                item.owner, row.path, max_bytes=limits.member_bytes
                            ),
                        )
                    )
            by_path = {row.path: row for row in entries}
            replaced = [
                row
                for row in updated
                if row.path in by_path
                or row.logical_id in {entry.logical_id for entry in entries}
            ]
            if any(
                row.path not in by_path
                or row.owner != item.owner
                or row.status != by_path[row.path].status
                for row in replaced
            ):
                raise ValueError("local_snapshot_created_owner_conflict")
            remap = {row.logical_id: by_path[row.path].logical_id for row in replaced}
            updated = [
                replace(
                    row,
                    dependencies=tuple(remap.get(key, key) for key in row.dependencies),
                )
                for row in updated
                if row not in replaced
            ]
            updated.extend(entries)
            mapped_roots[artifact.logical_id] = by_path[item.path].logical_id
            if bootstrap._control_records(bootstrap.default_bootstrap_root()) != before:
                raise ValueError("local_snapshot_created_scope_unverified")
            if lease is not None and generation not in _witnesses(item.path, lease):
                raise ValueError("local_snapshot_created_scope_unverified")
            observations.append(generation)
        finally:
            if lease is not None:
                lease.close()
    result = replace(
        target,
        items=tuple(sorted(updated, key=lambda row: row.logical_id)),
        scope_digest="",
    )
    return (
        replace(
            result,
            scope_digest=_evidence_digest(
                {
                    "target": TypeAdapter(type(result)).dump_python(
                        result, mode="json"
                    ),
                    "observed": observations,
                }
            ),
        ),
        mapped_roots,
        active_ancestors,
    )


def _created_builtin_members(plan, root, *, seen=()):
    """Prove a current inactive tree from the exact local created publication."""
    from .models import DiscoveryContext
    from .recovery_files import _tree_member_id
    from .staging import _items

    journal, _ = verify_snapshot_source(plan)
    identity = (str(journal.root), root.logical_id)
    if identity in seen or len(seen) >= 8:
        raise ValueError("local_snapshot_created_history_unverified")
    seen = (*seen, identity)
    original = load_plan(journal)
    with journal._locked(exclusive=False) as parent:
        rows = journal._records(parent)
    prepared = _Prepared.model_validate(
        next(row.evidence for row in rows if row.event == "prepared")
    )
    document, digest = _created_manifest(journal, original, prepared, rows)
    source_items = _items(document, original)
    source_items.update(_retained_config_items(original, document))
    artifacts = [
        row
        for row in prepared.artifacts
        if row.target == str(root.path)
        and row.action == "publish"
        and row.previous is None
        and row.candidate is not None
        and row.candidate.kind == "directory"
    ]
    if len(artifacts) != 1:
        raise ValueError("local_snapshot_created_history_unverified")
    source = source_items.get(artifacts[0].logical_id)
    if (
        source is None
        or source.owner != "persona.visual_identity_builtin"
        or source.metadata is None
        or source.metadata.parent_id is not None
        or source.path != root.path
        or any(
            row.path is not None and bootstrap._overlap(row.path, root.path)
            for row in original.target.items
            if row.logical_id in original.safety_scope
        )
    ):
        raise ValueError("local_snapshot_created_history_unverified")
    if source.logical_id != f"profile:{source.logical_id.split(':')[1]}:{source.owner}":
        _snapshot_builtin_members(original, digest, source, seen=seen)
    configs = [
        row
        for row in plan.target.items
        if row.owner == "config"
        and row.logical_id in root.dependencies
        and row.status == "included"
    ]
    if len(configs) != 1 or configs[0].path not in {
        row.path for row in source_items.values() if row.owner == "config"
    }:
        raise ValueError("local_snapshot_created_history_unverified")
    config = configs[0]
    context = DiscoveryContext(config.path, config.logical_id.split(":", 2)[1])
    members = [
        row
        for row in source_items.values()
        if row.metadata.root_id == source.logical_id
    ]
    expected = {}
    for row in members:
        key = _tree_member_id(context, source.owner, root.path, row.path)
        parent = (
            _tree_member_id(context, source.owner, root.path, row.path.parent)
            if row.metadata.parent_id is not None
            else None
        )
        expected[key] = (
            root.logical_id,
            parent,
            row.metadata.relative_path,
            row.metadata.kind,
        )
    actual = {
        row.logical_id: row
        for row in plan.target.items
        if row.metadata is not None and row.metadata.root_id == root.logical_id
    }
    if (
        set(actual) != set(expected)
        or root.logical_id not in expected
        or any(
            row.owner != source.owner
            or _topology(row) != expected[key]
            or row.path != root.path / row.metadata.relative_path
            or set(row.dependencies)
            != (
                {config.logical_id, row.metadata.parent_id}
                if row.metadata.parent_id
                else {config.logical_id}
            )
            for key, row in actual.items()
        )
    ):
        raise ValueError("local_snapshot_created_history_unverified")
    return actual


def _snapshot_builtin_members(plan, manifest_digest, root, *, seen=()):
    """Verify encrypted-copy coverage and its original created-tree provenance."""
    journal, proof = verify_snapshot_source(plan)
    if proof.manifest_digest != manifest_digest:
        raise ValueError("local_snapshot_source_changed")
    original = load_plan(journal)
    sources = {row.logical_id: row for row in original.target.items}
    source = sources.get(root.logical_id)
    with journal._locked(exclusive=False) as parent:
        rows = journal._records(parent)
    prepared = _Prepared.model_validate(
        next(row.evidence for row in rows if row.event == "prepared")
    )
    covered = [
        row
        for row in prepared.artifacts
        if row.target == str(source.path if source else None)
        and row.previous is not None
        and row.previous.kind == "directory"
        and proof.coverage.get(row.logical_id) == root.logical_id
    ]
    if (
        source is None
        or source.owner != "persona.visual_identity_builtin"
        or len(covered) != 1
        or dict(plan.restore).get(root.logical_id) != source.path
        or _topology(root) != _topology(source)
    ):
        raise ValueError("local_snapshot_created_history_unverified")
    return _created_builtin_members(original, source, seen=seen)


def _topology(item):
    meta = item.metadata
    return (
        (meta.root_id, meta.parent_id, meta.relative_path, meta.kind) if meta else None
    )


def _validate_created_builtin_members(expected, root, candidates, topology):
    """Validate every declared private member; no ID-only owner exemption."""
    from tldw_chatbook.Persona_Visual.recovery import _Assets

    from .owner_registry import install_adapters

    owner = next(
        row
        for row in install_adapters()
        if row.owner_id == "persona.visual_identity_builtin"
    )
    if type(owner) is not _Assets or {
        key: value for key, value in topology.items() if value[0] == root.logical_id
    } != {key: _topology(value) for key, value in expected.items()}:
        raise ValueError("local_snapshot_created_members_changed")
    for key, item in expected.items():
        if item.metadata.kind == "file" and (
            key not in candidates or owner.validate(candidates[key])
        ):
            raise ValueError("local_snapshot_created_member_invalid")


def validate_created_builtin_capture(plan, root, items, candidates):
    """Qualify this local retirement's private copies using its source history."""
    if plan.local_snapshot is None or root.logical_id.count(":") == 2:
        return False
    expected = _created_builtin_members(plan, root)
    _validate_created_builtin_members(
        expected,
        root,
        candidates,
        {item.logical_id: _topology(item) for item in items if item.metadata},
    )
    return True


def validate_snapshot_builtin_restore(
    plan, manifest_digest, root, candidates, topology
):
    """Handle only receipt-proved local inactive builtin copies at restore."""
    if (
        plan is None
        or plan.local_snapshot is None
        or root.owner != "persona.visual_identity_builtin"
        or root.metadata is None
        or root.metadata.parent_id is not None
        or root.logical_id.count(":") == 2
    ):
        return False
    expected = _snapshot_builtin_members(plan, manifest_digest, root)
    _validate_created_builtin_members(expected, root, candidates, topology)
    return True


def _known_absences(plan, original, prepared, created=None, *, active_ancestors=None):
    """Expose exact locally recorded absent targets as reviewed retirements."""
    from .projection_publication import dependent_retirements
    from .restore_plan import _fingerprint, _paths

    original_items = {item.logical_id: item for item in original.target.items}
    current = {item.logical_id: item for item in plan.target.items}
    retired = dict(plan.retire)
    for artifact in prepared.artifacts:
        if artifact.previous is not None or artifact.action != "publish":
            continue
        path = Path(artifact.target)
        if not path.exists() and not path.is_symlink():
            continue
        prior = original_items.get(artifact.logical_id)
        created_key = (created or {}).get(artifact.logical_id)
        item = current.get(created_key or artifact.logical_id)
        if created_key:
            prior = (
                item  # Exact original absence was proved from the publication above.
            )
        if (
            prior is None
            or item is None
            or artifact.candidate is None
            or prior.path != path
            or item.path != path
            or prior.owner != item.owner
            or item.logical_id in plan.safety_scope
            or item.status
            != (
                "included_directory"
                if artifact.candidate.kind == "directory"
                else "included"
            )
        ):
            raise ValueError("local_snapshot_absence_unclassified")
        info = os.stat(path, follow_symlinks=False)
        if (
            artifact.candidate.kind == "directory" and not stat.S_ISDIR(info.st_mode)
        ) or (artifact.candidate.kind == "file" and not stat.S_ISREG(info.st_mode)):
            raise ValueError("local_snapshot_absence_unclassified")
        selected = [item]
        if artifact.candidate.kind == "directory":
            declared = {
                entry.path: entry
                for entry in current.values()
                if entry.path is not None
            }
            for directory, names, files in os.walk(path, followlinks=False):
                for name in (*names, *files):
                    child = Path(directory) / name
                    entry = declared.get(child)
                    if (
                        entry is None
                        or entry.owner != item.owner
                        or entry.status not in {"included", "included_directory"}
                        or entry.logical_id in plan.safety_scope
                    ):
                        raise ValueError("local_snapshot_absence_unclassified")
                    child_mode = os.stat(child, follow_symlinks=False).st_mode
                    if (
                        entry.status == "included" and not stat.S_ISREG(child_mode)
                    ) or (
                        entry.status == "included_directory"
                        and not stat.S_ISDIR(child_mode)
                    ):
                        raise ValueError("local_snapshot_absence_unclassified")
                    selected.append(entry)
        else:
            selected += [
                entry
                for entry in current.values()
                if entry.owner == "sqlite.transient"
                and entry.dependencies == (item.logical_id,)
                and entry.path
                in {path.with_name(path.name + suffix) for suffix in ("-wal", "-shm")}
                and entry.path.exists()
            ]
        retained_ancestors = set()
        for key, ancestor, device, inode in (active_ancestors or {}).get(
            artifact.logical_id, ()
        ):
            row = current.get(key)
            if (
                created_key is None
                or item.owner != "persona.assets"
                or row is None
                or row.owner != item.owner
                or row.path != ancestor
                or ancestor not in item.path.parents
                or row.status != "included_directory"
                or row.metadata is None
                or item.metadata is None
                or row.metadata.kind != "directory"
                or row.metadata.root_id != item.metadata.root_id
                or (key, ancestor) not in plan.restore
                or ancestor in dict(plan.retire).values()
                or ancestor in dict(plan.containers).values()
            ):
                raise ValueError("local_snapshot_absence_overlap")
            with pinned_directory(ancestor) as descriptor:
                info = os.fstat(descriptor)
                if (info.st_dev, info.st_ino) != (device, inode):
                    raise ValueError("local_snapshot_created_root_changed")
            retained_ancestors.add((key, ancestor))
        if any(
            entry.path == restored
            or entry.path in restored.parents
            or restored in entry.path.parents
            and (key, restored) not in retained_ancestors
            for entry in selected
            for key, restored in plan.restore
        ):
            raise ValueError("local_snapshot_absence_overlap")
        retired.update((entry.logical_id, entry.path) for entry in selected)
    preserving = tuple(row for row in plan.preserve if row[0] not in retired)
    retirement, preserving, issues = dependent_retirements(
        plan.target,
        plan.restore,
        tuple(retired.items()),
        preserving,
        safety_scope=plan.safety_scope,
    )
    changed = {path for _, path in (*plan.restore, *retirement)}
    for item in plan.target.items:
        if (
            item.path in changed
            and item.shared_group
            and any(
                other.path not in changed
                for other in plan.target.items
                if other.shared_group == item.shared_group
            )
        ):
            raise ValueError("shared_scope_expansion_required")
    plan = replace(
        plan,
        retire=tuple(retirement),
        preserve=tuple(preserving),
        issues=tuple(dict.fromkeys((*plan.issues, *issues))),
    )
    return replace(plan, target_fingerprint=_fingerprint(_paths(plan), plan.target))


def _preview(journal, proof, archive, target, acknowledged, *, session=None):
    original = load_plan(journal)
    document = archive_reader.verify_sealed(archive)
    with journal._locked(exclusive=False) as parent:
        rows = journal._records(parent)
    prepared = _Prepared.model_validate(
        next(row.evidence for row in rows if row.event == "prepared")
    )
    if prepared.publication.bootstrap_root != str(bootstrap.default_bootstrap_root()):
        raise ValueError("local_snapshot_source_changed")
    originals = {item.logical_id: item for item in original.target.items}
    roots = {}
    for record in (*document.directories, *document.files):
        item = originals.get(record.logical_id)
        if item is None or item.logical_id in original.safety_scope:
            continue
        path = item.path
        for _part in Path(record.relative_path).parts:
            path = path.parent
        if record.root_id in roots and roots[record.root_id] != path:
            raise ValueError("local_snapshot_mapping_invalid")
        roots[record.root_id] = path
    for record in (*document.directories, *document.files):
        if record.root_id in roots and record.logical_id in original.safety_scope:
            raise ValueError("shared_scope_expansion_required")
    snapshot = LocalSnapshotSource(
        journal.root.parent, journal.operation_id, _evidence_digest(proof.model_dump())
    )
    _, authenticated = _verify_snapshot_source(
        snapshot, "replace", archive.digest, archive
    )
    if authenticated.safety_sources != prepared.safety_sources:
        raise ValueError("local_snapshot_preservation_unverified")
    target = _builtin_snapshot_target(
        original, authenticated, document, target, session=session
    )
    config_files = _created_config_file_targets(
        journal, original, prepared, rows, target, session=session
    )
    target, created, active_ancestors = _created_destination_target(
        journal,
        original,
        prepared,
        rows,
        target,
        session=session,
        config_files=config_files,
    )
    created.update(config_files)
    preserved = _preserved_snapshot_members(snapshot, archive, target)
    names, profile_map = {}, {}
    original_names = dict(original.profile_names)
    for profile in document.profile_ids:
        config = originals.get(f"profile:{profile}:config")
        if config is None or config.owner != "config":
            continue
        imported = [
            key.split(":")[1]
            for key, path in original.restore
            if path == config.path
            and key.startswith("profile:")
            and key.endswith(":config")
            and len(key.split(":")) == 3
        ]
        imported.extend(
            row.archive_config_id.split(":")[1]
            for row in original.retained_configs
            if row.config_path == config.path
        )
        for key in imported:
            if key in profile_map and profile_map[key] != profile:
                raise ValueError("local_snapshot_mapping_invalid")
            profile_map[key] = profile
        labels = {original_names[key] for key in imported if key in original_names}
        if len(labels) > 1:
            raise ValueError("local_snapshot_mapping_invalid")
        if labels:
            names[profile] = labels.pop()
    selectors = {}
    for key, path in original.selectors:
        parts = key.split(":", 2)
        if len(parts) != 3 or parts[0] != "profile" or parts[1] not in profile_map:
            raise ValueError("local_snapshot_mapping_invalid")
        local_key = f"profile:{profile_map[parts[1]]}:{parts[2]}"
        if local_key in selectors and selectors[local_key] != path:
            raise ValueError("local_snapshot_mapping_invalid")
        selectors[local_key] = path
    plan = plan_restore(
        archive,
        mode="replace",
        destinations={**roots, **selectors},
        target=target,
        profile_names=names,
        safety_scope=tuple(item.logical_id for item in preserved.values()),
        acknowledged_credential_issues=acknowledged,
        local_snapshot=snapshot,
        retained_configs=tuple(
            RetainedConfig(
                row.target_config_id,
                preserved[row.target_config_id].logical_id,
                row.config_path,
                retained_config_observation(row.config_path),
            )
            for row in original.retained_configs
        ),
    )
    plan = replace(
        plan,
        requested_groups=original.requested_groups,
        effective_groups=original.effective_groups,
        required_groups=original.required_groups,
    )
    plan = _known_absences(
        plan, original, prepared, created, active_ancestors=active_ancestors
    )
    if original.effective_groups:
        from .data_groups import resolve_inventory_groups
        from .restore_groups import (
            check_preserved_group_effects,
            required_target_groups,
        )

        scope = resolve_inventory_groups(plan.target.items, original.effective_groups)
        if (
            scope.effective_groups != original.effective_groups
            or required_target_groups(plan.target, original.effective_groups)
        ):
            raise ValueError("local_snapshot_group_scope_changed")
        check_preserved_group_effects(plan, scope)
    if not plan.restore and not plan.retire:
        raise ValueError("explicit_destination_required")
    verify_snapshot_source(plan, archive)
    _current_config_scope(plan, document)
    if "settings" in plan.effective_groups:
        from .destinations import check_config_destinations

        check_config_destinations(archive, plan, session=session)
    recheck_targets(plan)
    return plan


def preview_rollback(
    operation_id,
    *,
    control_root: Path,
    old_password: bytes,
    target,
    cancel: Event,
    acknowledged_credential_issues=(),
) -> RestorePlan:
    """Return explicit current replacement decisions without touching live data."""
    with _locked_copy(control_root, operation_id, exclusive=False) as (
        entry,
        journal,
        proof,
        _stream,
    ):
        if entry.pending_operation:
            raise ValueError("recovery_copy_pending")
        work = ensure_storage(control_root) / ("rollback-preview-" + uuid4().hex)
        create_private_directory(work)
        identity = (os.stat(work).st_dev, os.stat(work).st_ino)
        try:
            archive = _acquire(entry, proof, work, old_password, cancel)
            return _preview(
                journal, proof, archive, target, acknowledged_credential_issues
            )
        finally:
            _discard(work, identity)


def execute_rollback(
    operation_id, *, control_root, old_password, new_password, cancel, approved_plan
):
    from .control_records import UNBOUND_NAMESPACE, admission_authority
    from .replacement import replace as replace_current
    from .replacement import require_rollback_password
    from .staging import stage_restore

    if approved_plan is None:
        raise ValueError("preview_required")
    require_rollback_password(new_password)
    if (
        type(approved_plan) is not RestorePlan
        or approved_plan.local_snapshot is None
        or approved_plan.local_snapshot.operation_id != operation_id
        or approved_plan.local_snapshot.control_root != control_root
    ):
        raise ValueError("local_snapshot_source_changed")
    recheck_targets(approved_plan)
    with _locked_copy(control_root, operation_id, exclusive=False) as (
        entry,
        journal,
        proof,
        _stream,
    ):
        if entry.pending_operation:
            raise ValueError("recovery_copy_pending")
        work = ensure_storage(control_root) / ("later-rollback-" + uuid4().hex)
        create_private_directory(work)
        identity = (os.stat(work).st_dev, os.stat(work).st_ino)
        candidate = None
        completed = False
        try:
            archive = _acquire(entry, proof, work, old_password, cancel)
            current = _preview(
                journal,
                proof,
                archive,
                approved_plan.target,
                approved_plan.acknowledged_credential_issues,
            )
            if current != approved_plan:
                raise ValueError("target_changed")
            # Keep registry geometry and every retained dependency's full native
            # profile scope held while validating private payloads and copies.
            authority = admission_authority(bootstrap.default_bootstrap_root())
            names = {
                UNBOUND_NAMESPACE,
                *_builtin_stage_names(current),
                *retained_config_names(current),
            }
            if current.effective_groups:
                from .capture_service import _capture_names

                names.update(_capture_names(
                    authority, current.target,
                    include_absent_sqlite="settings" in current.effective_groups,
                ))
            with authority.maintenance(
                tuple(sorted(names)),
                30,
                cancel=cancel,
            ) as session:
                sources = tuple(
                    item.path
                    for item in _stage_read_sources(current)
                    if item.status == "included"
                )
                if sources or current.effective_groups:
                    with (
                        session.capture_scope(sources, work, limits=ArchiveLimits())
                        if sources
                        else nullcontext()
                    ):
                        held = _preview(
                            journal,
                            proof,
                            archive,
                            current.target,
                            current.acknowledged_credential_issues,
                            session=session,
                        )
                    if held != current:
                        raise ValueError("target_changed")
                candidate = stage_restore(
                    archive, current, work / "candidate", cancel, session=session
                )
            result = replace_current(
                current,
                candidate,
                control_root=control_root,
                rollback_password=new_password,
                cancel=cancel,
            )
            completed = True
            return result
        finally:
            # A prepared candidate may be referenced by a pending new operation.
            if candidate is None or completed:
                _discard(work, identity)
