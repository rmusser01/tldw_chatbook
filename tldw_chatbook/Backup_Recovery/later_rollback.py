"""New replacements from authenticated local stored-data recovery snapshots."""

import hashlib
import os
import shutil
import stat
import tomllib
from dataclasses import replace
from pathlib import Path
from threading import Event
from uuid import uuid4

from . import archive_reader, bootstrap
from .journal import _evidence_digest, _Prepared, _Rollback
from .limits import ArchiveLimits
from .native_files import create_private_directory, pinned_directory
from .plan_records import load_plan
from .recovery_copies import _journal, _locked_copy
from .restore_plan import (
    LocalSnapshotSource,
    RestorePlan,
    plan_restore,
    recheck_targets,
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
    if (
        target is None
        or not target.complete
        or original.target is None
        or not original.target.complete
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
        info = item.path.lstat()
        if (
            item.status == "included"
            and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1)
            or item.status == "included_directory"
            and not stat.S_ISDIR(info.st_mode)
        ):
            raise ValueError("local_snapshot_preservation_unverified")
        preserved[key] = matches[0]
    return preserved


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
    """Reobserve selected installed locators without expanding reviewed scope."""
    from .owner_registry import install_adapters
    from .staging import _config_targets

    owners = {owner.owner_id: owner for owner in install_adapters()}
    for item in plan.target.items:
        if item.owner != "config" or item.path not in dict(plan.restore).values():
            continue
        with archive_reader._regular(item.path) as stream:
            before = archive_reader._identity(os.fstat(stream.fileno()))
            content = stream.read(16 * 1024**2 + 1)
            if len(content) > 16 * 1024**2 or before != archive_reader._identity(
                os.fstat(stream.fileno())
            ):
                raise ValueError("target_changed")
        try:
            data = tomllib.loads(content.decode("utf-8"))
        except (ValueError, UnicodeError):
            # Exact typed local mapping, target fingerprint and held capture
            # remain authoritative when historical config cannot be parsed.
            continue
        _config_targets(
            data, item.logical_id.split(":", 2)[1], item.path, document, plan, owners
        )


def _discard(work, identity):
    with pinned_directory(work) as descriptor:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != identity:
            raise ValueError("recovery_workspace_changed")
    shutil.rmtree(work)


def _known_absences(plan, original, prepared):
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
        item = current.get(artifact.logical_id)
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
        info = path.lstat()
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
                    child_mode = child.lstat().st_mode
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
        if any(
            entry.path == restored
            or entry.path in restored.parents
            or restored in entry.path.parents
            for entry in selected
            for _, restored in plan.restore
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


def _preview(journal, proof, archive, target, acknowledged):
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
    preserved = _preserved_snapshot_members(snapshot, archive, target)
    plan = plan_restore(
        archive,
        mode="replace",
        destinations={**roots, **dict(original.selectors)},
        target=target,
        profile_names=dict(original.profile_names),
        safety_scope=tuple(item.logical_id for item in preserved.values()),
        acknowledged_credential_issues=acknowledged,
        local_snapshot=snapshot,
    )
    plan = _known_absences(plan, original, prepared)
    verify_snapshot_source(plan, archive)
    _current_config_scope(plan, document)
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
        identity = (work.stat().st_dev, work.stat().st_ino)
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
        identity = (work.stat().st_dev, work.stat().st_ino)
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
            # This phase reads only authenticated private payloads. The existing
            # unbound lease pins registry geometry without acquiring live owners.
            authority = admission_authority(bootstrap.default_bootstrap_root())
            with authority.maintenance(
                (UNBOUND_NAMESPACE,), 30, cancel=cancel
            ) as session:
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
