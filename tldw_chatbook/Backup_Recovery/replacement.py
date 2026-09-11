"""Held original capture and encrypted rollback verification for replacement.

This internal operation does not publish replacement data, apply credentials or
release admission. Only this call chain can produce SQLite-owner rollback proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from dataclasses import replace
from pathlib import Path
from threading import Event
from uuid import uuid4

from . import archive_reader as reader
from .archive_writer import write_archive
from .capture import CaptureResult, _item_validator, _manifest_for
from .credentials import _material, process_credentials
from .journal import _CandidateReceipt, _matches, _Prepared, observe_artifact
from .limits import ArchiveLimits
from .models import FileMetadata, Inventory, StorageItem
from .native_files import create_private_directory, create_private_file
from .owner_registry import install_adapters
from .publication import (
    _descriptor,
    _directory_state,
    _finalization_session,
    _pending,
    _plan_digest,
    _publication_targets,
    _rollback_sources,
)
from .restore_plan import RestorePlan, recheck_targets
from .sqlite_validation import validated_schema_version


def _checked_originals(plan, journal, session):
    """Reprove local reviewed mappings; a damaged selector is never reparsed."""
    from .bootstrap import _records, _registry
    from .storage_admission import _contains_owned_path

    if (
        type(plan) is not RestorePlan
        or plan.mode != "replace"
        or type(plan.target) is not Inventory
    ):
        raise ValueError("rollback_plan_required")
    with journal._locked(exclusive=True) as parent:
        records = journal._records(parent)
        if (
            not records
            or records[-1].event != "prepared"
            or records[0].event != "candidate_staged"
        ):
            raise ValueError("rollback_preparation_required")
        prepared = _Prepared.model_validate(records[-1].evidence)
        receipt = _CandidateReceipt.model_validate(records[0].evidence)
        context = prepared.publication
        if (
            context is None
            or context.plan_digest != _plan_digest(plan)
            or receipt.plan_digest != context.plan_digest
            or context.descriptor != receipt.descriptor
        ):
            raise ValueError("rollback_plan_changed")
        _finalization_session(session, context, prepared)
        _pending(journal, context, targets=_publication_targets(prepared), durable=True)
        if not _matches(receipt.descriptor, receipt.descriptor.path):
            raise ValueError("candidate_receipt_changed")
        recheck_targets(plan)
        previous = [row for row in prepared.artifacts if row.previous is not None]
        for row in previous:
            if not _matches(row.previous_metadata, row.target, metadata=True):
                raise ValueError("rollback_source_changed")
        for row in prepared.directory_metadata:
            if _directory_state(row.previous.path) != row.previous:
                raise ValueError("rollback_source_changed")
        _, profiles = _records(Path(context.bootstrap_root))
        registry = _registry(Path(context.bootstrap_root))
        bindings = [
            row
            for row in profiles
            if row["selector"] in context.selectors
            and set(row["namespaces"]) <= set(context.namespaces)
        ]
        if not bindings or any(
            row["roots"]
            != sorted(
                {path for name in row["namespaces"] for path in registry[name]["roots"]}
            )
            for row in bindings
        ):
            raise ValueError("rollback_local_binding_required")
        entries = []
        for item in plan.target.items:
            if item.path is None or not (
                any(
                    item.path == Path(row.target)
                    or Path(row.target) in item.path.parents
                    for row in previous
                )
                or any(
                    item.path == Path(row.previous.path)
                    for row in prepared.directory_metadata
                )
            ):
                continue
            if not any(
                _contains_owned_path(Path(root), item.path)
                for binding in bindings
                for root in binding["roots"]
            ):
                raise ValueError("rollback_source_binding_unverified")
            if item.status not in {
                "included",
                "included_directory",
            } and item.owner not in {"sqlite.transient", "rag.projections"}:
                raise ValueError("rollback_original_unavailable")
            entries.append(item)
        if not entries:
            raise ValueError("rollback_originals_required")
        paths = {item.path for item in entries}
        for row in previous:
            if Path(row.target) not in paths:
                raise ValueError("rollback_inventory_incomplete")
            if row.previous.kind == "directory":
                for directory, names, files in os.walk(row.target, followlinks=False):
                    if any(
                        Path(directory) / name not in paths for name in (*names, *files)
                    ):
                        raise ValueError("rollback_inventory_incomplete")
        owners = {owner.owner_id: owner for owner in install_adapters()}
        expected = _rollback_sources(
            plan, [row.model_dump() for row in prepared.artifacts], owners
        )
        if expected != [row.model_dump() for row in prepared.rollback_sources]:
            raise ValueError("rollback_owner_mapping_changed")
        from .projection_publication import normalized_originals

        entries = normalized_originals(
            Inventory(tuple(entries), True, plan.target_fingerprint, ())
        ).items
        directories = {
            item.path: item for item in entries if item.status == "included_directory"
        }
        normalized = []
        for item in entries:
            info = item.path.lstat()
            ancestors = [
                path
                for path in directories
                if path == item.path or path in item.path.parents
            ]
            meta = None
            if item.owner == "rag.projections":
                meta = replace(
                    item.metadata,
                    mode=stat.S_IMODE(info.st_mode),
                    mtime_ns=info.st_mtime_ns,
                )
            elif ancestors:
                root = min(ancestors, key=lambda path: len(path.parts))
                meta = FileMetadata(
                    1,
                    directories[root].logical_id,
                    str(item.path.relative_to(root)) if item.path != root else "",
                    directories[item.path.parent].logical_id
                    if item.path.parent in directories
                    else None,
                    "directory" if stat.S_ISDIR(info.st_mode) else "file",
                    stat.S_IMODE(info.st_mode),
                    info.st_mtime_ns,
                    "private",
                )
            normalized.append(replace(item, metadata=meta))
        journal._flush_records(parent)
        return prepared, Inventory(tuple(normalized), True, plan.target_fingerprint, ())


def _copy_verified_payload(archive, payload, destination, cancel):
    with (
        zipfile.ZipFile(archive.path) as source,
        source.open(payload.payload) as stream,
        create_private_file(destination) as fd,
    ):
        total = 0
        digest = hashlib.sha256()
        while chunk := stream.read(64 * 1024):
            reader._check(cancel)
            total += len(chunk)
            if total > payload.size:
                raise ValueError("rollback_payload_changed")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(fd, view)
                if not written:
                    raise OSError("rollback_write_failed")
                view = view[written:]
        if total != payload.size or digest.hexdigest() != payload.sha256:
            raise ValueError("rollback_payload_changed")


def capture_verify_rollback(
    candidate: Path,
    plan: RestorePlan,
    journal,
    destination: Path,
    *,
    session,
    password: bytes,
    work_root: Path,
    cancel: Event,
    acknowledged_credential_issues: tuple[str, ...],
) -> Path:
    """Capture, encrypt and verify exact local originals under retained admission.

    The caller retains bootstrap.unbound and all affected namespace leases through
    subsequent publication/finalization. No caller receipt or owner coverage map is
    accepted; acknowledged credential omissions cannot mask a storage failure.
    """
    from .space import require_capacity
    from .storage_admission import copy_capture_file

    if type(password) is not bytes or not password:
        raise ValueError("rollback_password_required")
    if type(acknowledged_credential_issues) is not tuple or any(
        type(issue) is not str for issue in acknowledged_credential_issues
    ):
        raise ValueError("rollback_credential_acknowledgements_invalid")
    reader._check(cancel)
    prepared, inventory = _checked_originals(plan, journal, session)
    if prepared.publication.descriptor.path != str(candidate / "candidate.json"):
        raise ValueError("candidate_receipt_changed")
    limits = ArchiveLimits()
    work_root = Path(work_root)
    destination = Path(destination)
    from .bootstrap import _overlap
    from .profile_paths import lexical_path

    work_root = lexical_path(work_root)
    private_roots = tuple(
        Path(path) for path in _descriptor(candidate, plan)["private_roots"]
    )
    protected = (
        *session._all_roots,
        session._control.parent,
        journal.root,
        candidate,
        *private_roots,
    )
    if any(_overlap(work_root, root) for root in protected):
        raise ValueError("rollback_workspace_overlaps_source")
    destination = lexical_path(destination)
    if any(_overlap(destination, root) for root in protected):
        raise ValueError("rollback_output_overlaps_source")
    create_private_directory(work_root)
    stage = work_root / ("originals-" + uuid4().hex)
    create_private_directory(stage)
    create_private_directory(stage / "payload")
    owners = {owner.owner_id: owner for owner in install_adapters()}
    staged, aliases, versions, physical = [], {}, {}, {}
    captured = {}
    raw_originals = {}
    entries = [item for item in inventory.items if item.status == "included"]
    groups = []
    with session._replacement_capture_scope(
        plan, journal, stage, limits=limits, byte_budget=limits.expanded_bytes
    ):
        for item in entries:
            reader._check(cancel)
            path = (
                stage / "payload" / hashlib.sha256(item.logical_id.encode()).hexdigest()
            )
            adapter = owners.get(item.owner)
            if adapter is None:
                raise ValueError("rollback_owner_unavailable")
            info = item.path.stat()
            key = (info.st_dev, info.st_ino)
            if key in physical:
                other, previous = physical[key]
                if not item.shared_group or previous.shared_group != item.shared_group:
                    raise ValueError("undeclared_alias")
                # Alias bytes are operation-private, with the same installed role.
                from .staging import _copy

                _copy(other, path, cancel)
            elif item.owner == "config":
                copy_capture_file(
                    "config", item.path, path, cancel, max_bytes=16 * 1024**2
                )
            else:
                adapter.capture(item, path, cancel)
            physical[key] = path, item
            validator = _item_validator(adapter, item)
            policy = validator.schema_policy()
            if policy is not None and policy.schema_sql:
                version = validated_schema_version(validator, path, cancel)
                if item.owner in versions and versions[item.owner] != version:
                    raise ValueError("rollback_mixed_owner_versions")
                versions[item.owner] = version
            elif item.owner != "config":
                issues = validator.validate(path)
                if issues:
                    raise ValueError(issues[0])
            captured[item.logical_id] = (
                path.stat().st_size,
                reader._hash(path, cancel),
            )
            if policy is None or not policy.schema_sql:
                raw_originals[item.logical_id] = observe_artifact(
                    item.path, metadata=True
                )
            staged.append((item, path))
            if item.shared_group:
                aliases.setdefault(item.shared_group, []).append(item.logical_id)
            total = sum(path.stat().st_size for _, path in staged)
            if (
                total > limits.expanded_bytes
                or path.stat().st_size > limits.member_bytes
            ):
                raise ValueError("rollback_capture_limit")
            require_capacity({stage: total, destination.parent: total * 5})
        from .rag_projection_validation import validate_groups

        candidates = {item.logical_id: path for item, path in staged}
        validate_groups(
            inventory.items, candidates, stage, cancel, limits, limits.expanded_bytes
        )
        rebound = replace(
            inventory, items=tuple(replace(item, path=path) for item, path in staged)
        )
        issues = process_credentials(stage, rebound, mode="rollback", encrypted=True)
        if set(issues) != set(acknowledged_credential_issues):
            raise ValueError("rollback_credential_coverage_changed")
        material = stage / "credential-recovery.json"
        if not material.is_file():
            raise ValueError("rollback_credential_material_missing")
        _material(stage)
        from .staging import _copy

        material_payload = stage / "payload" / "credential-recovery.json"
        _copy(material, material_payload, cancel)
        staged.append(
            (
                StorageItem(
                    "recovery.credentials", "credentials", material, "included", ()
                ),
                material_payload,
            )
        )
        source_owners = {
            source.logical_id: source.owner_id for source in prepared.rollback_sources
        }
        sidecar_owners = {
            item.logical_id: source_owners[item.dependencies[0]]
            for item in inventory.items
            if item.owner == "sqlite.transient"
        }
        archive_inventory = replace(
            inventory,
            items=tuple(
                replace(item, owner=sidecar_owners[item.logical_id])
                if item.logical_id in sidecar_owners
                else item
                for item in inventory.items
            ),
        )
        encoded = _manifest_for(
            archive_inventory,
            staged,
            aliases,
            {
                "root": stage,
                "cancel": cancel,
                "mode": "rollback",
                "encrypted": True,
                "versions": versions,
                "limits": limits,
            },
            (),
        )
        document = json.loads(encoded)
        document["report"]["lines"].extend(sorted(issues))
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        captured_doc = reader._manifest(encoded, limits, True)
        if captured_doc.consistency != "coherent":
            raise ValueError("rollback_storage_incoherent")
        from .publication import _archive_object

        files = {row.logical_id: row for row in captured_doc.files}
        for key, expected in captured.items():
            payload = files[key]
            if (payload.size, payload.sha256) != expected:
                raise ValueError("rollback_snapshot_changed")
        for key, original in raw_originals.items():
            if _archive_object(captured_doc, key, metadata=True) != (
                original["size"],
                original["sha256"],
            ):
                raise ValueError("rollback_raw_original_changed")
        directories = {row.logical_id: row for row in captured_doc.directories}
        for item in inventory.items:
            if item.status == "included_directory":
                archived = directories[item.logical_id]
                if (archived.metadata.mode, archived.metadata.mtime_ns) != (
                    item.metadata.mode,
                    item.metadata.mtime_ns,
                ):
                    raise ValueError("rollback_directory_metadata_changed")
        encoded = json.dumps(
            captured_doc.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        capture = CaptureResult(stage, archive_inventory, encoded)
        write_archive(capture, destination, password=password, cancel=cancel)
        ciphertext = observe_artifact(destination)
        archive = reader.acquire(
            destination, work_root / "verify", limits, password, cancel
        )
        doc = reader.verify_sealed(archive, cancel)
        if (
            archive.manifest_bytes != encoded
            or doc.consistency != "coherent"
            or doc.credential_policy != "rollback"
        ):
            raise ValueError("rollback_manifest_changed")
        by_id = {row.logical_id: row for row in doc.files}
        verified = stage / "verified"
        create_private_directory(verified)
        for source in prepared.rollback_sources:
            payload = by_id.get(source.logical_id)
            if payload is None or payload.owner_id != source.owner_id:
                raise ValueError("rollback_sqlite_coverage_mismatch")
            private = verified / hashlib.sha256(source.logical_id.encode()).hexdigest()
            _copy_verified_payload(archive, payload, private, cancel)
            version = validated_schema_version(owners[source.owner_id], private, cancel)
            if version != versions[source.owner_id]:
                raise ValueError("rollback_sqlite_schema_changed")
            groups.append(
                {
                    **source.model_dump(),
                    "schema_version": version,
                    "payload_size": payload.size,
                    "payload_digest": payload.sha256,
                }
            )
        projection_candidates = {}
        for item in inventory.items:
            if item.owner != "rag.projections" or item.status != "included":
                continue
            payload = by_id.get(item.logical_id)
            if payload is None or payload.owner_id != item.owner:
                raise ValueError("rollback_projection_coverage_mismatch")
            private = verified / hashlib.sha256(item.logical_id.encode()).hexdigest()
            _copy_verified_payload(archive, payload, private, cancel)
            projection_candidates[item.logical_id] = private
        validate_groups(
            inventory.items,
            projection_candidates,
            verified,
            cancel,
            limits,
            limits.expanded_bytes,
        )
        reader.verify_sealed(archive, cancel)
    # Positive scope retirement (including source/WAL rechecks) precedes proof.
    checked, _ = _checked_originals(plan, journal, session)
    if checked != prepared or observe_artifact(destination) != ciphertext:
        raise ValueError("rollback_capture_changed")
    coverage = {}
    by_path = {item.path: item for item in inventory.items}
    sidecars = {
        item.logical_id: item.dependencies[0]
        for item in inventory.items
        if item.owner == "sqlite.transient"
    }
    for row in prepared.artifacts:
        if row.previous is not None:
            item = by_path[Path(row.target)]
            coverage[row.logical_id] = sidecars.get(item.logical_id, item.logical_id)
    coverage.update(
        {
            row.logical_id: by_path[Path(row.previous.path)].logical_id
            for row in prepared.directory_metadata
        }
    )
    with journal._locked(exclusive=True) as parent:
        records = journal._records(parent)
        if (
            records[-1].event != "prepared"
            or _Prepared.model_validate(records[-1].evidence) != prepared
        ):
            raise ValueError("rollback_preparation_changed")
        _finalization_session(session, prepared.publication, prepared)
        _pending(journal, prepared.publication, durable=True)
        reader._check(cancel)
        journal._append(
            parent,
            "rollback_verified",
            {
                "ciphertext": ciphertext,
                "sealed_digest": archive.digest,
                "manifest_digest": hashlib.sha256(encoded).hexdigest(),
                "coverage": coverage,
                "sqlite_groups": groups,
                "projection_groups": list(
                    {
                        row.path: row.model_dump()
                        for artifact in prepared.artifacts
                        for row in artifact.rollback_projection_roots
                    }.values()
                ),
                "credential_issues": sorted(set(issues)),
            },
        )
        journal._flush_records(parent)
    return destination
