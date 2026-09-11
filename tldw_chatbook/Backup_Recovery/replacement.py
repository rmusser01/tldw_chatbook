"""Held original capture and encrypted rollback verification for replacement.

Replacement and interrupted recovery compose the existing held capture, credential,
publication and activation operations with authenticated local rollback evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from contextlib import contextmanager
from dataclasses import replace as _replace
from pathlib import Path
from threading import Event
from uuid import uuid4

from . import archive_reader as reader
from .archive_writer import write_archive
from .capture import (
    CaptureResult,
    CaptureReviewRequired,
    _item_validator,
    _manifest_for,
)
from .credentials import _material, process_credentials
from .journal import _CandidateReceipt, _matches, _Prepared, observe_artifact
from .limits import ArchiveLimits
from .models import FileMetadata, Inventory, StorageItem
from .native_files import (
    create_private_directory,
    create_private_file,
    pinned_directory,
)
from .owner_registry import install_adapters
from .publication import (
    _descriptor,
    _directory_state,
    _finalization_session,
    _pending,
    _plan_digest,
    _publication_scope_path,
    _publication_targets,
    _rollback_sources,
    _safety_source_matches,
)
from .restore_plan import RestorePlan, recheck_targets
from .sqlite_validation import validated_schema_version


class RollbackCredentialReviewRequired(CaptureReviewRequired):
    """Actual private capture omissions need renewed review before retry."""

    def __init__(self, issues):
        super().__init__(tuple(sorted(set(issues))))
        self.args = ("rollback_credential_coverage_changed",)


def _validate_builtin_safety(items, candidates, owners, plan):
    """Validate finite builtin roots against the actual captured core and files."""
    from tldw_chatbook.Persona_Visual.recovery import _Assets

    for item in items:
        if (
            item.owner == "persona.visual_identity_builtin"
            and item.status == "included_directory"
            and item.metadata is not None
            and item.metadata.parent_id is None
        ):
            owner = owners.get(item.owner)
            if type(owner) is not _Assets:
                raise ValueError("rollback_owner_unavailable")
            from .later_rollback import validate_created_builtin_capture

            if validate_created_builtin_capture(plan, item, items, candidates):
                continue
            issues = owner.validate_dependencies(item, item.path, candidates)
            if issues:
                raise ValueError(issues[0])


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
        if {row.logical_id for row in prepared.safety_sources} != set(
            plan.safety_scope
        ) or any(
            not _safety_source_matches(row.source)
            for row in prepared.safety_sources
        ):
            raise ValueError("rollback_safety_source_changed")
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
                or item.logical_id in plan.safety_scope
                or any(
                    item.path == Path(row.previous.path)
                    for row in prepared.directory_metadata
                )
            ):
                continue
            if not any(
                _contains_owned_path(
                    Path(root), _publication_scope_path(item.path, items=plan.target.items)
                )
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
                meta = _replace(
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
            normalized.append(_replace(item, metadata=meta))
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
        _validate_builtin_safety(inventory.items, candidates, owners, plan)
        rebound = _replace(
            inventory, items=tuple(_replace(item, path=path) for item, path in staged)
        )
        issues = process_credentials(stage, rebound, mode="rollback", encrypted=True)
        if set(issues) != set(acknowledged_credential_issues):
            raise RollbackCredentialReviewRequired(issues)
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
        archive_inventory = _replace(
            inventory,
            items=tuple(
                _replace(item, owner=sidecar_owners[item.logical_id])
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
        verified_candidates = {}
        for source in prepared.rollback_sources:
            payload = by_id.get(source.logical_id)
            if payload is None or payload.owner_id != source.owner_id:
                raise ValueError("rollback_sqlite_coverage_mismatch")
            private = verified / hashlib.sha256(source.logical_id.encode()).hexdigest()
            _copy_verified_payload(archive, payload, private, cancel)
            verified_candidates[source.logical_id] = private
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
        for item in inventory.items:
            if item.owner != "persona.visual_identity_builtin" or item.status != "included":
                continue
            payload = by_id.get(item.logical_id)
            if payload is None or payload.owner_id != item.owner:
                raise ValueError("rollback_builtin_coverage_mismatch")
            private = verified / hashlib.sha256(item.logical_id.encode()).hexdigest()
            _copy_verified_payload(archive, payload, private, cancel)
            verified_candidates[item.logical_id] = private
        _validate_builtin_safety(inventory.items, verified_candidates, owners, plan)
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
                "safety_sources": [row.model_dump() for row in prepared.safety_sources],
            },
        )
        journal._flush_records(parent)
    return destination


def require_rollback_password(password: bytes) -> None:
    """A supplied local password is necessary, never sufficient publication proof."""
    if type(password) is not bytes or not password:
        raise ValueError("rollback_password_required")


def _acquired_source(candidate: Path, plan: RestorePlan, cancel: Event):
    """Recheck the real stage producer's local source; never guess an input name."""
    from .archive_models import EncryptedSource, SealedArchive

    document = _descriptor(candidate, plan)
    source = document.get("archive_source")
    if not isinstance(source, dict) or set(source) != {
        "path",
        "digest",
        "manifest_digest",
        "encrypted_source",
    }:
        raise ValueError("candidate_acquisition_required")
    path = Path(source["path"])
    if (
        not path.is_absolute()
        or ".." in path.parts
        or source["digest"] != plan.archive_digest
    ):
        raise ValueError("candidate_acquisition_changed")
    if reader._hash(path, cancel) != source["digest"]:
        raise ValueError("candidate_acquisition_changed")
    encrypted = source["encrypted_source"]
    manifest = reader._inspect(
        path, ArchiveLimits(), encrypted is not None, cancel, source["digest"]
    )
    if hashlib.sha256(manifest).hexdigest() != source["manifest_digest"]:
        raise ValueError("candidate_acquisition_changed")
    provenance = None
    if encrypted is not None:
        if not isinstance(encrypted, dict) or set(encrypted) != {
            "path",
            "digest",
            "identity",
            "plaintext_digest",
            "manifest_digest",
        }:
            raise ValueError("candidate_acquisition_changed")
        provenance = EncryptedSource(
            Path(encrypted["path"]),
            encrypted["digest"],
            tuple(encrypted["identity"]),
            encrypted["plaintext_digest"],
            encrypted["manifest_digest"],
        )
    archive = SealedArchive(path, source["digest"], manifest, provenance)
    reader.verify_sealed(archive, cancel)
    return archive


def _first_binding_inventory(plan, selector):
    """Rediscover one actual current profile, never an imported source locator."""
    from .inventory import discover
    from .storage_admission import _preview_reads

    recheck_targets(plan)
    install_adapters()
    with _preview_reads():
        current = discover((selector,))
    if not current.complete:
        raise ValueError("replacement_current_scope_unavailable")
    approved = {item.logical_id: item for item in plan.target.items}
    for item in current.items:
        # These are independently recognized local protocol records, never payload.
        if item.owner == "recovery.control" and item.status == "intentionally_excluded":
            continue
        previous = approved.get(item.logical_id)
        if previous is None or (
            item.owner, item.path, item.status, item.dependencies
        ) != (
            previous.owner, previous.path, previous.status, previous.dependencies
        ):
            raise ValueError("replacement_current_scope_changed")
    recheck_targets(plan)
    return current


def _ensure_first_bindings(plan, selectors, root, cancel):
    """Compose first ordinary binding under actual unbound/source retirement."""
    from . import bootstrap
    from .capture_service import _capture_names
    from .control_records import UNBOUND_NAMESPACE, admission_authority, bind_profile

    pending, profiles = bootstrap._records(root)
    if pending:
        raise ValueError("recovery_pending")
    existing = {Path(row["selector"]): row for row in profiles}
    missing = tuple(path for path in selectors if path not in existing)
    if not missing:
        return
    authority = admission_authority(root)
    names_by_selector = {}
    inventories = {}
    for selector in missing:
        reader._check(cancel)
        inventory = _first_binding_inventory(plan, selector)
        inventories[selector] = inventory
        names_by_selector[selector] = tuple(
            name for name in _capture_names(authority, inventory)
            if name != UNBOUND_NAMESPACE
        )
    names = tuple(sorted({UNBOUND_NAMESPACE, *(name for group in names_by_selector.values() for name in group)}))
    registry = bootstrap._registry(root)
    if registry is None or any(name not in registry for name in names):
        raise ValueError("replacement_binding_changed")
    expected_registry = {name: registry[name] for name in names}
    with authority.maintenance(names, 30, cancel=cancel) as session:
        reader._check(cancel)
        pending, current_profiles = bootstrap._records(root)
        if pending or any(Path(row["selector"]) in missing for row in current_profiles):
            raise ValueError("replacement_binding_changed")
        # Reuse independently discovered footprints: rediscovery opens SQLite
        # readers and can change transient SHM after the approved observation.
        # Native registry locks and real sources remain held through recheck/write.
        registry = bootstrap._registry(root)
        if registry is None or any(
            registry.get(name) != entry for name, entry in expected_registry.items()
        ):
            raise ValueError("replacement_binding_changed")
        recheck_targets(plan)
        from .storage_admission import _contains_owned_path

        for selector in missing:
            roots = tuple(
                Path(path)
                for name in names_by_selector[selector]
                for path in registry[name]["roots"]
            )
            if any(
                not any(_contains_owned_path(root, item.path) for root in roots)
                for item in inventories[selector].items
                if item.path is not None
                and item.status in {"included", "included_directory"}
            ):
                raise ValueError("replacement_current_scope_changed")
            reader._check(cancel)
            bind_profile(
                root, selector, names_by_selector[selector], root / "admission",
                session=session,
            )
        recheck_targets(plan)


def _selected_config_container(plan, document, key, destination, registry, profiles):
    """Verify the one mapped config root and its actual local source footprint."""
    from .bootstrap import _overlap

    selected = dict(plan.restore)
    roots = {root for root, path in plan.destinations if path == destination}
    if key not in roots:
        return False
    configs = [
        row
        for row in document.files
        if row.root_id in roots and row.owner_id == "config"
    ]
    if len(configs) != 1:
        return False
    config = configs[0]
    selector = selected.get(config.logical_id)
    current = next((row for row in profiles if Path(row["selector"]) == selector), None)
    if (
        current is None
        or selector.parent != destination
        or any(
            Path(row["selector"]) != selector
            and _overlap(destination, Path(row["selector"]))
            for row in profiles
        )
    ):
        return False
    producers = {row.logical_id: row for row in document.producer_inventory}
    config_keys = {row.logical_id for row in document.files if row.owner_id == "config"}

    def config_dependencies(key):
        pending, visited, result = [key], set(), set()
        while pending:
            value = pending.pop()
            if value in visited:
                continue
            visited.add(value)
            if value in config_keys:
                result.add(value)
            elif value in producers:
                pending.extend(producers[value].dependencies)
        return result

    synthetic = {row.logical_id for row in document.directories if row.synthetic}
    members = [
        row
        for row in (*document.files, *document.directories)
        if row.root_id in roots and row.logical_id not in synthetic
    ]
    if {row.root_id for row in members} != roots or any(
        row.logical_id not in selected
        or config_dependencies(row.logical_id) != {config.logical_id}
        for row in members
    ):
        return False
    actual_paths = {
        item.path
        for item in plan.target.items
        if item.status in {"included", "included_directory"}
    }
    return all(
        name in current["namespaces"]
        and Path(path) in actual_paths
        and not any(name in row["namespaces"] for row in profiles if row is not current)
        for name, entry in registry.items()
        for path in entry["roots"]
        if _overlap(destination, Path(path))
    )


def _register_publication_parents(plan, authority, protected, *, document):
    """Fence only selected existing private parents of new mapped destinations."""
    from . import bootstrap
    from .restore_plan import _ancestor

    registry = bootstrap._registry(authority.control_root.parent)
    _, profiles = bootstrap._records(authority.control_root.parent)
    roots = [Path(path) for entry in registry.values() for path in entry["roots"]]
    for key, destination in plan.destinations:
        paths = [
            path
            for _, path in plan.restore
            if path == destination or destination in path.parents
        ]
        if all(
            any(root == path or root in path.parents for root in roots)
            for path in paths
        ):
            continue
        parent = _ancestor(destination)
        config_container = destination.is_dir() and _selected_config_container(
            plan, document, key, destination, registry, profiles
        )
        if any(bootstrap._overlap(parent, path) for path in protected) or (
            not config_container
            and (
                destination.exists()
                or parent == destination
                or any(bootstrap._overlap(parent, path) for path in roots)
            )
        ):
            raise ValueError("replacement_destination_parent_required")
        with pinned_directory(parent) as fd:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise ValueError("replacement_destination_parent_required")
        name = (
            "replacement.destination."
            + hashlib.sha256(str(parent).encode()).hexdigest()
        )
        authority.register(name, (parent,))
        roots.append(parent)
    recheck_targets(plan)


def replace(
    plan: RestorePlan,
    candidate: Path,
    *,
    control_root: Path,
    rollback_password: bytes,
    cancel: Event,
) -> str:
    """Replace reviewed data under one uninterrupted native maintenance session.

    Interrupted publication retains the exact journal and pending fence. A failed
    installed validation reverses provable originals while this session stays held.
    """
    from . import bootstrap
    from .control_records import (
        UNBOUND_NAMESPACE,
        admission_authority,
        register_pending,
    )
    from .journal import Journal
    from .profile_paths import lexical_path
    from .publication import finalize_candidate, publish_candidate
    from .space import require_capacity

    require_rollback_password(rollback_password)
    reader._check(cancel)
    if type(plan) is not RestorePlan or plan.mode != "replace" or plan.target is None:
        raise ValueError("replacement_plan_required")
    archive = _acquired_source(candidate, plan, cancel)
    descriptor = _descriptor(candidate, plan)
    if not set(descriptor.get("credential_issues", ())) <= set(
        plan.acknowledged_credential_issues
    ):
        raise ValueError("credential_omission_acknowledgement_required")
    recheck_targets(plan)
    control_root = lexical_path(control_root)
    root = bootstrap.default_bootstrap_root()
    document = reader.verify_sealed(archive, cancel)
    config_keys = {row.logical_id for row in document.files if row.owner_id == "config"}
    config_destinations = [path for key, path in plan.restore if key in config_keys]
    if len(config_destinations) != len(set(config_destinations)):
        raise ValueError("replacement_config_mapping_ambiguous")
    selectors = tuple(
        sorted(
            {
                item.path
                for item in plan.target.items
                if item.owner == "config"
                and item.path in config_destinations
            }
        )
    )
    if not selectors:
        raise ValueError("replacement_local_binding_required")
    _ensure_first_bindings(plan, selectors, root, cancel)
    _, profiles = bootstrap._records(root)
    registry = bootstrap._registry(root)
    selected = [row for row in profiles if Path(row["selector"]) in selectors]
    if len(selected) != len(selectors) or registry is None:
        raise ValueError("replacement_local_binding_required")
    if any(
        any(name not in registry for name in row["namespaces"])
        or row["roots"] != sorted({path for name in row["namespaces"] for path in registry[name]["roots"]})
        for row in selected
    ):
        raise ValueError("replacement_binding_changed")
    authority = admission_authority(root)
    _register_publication_parents(
        plan, authority, (root, control_root, candidate, archive.path), document=document
    )
    registry = bootstrap._registry(root)
    affected = [path for _, path in (*plan.restore, *plan.retire)]
    affected += [
        item.path for item in plan.target.items if item.logical_id in plan.safety_scope
    ]
    names = tuple(
        sorted(
            {name for row in selected for name in row["namespaces"]}
            | {
                name
                for name, entry in registry.items()
                if any(
                    bootstrap._overlap(Path(bound), path)
                    for bound in entry["roots"]
                    for path in affected
                )
            }
        )
    )
    if any(name not in registry for name in names) or any(
        not any(
            Path(bound) == path or Path(bound) in path.parents
            for name in names
            for bound in registry[name]["roots"]
        )
        for original in affected
        for path in (_publication_scope_path(original, items=plan.target.items),)
    ):
        raise ValueError("replacement_scope_uncovered")
    protected = [Path(bound) for name in names for bound in registry[name]["roots"]]
    if any(
        bootstrap._overlap(control_root, path)
        for path in (*protected, candidate, archive.path)
    ):
        raise ValueError("replacement_control_overlap")
    require_capacity(
        {
            control_root: sum(
                path.stat().st_size for path in affected if path.is_file()
            )
            * 5
        }
    )
    if not control_root.exists():
        create_private_directory(control_root)
    operation = uuid4().hex
    journal = Journal(control_root, operation)
    work = control_root / ("replacement-" + operation)
    create_private_directory(work)
    incoming = None
    if reader.verify_sealed(archive, cancel).credential_policy != "exclude":
        if archive.encrypted_source is None:
            raise ValueError("encrypted_acquisition_required")
        require_capacity({work: archive.encrypted_source.identity[2]})
        proof = reader.retain_encrypted(archive, work / "credentials.age", cancel)
        if reader._hash(proof.path, cancel) != proof.digest:
            raise ValueError("encrypted_retention_changed")
        ciphertext = observe_artifact(proof.path)
        if reader._identity(proof.path.stat(follow_symlinks=False)) != proof.identity:
            raise ValueError("encrypted_retention_changed")
        incoming = {
            "ciphertext": ciphertext,
            "plaintext_digest": proof.plaintext_digest,
            "manifest_digest": proof.manifest_digest,
        }
    journal.record_candidate(candidate, plan, archive)
    register_pending(root, operation, names, control_root, selectors)
    authority = admission_authority(root)
    with authority.maintenance((*names, UNBOUND_NAMESPACE), 30) as session:
        reader._check(cancel)
        reader.verify_sealed(archive, cancel)
        recheck_targets(plan)
        identities = tuple(
            {"config": str(selector), "installation_id": uuid4().hex}
            for selector in selectors
        )
        journal.prepare_publication(
            candidate,
            plan,
            bootstrap_root=root,
            namespaces=names,
            selectors=selectors,
            generation=operation,
            replacement_profiles=identities,
            incoming_credentials=incoming,
        )
        rollback = capture_verify_rollback(
            candidate,
            plan,
            journal,
            work / "rollback.tldw-backup.zip.age",
            session=session,
            password=rollback_password,
            work_root=work / "capture",
            cancel=cancel,
            acknowledged_credential_issues=tuple(
                issue
                for issue in plan.acknowledged_credential_issues
                if issue != "credential_isolated_retention_required"
            ),
        )
        reader._check(cancel)
        _apply_replacement_credentials(
            candidate, journal, session=session, cancel=cancel
        )
        reader._check(cancel)
        publish_candidate(candidate, plan, journal, rollback, session=session)
        try:
            finalize_candidate(candidate, plan, journal, session=session)
        except (OSError, ValueError, RuntimeError):
            with journal._locked(exclusive=False) as parent:
                rows = journal._records(parent)
            if any(row.event == "committed" for row in rows):
                raise
            prepared = _Prepared.model_validate(
                next(row.evidence for row in rows if row.event == "prepared")
            )
            # After live publication, complete a provable reversal under this same
            # held session; cancellation never abandons an in-flight native move.
            recovery_cancel = Event()
            from .control_records import _recover_activation_pairs

            with _unlock_recovery(
                journal, prepared, rollback_password, session, recovery_cancel
            ) as check_credentials:
                _recover_activation_pairs(journal, prepared, session)
                _rollback_replacement(
                    journal, prepared, session, check_credentials, recovery_cancel
                )
            raise ValueError("replacement_rolled_back") from None
    return operation


def _verify_applied_credentials(candidate, journal, *, session, records):
    """Recheck actual material and values under the caller's held journal lock."""
    from .credentials import (
        verify_replacement_credential,
    )
    from .journal import _evidence_digest, _Object

    prepared = _Prepared.model_validate(
        next(row.evidence for row in records if row.event == "prepared")
    )
    _finalization_session(session, prepared.publication, prepared)
    if not prepared.credential_scopes:
        return
    intent = next((row for row in records if row.event == "credential_intended"), None)
    complete = next(
        (row for row in records if row.event == "credentials_completed"), None
    )
    if (
        intent is None
        or complete is None
        or not _matches(
            _Object.model_validate(intent.evidence["material"]),
            str(candidate / "credential-recovery.json"),
        )
    ):
        raise ValueError("credential_application_incomplete")
    material = _read_credential_records(candidate, prepared, session)
    observed = {
        key: verify_replacement_credential(
            record, json.loads(prepared.credential_scopes[key])
        )
        for key, record in material.items()
    }
    if complete.evidence != {
        "intent_digest": _evidence_digest(intent.evidence),
        "applied_digest": _evidence_digest(observed),
    }:
        raise ValueError("credential_application_changed")


def _apply_replacement_credentials(candidate, journal, *, session, cancel):
    """Durable intent precedes every actual keyring write; progress is per value."""
    from .credentials import (
        apply_replacement_credential,
        check_replacement_credential,
        verify_replacement_credential,
    )
    from .journal import _evidence_digest, _Object

    with journal._locked(exclusive=True) as parent:
        records = journal._records(parent)
        prepared = _Prepared.model_validate(
            next(row.evidence for row in records if row.event == "prepared")
        )
        _finalization_session(session, prepared.publication, prepared)
        if not any(row.event == "rollback_verified" for row in records):
            raise ValueError("rollback_required")
        if not prepared.credential_scopes:
            return
        if any(row.event == "credentials_completed" for row in records):
            _verify_applied_credentials(
                candidate, journal, session=session, records=records
            )
            return
        intent = next(
            (row for row in records if row.event == "credential_intended"), None
        )
        material_path = candidate / "credential-recovery.json"
        if intent and not _matches(
            _Object.model_validate(intent.evidence["material"]), str(material_path)
        ):
            raise ValueError("credential_material_changed")
        material = _read_credential_records(candidate, prepared, session)
        plans = {
            key: json.loads(value) for key, value in prepared.credential_scopes.items()
        }
        for key, record in material.items():
            check_replacement_credential(
                record, plans[key], allow_existing=intent is not None
            )
        if intent is None:
            journal._append(
                parent,
                "credential_intended",
                {
                    "plan_digest": prepared.publication.plan_digest,
                    "descriptor_digest": prepared.publication.descriptor.sha256,
                    "material": observe_artifact(material_path),
                    "scopes": prepared.credential_scopes,
                },
            )
            journal._flush_records(parent)
            records = journal._records(parent)
            intent = records[-1]
        applied = {
            row.evidence["record_id"]: row.evidence
            for row in records
            if row.event == "credential_applied"
        }
        for key, record in sorted(material.items()):
            reader._check(cancel)
            _finalization_session(session, prepared.publication, prepared)
            if key in applied:
                if verify_replacement_credential(record, plans[key]) != applied[key]:
                    raise ValueError("credential_application_changed")
                continue
            proof = apply_replacement_credential(record, plans[key])
            journal._append(parent, "credential_applied", proof)
            journal._flush_records(parent)
            applied[key] = proof
        journal._append(
            parent,
            "credentials_completed",
            {
                "intent_digest": _evidence_digest(intent.evidence),
                "applied_digest": _evidence_digest(applied),
            },
        )
        journal._flush_records(parent)


def _read_credential_records(candidate, prepared, session):
    """Read only receipt-bound private material; no live sources enter this scope."""
    from .credentials import replacement_credential_records

    if prepared.credential_material is None or not _matches(
        prepared.credential_material, str(candidate / "credential-recovery.json")
    ):
        raise ValueError("credential_material_changed")
    limits = ArchiveLimits()
    with session._capture_bound_sources((), candidate, limits, limits.expanded_bytes):
        records = replacement_credential_records(candidate, prepared.credential_scopes)
    if not _matches(
        prepared.credential_material, str(candidate / "credential-recovery.json")
    ):
        raise ValueError("credential_material_changed")
    return records


@contextmanager
def _unlock_recovery(journal, prepared, password, session, cancel):
    """Authenticate this exact held rollback archive and expose its captured values."""
    from .journal import _Rollback
    from .space import require_capacity

    require_rollback_password(password)
    with journal._locked(exclusive=False) as parent:
        records = journal._records(parent)
    proof = _Rollback.model_validate(
        next(row.evidence for row in records if row.event == "rollback_verified")
    )
    if not _matches(proof.ciphertext, proof.ciphertext.path):
        raise ValueError("rollback_ciphertext_changed")
    work = journal.root.parent / ("recovery-readback-" + uuid4().hex)
    require_capacity({journal.root.parent: proof.ciphertext.size * 3})
    create_private_directory(work)
    try:
        archive = reader.acquire(
            Path(proof.ciphertext.path),
            work / "acquired",
            ArchiveLimits(),
            password,
            cancel,
        )
        doc = reader.verify_sealed(archive, cancel)
        if (
            archive.digest != proof.sealed_digest
            or hashlib.sha256(archive.manifest_bytes).hexdigest()
            != proof.manifest_digest
            or doc.credential_policy != "rollback"
        ):
            raise ValueError("rollback_archive_changed")
        require_capacity({work: sum(row.size for row in doc.files)})
        view = work / "material"
        create_private_directory(view)
        from .staging import _copy, _mkdirs

        for payload in doc.files:
            destination = view / payload.payload
            _mkdirs(destination.parent, view)
            _copy_verified_payload(archive, payload, destination, cancel)
        _copy(
            view / "payload" / "credential-recovery.json",
            view / "credential-recovery.json",
            cancel,
        )
        if not _matches(proof.ciphertext, proof.ciphertext.path):
            raise ValueError("rollback_ciphertext_changed")
        from .rollback_credentials import RollbackMaterial

        material = RollbackMaterial(archive, view, session, prepared)
        material.records()
        yield material

    finally:
        import shutil

        shutil.rmtree(work)


def _abort_context(journal, plan, records):
    """Bind only an untouched local receipt to its exact registered pending scope."""
    from . import bootstrap
    from .control_records import _Pending
    from .journal import _PublicationContext
    from .native_files import pinned_directory

    events = [row.event for row in records]
    if events and events[-1] == "prepublication_aborted":
        events = events[:-1]
    if events not in (["candidate_staged"], ["candidate_staged", "prepared"]):
        raise ValueError("prepublication_abort_unavailable")
    if plan.mode != "replace" or plan.target is None:
        raise ValueError("replacement_recovery_required")
    root = bootstrap.default_bootstrap_root()
    with pinned_directory(root) as parent:
        pending = _Pending.model_validate(
            bootstrap._read(
                parent, "pending-" + bootstrap._key(journal.operation_id) + ".json"
            )
        )
    _, profiles = bootstrap._records(root)
    registry = bootstrap._registry(root)
    receipt = _CandidateReceipt.model_validate(records[0].evidence)
    with reader._regular(journal.root / "verified-manifest.json") as stream:
        encoded = stream.read(ArchiveLimits().manifest_bytes + 1)
    if len(encoded) > ArchiveLimits().manifest_bytes or hashlib.sha256(encoded).hexdigest() != receipt.manifest_digest:
        raise ValueError("verified_manifest_changed")
    doc = reader._manifest(encoded, ArchiveLimits(), True)
    config_keys = {row.logical_id for row in doc.files if row.owner_id == "config"}
    config_paths = {path for key, path in plan.restore if key in config_keys}
    selectors = sorted(
        {
            str(item.path)
            for item in plan.target.items
            if item.owner == "config" and item.path in config_paths
        }
    )
    selected = [row for row in profiles if row["selector"] in selectors]
    affected = [path for _, path in (*plan.restore, *plan.retire)]
    affected += [
        item.path for item in plan.target.items if item.logical_id in plan.safety_scope
    ]
    if not selectors or len(selected) != len(selectors) or registry is None:
        raise ValueError("replacement_local_binding_required")
    names = sorted(
        {name for row in selected for name in row["namespaces"]}
        | {
            name
            for name, entry in registry.items()
            if any(
                bootstrap._overlap(Path(bound), path)
                for bound in entry["roots"]
                for path in affected
            )
        }
    )
    receipt = _CandidateReceipt.model_validate(records[0].evidence)
    context = _PublicationContext(
        bootstrap_root=str(root),
        namespaces=names,
        selectors=selectors,
        archive_digest=receipt.archive_digest,
        plan_digest=receipt.plan_digest,
        descriptor=receipt.descriptor,
    )
    if pending != _Pending(
        operation_id=journal.operation_id,
        namespaces=names,
        selectors=selectors,
        control_root=str(journal.root.parent),
    ):
        raise ValueError("publication_pending_mismatch")
    if (
        len(events) == 2
        and _Prepared.model_validate(records[1].evidence).publication != context
    ):
        raise ValueError("publication_pending_mismatch")
    return context


def _abort_prepublication(journal, plan, cancel, *, execute):
    """Prove unchanged native targets before recording abort and removing own fence."""
    from . import bootstrap
    from .control_records import UNBOUND_NAMESPACE, _existing_admission_authority
    from .journal import _evidence_digest, _states
    from .native_files import flush_directory, pinned_directory
    from .publication import _check_directory_states
    from .storage_admission import _contains_owned_path

    root = bootstrap.default_bootstrap_root()
    name = "pending-" + bootstrap._key(journal.operation_id) + ".json"
    affected = [path for _, path in (*plan.restore, *plan.retire, *plan.containers)]
    affected += [
        item.path for item in plan.target.items if item.logical_id in plan.safety_scope
    ]
    with journal._locked(exclusive=False) as parent:
        context = _abort_context(journal, plan, journal._records(parent))
    with (
        _existing_admission_authority(root).maintenance(
            (*context.namespaces, UNBOUND_NAMESPACE), 30
        ) as session,
        journal._locked(exclusive=True) as parent,
    ):
        records = journal._records(parent)
        if _abort_context(journal, plan, records) != context:
            raise ValueError("publication_pending_mismatch")
        session._check()
        if session._control != root / "admission" or any(
            not any(
                _contains_owned_path(bound, _publication_scope_path(path, items=plan.target.items))
                for bound in session._roots
            )
            for path in (*affected, *map(Path, context.selectors))
        ):
            raise ValueError("finalization_scope_uncovered")
        receipt = _CandidateReceipt.model_validate(records[0].evidence)
        prepared_row = next((row for row in records if row.event == "prepared"), None)

        def prove():
            reader._check(cancel)
            session._check()
            recheck_targets(plan)
            with pinned_directory(Path(receipt.stage.path)) as stage:
                info = os.fstat(stage)
                if (info.st_dev, info.st_ino) != (
                    receipt.stage.device,
                    receipt.stage.inode,
                ):
                    raise ValueError("candidate_receipt_changed")
            # Preparation adds exact empty retained directories under private roots;
            # its artifact receipts supersede the original whole-stage tree hash.
            if (
                prepared_row is None and not _matches(receipt.stage, receipt.stage.path)
            ) or not _matches(receipt.descriptor, receipt.descriptor.path):
                raise ValueError("candidate_receipt_changed")
            if prepared_row is not None:
                prepared = _Prepared.model_validate(prepared_row.evidence)
                _finalization_session(session, context, prepared)
                _check_directory_states(prepared, records)
                if any(value != "staged" for value in _states(prepared).values()):
                    raise ValueError("prepublication_originals_changed")
            _pending(journal, context, targets=affected, durable=execute)

        prove()
        evidence = {
            "candidate_digest": _evidence_digest(records[0].evidence),
            "prepared_digest": _evidence_digest(prepared_row.evidence)
            if prepared_row
            else None,
            "target_fingerprint": plan.target_fingerprint,
            "publication": context.model_dump(),
        }
        if (
            records[-1].event == "prepublication_aborted"
            and records[-1].evidence != evidence
        ):
            raise ValueError("prepublication_abort_unverified")
        if not execute:
            return
        with pinned_directory(root) as pointer:
            info = os.fstat(pointer)
            identity = info.st_dev, info.st_ino
            pending_before = observe_artifact(root / name)
        if records[-1].event != "prepublication_aborted":
            journal._append(parent, "prepublication_aborted", evidence)
        journal._flush_records(parent)
        prove()
        with pinned_directory(root) as pointer:
            info = os.fstat(pointer)
            if (info.st_dev, info.st_ino) != identity or observe_artifact(
                root / name
            ) != pending_before:
                raise ValueError("finalization_pending_changed")
            os.unlink(name, dir_fd=pointer)
            flush_directory(pointer)


def validate_replacement_abort(operation_id, *, control_root, cancel):
    """Check the actual pending operation under maintenance without clearing it."""
    from . import bootstrap
    from .journal import Journal
    from .plan_records import load_plan

    if (
        type(operation_id) is not str
        or not (control_root / ("operation-" + bootstrap._key(operation_id))).is_dir()
    ):
        raise ValueError("recovery_pending_missing")
    journal = Journal(control_root, operation_id)
    _abort_prepublication(journal, load_plan(journal), cancel, execute=False)


def _recovery_admission_record(journal, admission_root):
    """Bind recovery admission to the existing pending local replacement."""
    from . import bootstrap
    from .control_records import UNBOUND_NAMESPACE, _recovery_pending
    from .journal import Journal
    from .plan_records import load_plan

    root = bootstrap.default_bootstrap_root()
    if type(journal) is not Journal or admission_root != root / "admission":
        raise ValueError("replacement_recovery_required")
    plan = load_plan(journal)
    with journal._locked(exclusive=False) as parent:
        records = journal._records(parent)
    prepared = _Prepared.model_validate(
        next(row.evidence for row in records if row.event == "prepared")
    )
    if (
        prepared.mode != "replace"
        or plan.mode != prepared.mode
        or _plan_digest(plan) != prepared.publication.plan_digest
        or plan.archive_digest != prepared.publication.archive_digest
        or prepared.publication.bootstrap_root != str(root)
        or not any(row.event == "rollback_verified" for row in records)
    ):
        raise ValueError("replacement_recovery_required")
    _recovery_pending(root, journal, prepared)
    names = tuple(sorted({*prepared.publication.namespaces, UNBOUND_NAMESPACE}))
    return names, prepared, records


def _recovery_admission_aliases(journal, admission_root, names, roots):
    """Resolve only missing registered roots inside a checked native move gap."""
    from .journal import _absent, _MoveIntent, _RollbackCredentialPlan
    from .publication import _check_parents, _directory_state, _move_position

    expected, prepared, records = _recovery_admission_record(journal, admission_root)
    if names != expected:
        raise ValueError("replacement_recovery_scope_changed")
    latest, credential_artifacts = {}, {}
    for row in records:
        if row.event == "rollback_credentials_planned":
            for artifact in _RollbackCredentialPlan.model_validate(
                row.evidence
            ).artifacts:
                credential_artifacts.setdefault(artifact.logical_id, []).append(
                    artifact
                )
        elif row.event == "move_intended":
            intent = _MoveIntent.model_validate(row.evidence)
            latest[intent.logical_id] = (
                intent,
                None,
                tuple(credential_artifacts.get(intent.logical_id, ())),
            )
        elif row.event == "move_observed":
            latest[intent.logical_id] = (
                intent,
                row.evidence["moved"],
                latest[intent.logical_id][2],
            )
    aliases, checked = {}, {}
    for root in roots:
        resolved = root.resolve(strict=False)
        candidates = [
            item
            for item in prepared.artifacts
            if resolved == Path(item.target) or Path(item.target) in resolved.parents
        ]
        if len(candidates) != 1:
            raise ValueError("recovery_root_unverified")
        item = candidates[0]
        target = Path(item.target)
        if item.logical_id not in checked:
            if item.logical_id not in latest or not _absent(item.target):
                raise ValueError("recovery_root_unverified")
            intent, moved, alternatives = latest[item.logical_id]
            if intent.step not in {
                "retire",
                "publish",
                "unpublish",
                "restore",
                "credential_retire",
                "credential_publish",
                "credential_unpublish",
            }:
                raise ValueError("recovery_move_unqualified")
            if moved is None:
                moved = _move_position(intent) == "after"
            away = intent.step in {
                "retire",
                "unpublish",
                "credential_retire",
                "credential_unpublish",
            }
            if moved != away:
                raise ValueError("recovery_root_unverified")
            held = Path(intent.destination if away else intent.source.path)
            if (intent.source.path if away else intent.destination) != item.target:
                raise ValueError("recovery_root_unverified")
            original = intent.step in {"retire", "restore", "credential_retire"}
            expected_object = item.previous_metadata if original else item.candidate
            expected_path = (
                item.retained
                if original
                else item.candidate.path
                if item.candidate
                else None
            )
            metadata = original
            if intent.step in {"credential_publish", "credential_unpublish"}:
                alternate = (
                    (alternatives[-1] if alternatives else None)
                    if intent.step == "credential_publish"
                    else next(
                        (
                            row
                            for row in alternatives
                            if row.candidate.path == intent.destination
                        ),
                        None,
                    )
                )
                if alternate is None:
                    raise ValueError("recovery_root_unverified")
                expected_object, expected_path = (
                    alternate.metadata,
                    alternate.candidate.path,
                )
                metadata = True
            if (
                expected_object is None
                or str(held) != expected_path
                or not _matches(intent.source, str(held))
                or not _matches(expected_object, str(held), metadata=metadata)
            ):
                raise ValueError("recovery_root_unverified")
            _check_parents(item)
            for parent in intent.parents:
                current = _directory_state(parent.state.path)
                if (current.device, current.inode, current.mode) != (
                    parent.state.device,
                    parent.state.inode,
                    parent.state.mode,
                ):
                    raise ValueError("move_parent_changed")
            holders = [held]
            if not original and item.previous is not None:
                if not _matches(item.previous_metadata, item.retained, metadata=True):
                    raise ValueError("recovery_root_unverified")
                holders.append(Path(item.retained))
            checked[item.logical_id] = holders
        suffix = resolved.relative_to(target)
        held_children = tuple(
            path / suffix
            for path in checked[item.logical_id]
            if (path / suffix).exists()
        )
        if not held_children:
            raise ValueError("recovery_root_unverified")
        aliases[root] = resolved, held_children
    return aliases


def _recovery_staging_roots(journal, admission_root, names, roots):
    """Freeze original canonical source names for private-staging overlap checks."""
    canonical, missing = set(), []
    for root in roots:
        try:
            canonical.add(root.resolve(strict=True))
        except FileNotFoundError:
            missing.append(root)
    aliases = _recovery_admission_aliases(
        journal, admission_root, names, tuple(missing)
    )
    canonical.update(resolved for resolved, _ in aliases.values())
    return tuple(sorted(canonical))


def recover_replacement(
    operation_id: str,
    *,
    control_root: Path,
    action: str,
    rollback_password: bytes | None,
    cancel: Event,
) -> str:
    """Finish or reverse one still-fenced local replacement using durable evidence."""
    from . import bootstrap
    from .control_records import (
        _existing_admission_authority,
        _recover_activation_pairs,
        _recovery_pending,
    )
    from .journal import Journal
    from .plan_records import load_plan
    from .publication import finalize_candidate, publish_candidate

    if action not in {"finish", "rollback", "abort"}:
        raise ValueError("recovery_action_invalid")
    root = bootstrap.default_bootstrap_root()
    if (
        type(operation_id) is not str
        or not (control_root / ("operation-" + bootstrap._key(operation_id))).is_dir()
    ):
        raise ValueError("recovery_pending_missing")
    journal = Journal(control_root, operation_id)
    plan = load_plan(journal)
    if action == "abort":
        _abort_prepublication(journal, plan, cancel, execute=True)
        return "aborted"
    with journal._locked(exclusive=False) as parent:
        records = journal._records(parent)
    if not any(row.event == "rollback_verified" for row in records):
        raise ValueError("prepublication_abort_required")
    prepared = _Prepared.model_validate(
        next(row.evidence for row in records if row.event == "prepared")
    )
    if prepared.mode != "replace" or prepared.publication.bootstrap_root != str(root):
        raise ValueError("replacement_recovery_required")
    if records[-1].event == "committed" and action == "rollback":
        raise ValueError("later_rollback_required")
    _recovery_pending(root, journal, prepared)
    candidate = Path(records[0].evidence["stage"]["path"])
    with _existing_admission_authority(root)._replacement_recovery(
        journal, 30, cancel=cancel
    ) as session:
        _finalization_session(session, prepared.publication, prepared)
        with _unlock_recovery(
            journal, prepared, rollback_password, session, cancel
        ) as check_credentials:
            _recover_activation_pairs(journal, prepared, session)
            if action == "rollback":
                _rollback_replacement(
                    journal, prepared, session, check_credentials, cancel
                )
                return "rolled_back"
            if any(
                row.event in {"rollback_started", "rollback_credentials_planned"}
                for row in records
            ):
                raise ValueError("rollback_direction_selected")
            _apply_replacement_credentials(
                candidate, journal, session=session, cancel=cancel
            )
            rollback = next(
                row.evidence["ciphertext"]["path"]
                for row in records
                if row.event == "rollback_verified"
            )
            if records[-1].event != "committed":
                publish_candidate(
                    candidate, plan, journal, Path(rollback), session=session
                )
            try:
                finalize_candidate(candidate, plan, journal, session=session)
            except (OSError, ValueError, RuntimeError):
                with journal._locked(exclusive=False) as parent:
                    current = journal._records(parent)
                if any(row.event == "committed" for row in current):
                    raise
                _recover_activation_pairs(journal, prepared, session)
                _rollback_replacement(
                    journal, prepared, session, check_credentials, Event()
                )
                raise ValueError("replacement_rolled_back") from None
            return "committed"


def _rollback_replacement(journal, prepared, session, check_credentials, cancel):
    """Restore retained originals and authenticated remappable credential values."""
    from .activation import bind_activation
    from .archive_models import Metadata
    from .bootstrap import _key
    from .journal import _evidence_digest, _states
    from .native_files import flush_directory, pinned_directory
    from .plan_records import load_plan
    from .publication import (
        _activation_proof,
        _begin_move,
        _check_directory_states,
        _complete_move,
        _installed_metadata,
        _reconcile_moves,
        _reverse_native_move,
    )
    from .rollback_credentials import (
        alternate_artifacts,
        apply_credentials,
        current_phase,
        prepare_credentials,
        restore_alternate,
    )

    plan = load_plan(journal)
    context = prepared.publication
    root = Path(context.bootstrap_root)
    name = "pending-" + _key(journal.operation_id) + ".json"
    with pinned_directory(root) as bootstrap:
        root_identity = (os.fstat(bootstrap).st_dev, os.fstat(bootstrap).st_ino)
        pending_before = observe_artifact(root / name)
    with journal._locked(exclusive=True) as parent:
        records = journal._records(parent)
        if any(row.event == "committed" for row in records):
            raise ValueError("later_rollback_required")
        _finalization_session(session, context, prepared)
        _pending(journal, context, targets=_publication_targets(prepared), durable=True)
        _reconcile_moves(journal, parent, prepared, finish=False)
        credential_plan = prepare_credentials(
            check_credentials, journal, parent, prepared, plan, cancel
        )
        apply_credentials(check_credentials, journal, parent, credential_plan)
        records = journal._records(parent)
        phase = current_phase(records)
        alternatives = alternate_artifacts(credential_plan)
        started = next(
            (row for row in records if row.event == "rollback_started"), None
        )
        if started is None:
            _check_directory_states(prepared, records)
            if any(
                state == "uncertain" and key not in alternatives
                for key, state in _states(prepared).items()
            ):
                raise ValueError("rollback_originals_unverified")
            check_credentials.check(credential_plan)
            rollback = next(row for row in records if row.event == "rollback_verified")
            prepared_record = next(row for row in records if row.event == "prepared")
            journal._append(
                parent,
                "rollback_started",
                {
                    "prepared_digest": _evidence_digest(prepared_record.evidence),
                    "rollback_digest": _evidence_digest(rollback.evidence),
                    "generation": uuid4().hex,
                    "profiles": [
                        {"config": row.config, "installation_id": uuid4().hex}
                        for row in prepared.replacement_profiles
                    ],
                    "retained_credential_scopes": sorted(prepared.credential_scopes),
                },
            )
            journal._flush_records(parent)
            started = journal._records(parent)[-1]
        completed = any(row.event == "originals_validated" for row in phase)
        if not completed:
            for item in reversed(prepared.artifacts):
                reader._check(cancel)
                _finalization_session(session, context, prepared)
                check_credentials.check(credential_plan)
                if item.logical_id in alternatives:
                    restore_alternate(
                        journal, parent, prepared, item, alternatives[item.logical_id]
                    )
                    continue
                state = _states(prepared, logical_id=item.logical_id)[item.logical_id]
                if state == "uncertain":
                    raise ValueError("rollback_originals_unverified")
                if state == "published":
                    intent = _begin_move(journal, parent, item, "unpublish")
                    _reverse_native_move(intent)
                    _complete_move(journal, parent, prepared, intent, moved=True)
                    state = "retired" if item.previous else "staged"
                if state == "retired":
                    intent = _begin_move(journal, parent, item, "restore")
                    _reverse_native_move(intent)
                    _complete_move(journal, parent, prepared, intent, moved=True)
            for item in reversed(prepared.directory_metadata):
                records = current_phase(journal._records(parent))
                done = any(
                    row.event == "rollback_metadata_applied"
                    and row.evidence["logical_id"] == item.logical_id
                    for row in records
                )
                if done:
                    continue
                intent = next(
                    (
                        row
                        for row in records
                        if row.event == "rollback_metadata_started"
                        and row.evidence["logical_id"] == item.logical_id
                    ),
                    None,
                )
                applied = Metadata(
                    version=1, mode=item.previous.mode, mtime_ns=item.previous.mtime_ns
                )
                if intent is None:
                    before = _directory_state(item.previous.path)
                    journal._append(
                        parent,
                        "rollback_metadata_started",
                        {
                            "logical_id": item.logical_id,
                            "before": before.model_dump(),
                            "applied": applied.model_dump(),
                        },
                    )
                    journal._flush_records(parent)
                else:
                    from .journal import _DirectoryState

                    before = _DirectoryState.model_validate(intent.evidence["before"])
                _installed_metadata(
                    Path(item.previous.path),
                    (item.previous.device, item.previous.inode),
                    applied.model_dump(),
                    previous=before,
                    parent_identity=(item.parent.device, item.parent.inode),
                )
                journal._append(
                    parent,
                    "rollback_metadata_applied",
                    {
                        "logical_id": item.logical_id,
                        "observed": _directory_state(item.previous.path).model_dump(),
                    },
                )
                journal._flush_records(parent)
            proof = _originals_proof(
                prepared, started, check_credentials, credential_plan
            )
            journal._append(parent, "originals_validated", proof)
            journal._flush_records(parent)
        else:
            _originals_proof(prepared, started, check_credentials, credential_plan)
        records = current_phase(journal._records(parent))
        validated = next(
            row for row in reversed(records) if row.event == "originals_validated"
        )
        activation = next(
            (row for row in records if row.event == "rollback_activation_recorded"),
            None,
        )
        generation = started.evidence["generation"]
        selectors = sorted(row["config"] for row in started.evidence["profiles"])
        owners = sorted(
            owner.owner_id for owner in install_adapters() if owner.activation_required
        )
        if activation is None:
            for selector in selectors:
                bind_activation(
                    Path(context.bootstrap_root),
                    journal.operation_id,
                    Path(selector),
                    generation,
                    tuple(owners),
                    session=session,
                )
            proof = {
                "generation": generation,
                "selectors": selectors,
                "owners": owners,
                "originals_digest": _evidence_digest(validated.evidence),
                "records": _activation_proof(
                    journal, context, generation, selectors, owners
                ),
            }
            journal._append(parent, "rollback_activation_recorded", proof)
            journal._flush_records(parent)
            activation = journal._records(parent)[-1]
        if (
            _activation_proof(journal, context, generation, selectors, owners)
            != activation.evidence["records"]
        ):
            raise ValueError("rollback_activation_changed")
        _originals_proof(prepared, started, check_credentials, credential_plan)
        if journal._records(parent)[-1].event != "rolled_back":
            journal._append(
                parent,
                "rolled_back",
                {
                    "generation": generation,
                    "activation_digest": _evidence_digest(activation.evidence),
                },
            )
            journal._flush_records(parent)
        _finalization_session(session, context, prepared)
        _originals_proof(prepared, started, check_credentials, credential_plan)
        _pending(journal, context, targets=_publication_targets(prepared), durable=True)
        with pinned_directory(root) as bootstrap:
            info = os.fstat(bootstrap)
            if (info.st_dev, info.st_ino) != root_identity:
                raise ValueError("finalization_authority_changed")
            if observe_artifact(root / name) != pending_before:
                raise ValueError("finalization_pending_changed")
            os.unlink(name, dir_fd=bootstrap)
            flush_directory(bootstrap)


def _originals_proof(prepared, started, check_credentials, credential_plan=None):
    from .journal import _evidence_digest, _states
    from .rollback_credentials import alternate_artifacts, originals_proof

    alternatives = alternate_artifacts(credential_plan)
    if any(
        state != "staged" and key not in alternatives
        for key, state in _states(prepared).items()
    ):
        raise ValueError("rollback_originals_unverified")
    for row in prepared.safety_sources:
        if not _safety_source_matches(row.source):
            raise ValueError("safety_source_changed")
    check_credentials.check(credential_plan)
    return {
        "credential_plan_digest": _evidence_digest(credential_plan.evidence)
        if credential_plan
        else None,
        "rollback_started_digest": _evidence_digest(started.evidence),
        "artifacts": originals_proof(prepared, credential_plan),
        "directories": [
            _directory_state(item.previous.path).model_dump()
            for item in prepared.directory_metadata
        ],
    }
