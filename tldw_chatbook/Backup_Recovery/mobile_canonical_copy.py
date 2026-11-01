"""Authenticated mobile canonical captures; no replacement-plan fiction."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from tldw_chatbook.Utils.platform_files import os

from . import archive_reader, bootstrap
from .archive_models import SealedArchive
from .journal import (
    _evidence_digest,
    _mobile_state,
    _MobileCaptureVerified,
    _MobilePrepared,
)
from .mobile_preview_budget import path_cost
from .storage_admission import _CaptureFileDescriptors


@dataclass(frozen=True)
class _CanonicalCopy:
    """Source evidence only: a mobile capture has no recorded plaintext digest."""

    capture: _MobileCaptureVerified
    evidence_digest: str
    pending: bool
    prepared: _MobilePrepared

    @property
    def ciphertext(self):
        return self.capture.archive

    @property
    def manifest_digest(self):
        return self.capture.manifest_digest


def _canonical_copy(journal, *, _read_budget=None):
    """Re-read actual native history under its lock and one observational bound."""
    from .mobile_preview_budget import current

    scope = current()
    from .mobile_builtin_preservation import _preservation_budget

    budget = _preservation_budget(_read_budget)
    if scope is None:
        budget.reserve(512)
    custody = scope.custody if scope is not None else _CaptureFileDescriptors(None)
    try:
        with journal._locked(exclusive=False, _custody=custody) as parent:
            rows = journal._records(parent, _budget=budget, _custody=custody)
        return _canonical_evidence(journal, rows, budget, custody)
    finally:
        if scope is None:
            custody.retire()


def _canonical_evidence(journal, rows, budget, custody):
    """Derive evidence only for the two installed, native-owned journal readers."""
    if not rows or rows[0].event != "mobile_prepared":
        raise ValueError("mobile_canonical_copy_required")
    budget.value(rows[0].evidence, text_copies=4)
    prepared, _ = _mobile_state(rows)
    budget.container(len(rows))
    captures = [
        row
        for row in rows
        if row.event == "mobile_capture_verified"
        and row.evidence["purpose"] == "canonical"
    ]
    if len(captures) != 1:
        return None
    budget.value(captures[0].evidence, text_copies=8)
    capture = _MobileCaptureVerified.model_validate(captures[0].evidence)
    pending, _ = bootstrap._records(
        Path(prepared.bootstrap_root), _budget=budget, _custody=custody
    )
    budget.container(len(prepared.namespaces) + len(pending))
    unresolved = rows[-1].event != "mobile_committed" or any(
        row["operation_id"] == journal.operation_id
        or set(row["namespaces"]).intersection(prepared.namespaces)
        for row in pending
    )
    from .mobile_runs_goal_budget import _serialization

    _serialization(captures[0].evidence, budget, encoded=True)
    budget.reserve(512)
    return _CanonicalCopy(
        capture, _evidence_digest(captures[0].evidence), unresolved, prepared
    )


def _verify_canonical_archive(
    proof, archive, digest, *, _read_budget=None, cancel=None, _custody=None
):
    """Bind native encrypted acquisition to the retained cipher and plaintext."""
    if type(proof) is not _CanonicalCopy or type(archive) is not SealedArchive:
        raise ValueError("mobile_canonical_archive_required")
    if _custody is not None:
        if type(_custody) is not _CaptureFileDescriptors or _custody.close_failed:
            raise ValueError("mobile_canonical_custody_changed")
        from .mobile_preview_budget import current

        current_scope = current()
        if current_scope is not None and _custody is not current_scope.custody:
            raise ValueError("mobile_canonical_custody_changed")
    document = archive_reader.verify_sealed(archive)
    source = archive.encrypted_source
    if (
        source is None
        or source.identity[2] != proof.ciphertext.size
        or source.plaintext_digest != archive.digest
        or archive.digest != digest
        or source.manifest_digest != proof.manifest_digest
        or hashlib.sha256(archive.manifest_bytes).hexdigest() != proof.manifest_digest
        or document.credential_policy != "rollback"
    ):
        raise ValueError("local_snapshot_source_changed")
    # Native acquire records its private ciphertext COPY, not the original inode.
    # Its bytes must still equal the independently authenticated retained cipher.
    with archive_reader._regular(source.path) as stream:
        if archive_reader._identity(os.fstat(stream.fileno())) != source.identity:
            raise ValueError("encrypted_source_changed")
        if archive_reader._hash(source.path, Event()) != source.digest:
            raise ValueError("local_snapshot_source_changed")
        # The journal's artifact digest frames the raw file digest with its kind
        # and size. Compare native observations, never ciphertext to that frame.
        from .mobile_preview_budget import current
        from .mobile_rollback_io import observation

        scope = current()
        from .mobile_builtin_preservation import _preservation_budget

        budget = _preservation_budget(_read_budget)
        if scope is None:
            budget.reserve(512)
        custody = (
            _custody
            if _custody is not None
            else scope.custody
            if scope is not None
            else _CaptureFileDescriptors(None)
        )
        try:
            observed = observation(
                source.path, budget, custody, Event() if cancel is None else cancel
            )
            if (
                observed["kind"] != proof.ciphertext.kind
                or observed["size"] != proof.ciphertext.size
                or observed["sha256"] != proof.ciphertext.sha256
            ):
                raise ValueError("local_snapshot_source_changed")
        finally:
            if scope is None and _custody is None:
                custody.retire()
        if (
            archive_reader._identity(os.fstat(stream.fileno())) != source.identity
            or archive_reader._identity(os.stat(source.path, follow_symlinks=False))
            != source.identity
        ):
            raise ValueError("encrypted_source_changed")
    return document


def _current_canonical_target(proof, target, *, _read_budget=None):
    """Use unchanged authenticated selectors and fresh Complete native owners."""
    from .capture_service import _selectors
    from .mobile_consolidation import _selector_observation
    from .models import Inventory

    if (
        type(proof) is not _CanonicalCopy
        or type(target) is not Inventory
        or not target.complete
    ):
        raise ValueError("mobile_canonical_current_target_required")
    prepared = proof.prepared
    if prepared.bootstrap_root != str(bootstrap.default_bootstrap_root()):
        raise ValueError("local_snapshot_source_changed")
    from .mobile_builtin_preservation import _preservation_budget

    budget = _preservation_budget(_read_budget)
    budget.reserve(512)
    custody = _CaptureFileDescriptors(None)
    try:
        pending, profiles = bootstrap._records(
            Path(prepared.bootstrap_root), _budget=budget, _custody=custody
        )
        registry = bootstrap._registry(
            Path(prepared.bootstrap_root), _budget=budget, _custody=custody
        )
        if pending:
            raise bootstrap.RecoveryRequired("recovery_pending")
        budget.container(len(prepared.selector_review) * 8 + len(target.items) * 3)
        for item in target.items:
            if item.owner == "config" and item.path is not None:
                budget.reserve(path_cost(item.path, copies=2))
        configs = {
            str(item.path)
            for item in target.items
            if item.owner == "config" and item.path is not None
        }
        budget.value(
            (prepared.source_selector, prepared.target_selector), text_copies=8
        )
        selected = {prepared.source_selector, prepared.target_selector}
        if configs != selected:
            raise ValueError("mobile_canonical_selector_review_required")
        selectors = _selectors(
            (Path(prepared.source_selector), Path(prepared.target_selector)),
            include_known_profiles=True,
            _budget=budget,
            _custody=custody,
        )
        budget.container(len(selectors) * 2)
        if tuple(str(path) for path in selectors) != tuple(
            row.selector for row in prepared.selector_review
        ):
            raise ValueError("mobile_canonical_selector_review_required")
        for row in prepared.selector_review:
            budget.value(row.selector, text_copies=8)
            budget.container(len(row.witnesses) * 5 + 4)
            expected = (
                row.fingerprint,
                tuple((x.path, x.device, x.inode, x.mode) for x in row.witnesses),
            )
            if (
                _selector_observation(
                    Path(row.selector), _budget=budget, _custody=custody
                )
                != expected
            ):
                raise ValueError("mobile_canonical_selector_review_required")
            binding = bootstrap._binding(
                Path(row.selector), profiles, registry, _budget=budget, _custody=custody
            )
            budget.container(
                len(prepared.namespaces)
                + (len(binding["namespaces"]) if binding is not None else 0)
            )
            if row.selector in selected and (
                binding is None
                or not set(binding["namespaces"]) <= set(prepared.namespaces)
            ):
                raise ValueError("mobile_canonical_selector_review_required")
        for store in prepared.targets:
            budget.value(store.witnesses[-1].path, text_copies=8)
            budget.container(len(target.items))
            selected = Path(store.witnesses[-1].path)
            matches = [
                item
                for item in target.items
                if item.owner == store.owner_id and item.path == selected
            ]
            if len(matches) != 1:
                raise ValueError("mobile_canonical_store_review_required")
    finally:
        custody.retire()


def _mobile_preserved_members(
    proof, document, target, *, _read_budget=None, cancel=None
):
    from .mobile_builtin_preservation import (
        _ordinary_builtin_members,
        _preservation_budget,
        _verify_current_builtin_files,
    )

    budget = _preservation_budget(_read_budget)
    _current_canonical_target(proof, target, _read_budget=budget)
    members = _ordinary_builtin_members(document, target, budget)
    _verify_current_builtin_files(document, members, budget, cancel=cancel)
    return members


def _preview_canonical(
    journal, proof, archive, target, acknowledged, *, _read_budget=None, cancel=None
):
    """Plan a new replacement at the authenticated unchanged configuration."""
    from .later_rollback import _current_config_scope, _verify_snapshot_source
    from .mobile_builtin_preservation import _preservation_budget
    from .restore_plan import LocalSnapshotSource, plan_restore, recheck_targets

    budget = _preservation_budget(_read_budget)
    budget.reserve(1024 + path_cost(journal.root, copies=2))
    snapshot = LocalSnapshotSource(
        journal.root.parent, journal.operation_id, proof.evidence_digest
    )
    _, actual = _verify_snapshot_source(
        snapshot, "replace", archive.digest, _read_budget=budget, cancel=cancel
    )
    document = _verify_canonical_archive(
        actual, archive, archive.digest, _read_budget=budget, cancel=cancel
    )
    preserved = _mobile_preserved_members(
        actual, document, target, _read_budget=budget, cancel=cancel
    )
    budget.container(
        len(target.items) * 3
        + len(document.producer_inventory) * 3
        + len(document.files) * 4
        + len(document.directories) * 4
    )
    current = {item.logical_id: item for item in target.items}
    producers = {row.logical_id: row for row in document.producer_inventory}
    directories = {row.logical_id: row for row in document.directories}
    roots, names = {}, {}
    records = (*document.directories, *document.files)
    for record in records:
        if record.logical_id in preserved:
            continue  # Exact unchanged package members are preserve-only.
        if (
            record.logical_id in directories
            and directories[record.logical_id].synthetic
        ):
            continue
        if record.logical_id == "credentials":
            root = directories.get(record.root_id)
            budget.reserve(256)
            if (
                document.credential_policy != "rollback"
                or record.owner_id != "recovery.credentials"
                or record.payload != "payload/credential-recovery.json"
                or record.relative_path != "credential-recovery.json"
                or record.parent_id != record.root_id
                or root is None
                or not root.synthetic
                or sum(member.root_id == record.root_id for member in records) != 2
            ):
                raise ValueError("mobile_canonical_current_mapping_required")
            continue  # The installed planner/stager consumes this fixed control.
        producer = producers.get(record.logical_id)
        if producer is None:
            continue  # Native synthetic/credential controls are checked by plan_restore.
        item = current.get(record.logical_id)
        if item is None or item.path is None or item.owner != producer.owner_id:
            raise ValueError("mobile_canonical_current_mapping_required")
        if producer.status != item.status or item.status not in {
            "included",
            "included_directory",
        }:
            raise ValueError("mobile_canonical_current_mapping_required")
        meta = item.metadata
        root = directories.get(record.root_id)
        if root is not None and root.synthetic:
            budget.value(item.logical_id, text_copies=8)
            expected_root = (
                "root:" + hashlib.sha256(item.logical_id.encode()).hexdigest()
            )
            if (
                record.root_id != expected_root
                or record.parent_id != expected_root
                or record.relative_path != item.path.name
                or item.status != "included"
            ):
                raise ValueError("mobile_canonical_current_mapping_required")
        elif meta is None or (meta.root_id, meta.parent_id, meta.relative_path) != (
            record.root_id,
            record.parent_id,
            record.relative_path,
        ):
            raise ValueError("mobile_canonical_current_mapping_required")
        budget.reserve(path_cost(item.path, copies=len(item.path.parts) + 8))
        budget.value(record.relative_path, text_copies=8)
        path = item.path
        for _ in Path(record.relative_path).parts:
            path = path.parent
        if record.root_id in roots and roots[record.root_id] != path:
            raise ValueError("local_snapshot_mapping_invalid")
        roots[record.root_id] = path
        if item.owner == "config":
            parts = item.logical_id.split(":")
            if (
                len(parts) != 3
                or parts[0] != "profile"
                or parts[1] not in document.profile_ids
            ):
                raise ValueError("local_snapshot_mapping_invalid")
            names[parts[1]] = item.path.stem
    budget.container(len(preserved) + 4)
    plan = plan_restore(
        archive,
        mode="replace",
        destinations=roots,
        target=target,
        safety_scope=tuple(item.logical_id for item in preserved.values()),
        profile_names=names,
        acknowledged_credential_issues=acknowledged,
        local_snapshot=snapshot,
        _mobile_read_budget=budget,
        _mobile_cancel=cancel,
    )
    if not plan.restore and not plan.retire:
        raise ValueError("explicit_destination_required")
    _verify_snapshot_source(
        snapshot,
        "replace",
        plan.archive_digest,
        archive,
        _read_budget=budget,
        cancel=cancel,
    )
    _current_config_scope(plan, document)
    recheck_targets(plan)
    return plan
