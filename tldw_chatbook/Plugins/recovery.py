"""Recovery classification after exact authenticated evidence matching."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coordinator import PluginCoordinator
    from .models import PackageInspection
    from .review import OperationReceipt


def recovery_action(
    *,
    registry_new: bool,
    marker_new: bool,
    certificate_valid: bool,
    snapshot_valid: bool,
) -> str:
    """Classify matched evidence; registry phase alone never proves commitment."""
    if not snapshot_valid:
        return "review_required"
    if marker_new:
        return "reconcile_committed"
    if certificate_valid:
        return "complete_marker"
    return "review_required" if registry_new else "abort_prepared"


def retained_inspections(
    snapshot: dict, *, coordinator=None
) -> dict[tuple[str, str], PackageInspection]:
    """Reinspect immutable references; never fetch, execute or invent overlays."""
    from pathlib import Path

    from .package_files import capture_package
    from .retained_inspection import (
        _retained_candidates,
        _select_retained,
        _verify_retained_source,
    )

    result = {}
    for revision in snapshot["revisions"]:
        _verify_retained_source(revision, coordinator)
        root = revision["materialized_identity"]
        if not root:
            raise ValueError("missing immutable package reference")
        capture = capture_package(Path(root))
        if capture.errors:
            raise ValueError("unsafe retained package capture")
        identity = revision["installation_id"], revision["revision_digest"]
        components = [
            row
            for row in snapshot["components"]
            if (row["installation_id"], row["revision_digest"]) == identity
        ]
        inspection = _select_retained(
            revision, components, _retained_candidates(capture, revision)
        )
        if (
            inspection.catalog_binding is not None
            or inspection.acquisition_reference is not None
        ):
            from .source_service import validate_inspection_source

            validate_inspection_source(coordinator, inspection)
        result[identity] = inspection
    return result


def recover_coordinator(coordinator: PluginCoordinator) -> tuple[OperationReceipt, ...]:
    """Apply the evidence table after authenticating exact transition identities."""
    import sqlite3

    from .review import OperationReceipt

    authority, registry = coordinator.authority, coordinator.registry
    receipts = []
    operation_id = "recovery"
    try:
        marker = authority.load_marker()
        if marker is None:
            raise ValueError("authority setup required")
        current = authority.verify_current()
        evidence = []
        offset = 0
        while True:
            page = authority.list_transitions(limit=50, offset=offset)
            evidence.extend(page)
            if len(page) < 50:
                break
            offset += len(page)
        operations = {}
        offset = 0
        while True:
            page = registry.list_operations(limit=50, offset=offset)
            operations.update((item["result"]["operation_id"], item) for item in page)
            if len(page) < 50:
                break
            offset += len(page)
        by_id = {item.new.operation_id: item for item in evidence}
        current_result = current["operation_result"]
        marker_operation = (
            {marker.operation_id} if current_result is not None else set()
        )
        unmatched = set(operations) - by_id.keys() - marker_operation
        if (
            marker.operation_id in operations
            and operations[marker.operation_id]["result"] != current_result
        ):
            raise ValueError("marker operation result mismatch")
        # Walk the exact marker ancestry. Same generation or a phase flag is
        # insufficient; every retained committed branch must belong to this chain.
        lineage = {}
        accepted_markers = {marker}
        ancestor = marker
        while ancestor.generation:
            transition = by_id.get(ancestor.operation_id)
            if transition is None:
                # R29: exact reached endpoint authenticates a wholly absent prefix.
                # Every still-present disconnected branch is rejected below.
                if authority.metadata()["schema_version"] == 2 or (
                    ancestor == marker and not evidence
                ):
                    break
                raise ValueError("marker lineage mismatch")
            if transition.new != ancestor:
                raise ValueError("marker lineage mismatch")
            lineage[transition.new.operation_id] = transition
            ancestor = transition.old
            accepted_markers.add(ancestor)
        successors = [
            item for item in evidence if item.old == marker and item.committed
        ]
        if len(successors) > 1:
            raise ValueError("ambiguous certified successors")
        successor = successors[0] if successors else None
        allowed = set(lineage)
        if successor:
            allowed.add(successor.new.operation_id)
        if any(
            item.committed and item.new.operation_id not in allowed for item in evidence
        ):
            raise ValueError("unrelated certified transition")
        for item in evidence:
            operation_id = item.new.operation_id
            hint = operations.get(operation_id)
            if hint is not None and hint["result"] != item.snapshot["operation_result"]:
                raise ValueError("operation result mismatch")
            if operation_id not in allowed:
                if item.old not in accepted_markers:
                    raise ValueError("unrelated prepared transition")
                registry_new = hint is not None
                if item.old == marker and successor is None:
                    # Prepared-only abort requires the OLD authority to agree.
                    # A different valid projection still cannot prove COMMIT.
                    registry_new = (
                        registry_new
                        or registry.authority_projection(
                            operation_result=current_result
                        )
                        != current
                    )
                action = recovery_action(
                    registry_new=registry_new,
                    marker_new=False,
                    certificate_valid=item.committed,
                    snapshot_valid=True,
                )
                if action == "review_required":
                    return (
                        OperationReceipt(
                            operation_id,
                            "recovery_required",
                            False,
                            "missing_commit_certificate",
                        ),
                    )
                receipts.append(OperationReceipt(operation_id, "aborted", False))
        # All present proof and unresolved branches were checked first. An ID's
        # authenticated generation is issuance, never commitment evidence.
        expired = []
        if unmatched:
            legacy = {}
            if authority.metadata()["schema_version"] == 2:
                legacy = {
                    entry.operation_id: entry.result_digest
                    for entry in authority.verify_legacy_cutover().entries
                }
            recovered_generation = (
                successor.new.generation if successor else marker.generation
            )
            for identity in unmatched:
                result = operations[identity]["result"]
                if legacy.get(identity) == authority.result_digest(result):
                    expired.append(identity)
                    continue
                issued = authority.verify_operation_id(identity, result)
                if issued.generation > recovered_generation:
                    raise ValueError(
                        "newer registry operation lacks protected evidence"
                    )
                expired.append(identity)
        target = successor.snapshot if successor else current
        operation_id = (
            target["operation_result"]["operation_id"]
            if target["operation_result"]
            else "bootstrap"
        )
        inspections = retained_inspections(target, coordinator=coordinator)
        projection = registry.authority_projection(
            operation_result=target["operation_result"]
        )
        reconstructed = projection != target
        if expired:
            registry.forget_operation_hints(tuple(sorted(expired)))
        for mapping in target["mappings"]:
            coordinator.validate_mapping(
                mapping,
                inspections[(mapping["installation_id"], mapping["revision_digest"])],
                [
                    row
                    for row in target["mappings"]
                    if row["installation_id"] == mapping["installation_id"]
                    and row["revision_digest"] == mapping["revision_digest"]
                ],
            )
        if reconstructed:
            # Only a secure marker (or its matching certified successor) can
            # authorize reconstruction. Genesis never erases untrusted rows.
            if target["operation_result"] is None:
                raise ValueError("unexpected registry authority at bootstrap")
            registry.restore_authority(target, inspections)
        if successor:
            authority.advance_marker(successor.old, successor.new)
            lineage[successor.new.operation_id] = successor
        elif (
            marker.generation
            and marker.operation_id in by_id
            and by_id[marker.operation_id].committed
        ):
            # Qualify lost-response marker writes through the backend again.
            transition = by_id[marker.operation_id]
            authority.advance_marker(transition.old, transition.new)
        for item in sorted(lineage.values(), key=lambda value: value.new.generation):
            result = item.snapshot["operation_result"]
            with registry.transaction() as cursor:
                registry.write_operation(cursor, result, phase="complete")
            receipts.append(
                OperationReceipt(
                    item.new.operation_id,
                    "complete",
                    True,
                    "missing_runtime_provenance" if reconstructed else None,
                )
            )
        if not lineage and target["operation_result"] is not None:
            receipts.append(
                OperationReceipt(
                    operation_id,
                    "complete",
                    True,
                    "missing_runtime_provenance" if reconstructed else None,
                )
            )
        authority.ensure_issued_metadata(tuple(lineage.values()))
        coordinator._published = target
        coordinator.root_usage.initialize(target, reconstructed=reconstructed)
        return tuple(receipts)
    except (ValueError, OSError, RuntimeError, sqlite3.DatabaseError):
        # Closed metadata only. Exception strings may contain retained paths or
        # source bodies and are deliberately not copied into public receipts.
        return (
            OperationReceipt(
                operation_id,
                "recovery_required",
                False,
                "invalid_or_unavailable_evidence",
            ),
        )
