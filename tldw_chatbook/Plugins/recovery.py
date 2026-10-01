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


def retained_inspections(snapshot: dict) -> dict[tuple[str, str], PackageInspection]:
    """Reinspect immutable references; never fetch, execute or invent overlays."""
    from pathlib import Path

    from tldw_chatbook.Skills_Interop.skill_trust_crypto import (
        canonical_json,
        sha256_hex,
    )

    from .inspection import inspect_package
    from .package_files import parse_document

    result = {}
    for revision in snapshot["revisions"]:
        root = revision["materialized_identity"]
        if not root:
            raise ValueError("missing immutable package reference")
        inspection = inspect_package(Path(root), dialect=revision["dialect"])
        data = inspection.model_dump(mode="json")
        for key in (
            "dialect",
            "format_version",
            "adapter_version",
            "root_manifest",
            "overlay_identities",
            "content_digest",
            "activation_blockers",
            "rejected",
        ):
            if data[key] != revision[key]:
                raise ValueError("retained package interpretation mismatch")
        if (
            inspection.effective_digest != revision["revision_digest"]
            or sha256_hex(
                canonical_json(parse_document(inspection.variables_json.encode()))
            )
            != revision["variables_digest"]
        ):
            raise ValueError("retained package authority mismatch")
        identity = revision["installation_id"], revision["revision_digest"]
        expected = {
            row["component_id"]: row
            for row in snapshot["components"]
            if (row["installation_id"], row["revision_digest"]) == identity
        }
        if set(expected) != set(inspection.inventory):
            raise ValueError("retained inventory mismatch")
        for component_id, component in inspection.inventory.items():
            component_data = component.model_dump(mode="json")
            for key in ("availability", "evidence", "definition_json"):
                component_data.pop(key)
            component_data.update(
                installation_id=identity[0],
                revision_digest=identity[1],
                definition_digest=sha256_hex(
                    canonical_json(
                        parse_document(
                            b'{"definition":'
                            + component.definition_json.encode()
                            + b"}"
                        )["definition"]
                    )
                ),
            )
            if component_data != expected[component_id]:
                raise ValueError("retained component constraints mismatch")
        # Original links are intentionally regular files in owned storage. Their
        # acquisition provenance comes from authenticated authority, not reinspection.
        result[identity] = inspection.model_copy(
            update={
                key: revision[key]
                for key in (
                    "source_identity",
                    "source_digest",
                    "materialized_identity",
                    "link_targets",
                )
            }
        )
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
        inspections = retained_inspections(target)
        projection = registry.authority_projection(
            operation_result=target["operation_result"]
        )
        reconstructed = projection != target
        if expired:
            registry.forget_operation_hints(tuple(sorted(expired)))
        if reconstructed:
            # Only a secure marker (or its matching certified successor) can
            # authorize reconstruction. Genesis never erases untrusted rows.
            if target["operation_result"] is None:
                raise ValueError("unexpected registry authority at bootstrap")
            if target["mappings"]:
                raise ValueError("mapping reference owner verification required")
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
