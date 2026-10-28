"""Fixed historical projections for the actual owning operation's receipt bytes."""

import hashlib

from tldw_chatbook.DB.mobile_consolidation_receipts import (
    _DOMAINS_BY_NAME,
    _SPECS,
    _encode_owner_receipt,
    _row_digest,
)


def _array(value, budget):
    budget.container(len(value))
    return list(value)


def _projection(record, keys, budget):
    budget.container(len(keys) * 2)
    return {
        key: _array(getattr(record, key), budget)
        if type(getattr(record, key)) is tuple
        else getattr(record, key)
        for key in keys
    }


def _text_digest(value, budget):
    if value is None:
        return None
    budget.reserve(4 * len(value) + 128)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _receipt_children(plan, budget):
    """Serialize only the eleven reviewed native provenance families, never bodies."""
    budget.container(
        len(plan.rows) * 8
        + len(plan.source_rows) * 4
        + len(plan.identities.reservations) * 5
    )
    groups = {"chat": [], "workspaces": [], "runs": []}

    def emit(owner, kind, data):
        budget.container(6)
        groups[owner].append({"version": 1, "kind": kind, "data": data})

    def provenance(owner, family, record):
        budget.container(2)
        emit(owner, "provenance", {"family": family, "record": record})

    for reservation in plan.identities.reservations:
        owner = _DOMAINS_BY_NAME[reservation.domain][0]
        emit(
            owner,
            "identity",
            {
                "domain": reservation.domain,
                "original": _array(reservation.original, budget)
                if type(reservation.original) is tuple
                else reservation.original,
                "mapped": _array(reservation.mapped, budget)
                if type(reservation.mapped) is tuple
                else reservation.mapped,
            },
        )
    originals = {(row.table, row.original): row for row in plan.source_rows}
    for row in plan.rows:
        spec = _SPECS[row.table]
        budget.container(len(row.values) * 2 + 5)
        source = dict(originals[row.table, row.original].values)
        mapped = dict(row.values)
        if row.table == "automatic_work_chains":
            mapped["status"] = "review_required"
            mapped["pause_reason"] = "import_review_required"
        emit(
            spec.family,
            "row",
            {
                "table": row.table,
                "original": _array(row.original, budget),
                "mapped": [mapped[column] for column in spec.identity_columns],
                "source_sha256": _row_digest(row.table, source, budget),
                "mapped_sha256": _row_digest(row.table, mapped, budget),
            },
        )
    for record in plan.goal_provenance:
        provenance(
            "runs",
            "goal",
            _projection(
                record,
                (
                    "table",
                    "original",
                    "original_hash",
                    "mapped_hash",
                    "artifact_digest",
                ),
                budget,
            ),
        )
    for record in plan.note_provenance:
        provenance(
            "runs",
            "note_snapshot",
            _projection(
                record,
                (
                    "original_note",
                    "original_snapshot",
                    "mapped_snapshot",
                    "original_run",
                    "original_root",
                    "missing",
                ),
                budget,
            ),
        )
    provenance(
        "runs",
        "snapshot_allocator",
        _projection(
            plan.snapshot_allocator,
            ("source_sequence", "canonical_sequence", "required_frontier"),
            budget,
        ),
    )
    for record in plan.integrated.sync.provenance:
        provenance(
            "chat",
            "sync_source",
            _projection(
                record,
                (
                    "change_id",
                    "entity",
                    "original_id",
                    "operation",
                    "timestamp",
                    "client_id",
                    "version",
                    "source_digest",
                    "source_manifest_digest",
                ),
                budget,
            ),
        )
    for record in plan.sync_frontiers:
        data = _projection(
            record,
            (
                "entity",
                "original_id",
                "mapped_id",
                "version",
                "operation",
                "timestamp",
                "client_id",
                "payload_hash",
                "previous_hash",
                "previous_operation",
                "previous_timestamp",
                "previous_client_id",
                "current_disposition",
                "source_change_ids",
            ),
            budget,
        )
        budget.container(4)
        data.update(
            payload_sha256=_text_digest(record.payload, budget),
            previous_payload_sha256=_text_digest(record.previous_payload, budget),
            owner_table=record.owner[0],
            owner_original=_array(record.owner[1], budget),
        )
        provenance("chat", "sync_frontier", data)
    trace = plan.integrated.sync.trace
    for records in (trace.content.operational_provenance, trace.operational_provenance):
        for record in records:
            provenance(
                "chat",
                "operational",
                _projection(
                    record,
                    (
                        "table",
                        "original",
                        "source_digest",
                        "source_manifest_digest",
                        "disposition",
                    ),
                    budget,
                ),
            )
    for row in plan.operational_history:
        if row.table == "automatic_work_runtime_owner":
            budget.container(2)
            provenance("runs", "runtime_owner", dict(row.values))
    for record in plan.semantic_discharges:
        provenance(
            _SPECS[record.table].family,
            "semantic",
            _projection(
                record,
                ("table", "original", "handler", "domain", "original_id", "mapped_id"),
                budget,
            ),
        )
    for record in trace.citation_dispositions:
        provenance(
            "chat",
            "citation",
            _projection(
                record,
                (
                    "original",
                    "source_manifest_digest",
                    "canonical_manifest_digest",
                    "disposition",
                ),
                budget,
            ),
        )
    for record in trace.input_incidence:
        data = _projection(
            record,
            (
                "original_call_id",
                "mapped_call_id",
                "source_manifest_digest",
                "canonical_manifest_digest",
            ),
            budget,
        )
        budget.container(len(record.inputs) * 3 + 1)
        data["inputs"] = [
            [table, _array(identity, budget)] for table, identity in record.inputs
        ]
        provenance("chat", "trace_incidence", data)
    for record in trace.content.actors.workflows.provenance:
        provenance(
            "chat",
            "workflow",
            _projection(
                record,
                (
                    "table",
                    "original",
                    "source_identity",
                    "projected_identity",
                    "meaning",
                ),
                budget,
            ),
        )
    return groups


def _encode_receipts(plan, versions, children, command_count, budget):
    budget.container(len(versions) * 3 + 3)
    versions = {(purpose, family): version for purpose, family, version in versions}
    result = {}
    for owner in ("chat", "workspaces", "runs"):
        result[owner] = _encode_owner_receipt(
            owner,
            children[owner],
            source_manifest_sha256=plan.identities.source_manifest_digest
            if ("legacy", owner) in versions
            else None,
            canonical_manifest_sha256=plan.identities.canonical_manifest_digest
            if ("canonical", owner) in versions
            else None,
            source_runs_version=versions.get(("legacy", "runs"))
            if owner == "runs"
            else None,
            command_history_count=command_count if owner == "runs" else None,
            budget=budget,
        )
    return result
