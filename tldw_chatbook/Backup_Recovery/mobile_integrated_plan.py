"""Complete fixed captured plan composition, never an insertion capability."""

from dataclasses import dataclass, field, replace

from .mobile_content_graph import _ContentBlocker, _ContentRow
from .mobile_content_settings import _SettingsSuccessor
from .mobile_identity_plan import _COMPOSITES, _DOMAINS, _rows
from .mobile_plan_manifest import _TABLES

_NATIVE_QUARANTINE_DUTIES = {
    "console_dispatch_checkpoints": "S:imported_dispatch_quarantine",
    "console_hook_continuation_receipts": "S:imported_followup_quarantine",
    "console_machine_followup_receipts": "S:imported_followup_quarantine",
    "fleet_progress_messages": "S:imported_delivery_quarantine",
}


@dataclass(frozen=True)
class _PendingOwnerDuty:
    owner: str
    table: str
    original: tuple
    requirement: str


@dataclass(frozen=True)
class _IntegratedPlan:
    manifest: object = field(repr=False)
    sync: object = field(repr=False)
    workspace: object = field(repr=False)
    rows: tuple[_ContentRow, ...] = field(repr=False)
    source_rows: tuple[_ContentRow, ...] = field(repr=False)
    blockers: tuple[_ContentBlocker, ...]
    field_ownership: tuple
    pending_duties: tuple[_PendingOwnerDuty, ...]
    semantic_discharges: tuple[_SettingsSuccessor, ...]
    pending_handlers: tuple[str, ...] = ("A-runs", "owner-write", "public-cutover")

    @property
    def identities(self):
        return self.sync.identities

    def require_complete(self):
        # Semantic completion never proves native transaction/hold/receipt effects.
        raise ValueError("mobile_integrated_owner_handlers_pending")


def _workspace_rows(workspace, *, _read_budget):
    budget = _read_budget
    budget.value(workspace.source_rows)
    budget.value(workspace.rows)
    budget.container((len(workspace.source_rows) + len(workspace.rows)) * 8)
    budget.container(len(_TABLES) + len(_DOMAINS))
    specs = {s.table: s for s in _TABLES if s.handler == "I"}
    domain = {d.table: d for d in _DOMAINS if d.family == "workspaces"}
    original_rows = []
    transformed = []
    for row in workspace.source_rows:
        values = dict(row.values)
        original_rows.append(
            _ContentRow(
                row.table,
                tuple(values[k] for k in specs[row.table].identity_columns),
                row.values,
            )
        )
    for row in workspace.rows:
        values = dict(row.values)
        spec = specs[row.table]
        keys = tuple(values[k] for k in spec.identity_columns)
        if row.table in domain:
            d = domain[row.table]
            original = next(
                r.original
                for r in workspace.identities.reservations
                if r.domain == d.name and r.mapped == values[d.column]
            )
            keys = (original,)
        elif "workspace_id" in spec.identity_columns:
            original = next(
                r.original
                for r in workspace.identities.reservations
                if r.domain == "workspace" and r.mapped == values["workspace_id"]
            )
            keys = (original,)
        source = next(
            r for r in original_rows if r.table == row.table and r.original == keys
        )
        before = dict(source.values)
        dependencies = []
        if "workspace_id" in before and row.table != "workspace_records":
            dependencies.append(("workspace", before["workspace_id"]))
        if row.table == "workspace_memberships":
            dependencies.append((before["item_type"], before["item_id"]))
        if row.table == "workspace_rag_scopes":
            from .mobile_workspace_plan import _object

            budget.json(before["payload"])
            for item in _object(before["payload"])["items"]:
                budget.container(2)
                dependencies.append((item["source_type"], item["source_id"]))
        transformed.append(
            _ContentRow(row.table, keys, row.values, dependencies=tuple(dependencies))
        )
    return tuple(transformed), tuple(original_rows)


def _close_dependencies(rows, originals, blockers, *, _read_budget):
    budget = _read_budget
    budget.value(originals)
    budget.container(len(originals) * 4)
    known = {}
    for row in originals:
        values = dict(row.values)
        for domain in _DOMAINS:
            if domain.family != "runs" and domain.table == row.table:
                budget.container(5)
                known.setdefault((domain.name, values[domain.column]), set()).add(
                    (row.table, row.original)
                )
        for name, family, table, fields in _COMPOSITES:
            if family != "runs" and table == row.table:
                budget.container(len(fields) + 5)
                identity = tuple(values[column] for column, _kind in fields)
                known.setdefault((name, identity), set()).add((row.table, row.original))
    # Only I's proved Default equivalence may omit the imported Default row.
    if not any(
        b.table == "workspace_records" and b.original == ("workspace-default",)
        for b in blockers
    ):
        known.pop(("workspace", "workspace-default"), None)
    while True:
        budget.container(len(rows) * 8)
        accepted = {(r.table, r.original) for r in rows}
        retained = []
        for row in rows:
            missing = next(
                (
                    ref
                    for ref in row.dependencies
                    if ref in known and not known[ref] & accepted
                ),
                None,
            )
            if missing is None:
                retained.append(row)
            else:
                budget.container(6)
                blockers.append(
                    _ContentBlocker(
                        row.table,
                        row.original,
                        "integrated_dependency_blocked",
                        *missing,
                    )
                )
        if len(retained) == len(rows):
            return tuple(retained)
        rows = retained


def _plan_integrated(reads):
    from .mobile_captured_sqlite import _CapturedSQLite

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_integrated_native_snapshot_required")
    reads.check()
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    from .mobile_sync_plan import _plan_sync

    budget = _TraceReadBudget()
    sync = _plan_sync(reads, _read_budget=budget)
    trace = sync.trace
    manifest = reads.plan_manifest()
    from .mobile_workspace_plan import _plan_workspaces

    workspace = _plan_workspaces(reads, _identities=sync.identities)
    if workspace.identities is not sync.identities:
        raise ValueError("mobile_integrated_identity_binding_changed")
    workspace_rows, workspace_originals = _workspace_rows(
        workspace, _read_budget=budget
    )
    budget.container(len(trace.content.rows) * 4)
    budget.container(len(trace.authority_discharges) * 6)
    content_rows = {(r.table, r.original): r for r in trace.content.rows}
    for discharge in trace.authority_discharges:
        previous = content_rows[discharge.table, discharge.original]
        if (
            previous.pending_columns != ("assistant_authority_id",)
            or discharge.pending_columns
        ):
            raise ValueError("mobile_integrated_authority_slot_mismatch")
        budget.value(previous.values)
        budget.value(discharge.values)
        before, after = dict(previous.values), dict(discharge.values)
        if {k: v for k, v in after.items() if k != "assistant_authority_id"} != before:
            raise ValueError("mobile_integrated_authority_owner_changed")
        content_rows[discharge.table, discharge.original] = discharge
    budget.container((len(content_rows) + len(trace.rows) + len(workspace_rows)) * 3)
    budget.container(
        (
            len(trace.content.source_rows)
            + len(trace.source_rows)
            + len(workspace_originals)
        )
        * 3
    )
    budget.container(
        (
            len(trace.content.blockers)
            + len(trace.blockers)
            + len(sync.blockers)
            + len(workspace.blockers)
            + len(manifest.unsupported_records)
        )
        * 8
    )
    rows = tuple(content_rows.values()) + trace.rows + workspace_rows
    originals = trace.content.source_rows + trace.source_rows + workspace_originals
    blockers = list(trace.content.blockers + trace.blockers + sync.blockers)
    blockers.extend(
        _ContentBlocker(
            b.table, (b.identity,), b.reason, b.dependency_domain, b.dependency_id
        )
        for b in workspace.blockers
    )
    blockers.extend(
        _ContentBlocker(r.table, r.identity, "unsupported_populated_family", r.family)
        for r in manifest.unsupported_records
    )
    from .mobile_integrated_closure import _close_qualified_inputs
    from .mobile_integrated_semantics import _close_semantics

    # Each pass starts from the actual qualified projections, so semantic tags
    # removed on an earlier pass cannot conceal a newly lost prerequisite.
    base_rows = rows
    budget.container((len(blockers) + len(rows)) * 4)
    blocked = {(b.table, b.original) for b in blockers}
    accepted = {(r.table, r.original) for r in rows} - blocked
    while True:
        budget.container((len(base_rows) + len(blockers)) * 12)
        rows = tuple(r for r in base_rows if (r.table, r.original) in accepted)
        rows, semantic_discharges, semantic_blockers = _close_semantics(
            sync, workspace, rows, originals, _read_budget=budget
        )
        blockers.extend(semantic_blockers)
        for row in rows:
            budget.container(len(row.required_proofs) * 3 + 6)
            if any(
                proof.split(":", 1)[0]
                in {"I", "C1", "C2a", "C2b", "C2c", "C3", "T", "S"}
                and proof != _NATIVE_QUARANTINE_DUTIES.get(row.table)
                for proof in row.required_proofs
            ):
                blockers.append(
                    _ContentBlocker(
                        row.table, row.original, "integrated_semantic_proof_unclosed"
                    )
                )
        blocked = {(b.table, b.original) for b in blockers}
        rows = tuple(r for r in rows if (r.table, r.original) not in blocked)
        rows = _close_dependencies(rows, originals, blockers, _read_budget=budget)
        rows = _close_qualified_inputs(trace, rows, blockers, _read_budget=budget)
        remaining = {(r.table, r.original) for r in rows}
        if remaining == accepted:
            break
        accepted = remaining
    budget.container(
        (len(semantic_discharges) + len(sync.frontiers) + len(blockers)) * 8
    )
    semantic_discharges = tuple(
        proof
        for proof in semantic_discharges
        if (proof.table, proof.original) in accepted
    )
    for frontier in sync.frontiers:
        if frontier.owner not in accepted:
            blockers.append(
                _ContentBlocker(
                    "sync_log",
                    frontier.source_change_ids,
                    "integrated_sync_owner_blocked",
                )
            )
    sync = replace(
        sync, frontiers=tuple(f for f in sync.frontiers if f.owner in accepted)
    )
    blockers = list(dict.fromkeys(blockers))
    budget.container(len(_TABLES) * 12)
    specs = {s.table: s for s in _TABLES}
    ownership = dict(
        trace.content.field_ownership + trace.field_ownership + sync.field_ownership
    )
    ownership.update({s.table: s.columns for s in _TABLES if s.handler == "I"})
    expected = {
        s.table: s.columns for s in _TABLES if s.handler in {"I", "C", "T", "S"}
    }
    if ownership != expected:
        raise ValueError("mobile_integrated_field_ownership_incomplete")
    for row in rows:
        budget.container((len(row.values) + len(row.pending_columns)) * 4)
        if set(dict(row.values)) | set(row.pending_columns) != set(
            specs[row.table].columns
        ):
            raise ValueError("mobile_integrated_row_fields_incomplete")
    budget.container(len(semantic_discharges) * 5)
    duties = [
        _PendingOwnerDuty(
            "owner-write",
            s.table,
            s.original,
            "native_imported_message_passive_quarantine",
        )
        for s in semantic_discharges
        if s.handler == "S"
    ]
    for spec in _TABLES:
        if spec.handler == "A-runs" and not spec.canonical_only:
            for value in _rows(
                reads, "legacy", spec.family, spec.table, _read_budget=budget
            ):
                budget.container(len(spec.identity_columns) + 6)
                duties.append(
                    _PendingOwnerDuty(
                        "A-runs",
                        spec.table,
                        tuple(value[k] for k in spec.identity_columns),
                        "native_run_goal_handler_and_complete_closure",
                    )
                )
    # A row tag is an obligation, never proof of passive native recovery or a hold.
    for row in rows:
        for proof in row.required_proofs:
            budget.container(6)
            duties.append(
                _PendingOwnerDuty(
                    "A-runs" if proof.startswith("A-runs:") else "owner-write",
                    row.table,
                    row.original,
                    proof,
                )
            )
    for binding in workspace.binding_holds:
        budget.container(6)
        duties.append(
            _PendingOwnerDuty(
                "owner-write",
                "workspace_runtime_bindings",
                (binding,),
                "atomic_exact_binding_hold_and_native_unavailability",
            )
        )
    budget.container(32)
    duties.extend(
        (
            _PendingOwnerDuty(
                "A-runs",
                "*",
                (),
                "all_owner_preflight_before_files_or_first_data_effect",
            ),
            _PendingOwnerDuty(
                "A-runs",
                "*",
                (),
                "complete_reference_artifact_closure_and_native_quarantines",
            ),
            _PendingOwnerDuty(
                "owner-write",
                "*",
                (),
                "native_lifecycle_created_at_distinct_artifact_ids_atomic_receipts",
            ),
            _PendingOwnerDuty(
                "owner-write",
                "sync_log",
                (),
                "unique_native_current_previous_and_body_free_tombstone_frontiers",
            ),
            _PendingOwnerDuty(
                "public-cutover",
                "*",
                (),
                "reviewed_recovery_cutover_native_sync_continuation_and_zero_dispatch",
            ),
        )
    )
    from .mobile_trace_inputs import _check_trace_members

    _check_trace_members(reads)
    budget.container(len(blockers) + len(duties) + len(_TABLES) * 5 + 10)
    return _IntegratedPlan(
        manifest,
        sync,
        workspace,
        rows,
        originals,
        tuple(blockers),
        tuple((s.family, s.table, s.columns, s.handler) for s in _TABLES),
        tuple(duties),
        semantic_discharges,
    )
