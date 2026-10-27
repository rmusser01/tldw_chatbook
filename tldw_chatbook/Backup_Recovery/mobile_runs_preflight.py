"""Fixed captured Run/Goal preflight. Native owning effects remain mandatory."""

import json
from dataclasses import dataclass, field, replace

from .mobile_content_graph import (
    _ContentBlocker,
    _ContentRow,
    _mapped,
    _UnprovedReference,
)
from .mobile_identity_plan import _Domain, _Reservation, _reserve, _rows
from .mobile_integrated_plan import _PendingOwnerDuty
from .mobile_plan_manifest import _TABLES

_REFS = {
    "agent_definitions": (("id", "agent_definition"),),
    "agent_runs": (
        ("id", "run"),
        ("conversation_id", "conversation"),
        ("parent_run_id", "run"),
        ("resumed_from_run_id", "run"),
        ("assistant_message_id", "message"),
        ("work_chain_id", "work_chain"),
    ),
    "agent_run_steps": (("run_id", "run"),),
    "agent_worktrees": (
        ("run_id", "run"),
        ("workspace_id", "workspace"),
        ("binding_id", "binding"),
    ),
    "automatic_work_chains": (
        ("id", "work_chain"),
        ("conversation_id", "conversation"),
        ("allowance_root_chain_id", "work_chain"),
    ),
    "automatic_work_reservations": (
        ("id", "work_reservation"),
        ("chain_id", "work_chain"),
    ),
    "automatic_wake_attempts": (
        ("id", "wake_attempt"),
        ("chain_id", "work_chain"),
        ("conversation_id", "conversation"),
        ("generation_reservation_id", "work_reservation"),
    ),
    "automatic_chat_start_attempts": (
        ("id", "chat_start_attempt"),
        ("source_run_id", "run"),
        ("source_chain_id", "work_chain"),
        ("chain_id", "work_chain"),
        ("conversation_id", "conversation"),
        ("generation_reservation_id", "work_reservation"),
    ),
    "automatic_wake_claims": (("run_id", "run"), ("attempt_id", "wake_attempt")),
    "automatic_progress_wake_claims": (
        ("message_id", "fleet_message"),
        ("source_run_id", "run"),
        ("attempt_id", "wake_attempt"),
    ),
    "change_snapshots": (("id", "change_snapshot"), ("run_id", "run")),
    "change_notes": (
        ("id", "change_note"),
        ("run_id", "run"),
        ("delivered_by_run_id", "run"),
        ("snapshot_id", "change_snapshot"),
    ),
    "console_activity_receipts": (
        ("activity_id", "activity"),
        ("conversation_id", "conversation"),
        ("run_id", "run"),
        ("assistant_message_id", "message"),
    ),
}
_HOLDS = {
    "agent_runs": ("run", "id"),
    "automatic_work_chains": ("chain", "id"),
    "automatic_wake_attempts": ("wake_attempt", "id"),
    "automatic_chat_start_attempts": ("chat_start_attempt", "id"),
    "automatic_progress_wake_claims": ("progress", "message_id"),
    "goal_runs": ("goal", "id"),
}


@dataclass(frozen=True)
class _OperationalHistory:
    table: str
    original: tuple
    values: tuple = field(repr=False)
    associations: tuple = ()


@dataclass(frozen=True)
class _RunsPreflight:
    integrated: object = field(repr=False)
    identities: object = field(repr=False)
    rows: tuple[_ContentRow, ...] = field(repr=False)
    source_rows: tuple[_ContentRow, ...] = field(repr=False)
    operational_history: tuple[_OperationalHistory, ...] = field(repr=False)
    history: tuple = field(repr=False)
    history_observations: tuple = field(repr=False)
    goal_provenance: tuple
    note_provenance: tuple
    snapshot_allocator: object
    sync_frontiers: tuple
    semantic_discharges: tuple
    blockers: tuple[_ContentBlocker, ...]
    pending_duties: tuple[_PendingOwnerDuty, ...]
    field_ownership: tuple

    def require_complete(self):
        raise ValueError("mobile_runs_native_owners_pending")


def _history_observations(history, *, _read_budget):
    """Admit fixed shallow native replacements without traversing source bodies."""
    budget = _read_budget
    budget.container(len(history))
    observations = []
    for proof in history:
        count = len(proof.__dataclass_fields__)
        budget.container(count)  # replace expands its changes dictionary.
        budget.container(count)  # Constructor keyword dictionary.
        budget.container(count)  # New shallow result fields.
        observations.append(replace(proof, associations=(), owner_association=None))
    budget.container(len(observations))
    return tuple(observations)


def _plan_runs(reads, *, _read_budget=None):
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.console_command_receipts import (
        ConsoleCommandReceiptRepository,
    )

    from .mobile_captured_sqlite import _CapturedSQLite
    from .mobile_integrated_plan import _close_dependencies, _plan_integrated
    from .mobile_runs_goals import _plan_goal_rows
    from .mobile_trace_inputs import _check_trace_members

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_runs_native_snapshot_required")
    reads.check()
    _check_trace_members(reads)
    budget = _TraceReadBudget() if _read_budget is None else _read_budget
    if type(budget) is not _TraceReadBudget:
        raise TypeError("mobile_runs_native_budget_required")
    integrated = _plan_integrated(reads, _read_budget=budget)
    goals = _plan_goal_rows(reads, integrated.identities, _read_budget=budget)
    identities = goals.identities
    budget.container(len(_TABLES) * 6)
    specs = {s.table: s for s in _TABLES if s.handler == "A-runs"}
    remaining = tuple(_REFS) + (
        "automatic_work_runtime_owner",
        "console_command_receipts",
    )
    source = {
        t: _rows(reads, "legacy", "runs", t, _read_budget=budget) for t in remaining
    }
    canonical = {
        t: _rows(reads, "canonical", "runs", t, _read_budget=budget)
        for t in (
            "automatic_work_chains",
            "agent_definitions",
            "console_activity_receipts",
            "change_snapshots",
            "change_notes",
        )
    }
    budget.container(sum(len(v) for v in source.values()) * 12)
    budget.value(goals.source_rows)
    budget.container(
        len(goals.rows)
        + len(integrated.blockers)
        + len(goals.blockers)
        + len(integrated.pending_duties)
    )
    rows = list(goals.rows)
    originals = list(goals.source_rows)
    blockers = list(integrated.blockers + goals.blockers)
    duties = [
        d
        for d in integrated.pending_duties
        if d.requirement != "native_run_goal_handler_and_complete_closure"
    ]
    historical = []
    from .mobile_runs_snapshots import _extend_snapshots

    identities, note_provenance, snapshot_allocator, snapshot_blockers = (
        _extend_snapshots(reads, identities, source, canonical, _read_budget=budget)
    )
    blockers.extend(snapshot_blockers)
    duties.append(
        _PendingOwnerDuty(
            "owner-write",
            "change_snapshots",
            (),
            "native_snapshot_allocator_admission_before_effects",
        )
    )
    source_runs = {r["id"]: r for r in source["agent_runs"]}
    source_goals = {
        dict(r.values)["chain_id"]: dict(r.values)
        for r in goals.source_rows
        if r.table == "goal_runs"
    }
    domain = _Domain(
        "work_submission", "runs", "automatic_work_chains", "root_submission_id"
    )
    budget.container(len(source[domain.table]) * 4)
    goal_keys = {
        "goal:" + identities.mapped("goal", r["id"]): r["chain_id"]
        for r in source_goals.values()
    }
    occupied_submissions = {r["root_submission_id"] for r in canonical[domain.table]}
    source_submissions = {
        r["root_submission_id"]: r["id"] for r in source[domain.table]
    }
    coupled = {}
    for chain in source[domain.table]:
        goal = source_goals.get(chain["id"])
        if goal is None:
            continue
        if chain["root_submission_id"] != "goal:" + goal["id"]:
            blockers.append(
                _ContentBlocker(
                    domain.table, (chain["id"],), "native_goal_submission_unproved"
                )
            )
            continue
        target = "goal:" + identities.mapped("goal", goal["id"])
        if target in occupied_submissions or (
            target in source_submissions and source_submissions[target] != chain["id"]
        ):
            blockers.append(
                _ContentBlocker(
                    domain.table, (chain["id"],), "native_goal_submission_conflict"
                )
            )
        coupled[chain["root_submission_id"]] = target
    additions = []
    for reservation in _reserve(
        domain,
        source[domain.table],
        canonical[domain.table],
        identities.source_fingerprint,
        _read_budget=budget,
    ):
        if reservation.original in coupled:
            reservation = _Reservation(
                domain.name, reservation.original, coupled[reservation.original]
            )
        elif reservation.mapped in goal_keys:
            blockers.append(
                _ContentBlocker(
                    domain.table,
                    (source_submissions[reservation.original],),
                    "native_goal_submission_conflict",
                )
            )
        additions.append(reservation)
    budget.container(len(identities.reservations) + len(additions))
    identities = replace(
        identities, reservations=identities.reservations + tuple(additions)
    )
    definition_names = {
        r["name"] for r in canonical["agent_definitions"] if not r["deleted"]
    }

    for table in remaining:
        spec = specs[table]
        for original in source[table]:
            budget.value(original)
            budget.container(len(original) * 5)
            key = tuple(original[c] for c in spec.identity_columns)
            originals.append(_ContentRow(table, key, tuple(original.items())))
            dependencies = []

            def ref(name, value, *, _dependencies=dependencies):
                result = _mapped(identities, name, value)
                budget.container(2)
                _dependencies.append((name, value))
                return result

            try:
                if table == "automatic_work_runtime_owner":
                    historical.append(
                        _OperationalHistory(table, key, tuple(original.items()))
                    )
                    duties.append(
                        _PendingOwnerDuty(
                            "owner-write",
                            table,
                            key,
                            "historical_runtime_owner_only_canonical_owner_unchanged",
                        )
                    )
                    continue
                if table == "console_command_receipts":
                    budget.json(original["owner_epoch"])
                    budget.json(original["target_json"])
                    receipt = ConsoleCommandReceiptRepository._decode(original)
                    associations = []
                    run = getattr(receipt.target, "run_id", None)
                    if run is not None:
                        if run not in source_runs:
                            raise _UnprovedReference("run", run)
                        associations.append(("run", run, ref("run", run)))
                    historical.append(
                        _OperationalHistory(
                            table, key, tuple(original.items()), tuple(associations)
                        )
                    )
                    duties.append(
                        _PendingOwnerDuty(
                            "owner-write",
                            table,
                            key,
                            "native_passive_command_history_retention",
                        )
                    )
                    continue
                values = dict(original)
                for column, name in _REFS[table]:
                    if original[column] is not None:
                        values[column] = ref(name, original[column])
                if table == "agent_definitions":
                    if not original["deleted"] and original["name"] in definition_names:
                        blockers.append(
                            _ContentBlocker(table, key, "native_unique_conflict")
                        )
                    duties.append(
                        _PendingOwnerDuty(
                            "owner-write",
                            table,
                            key,
                            "native_definition_advertising_routing_and_current_authority",
                        )
                    )
                elif table == "agent_runs":
                    if original["spawn_event_id"] is not None:
                        values["spawn_event_id"] = _event_reference(
                            original["spawn_event_id"],
                            ref,
                            source_runs,
                            source["agent_run_steps"],
                            budget,
                        )
                    budget.json(original["steps"])
                    steps = json.loads(original["steps"])
                    if not isinstance(steps, list):
                        raise ValueError("run_steps_invalid")
                    if steps:
                        values["steps"] = _steps(
                            steps, ref, source_runs, source["agent_run_steps"], budget
                        )
                elif table == "agent_run_steps":
                    budget.json(original["payload"])
                    values["payload"] = _steps(
                        [json.loads(original["payload"])],
                        ref,
                        source_runs,
                        source["agent_run_steps"],
                        budget,
                        single=True,
                    )
                elif table == "automatic_work_chains":
                    goal = source_goals.get(original["id"])
                    if (
                        goal is not None
                        and original["root_submission_id"] == "goal:" + goal["id"]
                    ):
                        value = "goal:" + ref("goal", goal["id"])
                        if (
                            value in occupied_submissions
                            or goal_keys.get(value) != original["id"]
                        ):
                            raise ValueError("goal_submission_conflict")
                        values["root_submission_id"] = value
                    else:
                        values["root_submission_id"] = ref(
                            "work_submission", original["root_submission_id"]
                        )
                        if values["root_submission_id"] in goal_keys:
                            raise ValueError("goal_submission_conflict")
                elif table == "automatic_wake_attempts":
                    for column, name in (
                        ("run_ids_json", "run"),
                        ("message_ids_json", "fleet_message"),
                    ):
                        budget.json(original[column])
                        items = json.loads(original[column])
                        if not isinstance(items, list):
                            raise TypeError("wake_list_invalid")
                        budget.value(items, text_copies=3)
                        values[column] = json.dumps([ref(name, item) for item in items])
                    if original["attempt_kind"] == "goal_iteration":
                        if original["session_id"] != original["conversation_id"]:
                            raise ValueError("goal_session_unproved")
                        values["session_id"] = values["conversation_id"]
                elif table == "change_snapshots":
                    duties.append(
                        _PendingOwnerDuty(
                            "owner-write",
                            table,
                            key,
                            "native_original_root_snapshot_full_diff_revert_physical_dependency",
                        )
                    )
                elif table == "change_notes":
                    snapshot = next(
                        (
                            r
                            for r in source["change_snapshots"]
                            if r["id"] == original["snapshot_id"]
                        ),
                        None,
                    )
                    if snapshot is not None and (
                        snapshot["run_id"],
                        snapshot["root"],
                    ) != (original["run_id"], original["root"]):
                        raise ValueError("change_note_snapshot_owner_unproved")
                    duties.append(
                        _PendingOwnerDuty(
                            "owner-write",
                            table,
                            key,
                            "native_note_delivery_without_imported_execution",
                        )
                    )
                elif table == "agent_worktrees":
                    blockers.append(
                        _ContentBlocker(
                            table, key, "physical_worktree_dependencies_unproved"
                        )
                    )
                elif table == "console_activity_receipts":
                    if (
                        original["origin"] == "fleet_survivor"
                        and original["run_id"] is not None
                        and original["logical_outcome_id"]
                        == "fleet-run:" + original["run_id"]
                    ):
                        values["logical_outcome_id"] = "fleet-run:" + values["run_id"]
                    if any(
                        (r["origin"], r["logical_outcome_id"], r["transition_revision"])
                        == (
                            values["origin"],
                            values["logical_outcome_id"],
                            values["transition_revision"],
                        )
                        for r in canonical[table]
                    ):
                        blockers.append(
                            _ContentBlocker(table, key, "native_unique_conflict")
                        )
                    duties.append(
                        _PendingOwnerDuty(
                            "owner-write",
                            table,
                            key,
                            "native_passive_activity_history_and_truthful_acknowledgement",
                        )
                    )
                if table in _HOLDS:
                    kind, _column = _HOLDS[table]
                    duties.append(
                        _PendingOwnerDuty(
                            "owner-write",
                            table,
                            key,
                            "atomic_exact_import_hold:" + kind,
                        )
                    )
                budget.container(len(values) * 2 + len(dependencies) * 2)
                rows.append(
                    _ContentRow(
                        table,
                        key,
                        tuple(values.items()),
                        dependencies=tuple(dependencies),
                    )
                )
            except _UnprovedReference as error:
                blockers.append(
                    _ContentBlocker(
                        table, key, "dependency_unproven", error.domain, error.identity
                    )
                )
            except (ValueError, TypeError, KeyError):
                blockers.append(
                    _ContentBlocker(table, key, "native_run_content_invalid")
                )
    from .mobile_runs_history import _qualify_history

    history, history_blockers = _qualify_history(
        reads, source_runs, identities, _read_budget=budget
    )
    blockers.extend(history_blockers)
    budget.container((len(rows) + len(originals) + len(blockers)) * 4)
    all_rows = integrated.rows + tuple(rows)
    all_originals = integrated.source_rows + tuple(originals)
    blocked = {(b.table, b.original) for b in blockers}
    all_rows = tuple(r for r in all_rows if (r.table, r.original) not in blocked)
    from .mobile_integrated_closure import _close_qualified_inputs

    while True:
        previous_count = len(all_rows)
        budget.container((len(all_rows) + len(originals)) * 4)
        present = {(r.table, r.original) for r in all_rows}
        lost_runs = {
            r["run_id"]
            for r in source["agent_run_steps"]
            if ("agent_run_steps", (r["run_id"], r["seq"])) not in present
        }
        lost_goals = {
            dict(r.values)["goal_id"]
            for r in goals.source_rows
            if r.table != "goal_runs" and (r.table, r.original) not in present
        }
        retained = []
        for row in all_rows:
            reason = None
            if row.table == "agent_runs" and row.original[0] in lost_runs:
                reason = "native_run_step_input_lost"
            elif row.table == "goal_runs" and row.original[0] in lost_goals:
                reason = "native_goal_input_lost"
            if reason:
                blockers.append(_ContentBlocker(row.table, row.original, reason))
            else:
                retained.append(row)
        all_rows = tuple(retained)
        all_rows = _close_dependencies(
            all_rows, all_originals, blockers, _read_budget=budget, _include_runs=True
        )
        all_rows = _close_qualified_inputs(
            integrated.sync.trace, all_rows, blockers, _read_budget=budget
        )
        if len(all_rows) == previous_count:
            break
    budget.container(
        (
            len(all_rows)
            + len(integrated.sync.frontiers)
            + len(integrated.semantic_discharges)
        )
        * 4
    )
    accepted = {(r.table, r.original) for r in all_rows}
    frontiers = tuple(f for f in integrated.sync.frontiers if f.owner in accepted)
    discharges = tuple(
        p for p in integrated.semantic_discharges if (p.table, p.original) in accepted
    )
    history_observations = _history_observations(history, _read_budget=budget)
    budget.container(len(history))
    qualified_history = []
    for proof in history:
        for _ in range(
            len(proof.run_ids)
            + len(proof.historical_run_ids)
            + (proof.owner_run_id is not None)
        ):
            budget.container(1)  # Original Run identity tuple.
            budget.container(2)  # Accepted-owner lookup key.
        budget.container(2)  # Framed/historical reference groups.
        if (
            proof.owner_run_id is not None
            and ("agent_runs", (proof.owner_run_id,)) not in accepted
        ) or any(
            ("agent_runs", (run,)) not in accepted
            for group in (proof.run_ids, proof.historical_run_ids)
            for run in group
        ):
            budget.container(1)
            budget.container(5)
            budget.reserve(32)  # New blocker list entry.
            blockers.append(
                _ContentBlocker(
                    "agents.history",
                    (proof.logical_id,),
                    "native_history_run_dependency_blocked",
                )
            )
        else:
            qualified_history.append(proof)
    budget.container(len(qualified_history))
    qualified_members = {proof.logical_id for proof in qualified_history}
    budget.container(len(qualified_history))
    retained_history = []
    for proof in qualified_history:
        if any(segment not in qualified_members for segment in proof.segment_ids):
            budget.container(1)
            budget.container(5)
            budget.reserve(32)
            blockers.append(
                _ContentBlocker(
                    "agents.history",
                    (proof.logical_id,),
                    "native_history_segment_dependency_blocked",
                )
            )
        else:
            retained_history.append(proof)
    budget.container(len(retained_history))
    history = tuple(retained_history)
    budget.container(len(historical))
    closed_history = []
    for fact in historical:
        budget.container(len(fact.associations))
        associations = []
        for domain, original, mapped in fact.associations:
            budget.container(1)
            budget.container(2)
            if ("agent_runs", (original,)) in accepted:
                budget.container(3)
                associations.append((domain, original, mapped))
            else:
                budget.container(5)
                budget.reserve(32)
                blockers.append(
                    _ContentBlocker(
                        fact.table,
                        fact.original,
                        "native_operational_run_dependency_blocked",
                        domain,
                        original,
                    )
                )
        budget.container(len(associations))
        count = len(fact.__dataclass_fields__)
        budget.container(count)  # Expanded replace changes; values stay shared.
        budget.container(count)  # Constructor keywords.
        budget.container(count)  # Shallow result fields.
        closed_history.append(replace(fact, associations=tuple(associations)))
    historical = closed_history
    budget.container(len(historical))
    historical_owners = set()
    for fact in historical:
        budget.container(2)
        historical_owners.add((fact.table, fact.original))
    budget.container(len(duties))
    retained_duties = []
    for duty in duties:
        if not duty.original:
            retained_duties.append(duty)
        else:
            budget.container(2)
            owner = (duty.table, duty.original)
            if owner in accepted or owner in historical_owners:
                retained_duties.append(duty)
    duties = retained_duties
    for frontier in integrated.sync.frontiers:
        if frontier.owner not in accepted:
            blockers.append(
                _ContentBlocker(
                    "sync_log",
                    frontier.source_change_ids,
                    "integrated_sync_owner_blocked",
                )
            )
    for row in goals.source_rows:
        if row.table == "goal_runs" and (row.table, row.original) in accepted:
            budget.container(10)
            duties.extend(
                _PendingOwnerDuty("owner-write", row.table, row.original, requirement)
                for requirement in (
                    "atomic_exact_import_hold:goal",
                    "native_goal_current_binding_authority_and_fresh_result_review",
                )
            )
    for row in all_rows:
        if set(dict(row.values)) | set(row.pending_columns) != set(
            next(s.columns for s in _TABLES if s.table == row.table)
        ):
            raise ValueError("mobile_runs_fields_incomplete")
    _check_trace_members(reads)
    budget.container(
        (
            len(all_rows)
            + len(all_originals)
            + len(blockers)
            + len(duties)
            + len(historical)
            + len(history)
        )
        * 4
    )
    # The integrated manifest already names every fixed owner, including A-runs.
    # Preserve its exact family/table/columns/handler contract and provenance.
    ownership = integrated.field_ownership
    return _RunsPreflight(
        integrated,
        identities,
        all_rows,
        all_originals,
        tuple(historical),
        history,
        history_observations,
        goals.provenance,
        note_provenance,
        snapshot_allocator,
        frontiers,
        discharges,
        tuple(dict.fromkeys(blockers)),
        tuple(duties),
        ownership,
    )


def _event_reference(value, ref, runs, steps, budget):
    budget.value(value)
    if not isinstance(value, str):
        raise TypeError("run_event_invalid")
    matches = []
    for run, record in runs.items():
        budget.value(run, text_copies=4)
        if value == "agent-run:" + run:
            matches.append((run, None))
        budget.json(record["steps"])
        legacy = json.loads(record["steps"])
        budget.container(len(legacy) + len(steps))
        owned = list(legacy)
        for row in steps:
            if row["run_id"] == run:
                budget.json(row["payload"])
                owned.append(json.loads(row["payload"]))
        for item in owned:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            if type(index) is int and index >= 0:
                budget.value((run, index), text_copies=3)
                if value == "agent-step:" + run + ":" + str(index):
                    matches.append((run, index))
    if len(matches) != 1:
        raise ValueError("run_event_unproved")
    run, index = matches[0]
    mapped = ref("run", run)
    return (
        "agent-run:" + mapped
        if index is None
        else "agent-step:" + mapped + ":" + str(index)
    )


def _steps(items, ref, runs, steps, budget, *, single=False):
    from tldw_chatbook.Agents.agent_models import AgentStep

    budget.value(items, text_copies=3)
    values = []
    for item in items:
        AgentStep(**item)
        data = dict(item)
        for column in ("parent_event_id", "source_event_id", "replacement_event_id"):
            if data.get(column) is not None:
                data[column] = _event_reference(data[column], ref, runs, steps, budget)
        values.append(data)
    budget.value(values, text_copies=3)
    return json.dumps(values[0] if single else values)
