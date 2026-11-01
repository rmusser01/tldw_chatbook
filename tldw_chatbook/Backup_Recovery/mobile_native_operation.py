"""Fixed native owning transactions from one authenticated, retired input cohort."""

from datetime import UTC

from . import archive_reader, control_records
from .mobile_content_graph import _ContentBlocker


class _NativeOperationBlocked(ValueError):
    """Exact body-free outstanding work, never an import-completion receipt."""

    def __init__(self, blockers):
        super().__init__("mobile_native_operation_blocked")
        self.blockers = blockers


def _require_readers_retired(session, *, _reader=None):
    """Check actual native custody, including scopes whose local marker is gone."""
    from .mobile_captured_sqlite import _CapturedSQLite
    from .mobile_consolidation_apply import (
        _CapturedMobile,
        _MobileCapturePartition,
        _SourceClaim,
    )
    from .mobile_history_files import _HistoryFiles
    from .storage_admission import (
        _CaptureFileDescriptors,
        _CaptureScope,
        _DiscoveryScope,
        _local,
        _mobile_admission,
        _MobileScope,
        _PreviewScope,
    )

    session._check()
    if _reader is not None and (
        type(_reader) is not _CapturedSQLite
        or _reader.session is not session
        or _reader not in session._scopes
        or getattr(_local, "mobile_sqlite_scope", None) is not _reader
    ):
        raise ValueError("mobile_native_reader_changed")
    bindings = None
    if _reader is not None:
        admission = _mobile_admission(session)
        if (
            admission is not _reader.admission
            or admission[3] is not _reader.discovery_bindings
        ):
            raise ValueError("mobile_sqlite_admission_changed")
        bindings = admission[3]
    for scope in session._scopes:
        if type(scope) not in (
            _CapturedMobile,
            _MobileCapturePartition,
            _SourceClaim,
            _CapturedSQLite,
            _HistoryFiles,
            _CaptureScope,
            _DiscoveryScope,
            _MobileScope,
            _CaptureFileDescriptors,
        ):
            raise ValueError("mobile_native_scope_unqualified")
        if scope is _reader:
            if scope.custody.close_failed or any(
                getattr(resource, "close_failed", False)
                or getattr(resource, "resource_close_failed", False)
                for resource in scope.resources
            ):
                raise ValueError("mobile_native_readers_not_retired")
            continue
        if (
            type(scope) is _DiscoveryScope
            and scope.active
            and bindings is not None
            and scope in bindings
        ):
            expected = bindings[scope]
            copies = scope.sqlite_copies
            if (
                len(expected) != 4
                or expected[0] is not _reader
                or scope.session is not session
                or getattr(_local, "discovery_scope", None) is not scope
                or type(copies) is not _PreviewScope
                or expected[1] is not copies
                or expected[2] is not scope.resources
                or expected[3] is not copies.resources
                or copies.native_scope is not scope
            ):
                raise ValueError("mobile_native_discovery_child_changed")
            scope.check()
            copies.check()
            _reader.read_budget.container(4)
            for holder in (scope, copies):
                if (
                    getattr(holder, "close_failed", False)
                    or getattr(holder, "resource_close_failed", False)
                    or any(
                        getattr(resource, "close_failed", False)
                        or getattr(resource, "resource_close_failed", False)
                        for resource in holder.resources
                    )
                ):
                    raise ValueError("mobile_native_readers_not_retired")
            continue
        if type(scope) is _MobileScope and any(
            getattr(scope, name, None) is not None
            for name in (
                "execution_owner",
                "execution_use",
                "execution_stream",
                "execution_fd",
                "mode_fd",
            )
        ):
            raise ValueError("mobile_native_readers_not_retired")
        if type(scope) is _SourceClaim and scope.custody.close_failed:
            raise ValueError("mobile_native_readers_not_retired")
        if (
            getattr(scope, "close_failed", False)
            or getattr(scope, "resource_close_failed", False)
            or getattr(scope, "resources", ())
        ):
            raise ValueError("mobile_native_readers_not_retired")
        if (
            type(scope)
            in (
                _CapturedSQLite,
                _HistoryFiles,
                _CaptureScope,
                _DiscoveryScope,
                _MobileScope,
            )
            and scope.active
        ):
            raise ValueError("mobile_native_readers_not_retired")
        if type(scope) is _CaptureScope and scope.sqlite_directory is not None:
            raise ValueError("mobile_native_readers_not_retired")
        if type(scope) is _CapturedSQLite and scope.roots:
            raise ValueError("mobile_native_readers_not_retired")
        if type(scope) is _CaptureFileDescriptors and (
            scope.fds or scope.directory_iterators
        ):
            raise ValueError("mobile_native_readers_not_retired")
        if type(scope) is _DiscoveryScope and scope.sqlite_copies is not None:
            copies = scope.sqlite_copies
            if copies.active or copies.resources or copies.directory is not None:
                raise ValueError("mobile_native_readers_not_retired")


def _native_capture_check(captured, cancel, *, _pricing_inputs=None):
    """Recheck the original captured cohort before native owner effects."""
    from .journal import _mobile_state
    from .mobile_consolidation import _path_witnesses
    from .storage_admission import _mobile_check_witnesses

    captured.check()
    archive_reader._check(cancel)
    if captured.claim.custody.close_failed:
        raise ValueError("mobile_native_claim_retirement_uncertain")
    if (
        _path_witnesses((captured.preview.source_dir,))[0][1]
        != captured.preview.source_witnesses
    ):
        raise ValueError("mobile_source_changed")
    with captured.journal._locked(exclusive=False) as fd:
        records = captured.journal._records(fd)
    prepared, witnesses = _mobile_state(records)
    control_records._mobile_pending(captured.session, captured.journal, prepared)
    if prepared.source_fingerprint != captured.preview.source_fingerprint:
        raise ValueError("mobile_native_source_changed")
    _mobile_check_witnesses(records, witnesses, recovering_mode=True)
    archive_reader._check(cancel)
    if _pricing_inputs is not None:
        if type(_pricing_inputs) is not list or _pricing_inputs:
            raise ValueError("mobile_terminal_pricing_inputs_changed")
        # Caller-prepaid references only; no copy, readiness or owner authority.
        _pricing_inputs.append(records)
        _pricing_inputs.append(prepared)


def _native_preopen(captured, cancel, *, _pricing_inputs=None):
    """Refuse retained resources before even native path or journal creation."""
    if _pricing_inputs is None:
        _native_capture_check(captured, cancel)
    else:
        _native_capture_check(captured, cancel, _pricing_inputs=_pricing_inputs)
    _require_readers_retired(captured.session)


def _native_trace_refusals(plan, budget):
    """Refuse the two native lifecycles that cannot carry portable authority."""
    source = plan.integrated.sync.trace.inputs.source.rows
    budget.container(len(source) * 4)
    by_table = {}
    for row in source:
        if row.table in {
            "console_trace_owners",
            "console_trace_segments",
            "console_trace_calls",
            "console_trace_surface_nodes",
            "console_trace_surface_replacements",
            "console_trace_events",
            "console_trace_semantic_revisions",
            "messages",
        }:
            budget.container(len(row.values) * 2 + 2)
            by_table.setdefault(row.table, []).append(dict(row.values))
    owners = by_table.get("console_trace_owners", ())
    segments = by_table.get("console_trace_segments", ())
    budget.container(len(owners) * 3 + len(segments) * 2)
    root_owners = {row["root_segment_id"]: row for row in owners}
    parents = {row["segment_id"]: row["parent_segment_id"] for row in segments}
    blocked, populated = [], set()
    for table in (
        "console_trace_calls",
        "console_trace_surface_nodes",
        "console_trace_surface_replacements",
        "console_trace_events",
        "console_trace_segments",
    ):
        for row in by_table.get(table, ()):
            if (
                table == "console_trace_calls"
                and row["reservation_provenance"] == "post_dispatch_promoted"
            ):
                budget.container(6)
                blocked.append(
                    _ContentBlocker(
                        table,
                        (row["call_id"],),
                        "native_promoted_trace_current_voice_authority_required",
                    )
                )
            segment = (
                row["parent_segment_id"]
                if table == "console_trace_segments"
                else row["segment_id"]
            )
            budget.container(len(parents))
            seen = set()
            while segment is not None and segment not in seen:
                seen.add(segment)
                owner = root_owners.get(segment)
                if owner is not None:
                    if owner["attached"] == 0:
                        budget.container(1)
                        populated.add(owner["owner_id"])
                    break
                segment = parents.get(segment)
    # A retained original conversation requires a native relationship witness.
    # Coincidental canonical IDs or any caller-selected replacement are no proof.
    for owner in owners:
        if owner["owner_id"] not in populated:
            continue
        budget.container(len(source) * 2)
        conversations = set()
        for node in by_table.get("console_trace_surface_nodes", ()):
            segment = node["segment_id"]
            for _ in range(len(parents) + 1):
                if segment in root_owners or segment is None:
                    break
                segment = parents.get(segment)
            if segment != owner["root_segment_id"]:
                continue
            for revision in by_table.get("console_trace_semantic_revisions", ()):
                if revision["revision_id"] == node["semantic_revision_id"]:
                    conversations.add(revision["source_conversation_id"])
        for call in by_table.get("console_trace_calls", ()):
            if call["owner_id"] == owner["owner_id"]:
                for message in by_table.get("messages", ()):
                    if message["id"] == call["turn_id"]:
                        conversations.add(message["conversation_id"])
        proved = tuple(
            r
            for r in plan.identities.reservations
            if r.domain == "conversation" and r.original in conversations
        )
        if len(proved) != 1:
            budget.container(6)
            blocked.append(
                _ContentBlocker(
                    "console_trace_owners",
                    (owner["owner_id"],),
                    "native_detached_trace_original_owner_unproved",
                )
            )
    budget.container(len(blocked))
    return tuple(blocked)


def _goal_note_closure(plan, budget):
    """Check the actual mapped native graph that this operation will hold atomically."""
    tables = {
        "goal_runs",
        "goal_iterations",
        "goal_checkpoints",
        "goal_evidence",
        "goal_reports",
        "goal_payload_reservations",
        "goal_waits",
        "automatic_work_chains",
        "automatic_wake_attempts",
        "agent_runs",
        "change_notes",
        "change_snapshots",
    }
    budget.container(len(plan.rows) * 4 + len(plan.note_provenance) * 2)
    values = {}
    by_original = {}
    for row in plan.rows:
        if row.table in tables:
            budget.container(len(row.values) * 2 + 4)
            data = dict(row.values)
            values.setdefault(row.table, []).append(data)
            by_original[row.table, row.original] = data
    budget.container(sum(len(rows) for rows in values.values()) * 3)
    goals = {row["id"]: row for row in values.get("goal_runs", ())}
    chains = {row["id"]: row for row in values.get("automatic_work_chains", ())}
    attempts = {row["id"]: row for row in values.get("automatic_wake_attempts", ())}
    runs = {row["id"]: row for row in values.get("agent_runs", ())}
    admitted = set()
    for row in plan.rows:
        if row.table not in tables:
            continue
        data = by_original[row.table, row.original]
        if row.table.startswith("goal_"):
            goal = data if row.table == "goal_runs" else goals.get(data["goal_id"])
            chain = chains.get(goal["chain_id"]) if goal else None
            valid = bool(
                goal and chain and chain["conversation_id"] == goal["conversation_id"]
            )
            attempt_id = data.get("attempt_id")
            if attempt_id is not None:
                attempt = attempts.get(attempt_id)
                valid = valid and bool(
                    attempt
                    and attempt["chain_id"] == goal["chain_id"]
                    and attempt["conversation_id"] == goal["conversation_id"]
                )
            run_id = data.get("run_id")
            if run_id is not None:
                run = runs.get(run_id)
                valid = valid and bool(
                    run
                    and run["work_chain_id"] == goal["chain_id"]
                    and run["conversation_id"] == goal["conversation_id"]
                )
            if valid:
                budget.container(3)
                admitted.add((row.table, row.original))
        elif row.table == "change_notes":
            # A dangling snapshot is retained as a locator, never synthesized.
            # The imported run is held; the native delivery consumer separately
            # requires the actual delivering run's current authority in its TX.
            records = [
                record
                for record in plan.note_provenance
                if record.original_note == row.original[0]
            ]
            budget.container(len(records))
            if len(records) != 1 or data["run_id"] not in runs:
                continue
            record = records[0]
            run_mapping = next(
                (
                    item
                    for item in plan.identities.reservations
                    if item.domain == "run" and item.original == record.original_run
                ),
                None,
            )
            if (
                run_mapping is None
                or run_mapping.mapped != data["run_id"]
                or record.original_root != data["root"]
                or record.mapped_snapshot != data["snapshot_id"]
            ):
                continue
            snapshots = [
                item
                for item in values.get("change_snapshots", ())
                if item["id"] == data["snapshot_id"]
            ]
            budget.container(len(snapshots))
            if record.missing:
                valid = not snapshots
            else:
                valid = data["snapshot_id"] is None or (
                    len(snapshots) == 1
                    and (snapshots[0]["run_id"], snapshots[0]["root"])
                    == (data["run_id"], data["root"])
                )
            if valid:
                budget.container(3)
                admitted.add((row.table, row.original))
    return admitted


def _passive_chat_closure(plan, budget):
    """Qualify exact native checkpoint and fleet rows for committed consumers."""
    from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository

    from .mobile_content_workflows import _checkpoint_join

    budget.container(len(plan.rows) * 6 + len(plan.identities.reservations) * 3)
    rows, by_original = {}, {}
    for row in plan.rows:
        if row.table in {
            "messages",
            "conversations",
            "console_dispatch_checkpoints",
            "fleet_progress_messages",
            "console_hook_continuation_receipts",
            "console_machine_followup_receipts",
            "agent_runs",
            "automatic_work_chains",
        }:
            budget.container(len(row.values) * 2 + 4)
            value = dict(row.values)
            rows.setdefault(row.table, []).append(value)
            by_original[row.table, row.original] = value
    budget.container(sum(len(value) for value in rows.values()) * 4)
    messages = {row["id"]: row for row in rows.get("messages", ())}
    conversations = {row["id"]: row for row in rows.get("conversations", ())}
    runs = {row["id"]: row for row in rows.get("agent_runs", ())}
    identities = {(row.domain, row.mapped) for row in plan.identities.reservations}
    chains = {row["id"]: row for row in rows.get("automatic_work_chains", ())}
    admitted = set()
    # Current checkpoint-free recovery and continuation owner lookup both use
    # the complete committed message census before advertising any local owner.
    passive_messages = {
        row["id"]
        for row in messages.values()
        if row["role"] == "assistant"
        and ("message", row["id"]) in identities
        and (
            row["assistant_generation_state"]
            in {"accepted", "dispatch_started", "continuation_active"}
            or row["provider_continuation_json"] is not None
        )
    }
    for row in plan.rows:
        if row.table == "console_dispatch_checkpoints":
            value = by_original[row.table, row.original]
            assistant = messages.get(value["assistant_message_id"])
            user = messages.get(value["user_message_id"])
            if (
                not assistant
                or not user
                or value["conversation_id"] not in conversations
            ):
                continue
            if any(
                ("message", identity) not in identities
                for identity in (assistant["id"], user["id"])
            ):
                continue
            # The required live consumer offers only explicit native Discard.
            # Its durable association is emitted with these exact message maps
            # and checkpoint row in the same Chat owner transaction.
            budget.container(len(value) + 15)
            joined = _checkpoint_join(value, conversations, messages)
            for name in (
                "frozen_authority_json",
                "resolved_destination_json",
                "reconstructability_json",
                "user_metadata_json",
                "provider_continuation_json",
                "thinking_blocks_json",
            ):
                if joined.get(name) is not None:
                    budget.json(joined[name])
            checkpoint, error = ConsoleDispatchRepository._checkpoint_from_row(joined)
            if (
                checkpoint is None
                or error is not None
                or assistant["provider_continuation_json"] is not None
            ):
                continue
            budget.container(5)
            admitted.add((row.table, row.original))
            passive_messages.add(assistant["id"])
        elif row.table in {
            "console_hook_continuation_receipts",
            "console_machine_followup_receipts",
        }:
            value = by_original[row.table, row.original]
            parent = messages.get(value["parent_assistant_message_id"])
            child = messages.get(value["assistant_message_id"])
            if any(
                message is None
                or message["role"] != "assistant"
                or message["conversation_id"] != value["conversation_id"]
                or ("message", message["id"]) not in identities
                for message in (parent, child)
            ):
                continue
            budget.container(3)
            admitted.add((row.table, row.original))
        elif row.table == "fleet_progress_messages":
            value = by_original[row.table, row.original]
            child, parent = runs.get(value["run_id"]), runs.get(value["parent_run_id"])
            if (
                child is None
                or parent is None
                or child["parent_run_id"] != parent["id"]
                or child["conversation_id"] != value["conversation_id"]
                or parent["conversation_id"] != value["conversation_id"]
                or ("fleet_message", value["message_id"]) not in identities
            ):
                continue
            if value["chain_id"] is not None:
                chain = chains.get(value["chain_id"])
                if (
                    chain is None
                    or chain["conversation_id"] != value["conversation_id"]
                    or parent["work_chain_id"] != chain["id"]
                    or child["work_chain_id"] not in (None, chain["id"])
                ):
                    continue
            budget.container(3)
            admitted.add((row.table, row.original))
    budget.container(len(passive_messages) + len(admitted))
    return admitted, passive_messages


def _failed_snapshot_rows(plan, budget):
    """Qualify native no-Git failure history against its actual mapped Run."""
    budget.container(len(plan.rows) * 3 + len(plan.identities.reservations) * 3)
    runs = {dict(row.values)["id"] for row in plan.rows if row.table == "agent_runs"}
    reserved = {
        item.mapped for item in plan.identities.reservations if item.domain == "run"
    }
    admitted = set()
    for row in plan.rows:
        if row.table != "change_snapshots":
            continue
        budget.container(len(row.values) * 2)
        value = dict(row.values)
        if (
            type(value["tracking_error"]) is str
            and value["tracking_error"]
            and value["run_id"] in runs
            and value["run_id"] in reserved
        ):
            budget.container(2)
            admitted.add(row.original)
    return admitted


def _required_handlers(plan, budget):
    """Refuse actual unresolved native consumer/physical dependencies."""
    budget.container(len(plan.blockers) + len(plan.rows) * 3 + len(plan.history) * 2)
    blockers = list(plan.blockers + _native_trace_refusals(plan, budget))
    goal_notes = _goal_note_closure(plan, budget)
    passive_rows, passive_messages = _passive_chat_closure(plan, budget)
    failed_snapshots = _failed_snapshot_rows(plan, budget)
    for row in plan.rows:
        for column in row.pending_columns:
            budget.container(6)
            blockers.append(
                _ContentBlocker(
                    row.table,
                    row.original,
                    "native_field_owner_required",
                    "column",
                    column,
                )
            )
        for proof in row.required_proofs:
            if (
                (
                    row.table == "console_dispatch_checkpoints"
                    and proof == "S:imported_dispatch_quarantine"
                )
                or (
                    row.table == "fleet_progress_messages"
                    and proof
                    in {
                        "S:imported_delivery_quarantine",
                        "A-runs:fleet_lineage",
                        "A-runs:run",
                        "A-runs:work_chain",
                    }
                )
                or (
                    row.table
                    in {
                        "console_hook_continuation_receipts",
                        "console_machine_followup_receipts",
                    }
                    and proof == "S:imported_followup_quarantine"
                )
            ) and (row.table, row.original) in passive_rows:
                continue
            if (
                proof == "A-runs:exact_goal_chain_attempt_holds"
                and (row.table, row.original) in goal_notes
            ):
                continue
            budget.container(6)
            blockers.append(
                _ContentBlocker(
                    row.table,
                    row.original,
                    "native_consumer_required",
                    "owner-write",
                    proof,
                )
            )
        if row.table == "agent_worktrees" or (
            row.table == "change_snapshots" and row.original not in failed_snapshots
        ):
            budget.container(6)
            blockers.append(
                _ContentBlocker(
                    row.table, row.original, "native_physical_dependency_unproved"
                )
            )
        if row.table == "change_notes" and (row.table, row.original) not in goal_notes:
            budget.container(6)
            blockers.append(
                _ContentBlocker(
                    row.table, row.original, "native_note_delivery_consumer_required"
                )
            )
    for proof in plan.semantic_discharges:
        if proof.handler == "S":
            if (
                proof.table == "messages"
                and proof.domain in {"dispatch_owner", "provider_continuation"}
                and proof.mapped_id in passive_messages
            ):
                continue
            budget.container(6)
            blockers.append(
                _ContentBlocker(
                    proof.table,
                    proof.original,
                    "native_imported_message_passive_quarantine_required",
                )
            )
    budget.container(len(blockers))
    return tuple(blockers)


def _consolidate(captured, *, password, cancel):
    """Derive complete input in this call; retain no reusable readiness object."""
    from datetime import datetime

    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.console_command_history import (
        _commit_command_history,
        _prepare_command_history,
    )

    from .mobile_native_receipts import _encode_receipts, _receipt_children
    from .mobile_native_rows import (
        _advance_snapshot_allocator,
        _insert_row,
        _prepare_rows,
        _verify_row,
    )
    from .mobile_native_sync import _apply_sync, _prepare_sync, _verify_sync
    from .mobile_plan_manifest import _TABLES
    from .mobile_runs_preflight import _HOLDS, _plan_runs

    captured.check()
    archive_reader._check(cancel)
    if captured.session._mobile_attempted_journal is not None:
        raise ValueError("mobile_native_operation_already_attempted")
    # Denial-only session state survives reminting this operation's lifecycle.
    captured.session._mobile_attempted_journal = captured.journal
    from .storage_admission import _mobile_admission, _mobile_coordinator_budget

    admission = _mobile_admission(captured.session)
    budget = admission[2]
    if type(budget) is not _TraceReadBudget:
        raise ValueError("mobile_native_budget_required")
    budget.container(2)
    admission[3] = {}
    budget.container(12)
    budget.value(captured.preview.source_witnesses)
    budget.value(captured.preview.destination_witnesses)
    if captured.preview.has_data:
        _native_preopen(captured, cancel)
    else:
        _mobile_coordinator_budget(captured.session).container(2)
        terminal_inputs = []
        _native_preopen(captured, cancel, _pricing_inputs=terminal_inputs)
    budget.container(len(captured.session._scopes))
    issued_scopes = tuple(captured.session._scopes)
    cohort = captured.binding
    source_fingerprint = captured.preview.source_fingerprint
    operation_id = captured.journal.operation_id

    from .mobile_terminal_pricing import _check_terminal_dimensions, _preprice_terminal

    def commit_terminal():
        # This closure retains the exact owner receipts verified by this call.
        # No caller supplies completion evidence, a plan, or a readiness token.
        import hashlib
        import json
        from pathlib import Path

        from tldw_chatbook.Utils.platform_files import os

        from . import bootstrap
        from .journal import (
            _mobile_ref,
            _mobile_sidecar_state,
            _mobile_state,
            observe_artifact,
        )
        from .mobile_consolidation_apply import _SourceClaim
        from .mobile_journal_frame import _locked_mobile
        from .native_files import flush_directory
        from .storage_admission import (
            _mobile_check_witnesses,
            _mobile_coordinator_budget,
        )

        terminal_budget = _mobile_coordinator_budget(captured.session)

        _native_preopen(captured, cancel)
        custody = captured.claim.custody
        terminal_budget.container(3)
        owners = []
        if captured.preview.has_data:
            for family, owner_id in (
                ("chat", "db.chachanotes.primary"),
                ("workspaces", "db.workspaces"),
                ("runs", "db.agent_runs"),
            ):
                raw = next(raw for key, raw in receipts[family] if key == family)
                terminal_budget.json(raw)
                terminal_budget.value(raw)
                master = json.loads(raw)
                terminal_budget.container(6)
                owners.append(
                    {
                        "owner_id": owner_id,
                        "master_sha256": hashlib.sha256(
                            raw.encode("utf-8")
                        ).hexdigest(),
                        "entry_count": master["entry_count"],
                        "entries_sha256": master["entries_sha256"],
                        "command_history_count": master["command_history_count"],
                    }
                )
        with _locked_mobile(
            captured.journal, captured.session, custody, cancel
        ) as frame:
            records = captured.journal._records(
                frame.parent, _budget=terminal_budget, _custody=custody
            )
            prepared, witnesses = _mobile_state(records)
            _SourceClaim._check_binding(captured.claim)
            control_records._mobile_pending(
                captured.session, captured.journal, prepared
            )
            _mobile_check_witnesses(records, witnesses, recovering_mode=True)
            terminal_budget.value(witnesses, text_copies=3)
            terminal_budget.container(3)
            targets = []
            for target, version in zip(prepared.targets, (82, 9, 26), strict=True):
                terminal_budget.value(target.model_dump(), text_copies=2)
                targets.append(
                    {
                        "owner_id": target.owner_id,
                        "witnesses": [witnesses[row.path] for row in target.witnesses],
                        "sidecars": list(
                            _mobile_sidecar_state(records, target.owner_id).values()
                        ),
                        "schema_version": version,
                    }
                )
            root = Path(prepared.bootstrap_root)
            name = "pending-" + bootstrap._key(operation_id) + ".json"
            with custody.pinned_directory(root) as pending_parent:
                info = os.fstat(pending_parent)
                pending_info = os.stat(
                    name, dir_fd=pending_parent, follow_symlinks=False
                )
                if pending_info.st_size > bootstrap.MAX_RECORD:
                    raise ValueError("mobile_pending_scope_changed")
                terminal_budget.reserve(4096 + pending_info.st_size * 8)
                pending = observe_artifact(root / name, _custody=custody)
                terminal_budget.container(len(records) * 3 + 20)
                for record in records:
                    if record.event in {
                        "mobile_prepared",
                        "mobile_claim_observed",
                        "mobile_capture_verified",
                        "mobile_history_verified",
                    }:
                        terminal_budget.value(record.evidence, text_copies=4)
                terminal = {
                    "version": 1,
                    "kind": "imported" if captured.preview.has_data else "no_data",
                    "source_fingerprint": source_fingerprint,
                    "destination_identity": captured.preview.destination.identity,
                    "prepared": _mobile_ref(records[0]),
                    "claim": _mobile_ref(
                        next(
                            row
                            for row in records
                            if row.event == "mobile_claim_observed"
                        )
                    ),
                    "captures": [
                        {
                            "purpose": row.evidence["purpose"],
                            "verified": _mobile_ref(row),
                        }
                        for row in records
                        if row.event == "mobile_capture_verified"
                    ],
                    "targets": targets,
                    "owners": owners,
                    "history": next(
                        (
                            _mobile_ref(row)
                            for row in records
                            if row.event == "mobile_history_verified"
                        ),
                        None,
                    ),
                    "pending": pending,
                    "bootstrap": {
                        "path": str(root),
                        "device": info.st_dev,
                        "inode": info.st_ino,
                    },
                }
                _check_terminal_dimensions(terminal, terminal_budget)
                frame.append("mobile_committed", terminal)
                # The journal barrier completed. Retire only the exact witnessed
                # pointer while bootstrap.unbound and the source flock remain held.
                current = os.stat(name, dir_fd=pending_parent, follow_symlinks=False)
                if (current.st_dev, current.st_ino, current.st_size) != (
                    pending["device"],
                    pending["inode"],
                    pending["size"],
                ):
                    raise ValueError("mobile_pending_scope_changed")
                frame.verify()
                bootstrap.advance_admission_epoch()
                os.unlink(name, dir_fd=pending_parent)
                flush_directory(pending_parent)
                bootstrap.advance_admission_epoch()

    if not captured.preview.has_data:
        _preprice_terminal(
            captured,
            native_inputs=terminal_inputs,
            receipts=None,
            history_location=None,
        )
        del terminal_inputs
        for owner_id in ("db.chachanotes.primary", "db.workspaces", "db.agent_runs"):
            _native_preopen(captured, cancel)
            with captured.owner(owner_id):
                pass
        commit_terminal()
        return
    from contextlib import nullcontext

    from tldw_chatbook.Chat.console_trace_read_budget import _mobile_budget_pair

    from .mobile_journal_frame import _locked_mobile, _verify_current_mobile

    if admission[4] is not None:
        admission[4].reserve(512)
        if not _mobile_budget_pair(admission[4], budget):
            raise ValueError("mobile_budget_pair_changed")
    # Keep only the actual public read/planning phase in the existing frame.
    # Readers retire before its flock closes; all target effects follow below.
    with (
        (
            _locked_mobile(
                captured.journal, captured.session, captured.claim.custody, cancel
            )
            if admission[4] is not None
            else nullcontext()
        ),
        captured.sqlite_reads(password=password, cancel=cancel) as reads,
    ):
        budget.container(6)
        native_input = (
            reads.binding,
            reads.versions,
            reads._source_archive,
            reads._canonical_archive,
        )
        plan = _plan_runs(reads, _read_budget=budget)
        blockers = _required_handlers(plan, budget)
        if not blockers:
            ordered = _prepare_rows(plan, budget)
            sync = _prepare_sync(plan, budget)
            budget.container(len(_TABLES) * 2 + len(ordered) * 8)
            families = {spec.table: spec.family for spec in _TABLES}
            grouped = {"chat": [], "workspaces": [], "runs": []}
            holds = {"workspaces": [], "runs": []}
            for row, values, detach in ordered:
                family = families[row.table]
                budget.container(len(values) * 2 + 3)
                final_values = dict(row.values)
                if row.table == "automatic_work_chains":
                    final_values.update(
                        status="review_required", pause_reason="import_review_required"
                    )
                grouped[family].append((row, values, final_values, detach))
                if row.table in _HOLDS:
                    kind, column = _HOLDS[row.table]
                    holds["runs"].append((kind, values[column]))
                elif row.table == "workspace_runtime_bindings":
                    holds["workspaces"].append(("binding", values["binding_id"]))
            budget.container(len(plan.operational_history))
            commands = tuple(
                row
                for row in plan.operational_history
                if row.table == "console_command_receipts"
            )
            passive = _prepare_command_history(
                operation_id, source_fingerprint, commands, budget
            )
            receipt_children = _receipt_children(plan, budget)
            receipts = _encode_receipts(
                plan, reads.versions, receipt_children, len(commands), budget
            )
            for family in ("chat", "workspaces", "runs"):
                # All future SQL parameters, bounded scalar observations, native
                # hold clones and timestamp representations are pre-admitted.
                entries = receipts[family] + (passive if family == "runs" else ())
                budget.container(
                    len(entries) * 28
                    + len(grouped[family]) * 12
                    + len(holds.get(family, ())) * 20
                    + 32
                )
                budget.reserve(1024)
        reads.check()
    # The same frame retains the actual immutable proofs while their readers close.
    if (
        captured.binding is not cohort
        or reads not in captured.session._scopes
        or len(captured.session._scopes) < len(issued_scopes)
        or any(
            captured.session._scopes[index] is not scope
            for index, scope in enumerate(issued_scopes)
        )
        or captured.preview.source_fingerprint != source_fingerprint
        or captured.journal.operation_id != operation_id
        or reads.binding is not native_input[0]
        or reads.versions is not native_input[1]
        or reads._source_archive is not native_input[2]
        or reads._canonical_archive is not native_input[3]
    ):
        raise ValueError("mobile_native_input_cohort_changed")
    budget.value(captured.preview.source_witnesses)
    budget.value(captured.preview.destination_witnesses)
    _mobile_coordinator_budget(captured.session).container(2)
    terminal_inputs = []
    _native_preopen(captured, cancel, _pricing_inputs=terminal_inputs)
    budget.container(3)
    if blockers:
        raise _NativeOperationBlocked(blockers)

    history_shapes = None
    if plan.history_members:
        import json
        from pathlib import Path

        from .journal import _mobile_active_owner
        from .mobile_native_history import _history_shapes, _target_history_location

        history_data_root, history_location = _target_history_location(captured, budget)
        history_shapes, history_expected = _history_shapes(
            plan, history_location, budget
        )
        with captured.journal._locked(exclusive=False) as fd:
            history_records = captured.journal._records(fd)
        intents = [
            row.evidence
            for row in history_records
            if row.event == "mobile_history_intended"
        ]
        if intents:
            if len(intents) != 1 or intents[0]["location"] != str(history_location):
                raise ValueError("mobile_history_review_changed")
            publication_path = Path(intents[0]["publication"])
        else:
            publication_path = history_location
            while (
                publication_path.parent != history_data_root
                and not publication_path.parent.exists()
            ):
                publication_path = publication_path.parent
        budget.container(32)
        prospective_publication = {
            "version": 1,
            "kind": "history_publication",
            "data": {
                "destination_identity": captured.preview.destination.identity,
                "source_fingerprint": source_fingerprint,
                "location": str(history_location),
                "artifact": {
                    "path": str(publication_path),
                    "device": 2**64 - 1,
                    "inode": 2**64 - 1,
                    "kind": "directory",
                    "size": sum(member.size for member in plan.history_members),
                    "sha256": "f" * 64,
                },
            },
        }
        from tldw_chatbook.DB.mobile_consolidation_receipts import _encode_owner_receipt

        budget.container((len(history_shapes) + 1) * 28)
        prospective_children = (
            receipt_children["runs"] + history_shapes + [prospective_publication]
        )
        version_map = {
            (purpose, family): version for purpose, family, version in reads.versions
        }
        history_codec_fields = {
            "source_manifest_sha256": plan.identities.source_manifest_digest
            if ("legacy", "runs") in version_map
            else None,
            "canonical_manifest_sha256": plan.identities.canonical_manifest_digest
            if ("canonical", "runs") in version_map
            else None,
            "source_runs_version": version_map.get(("legacy", "runs")),
            "command_history_count": len(commands),
        }
        before_codec = budget._remaining
        # Maximum-width prospective observations qualify the exact codec shape;
        # their encoded values are discarded and can never enter an owner TX.
        _encode_owner_receipt(
            "runs", prospective_children, **history_codec_fields, budget=budget
        )
        codec_bound = (
            before_codec - budget._remaining + 256 * len(plan.history_members) + 1024
        )
        from .mobile_native_history import _history_observation_bounds

        observation_bounds = _history_observation_bounds(
            captured,
            plan,
            history_data_root,
            history_location,
            publication_path,
            budget,
        )
        budget.reserve(
            8 * len(captured.source.manifest_bytes) + 32768 * len(plan.history_members)
        )
        # Keep the separately pre-admitted late codec space on this same budget;
        # no refund, replacement allowance, or pre-admitted flag is passed on.
        budget.require_capacity(codec_bound)

    # These closures capture this call's actual source, native cohort and original
    # allowance. No caller plan, callback, writer registry or readiness flag enters.
    def verify_receipt(connection, family, entries):
        current_admission = _mobile_admission(captured.session)
        if current_admission[2] is not budget:
            raise ValueError("mobile_native_budget_changed")
        if current_admission[4] is not None:
            current_admission[4].reserve(512)
            if not _mobile_budget_pair(current_admission[4], budget):
                raise ValueError("mobile_budget_pair_changed")
        # Reuse only this transaction's actual continuously held native frame.
        with (
            _verify_current_mobile(
                captured.journal, captured.session, captured.claim.custody, cancel
            )
            if current_admission[4] is not None
            else nullcontext()
        ):
            count = connection.execute(
                "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id=?",
                (operation_id,),
            ).fetchone()[0]
            if count == 0:
                return False
            if count != len(entries):
                raise ValueError("mobile_native_owner_receipt_census_changed")
            for key, raw in entries:
                if (
                    connection.execute(
                        "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id=? AND source_fingerprint=? AND batch_key=? AND mapping_json=? COLLATE BINARY AND typeof(committed_at)='text' AND length(committed_at) BETWEEN 1 AND 128",
                        (operation_id, source_fingerprint, key, raw),
                    ).fetchone()[0]
                    != 1
                ):
                    raise ValueError("mobile_native_owner_receipt_changed")
            return True

    def verify_rows(connection, family):
        current_admission = _mobile_admission(captured.session)
        if current_admission[2] is not budget:
            raise ValueError("mobile_native_budget_changed")
        if current_admission[4] is not None:
            current_admission[4].reserve(512)
            if not _mobile_budget_pair(current_admission[4], budget):
                raise ValueError("mobile_budget_pair_changed")
        # Reuse only this transaction's actual continuously held native frame.
        with (
            _verify_current_mobile(
                captured.journal, captured.session, captured.claim.custody, cancel
            )
            if current_admission[4] is not None
            else nullcontext()
        ):
            for row, _values, final_values, _detach in grouped[family]:
                _verify_row(connection, row, final_values)
            if family == "chat":
                _verify_sync(connection, sync)
            else:
                expected = holds[family]
                if connection.execute(
                    "SELECT count(*) FROM mobile_import_authority_holds WHERE operation_id=?",
                    (operation_id,),
                ).fetchone()[0] != len(expected):
                    raise ValueError("mobile_native_exact_hold_census_changed")
                for kind, identity in expected:
                    if (
                        connection.execute(
                            "SELECT count(*) FROM mobile_import_authority_holds WHERE operation_id=? AND authority_kind=? AND authority_id=? AND reviewed_at IS NULL",
                            (operation_id, kind, identity),
                        ).fetchone()[0]
                        != 1
                    ):
                        raise ValueError("mobile_native_exact_hold_changed")
                if family == "runs":
                    sequence = connection.execute(
                        "SELECT seq FROM sqlite_sequence WHERE name='change_snapshots'"
                    ).fetchone()
                    if (
                        sequence is None
                        or sequence[0] < plan.snapshot_allocator.required_frontier
                    ):
                        raise ValueError("mobile_native_snapshot_frontier_changed")

    def verify_recovery_history(connection):
        current_admission = _mobile_admission(captured.session)
        if current_admission[2] is not budget:
            raise ValueError("mobile_native_budget_changed")
        if current_admission[4] is not None:
            current_admission[4].reserve(512)
            if not _mobile_budget_pair(current_admission[4], budget):
                raise ValueError("mobile_budget_pair_changed")
        # Reuse only this transaction's actual continuously held native frame.
        with (
            _verify_current_mobile(
                captured.journal, captured.session, captured.claim.custody, cancel
            )
            if current_admission[4] is not None
            else nullcontext()
        ):
            from tldw_chatbook.DB.mobile_consolidation_receipts import (
                _read_owner_receipt,
            )

            result = _read_owner_receipt(
                connection, operation_id, source_fingerprint, "runs", budget
            )
            if result is None:
                if connection.execute(
                    "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id=?",
                    (operation_id,),
                ).fetchone()[0]:
                    raise ValueError("mobile_native_owner_receipt_census_changed")
                return False
            master, children = result
            budget.container(len(children) * 3)
            expected_master_raw = next(
                raw for key, raw in receipts["runs"] if key == "runs"
            )
            budget.json(expected_master_raw)
            expected_master = json.loads(expected_master_raw)
            if any(
                master[key] != value
                for key, value in expected_master.items()
                if key not in {"entry_count", "entries_sha256"}
            ):
                raise ValueError("mobile_native_owner_receipt_changed")
            actual_base = [
                child
                for child in children
                if child["kind"] not in {"runlog", "runlog_root", "history_publication"}
            ]
            if len(actual_base) != len(receipt_children["runs"]) or any(
                child not in actual_base for child in receipt_children["runs"]
            ):
                raise ValueError("mobile_native_owner_receipt_changed")
            actual_history = [
                child
                for child in children
                if child["kind"] in {"runlog", "runlog_root"}
            ]
            if len(actual_history) != len(history_shapes):
                raise ValueError("mobile_native_owner_receipt_changed")
            for expected in history_shapes:
                key = "logical_id" if expected["kind"] == "runlog" else "root_id"
                matches = [
                    child
                    for child in actual_history
                    if child["kind"] == expected["kind"]
                    and child["data"][key] == expected["data"][key]
                ]
                if (
                    len(matches) != 1
                    or any(
                        matches[0]["data"][name] != value
                        for name, value in expected["data"].items()
                        if name != "artifact"
                    )
                    or any(
                        matches[0]["data"]["artifact"][name] != value
                        for name, value in expected["data"]["artifact"].items()
                        if name not in {"device", "inode"}
                    )
                ):
                    raise ValueError("mobile_native_owner_receipt_changed")
            publications = [
                child for child in children if child["kind"] == "history_publication"
            ]
            if (
                len(publications) != 1
                or any(
                    publications[0]["data"][key] != prospective_publication["data"][key]
                    for key in (
                        "destination_identity",
                        "source_fingerprint",
                        "location",
                    )
                )
                or publications[0]["data"]["artifact"]["path"] != str(publication_path)
            ):
                raise ValueError("mobile_native_owner_receipt_changed")
            if connection.execute(
                "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id=?",
                (operation_id,),
            ).fetchone()[0] != len(children) + 1 + len(passive):
                raise ValueError("mobile_native_owner_receipt_census_changed")
            for key, raw in passive:
                if (
                    connection.execute(
                        "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id=? AND source_fingerprint=? AND batch_key=? AND mapping_json=? COLLATE BINARY",
                        (operation_id, source_fingerprint, key, raw),
                    ).fetchone()[0]
                    != 1
                ):
                    raise ValueError("mobile_native_owner_receipt_changed")
            return True

    _preprice_terminal(
        captured,
        native_inputs=terminal_inputs,
        receipts=receipts,
        history_location=history_location if history_shapes is not None else None,
    )
    del terminal_inputs

    # A cold process can retain a later owner's unmatched OPEN. Settle only
    # that actual native owner before returning to the fixed write order. This
    # path inspects committed evidence and never imports an absent receipt.
    from .journal import _mobile_active_owner

    budget.container(8)
    with captured.journal._locked(exclusive=False) as fd:
        recovery_records = captured.journal._records(fd)
    stale_owner = _mobile_active_owner(recovery_records)
    if stale_owner is not None:
        budget.value(stale_owner)
        recovery_family = {
            "db.chachanotes.primary": "chat",
            "db.workspaces": "workspaces",
            "db.agent_runs": "runs",
        }[stale_owner["owner_id"]]
        recovery_entries = receipts[recovery_family] + (
            passive if recovery_family == "runs" else ()
        )
        budget.container(
            len(recovery_entries) * 28
            + len(grouped[recovery_family]) * 12
            + len(holds.get(recovery_family, ())) * 20
            + 32
        )
        budget.reserve(1024)
        _native_preopen(captured, cancel)
        with captured.owner(
            stale_owner["owner_id"]
        ) as owner:  # Keep the explicit native resource/fault lifetime boundary.
            if admission[4] is not None:
                admission[4].reserve(512)
            # Native construction is complete; retirement begins after unlock.
            with (
                _locked_mobile(
                    captured.journal, captured.session, captured.claim.custody, cancel
                )
                if admission[4] is not None
                else nullcontext(),
                owner.transaction() as connection,
            ):
                _native_capture_check(captured, cancel)
                complete = (
                    verify_recovery_history(connection)
                    if history_shapes is not None and recovery_family == "runs"
                    else verify_receipt(connection, recovery_family, recovery_entries)
                )
                if complete:
                    verify_rows(connection, recovery_family)
                _native_capture_check(captured, cancel)
        _native_preopen(captured, cancel)

    if history_shapes is not None:
        from .mobile_consolidation_apply import _read_mobile_archive
        from .mobile_history_files import _HistoryFiles
        from .mobile_native_history import _observe_history_children
        from .storage_admission import _local

        _native_preopen(captured, cancel)
        budget.require_capacity(codec_bound)
        with captured.journal._locked(exclusive=False) as fd:
            if _mobile_active_owner(captured.journal._records(fd)) is not None:
                raise ValueError("mobile_history_scope_busy")
        # This fixed issuer is inside the operation that qualified the census.
        # No caller can supply a plan, member list, publisher or readiness flag.
        scope = object.__new__(_HistoryFiles)
        scope.session, scope.captured, scope.journal = (
            captured.session,
            captured,
            captured.journal,
        )
        scope.active, scope.resources, scope.cancel = True, [], cancel
        scope.observation_bounds = scope.observation_binding = observation_bounds
        scope.observation_counts = scope.observation_count_binding = [0, 0, 0, set()]
        scope.data_root, scope.location = history_data_root, history_location
        captured.session._scopes.append(scope)
        _local.mobile_history_scope = scope
        try:
            with _read_mobile_archive(
                captured.source.path, password, cancel
            ) as archive:
                if archive.manifest_bytes != captured.source.manifest_bytes:
                    raise ValueError("mobile_history_capture_changed")
                scope.archive, scope.members = archive, plan.history_members
                scope.binding = (
                    scope.session,
                    captured,
                    scope.journal,
                    scope.data_root,
                    scope.location,
                    archive,
                    scope.members,
                )
                scope.check()
                if scope.publish() is not plan.history_members:
                    raise ValueError("mobile_history_members_changed")
                actual_children = _observe_history_children(
                    scope, history_shapes, history_expected, budget
                )
                receipt_children["runs"].extend(actual_children)
                receipts["runs"] = _encode_owner_receipt(
                    "runs",
                    receipt_children["runs"],
                    **history_codec_fields,
                    budget=budget,
                )
        finally:
            try:
                scope.retire()
            finally:
                _local.mobile_history_scope = None
        _native_preopen(captured, cancel)

    for family, owner_id in (
        ("chat", "db.chachanotes.primary"),
        ("workspaces", "db.workspaces"),
        ("runs", "db.agent_runs"),
    ):
        _native_preopen(captured, cancel)
        with captured.owner(
            owner_id
        ) as owner:  # Keep the explicit native resource/fault lifetime boundary.
            if admission[4] is not None:
                admission[4].reserve(512)
            # Native construction is complete; retirement begins after unlock.
            with (
                _locked_mobile(
                    captured.journal, captured.session, captured.claim.custody, cancel
                )
                if admission[4] is not None
                else nullcontext(),
                owner.transaction() as connection,
            ):
                _native_capture_check(captured, cancel)
                entries = receipts[family] + (passive if family == "runs" else ())
                if verify_receipt(connection, family, entries):
                    verify_rows(connection, family)
                    continue
                for row, values, _final, detach in grouped[family]:
                    _insert_row(connection, row, values, detach=detach)
                if family == "chat":
                    _apply_sync(connection, sync)
                elif family == "workspaces":
                    owner.hold_imported_bindings(
                        connection,
                        operation_id,
                        tuple(identity for _kind, identity in holds[family]),
                    )
                else:
                    _advance_snapshot_allocator(connection, plan.snapshot_allocator)
                    owner.hold_imported_authority(
                        connection, operation_id, tuple(holds[family])
                    )
                    _commit_command_history(
                        owner, connection, operation_id, source_fingerprint, passive
                    )
                verify_rows(connection, family)
                stamp = datetime.now(UTC).isoformat()
                for key, raw in receipts[family]:
                    connection.execute(
                        "INSERT INTO mobile_consolidation_batches(operation_id,source_fingerprint,batch_key,mapping_json,committed_at) VALUES (?,?,?,?,?)",
                        (operation_id, source_fingerprint, key, raw, stamp),
                    )
                verify_receipt(connection, family, entries)
                _native_capture_check(captured, cancel)
    _native_preopen(captured, cancel)
    commit_terminal()
