"""Native child trace links survive captured planning and real owning imports."""

from contextlib import contextmanager
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_native_operation import captured_operation
from Tests.Backup_Recovery.test_mobile_owning_transactions import (
    _ordinary_imported_owner,
    _public_owning_operation,
    _small_native_seed,
)

_TABLE = "console_trace_child_segments"
_COLUMNS = ("segment_id", "owner_id", "turn_id", "run_id", "created_at")


def _install_child_seed(monkeypatch, *, version=87, detached=False):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Chat.console_trace_models import TraceCallState, TraceContentRef
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository

    seeded = {}

    def children(chat, _workspace, _runs, _registry, result, legacy):
        selected = "legacy" if legacy else "canonical"
        result["child_links"] = ()
        if legacy and version < 87:
            seeded[selected] = result
            return
        repository = ConsoleTraceRepository()
        conversation = chat.add_conversation(
            {"id": "child-only-chat", "title": "Native child-only history"}
        )
        turn = chat.add_message(
            {
                "id": "child-only-turn",
                "conversation_id": conversation,
                "sender": "User",
                "role": "user",
                "content": "Spawning turn",
            }
        )
        with chat.transaction(immediate=True) as cursor:
            root = repository.create_segment(cursor)
            child_owner = repository.attach_owner(
                cursor, conversation_id=conversation, root_segment_id=root.segment_id
            )
            result.update(
                child_owner=child_owner.owner_id,
                child_conversation=conversation,
                child_message=turn,
            )
            policy_id = cursor.execute(
                "SELECT policy_id FROM console_trace_calls WHERE owner_id=?",
                (result["owner"],),
            ).fetchone()[0]
            empty = repository.create_child_segment(
                cursor,
                owner_id=child_owner.owner_id,
                turn_id=turn,
                run_id="opaque child / empty",
            )
            populated = repository.create_child_segment(
                cursor,
                owner_id=child_owner.owner_id,
                turn_id=turn,
                run_id="opaque child / populated",
            )
            node = repository.append_surface_node(
                cursor,
                segment_id=populated.segment_id,
                sequence=0,
                predecessor_node_id=None,
                component_kind="native_message",
                reference=TraceContentRef(result["artifact"], "native_message"),
            )
            repository.append_event(
                cursor,
                segment_id=populated.segment_id,
                sequence=0,
                event_type="surface_append",
                surface_node_id=node.node_id,
            )
            call = repository.reserve_call(
                cursor,
                owner_id=child_owner.owner_id,
                segment_id=populated.segment_id,
                turn_id=turn,
                run_id=populated.run_id,
                call_sequence=0,
                idempotency_key="child-link-" + selected,
                policy_id=policy_id,
            )
            repository.append_event(
                cursor,
                segment_id=populated.segment_id,
                sequence=1,
                event_type="call_boundary",
                call_id=call.call_id,
            )
            repository.advance_call_state(
                cursor,
                call_id=call.call_id,
                target=TraceCallState.NOT_DISPATCHED,
                occurred_at=cursor.execute(
                    "SELECT created_at FROM console_trace_calls WHERE call_id=?",
                    (call.call_id,),
                ).fetchone()[0],
            )
            result["child_links"] = tuple(
                dict(zip(_COLUMNS, row, strict=True))
                for row in cursor.execute(
                    "SELECT segment_id,owner_id,turn_id,run_id,created_at "
                    "FROM console_trace_child_segments ORDER BY run_id"
                )
            )
            result["child_call"] = call.call_id
            result["child_node"] = node.node_id
            result["empty_segment"] = empty.segment_id
            if legacy and detached:
                repository.detach_owner(
                    cursor,
                    owner_id=child_owner.owner_id,
                    detached_at="2026-11-23T00:00:00Z",
                )
                root = repository.create_segment(cursor)
                repository.attach_owner(
                    cursor,
                    conversation_id=conversation,
                    root_segment_id=root.segment_id,
                )
        seeded[selected] = result

    def seed(stores, **kwargs):
        kwargs["legacy_chat_version"] = version
        return _small_native_seed(stores, **kwargs, native_seed=children)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    return seeded


def _identities(children):
    return {
        (entry["data"]["domain"], entry["data"]["original"]): entry["data"]["mapped"]
        for entry in children
        if entry["kind"] == "identity" and type(entry["data"]["original"]) is str
    }


@pytest.mark.parametrize("version", [86, 87])
def test_child_link_manifest_uses_exact_native_absence(tmp_path, monkeypatch, version):
    seeded = _install_child_seed(monkeypatch, version=version)
    with captured_operation(tmp_path, monkeypatch) as captured:
        with captured.sqlite_reads(
            password=b"test-private-mobile-rollback", cancel=Event()
        ) as reads:
            manifest = reads.plan_manifest()
            tables = {
                (row.purpose, row.family, row.table): row for row in manifest.tables
            }
            assert tables["canonical", "chat", _TABLE].columns == _COLUMNS
            assert (("legacy", "chat", _TABLE) in tables) is (version == 87)
            assert tables["canonical", "chat", _TABLE].handler == "T"
            mixins = (
                "console_prompt_mixin_attempt_items",
                "console_prompt_mixin_attempts",
                "console_prompt_mixin_preferences",
                "console_prompt_mixin_revisions",
            )
            assert all(
                tables["legacy", "chat", table].handler == "unsupported-populated"
                and tables["canonical", "chat", table].handler == "C"
                for table in mixins
            )
            assert not manifest.unsupported_records
            from tldw_chatbook.Backup_Recovery.mobile_captured_sqlite import (
                _legacy_absent,
            )

            assert _legacy_absent(reads, "legacy", "chat", _TABLE) is (version == 86)
            assert not _legacy_absent(reads, "canonical", "chat", _TABLE)
            trace = reads.trace_plan()
            assert not trace.blockers
            assert dict(trace.field_ownership)[_TABLE] == _COLUMNS
            source = [row for row in trace.source_rows if row.table == _TABLE]
            assert len(source) == len(seeded["legacy"]["child_links"])


def test_child_link_without_calls_reserves_canonical_group_collision(
    tmp_path, monkeypatch
):
    seeded = _install_child_seed(monkeypatch)
    with captured_operation(tmp_path, monkeypatch) as captured:
        with captured.sqlite_reads(
            password=b"test-private-mobile-rollback", cancel=Event()
        ) as reads:
            reads.plan_manifest()
            trace = reads.trace_plan()
            assert not trace.blockers
            for original in seeded["legacy"]["child_links"]:
                group = original["run_id"]
                reservations = [
                    row
                    for row in trace.identities.reservations
                    if row.domain == "trace_group" and row.original == group
                ]
                assert len(reservations) == 1
                assert reservations[0].mapped != group
                mapped = next(
                    dict(row.values)
                    for row in trace.rows
                    if row.table == _TABLE and row.original == (original["segment_id"],)
                )
                assert mapped["run_id"] == reservations[0].mapped
                assert mapped["turn_id"] == trace.identities.mapped(
                    "message", original["turn_id"]
                )
                assert mapped["created_at"] == original["created_at"]
            assert not any(
                row.table == "console_trace_calls"
                and dict(row.values)["run_id"] == "opaque child / empty"
                for row in trace.source_rows
            )


def test_child_call_closure_requires_its_ownership_link(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.mobile_integrated_closure import (
        _close_qualified_inputs,
    )
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    seeded = _install_child_seed(monkeypatch)
    with captured_operation(tmp_path, monkeypatch) as captured:
        with captured.sqlite_reads(
            password=b"test-private-mobile-rollback", cancel=Event()
        ) as reads:
            reads.plan_manifest()
            trace = reads.trace_plan()
            assert not trace.blockers
            original = seeded["legacy"]
            call_key = ("console_trace_calls", (original["child_call"],))
            link = next(
                row
                for row in trace.source_rows
                if row.table == _TABLE
                and dict(row.values)["run_id"] == "opaque child / populated"
            )
            link_key = (_TABLE, link.original)
            proof = next(
                row
                for row in trace.input_incidence
                if row.original_call_id == original["child_call"]
            )
            assert link_key in proof.inputs
            rows = tuple(
                row
                for row in (*trace.content.rows, *trace.rows)
                if (row.table, row.original) != link_key
            )
            blockers = []
            closed = _close_qualified_inputs(
                trace, rows, blockers, _read_budget=_TraceReadBudget()
            )
            assert call_key not in {(row.table, row.original) for row in closed}
            assert any(
                row.table == call_key[0]
                and row.original == call_key[1]
                and row.reason == "integrated_native_input_lost"
                for row in blockers
            )


@pytest.mark.parametrize("detached", [False, True])
def test_actual_child_link_import_preserves_native_ownership_and_receipts(
    tmp_path, monkeypatch, detached
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery import mobile_native_rows
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.DB.mobile_consolidation_receipts import (
        _read_owner_receipt,
        _row_digest,
    )

    seeded = _install_child_seed(monkeypatch, detached=detached)
    observed = {}
    prepare = mobile_native_rows._prepare_rows

    def prepared(plan, budget):
        ordered = prepare(plan, budget)
        observed["ordered"] = ordered
        return ordered

    monkeypatch.setattr(mobile_native_rows, "_prepare_rows", prepared)
    with _public_owning_operation(tmp_path, monkeypatch) as (
        service,
        preview,
        captures,
    ):
        stores = preview.destination
        source = preview.source_dir
        before = mobile_consolidation_fixture.fingerprints(source)
        fingerprint = preview.source_fingerprint
        operation = service.start_mobile_consolidation(
            preview,
            destination_confirmed=True,
            rollback_password=b"test-private-mobile-rollback",
        )
        state = service.wait(operation)
        assert state["operation_id"] == operation
        assert state["state"] == "succeeded", dict(state)
        assert state["result"]["mobile_consolidated"] is True
        assert state["result"]["journal_operation_id"] == operation
        assert service.pending_operations() == ()
        (captured,) = captures
        assert captured.journal.operation_id == operation
        assert captured.preview.source_fingerprint == fingerprint
    assert mobile_consolidation_fixture.fingerprints(source) == before
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_recovery import (
        _require_mobile_cutover,
    )

    _require_mobile_cutover(source, stores)
    with (
        _ordinary_imported_owner(stores, "db.chachanotes.primary") as owner,
        owner.transaction() as cursor,
    ):
        receipt = _read_owner_receipt(
            cursor.connection, operation, fingerprint, "chat", _TraceReadBudget()
        )
        assert receipt is not None
        children = receipt[1]
        mappings = _identities(children)
        source_values = seeded["legacy"]
        mapped_owner = mappings["trace_owner", source_values["child_owner"]]
        repository = ConsoleTraceRepository()
        for original in source_values["child_links"]:
            expected = dict(original)
            for column, domain in (
                ("segment_id", "trace_segment"),
                ("owner_id", "trace_owner"),
                ("turn_id", "message"),
                ("run_id", "trace_group"),
            ):
                expected[column] = mappings[domain, original[column]]
            actual = cursor.execute(
                "SELECT segment_id,owner_id,turn_id,run_id,created_at "
                "FROM console_trace_child_segments WHERE segment_id=?",
                (expected["segment_id"],),
            ).fetchone()
            assert tuple(actual) == tuple(expected[column] for column in _COLUMNS)
            assert (
                repository.get_child_segment(cursor, expected["run_id"]).segment_id
                == expected["segment_id"]
            )
            effective = repository.get_effective_owner(cursor, expected["segment_id"])
            assert effective.owner_id == mapped_owner
            assert effective.attached is (not detached)
            assert (
                cursor.execute(
                    "SELECT parent_segment_id FROM console_trace_segments WHERE segment_id=?",
                    (expected["segment_id"],),
                ).fetchone()[0]
                is None
            )
            row_receipt = next(
                entry["data"]
                for entry in children
                if entry["kind"] == "row"
                and entry["data"]["table"] == _TABLE
                and entry["data"]["original"] == [original["segment_id"]]
            )
            assert row_receipt["mapped"] == [expected["segment_id"]]
            assert row_receipt["source_sha256"] == _row_digest(
                _TABLE, original, _TraceReadBudget()
            )
            assert row_receipt["mapped_sha256"] == _row_digest(
                _TABLE, expected, _TraceReadBudget()
            )
        for original in seeded["canonical"]["child_links"]:
            actual = cursor.execute(
                "SELECT segment_id,owner_id,turn_id,run_id,created_at "
                "FROM console_trace_child_segments WHERE segment_id=?",
                (original["segment_id"],),
            ).fetchone()
            assert tuple(actual) == tuple(original[column] for column in _COLUMNS)
        call = repository.get_call(
            cursor, mappings["trace_call", source_values["child_call"]]
        )
        assert call.owner_id == mapped_owner
        assert call.turn_id == mappings["message", source_values["child_message"]]
        assert call.run_id == mappings["trace_group", "opaque child / populated"]
        assert call.surface_node_id is None
        selected = repository.read_child_calls(
            cursor, mappings["conversation", source_values["child_conversation"]]
        )
        assert [row.call_id for row in selected] == ([] if detached else [call.call_id])
        ordered = observed["ordered"]
        owner_position = next(
            index
            for index, (row, values, detach) in enumerate(ordered)
            if row.table == "console_trace_owners"
            and values["owner_id"] == mapped_owner
            and not detach
        )
        child_positions = [
            index
            for index, (row, values, _) in enumerate(ordered)
            if row.table == _TABLE and values["owner_id"] == mapped_owner
        ]
        assert len(child_positions) == 2
        assert owner_position < min(child_positions)
        populated_segment = call.segment_id
        link_position = next(
            index
            for index, (row, values, _) in enumerate(ordered)
            if row.table == _TABLE and values["segment_id"] == populated_segment
        )
        guarded = [
            index
            for index, (row, values, _) in enumerate(ordered)
            if row.table
            in {
                "console_trace_surface_nodes",
                "console_trace_calls",
                "console_trace_events",
            }
            and values["segment_id"] == populated_segment
        ]
        assert guarded and link_position < min(guarded)
        if detached:
            detach_position = next(
                index
                for index, (row, values, detach) in enumerate(ordered)
                if row.table == "console_trace_owners"
                and values["owner_id"] == mapped_owner
                and detach
            )
            assert detach_position > max((*child_positions, *guarded))
    with (
        _ordinary_imported_owner(stores, "db.agent_runs") as runs,
        runs.transaction() as connection,
    ):
        holds = connection.execute(
            "SELECT authority_kind,authority_id,reviewed_at "
            "FROM mobile_import_authority_holds"
        ).fetchall()
        assert len(holds) == 1
        assert tuple(holds[0])[::2] == ("run", None)
        assert holds[0][1] != seeded["canonical"]["run"]


def test_child_link_receipts_resume_under_fresh_public_admission(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import mobile_native_rows
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    _install_child_seed(monkeypatch)
    insert = mobile_native_rows._insert_row
    workspace_faults = []

    def fail_workspace(connection, row, values, detach=False):
        if row.table == "workspace_records":
            workspace_faults.append(row.table)
            raise OSError("child-link fixture interrupted after Chat commit")
        return insert(connection, row, values, detach=detach)

    with _public_owning_operation(tmp_path, monkeypatch) as (
        service,
        preview,
        captures,
    ):
        with monkeypatch.context() as fault:
            fault.setattr(mobile_native_rows, "_insert_row", fail_workspace)
            operation = service.start_mobile_consolidation(
                preview,
                destination_confirmed=True,
                rollback_password=b"test-private-mobile-rollback",
            )
            state = service.wait(operation)
        assert state["state"] == "recovery_required", dict(state)
        assert workspace_faults == ["workspace_records"]
        (captured,) = captures
        journal = captured.journal
        stores = preview.destination

    def refuse_chat_replay(connection, row, values, detach=False):
        assert not row.table.startswith("console_trace_"), "committed Chat replayed"
        return insert(connection, row, values, detach=detach)

    with monkeypatch.context() as observation:
        observation.setattr(mobile_native_rows, "_insert_row", refuse_chat_replay)
        fresh = RecoveryService(journal.root.parent)
        try:
            resumed = fresh.start_recovery(
                operation,
                action="finish",
                rollback_password=b"test-private-mobile-rollback",
            )
            state = fresh.wait(resumed)
            assert state["state"] == "succeeded", dict(state)
            assert state["result"]["mobile_consolidated"] is True
            assert fresh.pending_operations() == ()
        finally:
            fresh.close()
    with (
        _ordinary_imported_owner(stores, "db.chachanotes.primary") as owner,
        owner.transaction() as cursor,
    ):
        assert (
            cursor.execute(
                "SELECT count(*) FROM console_trace_child_segments"
            ).fetchone()[0]
            == 4
        )


@contextmanager
def _native_child_projection(*, tied=True, opaque=False):
    """Read real in-memory native rows; no capture/session authority is issued."""
    from tldw_chatbook.Backup_Recovery.mobile_content_graph import _ContentRow
    from tldw_chatbook.Chat.console_trace_models import (
        FrozenTracePolicy,
        TraceCallState,
        TraceContentRef,
        new_opaque_id,
    )
    from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(":memory:", "child-preflight-data")
    repository = ConsoleTraceRepository()
    try:
        conversation = db.add_conversation({"id": "unit-child-chat", "title": "Child"})
        saved_turn = db.add_message(
            {
                "id": "unit-child-turn",
                "conversation_id": conversation,
                "sender": "User",
                "role": "user",
                "content": "Spawning turn",
            }
        )
        turn = "retired opaque child turn / 5" if opaque else saved_turn
        call_ids = (
            "00000000-0000-4000-8000-000000000001",
            "00000000-0000-4000-8000-000000000002",
        )
        with db.transaction(immediate=True) as cursor:
            root = repository.create_segment(cursor)
            owner = repository.attach_owner(
                cursor, conversation_id=conversation, root_segment_id=root.segment_id
            )
            bare = repository.create_segment(cursor)
            policy = repository.ensure_policy(
                cursor, FrozenTracePolicy(new_opaque_id(), "unit-v1", False, None)
            )
            artifact = repository.store_sanitized_artifact(
                cursor,
                sanitized_bytes=b'"native child"',
                media_type="application/json",
                normalization_version="unit-v1",
            )
            for index, call_id in enumerate(call_ids):
                child = repository.create_child_segment(
                    cursor,
                    owner_id=owner.owner_id,
                    turn_id=turn,
                    run_id=f"opaque unit child / {index}",
                )
                node = repository.append_surface_node(
                    cursor,
                    segment_id=child.segment_id,
                    sequence=0,
                    predecessor_node_id=None,
                    component_kind="native_message",
                    reference=TraceContentRef(artifact.artifact_id, "native_message"),
                )
                repository.append_event(
                    cursor,
                    segment_id=child.segment_id,
                    sequence=0,
                    event_type="surface_append",
                    surface_node_id=node.node_id,
                )
                created = (
                    "2026-11-23T00:00:00Z"
                    if tied or index == 0
                    else "2026-11-23T00:00:01Z"
                )
                # Every native trigger remains installed. Explicit legal clocks
                # make a sibling tie deterministic without rewriting history.
                cursor.execute(
                    "INSERT INTO console_trace_calls "
                    "(call_id,owner_id,segment_id,turn_id,run_id,call_sequence,"
                    "idempotency_key,policy_id,created_at) VALUES (?,?,?,?,?,0,?,?,?)",
                    (
                        call_id,
                        owner.owner_id,
                        child.segment_id,
                        turn,
                        child.run_id,
                        "unit-child-" + str(index),
                        policy.policy_id,
                        created,
                    ),
                )
                repository.advance_call_state(
                    cursor,
                    call_id=call_id,
                    target=TraceCallState.NOT_DISPATCHED,
                    occurred_at=created,
                )
            actual_order = tuple(
                row.call_id for row in repository.read_child_calls(cursor, conversation)
            )
            assert actual_order == call_ids
            assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []
            assert (
                cursor.execute(
                    "SELECT 1 FROM sqlite_schema WHERE name=?", (_TABLE,)
                ).fetchone()
                is not None
            )
            records = []
            for table, identity in (
                ("conversations", "id"),
                ("messages", "id"),
                ("console_trace_owners", "owner_id"),
                ("console_trace_segments", "segment_id"),
                ("console_trace_policies", "policy_id"),
                ("console_trace_artifacts", "artifact_id"),
                (_TABLE, "segment_id"),
                ("console_trace_calls", "call_id"),
                ("console_trace_surface_nodes", "node_id"),
                ("console_trace_events", "event_id"),
                ("console_trace_semantic_revisions", "revision_id"),
            ):
                cursor.execute('SELECT * FROM "' + table + '"')
                columns = tuple(column[0] for column in cursor.description)
                for row in cursor.fetchall():
                    values = dict(zip(columns, row, strict=True))
                    records.append(
                        _ContentRow(table, (values[identity],), tuple(values.items()))
                    )
        yield (
            db,
            tuple(records),
            {
                "bare": bare.segment_id,
                "turn": turn,
                "order": actual_order,
            },
        )
    finally:
        db.close_connection()


def _child_preflight_data(source, canonical=()):
    """Pure fixed-plan data for a refusal predicate, never an owner API input."""
    import hashlib
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.mobile_identity_plan import (
        _DOMAINS,
        _IdentityPlan,
        _reserve,
    )
    from tldw_chatbook.Backup_Recovery.mobile_trace_plan import _REFERENCES
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    budget = _TraceReadBudget()
    reservations = []
    # This deterministic allocator vector is data, not a claimed capture hash.
    allocation_vector = "a" * 64
    for domain in _DOMAINS:
        partitions = []
        for rows in (source, canonical):
            selected = [dict(row.values) for row in rows if row.table == domain.table]
            if domain.name in {"message", "trace_group"}:
                column = "turn_id" if domain.name == "message" else "run_id"
                budget.container(4 * (len(rows) + len(selected)) + 1)
                occupied = {row[domain.column] for row in selected}
                extra = {
                    dict(row.values)[column]
                    for row in rows
                    if row.table == _TABLE and dict(row.values)[column] not in occupied
                }
                selected += [{domain.column: value} for value in sorted(extra)]
            partitions.append(tuple(selected))
        reservations.extend(
            _reserve(domain, *partitions, allocation_vector, _read_budget=budget)
        )
    identities = _IdentityPlan(
        allocation_vector,
        hashlib.sha256(repr(source).encode()).hexdigest(),
        hashlib.sha256(repr(canonical).encode()).hexdigest(),
        tuple(reservations),
    )
    projected = []
    for row in source:
        values = dict(row.values)
        references = _REFERENCES.get(row.table, {})
        if row.table == "messages":
            references = {
                "id": "message",
                "conversation_id": "conversation",
                "parent_message_id": "message",
            }
        elif row.table == "conversations":
            references = {"id": "conversation"}
        for column, domain in references.items():
            if values[column] is not None:
                values[column] = identities.mapped(domain, values[column])
        projected.append(type(row)(row.table, row.original, tuple(values.items())))
    # Only the existing private pure predicate consumes these data projections.
    # No MaintenanceSession, captured reader, path, permission or owner is mocked.
    trace = SimpleNamespace(inputs=SimpleNamespace(source=SimpleNamespace(rows=source)))
    return SimpleNamespace(
        identities=identities,
        rows=tuple(projected),
        integrated=SimpleNamespace(sync=SimpleNamespace(trace=trace)),
    )


@pytest.mark.parametrize("guarded", ["call", "surface_event"])
def test_guarded_orphan_trace_is_rejected_by_pure_preflight(guarded):
    from tldw_chatbook.Backup_Recovery.mobile_native_operation import (
        _native_trace_refusals,
    )
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    with _native_child_projection() as (db, rows, native):
        # Deliberate private-plan data-loss fault, not a mutated source catalog.
        # Calls keep no head; the standalone case retains only surface events.
        source = []
        for row in rows:
            if row.table == _TABLE:
                continue
            if guarded == "call" and row.table in {
                "console_trace_surface_nodes",
                "console_trace_events",
            }:
                continue
            if guarded == "surface_event" and row.table == "console_trace_calls":
                continue
            source.append(row)
        plan = _child_preflight_data(tuple(source))
        before = db.get_connection().total_changes
        blockers = _native_trace_refusals(plan, _TraceReadBudget())
        assert db.get_connection().total_changes == before
        expected_tables = (
            {"console_trace_calls"}
            if guarded == "call"
            else {"console_trace_surface_nodes", "console_trace_events"}
        )
        refused = [
            row
            for row in blockers
            if row.reason == "native_guarded_trace_owner_unproved"
        ]
        assert expected_tables <= {row.table for row in refused}
        assert all(row.original != (native["bare"],) for row in refused)


def test_native_child_preflight_keeps_empty_unowned_and_opaque_turns():
    from tldw_chatbook.Backup_Recovery.mobile_native_operation import (
        _native_trace_refusals,
    )
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    with _native_child_projection(opaque=True) as (db, rows, native):
        plan = _child_preflight_data(rows)
        before = db.get_connection().total_changes
        assert _native_trace_refusals(plan, _TraceReadBudget()) == ()
        assert db.get_connection().total_changes == before
        assert (
            db.get_connection()
            .execute("SELECT count(*) FROM messages WHERE id=?", (native["turn"],))
            .fetchone()[0]
            == 0
        )
        assert (
            db.get_connection()
            .execute(
                "SELECT count(*) FROM console_trace_segments WHERE segment_id=? "
                "AND parent_segment_id IS NULL",
                (native["bare"],),
            )
            .fetchone()[0]
            == 1
        )


@pytest.mark.parametrize(
    "tied,collision,changed",
    [(True, True, True), (False, True, False), (True, False, False)],
    ids=["tied-collision", "untied-collision", "tied-unoccupied"],
)
def test_native_child_order_collision_is_preserved_or_blocked(tied, collision, changed):
    from tldw_chatbook.Backup_Recovery.mobile_native_operation import (
        _native_trace_refusals,
    )
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    with (
        _native_child_projection(tied=tied) as (db, source, native),
        _native_child_projection(tied=tied) as (_canonical_db, canonical, _),
    ):
        plan = _child_preflight_data(source, canonical if collision else ())
        source_calls = [row for row in source if row.table == "console_trace_calls"]
        mapped = {
            row.original: dict(row.values)
            for row in plan.rows
            if row.table == "console_trace_calls"
        }
        if collision:
            assert {row.original for row in source_calls} <= {
                row.original for row in canonical if row.table == "console_trace_calls"
            }
            assert all(
                mapped[row.original]["call_id"] != row.original[0]
                for row in source_calls
            )
        parameters = []
        for row in source_calls:
            values = mapped[row.original]
            parameters.extend(
                (
                    row.original[0],
                    values["created_at"],
                    values["call_sequence"],
                    values["call_id"],
                )
            )
        # SELECT-only SQLite oracle uses the independent native reader's exact
        # tie keys. It neither inserts projected rows nor bypasses any guard.
        before = db.get_connection().total_changes
        actual = tuple(
            row[0]
            for row in db.get_connection().execute(
                "SELECT original FROM "
                "(SELECT ? AS original,? AS created_at,? AS call_sequence,? AS call_id "
                "UNION ALL SELECT ?,?,?,?) "
                "ORDER BY created_at,call_sequence,call_id",
                parameters,
            )
        )
        assert (actual != native["order"]) is changed
        blockers = _native_trace_refusals(plan, _TraceReadBudget())
        assert db.get_connection().total_changes == before
        ordering = [
            row for row in blockers if row.reason == "native_child_call_order_changed"
        ]
        assert bool(ordering) is changed
        if not changed:
            assert blockers == ()


@pytest.mark.parametrize(
    "version,history,canonical",
    (
        (86, "preferences", False),
        (87, "preferences", False),
        (87, "attempts", False),
        (87, "attempt_items", False),
        (87, "revisions", False),
        (87, "attempt_items", True),
    ),
)
def test_native_mixin_history_boundary_refuses_before_effects(
    tmp_path, monkeypatch, version, history, canonical
):
    import hashlib
    import re

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_consolidation_admission import (
        prepare,
        prepared,
    )
    from Tests.DB.test_console_prompt_mixins import attempt, saved_turn
    from tldw_chatbook.Backup_Recovery import mobile_consolidation
    from tldw_chatbook.Chat.console_prompt_mixins import MixinPreferences
    from tldw_chatbook.DB.console_prompt_mixins import save_preferences, write_attempt

    snapshots, legacy_paths, statements = {}, set(), []
    original_open = mobile_consolidation._open_preview_read

    def observed_open(path):
        connection = original_open(path)
        connection.set_trace_callback(lambda sql: statements.append((path, sql)))
        return connection

    def native(chat, _workspace, _runs, _registry, _result, legacy):
        if legacy == canonical:
            return
        conversation, native_ids = saved_turn(chat)
        with chat.transaction() as cursor:
            if history == "preferences":
                save_preferences(
                    cursor, conversation, expected_revision=0, value=MixinPreferences()
                )
            else:
                write_attempt(
                    cursor,
                    conversation,
                    batch=attempt(empty=history == "attempts"),
                    native_message_ids=native_ids,
                )
                if history == "revisions":
                    # Native deletion permits retained snapshots after their attempt.
                    cursor.execute("DELETE FROM console_prompt_mixin_attempt_items")
                    cursor.execute("DELETE FROM console_prompt_mixin_attempts")
            assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []

    def seed(stores, **kwargs):
        result = _small_native_seed(
            stores, **kwargs, native_seed=native, legacy_chat_version=version
        )
        snapshots.update(
            {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in stores.paths}
        )
        if kwargs["legacy"]:
            legacy_paths.update(stores.paths)
        return result

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    monkeypatch.setattr(mobile_consolidation, "_open_preview_read", observed_open)
    reason = "mobile_source_mixin_history_unsupported:console_prompt_mixin_" + history
    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _stores,
        preview,
        inventory,
        _,
    ):
        if canonical:
            assert not preview.issues, preview.issues
        else:
            assert reason in preview.issues, preview.issues
            assert "PRIVATE" not in repr(preview.issues)
            with pytest.raises(ValueError, match="mobile"):
                prepare(session, journal, config, preview, inventory)
        assert snapshots == {
            p: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshots
        }
        with journal._locked(exclusive=False) as directory:
            assert not journal._records(directory)
    if not canonical:
        mixin_select = re.compile(
            r"^\s*SELECT\b.*?\bFROM\s+"
            r'(?:["`\[]?\w+["`\]]?\s*\.\s*)?'
            r'["`\[]?console_prompt_mixin_\w+["`\]]?(?=\s|;|$)',
            re.IGNORECASE | re.DOTALL,
        )
        observed = [
            sql
            for path, sql in statements
            if path in legacy_paths and mixin_select.search(sql)
        ]
        expected_probes = {
            "SELECT substr(conversation_id,1,128), substr(attempt_id,1,128), ordinal FROM console_prompt_mixin_attempt_items LIMIT 1",
            "SELECT substr(conversation_id,1,128), substr(attempt_id,1,128) FROM console_prompt_mixin_attempts LIMIT 1",
            "SELECT substr(conversation_id,1,128) FROM console_prompt_mixin_preferences LIMIT 1",
            "SELECT substr(conversation_id,1,128), substr(snapshot_id,1,128) FROM console_prompt_mixin_revisions LIMIT 1",
        }
        assert observed
        assert all(sql in expected_probes for sql in observed), observed
