"""Explicit local privacy on native held historical Goal accounting."""

import pytest

from Tests.Agents.test_goal_models import request
from tldw_chatbook.Backup_Recovery.mobile_identity_plan import _DOMAINS
from tldw_chatbook.Backup_Recovery.mobile_runs_preflight import _HOLDS
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
from tldw_chatbook.DB.console_command_history import (
    _commit_command_history,
    _prepare_command_history,
)
from tldw_chatbook.DB.mobile_consolidation_receipts import (
    _SPECS,
    _encode_owner_receipt,
    _row_digest,
)

_TABLES = (
    "automatic_work_chains",
    "agent_runs",
    "automatic_work_reservations",
    "automatic_wake_attempts",
    "automatic_chat_start_attempts",
    "automatic_wake_claims",
    "automatic_progress_wake_claims",
    "goal_runs",
    "goal_iterations",
    "goal_checkpoints",
    "goal_evidence",
    "goal_reports",
    "goal_payload_reservations",
    "goal_waits",
)


@pytest.fixture
def native_db(tmp_path):
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    path = tmp_path / "runs.db"
    with ConsoleExecutionOwner.acquire(path) as owner:
        db = AgentRunsDB(path, execution_owner=owner)
        try:
            yield db
        finally:
            db.close()


def _hold_and_receipt(db, *, excluded_reservation=None):
    """Actual native holds and committed owner fixture; no producer E2E claim."""
    with db.transaction() as conn:
        holds = []
        for table, (kind, column) in _HOLDS.items():
            holds.extend(
                (kind, row[0])
                for row in conn.execute(f'SELECT "{column}" FROM "{table}"')
            )
        db.hold_imported_authority(conn, "operation", holds)
        children = []
        identities = set()
        for table in _TABLES:
            spec = _SPECS[table]
            for row in conn.execute(f'SELECT * FROM "{table}"'):
                data = dict(row)
                if (
                    table == "automatic_work_reservations"
                    and data["id"] == excluded_reservation
                ):
                    continue
                original = [
                    "original-" + data[key] if type(data[key]) is str else data[key]
                    for key in spec.identity_columns
                ]
                children.append(
                    {
                        "version": 1,
                        "kind": "row",
                        "data": {
                            "table": table,
                            "original": original,
                            "mapped": [data[key] for key in spec.identity_columns],
                            "source_sha256": "d" * 64,
                            "mapped_sha256": _row_digest(
                                table, data, _TraceReadBudget()
                            ),
                        },
                    }
                )
                for domain in _DOMAINS:
                    if domain.table == table and data[domain.column] is not None:
                        value = data[domain.column]
                        if (domain.name, value) in identities:
                            continue
                        identities.add((domain.name, value))
                        children.append(
                            {
                                "version": 1,
                                "kind": "identity",
                                "data": {
                                    "domain": domain.name,
                                    "original": "original-" + value
                                    if type(value) is str
                                    else value,
                                    "mapped": value,
                                },
                            }
                        )
        children.append(
            {
                "version": 1,
                "kind": "provenance",
                "data": {
                    "family": "snapshot_allocator",
                    "record": {
                        "source_sequence": None,
                        "canonical_sequence": None,
                        "required_frontier": 0,
                    },
                },
            }
        )
        encoded = _encode_owner_receipt(
            "runs",
            children,
            source_manifest_sha256="a" * 64,
            canonical_manifest_sha256="b" * 64,
            source_runs_version=25,
            command_history_count=0,
            budget=_TraceReadBudget(),
        )
        passive = _prepare_command_history(
            "operation", "c" * 64, (), _TraceReadBudget()
        )
        _commit_command_history(db, conn, "operation", "c" * 64, passive)
        conn.executemany(
            "INSERT INTO mobile_consolidation_batches VALUES (?,?,?,?,?)",
            (
                ("operation", "c" * 64, key, raw, "2026-10-27T00:00:00Z")
                for key, raw in encoded
            ),
        )


def _accounting(db):
    with db.transaction() as conn:
        return {
            table: tuple(tuple(row) for row in conn.execute(f'SELECT * FROM "{table}"'))
            for table in (
                "automatic_work_chains",
                "automatic_work_reservations",
                "automatic_wake_attempts",
                "automatic_chat_start_attempts",
                "automatic_wake_claims",
                "automatic_progress_wake_claims",
                "automatic_work_runtime_owner",
                "mobile_import_authority_holds",
            )
        }


@pytest.mark.parametrize(
    "state", ["plain", "reserved", "committed", "uncertain", "prepared", "accepted"]
)
def test_imported_goal_privacy_erases_only_local_bodies(native_db, state):
    db = native_db
    try:
        goal = db.goal_runs.create(request(), launch_id="history")
        goal = db.goal_runs.set_provisioning(goal, status="ready")
        if state in {"reserved", "committed", "uncertain"}:
            reservation = db.automatic_work.reserve(
                goal.chain_id,
                reservation_id="tokens",
                owner_id="owner",
                kind="tokens",
                amount=1,
            )
            if state != "reserved":
                db.automatic_work.commit(reservation.id, owner_id="owner")
            if state == "uncertain":
                db.automatic_work.settle(
                    reservation.id, owner_id="owner", actual_amount=None
                )
        elif state in {"prepared", "accepted"}:
            run = db.create_run(
                conversation_id=goal.conversation_id,
                agent_kind="primary",
                work_chain_id=goal.chain_id,
            )
            child = db.automatic_work.prepare_chat_start(
                attempt_id="child",
                source_run_id=run,
                target_conversation_id="child-chat",
                target_session_id="child-session",
                target_session_incarnation="incarnation",
                owner_id="owner",
                draft_revision=0,
                context_epoch=0,
                request_fingerprint="a" * 64,
            )
            if state == "accepted":
                assert db.automatic_work.accept_chat_start(child.id, owner_id="owner")
        goal = db.goal_runs.control(goal.id)
        assert goal.status == "paused"
        _hold_and_receipt(db)
        before = _accounting(db)
        removed = db.goal_runs.remove_payloads(goal.id)
        assert removed.status == "removed" and removed.request is None
        assert _accounting(db) == before
        assert db.goal_runs.remove_payloads(goal.id) == removed
        from tldw_chatbook.DB.automatic_work import AutomaticWorkRefused

        with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
            db.goal_runs.set_provisioning(removed, status="ready")
    finally:
        db.close()


def test_current_settled_goal_privacy_is_unchanged(native_db):
    db = native_db
    try:
        goal = db.goal_runs.create(request(), launch_id="current")
        goal = db.goal_runs.control(goal.id)
        assert db.goal_runs.remove_payloads(goal.id).status == "removed"
    finally:
        db.close()


@pytest.mark.parametrize(
    "damage",
    [
        "changed_chain",
        "changed_goal",
        "missing_hold",
        "reviewed_hold",
        "missing_master",
        "missing_command",
        "missing_identity",
        "current_reserved",
        "current_uncertain",
        "missing_accounting",
    ],
)
def test_imported_goal_privacy_requires_complete_unchanged_family(native_db, damage):
    db = native_db
    goal = db.goal_runs.create(request(), launch_id="history-negative")
    excluded = None
    if damage.startswith("current_") or damage == "missing_accounting":
        reservation = db.automatic_work.reserve(
            goal.chain_id,
            reservation_id="current-tokens",
            owner_id="owner",
            kind="tokens",
            amount=1,
        )
        if damage == "current_uncertain":
            db.automatic_work.commit(reservation.id, owner_id="owner")
            db.automatic_work.settle(
                reservation.id, owner_id="owner", actual_amount=None
            )
        if damage.startswith("current_"):
            excluded = reservation.id
    goal = db.goal_runs.control(goal.id)
    _hold_and_receipt(db, excluded_reservation=excluded)
    with db.transaction() as conn:
        if damage == "missing_accounting":
            conn.execute(
                "DELETE FROM automatic_work_reservations WHERE id=?", (reservation.id,)
            )
        elif damage == "changed_chain":
            conn.execute(
                "UPDATE automatic_work_chains SET pause_reason='changed' WHERE id=?",
                (goal.chain_id,),
            )
        elif damage == "changed_goal":
            conn.execute(
                "UPDATE goal_runs SET revision=revision+1 WHERE id=?", (goal.id,)
            )
        elif damage == "missing_hold":
            conn.execute(
                "DELETE FROM mobile_import_authority_holds WHERE authority_kind='chain'"
            )
        elif damage == "reviewed_hold":
            conn.execute(
                "UPDATE mobile_import_authority_holds SET reviewed_at='2026-10-27' WHERE authority_kind='goal'"
            )
        elif damage == "missing_master":
            conn.execute(
                "DELETE FROM mobile_consolidation_batches WHERE batch_key='runs'"
            )
        elif damage == "missing_command":
            conn.execute(
                "DELETE FROM mobile_consolidation_batches WHERE batch_key='history.command'"
            )
        elif damage == "missing_identity":
            row = conn.execute(
                "SELECT batch_key FROM mobile_consolidation_batches WHERE batch_key LIKE 'runs.entry:%' AND mapping_json LIKE '%identity%' LIMIT 1"
            ).fetchone()
            conn.execute(
                "DELETE FROM mobile_consolidation_batches WHERE batch_key=?", (row[0],)
            )
        before = tuple(tuple(row) for row in conn.execute("SELECT * FROM goal_runs"))
    accounting = _accounting(db)
    with pytest.raises((ValueError, RuntimeError)):
        db.goal_runs.remove_payloads(goal.id)
    with db.transaction() as conn:
        assert (
            tuple(tuple(row) for row in conn.execute("SELECT * FROM goal_runs"))
            == before
        )
    assert _accounting(db) == accounting


def test_prior_native_privacy_deletion_is_not_restored_or_execution_release(native_db):
    db = native_db
    goals = [
        db.goal_runs.create(request(), launch_id=key) for key in ("first", "second")
    ]
    for goal in goals:
        db.goal_runs.control(goal.id)
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO goal_reports(id,goal_id,payload_json) VALUES (?,?,?)",
                ("report-" + goal.id, goal.id, '{"summary":"Retained private result"}'),
            )
    _hold_and_receipt(db)
    accounting = _accounting(db)
    first = db.goal_runs.remove_payloads(goals[0].id)
    assert first.status == "removed" and first.reports == ()
    second = db.goal_runs.remove_payloads(goals[1].id)
    assert second.status == "removed" and second.reports == ()
    assert db.goal_runs.remove_payloads(goals[0].id) == first
    assert _accounting(db) == accounting
    with db.transaction() as conn:
        assert conn.execute("SELECT count(*) FROM goal_reports").fetchone()[0] == 0
