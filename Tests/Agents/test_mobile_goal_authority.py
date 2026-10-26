"""Imported Goal evidence stays readable but cannot become fresh native authority."""

import pytest

from Tests.Agents.test_goal_models import request
from Tests.Chat.test_mobile_import_dispatch import stores as _stores_fixture
from Tests.DB.test_mobile_import_holds import db as _db_fixture
from Tests.DB.test_mobile_import_holds import hold

stores = _stores_fixture
db = _db_fixture
pytestmark = pytest.mark.bootstrap_profile


def _rows(database):
    tables = (
        "goal_runs",
        "goal_iterations",
        "goal_checkpoints",
        "goal_evidence",
        "goal_reports",
        "goal_waits",
        "goal_payload_reservations",
        "automatic_work_chains",
        "automatic_work_reservations",
        "automatic_wake_attempts",
        "mobile_import_authority_holds",
        "mobile_consolidation_batches",
    )
    with database.connection() as connection:
        return {
            table: tuple(
                tuple(row) for row in connection.execute(f"SELECT * FROM {table}")
            )
            for table in tables
        }


def _hold_goal(database, goal, attempt, kind):
    hold(
        database,
        (
            kind,
            {"goal": goal.id, "chain": goal.chain_id, "wake_attempt": attempt.id}[kind],
        ),
    )


@pytest.mark.parametrize("kind", ["goal", "chain", "wake_attempt"])
@pytest.mark.parametrize(
    "action", ["create", "control", "settle_control", "provision", "reserve"]
)
def test_direct_goal_store_mutations_refuse_exact_imported_family(db, kind, action):
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    owner = db.goal_runs
    goal = owner.create(request(), launch_id="retained")
    goal = owner.set_provisioning(goal, status="ready")
    attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="fixture")
    _hold_goal(db, goal, attempt, kind)
    before = _rows(db)
    with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
        if action == "create":
            owner.create(goal.request, launch_id=goal.launch_id)
        elif action == "control":
            owner.control(goal.id, stop=True)
        elif action == "settle_control":
            owner.settle_control(goal.id)
        elif action == "provision":
            owner.set_provisioning(goal, status="paused")
        else:
            with db.automatic_work.transaction() as connection:
                owner.reserve_result(connection, goal, attempt.id)
    assert _rows(db) == before
    assert owner.get(goal.id).request == goal.request
    fresh = owner.create(request(), launch_id="fresh")
    assert owner.set_provisioning(fresh, status="ready").status == "ready"
    assert owner.control(fresh.id, stop=True).status == "stopped"


@pytest.mark.parametrize("kind", ["goal", "chain", "wake_attempt"])
@pytest.mark.parametrize(
    "action",
    [
        "checkpoint_failed",
        "completion_check",
        "defer",
        "clear_wait",
        "resume",
        "review_result",
        "resolve_recovery",
        "create",
        "pause",
        "stop",
    ],
)
@pytest.mark.asyncio
async def test_goal_service_callbacks_refuse_before_reusing_evidence(
    stores, kind, action
):
    from Tests.Chat.test_goal_conversation_provisioning import service
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Agents.goal_models import RecoveryResolution

    database, _, _, req = stores
    owner = service(stores)
    goal = owner.create(req, launch_id="retained")
    attempt = database.automatic_work.prepare_goal_iteration(
        goal.id, owner_id="fixture"
    )
    _hold_goal(database, goal, attempt, kind)
    before = _rows(database)
    with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
        if action == "defer":
            owner.defer(goal.id, reason="primary_capacity", delay=1)
        elif action == "resume":
            owner.resume(goal.id, expected_revision=goal.revision)
        elif action == "review_result":
            owner.review_result(
                goal.id,
                expected_revision=goal.revision,
                checkpoint_id=attempt.id,
                artifact_digest="0" * 64,
                accepted=True,
            )
        elif action == "resolve_recovery":
            owner.resolve_recovery(
                goal.id,
                expected_revision=goal.revision,
                resolution=RecoveryResolution.CLOSE_UNCERTAIN,
            )
        elif action == "create":
            owner.create(req, launch_id=goal.launch_id)
        else:
            getattr(owner, action)(goal.id)
    assert _rows(database) == before
    assert owner.get(goal.id).id == goal.id
    assert owner.list_goals()[0].id == goal.id
    assert owner.evidence(goal.id) == ()
    fresh = owner.create(req, launch_id="fresh")
    assert fresh.status == "ready"
    assert owner.stop(fresh.id).status == "stopped"


@pytest.mark.asyncio
async def test_held_saved_checkpoint_and_quality_review_never_refresh_evidence(
    stores, monkeypatch, tmp_path
):
    from Tests.Agents.test_goal_progress import run_verifier
    from tldw_chatbook.Agents import goal_iteration
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    coordinator, goal, result, *_ = await run_verifier(
        stores, monkeypatch, tmp_path, human=True
    )
    owner = coordinator.service
    saved = owner.checkpoint(result)
    assert saved.status == "awaiting_result_review"
    checkpoint = saved.checkpoints[-1]
    assert owner.db.goal_runs.checkpoint(result) == saved
    hold(owner.db, ("goal", goal.id))
    before = _rows(owner.db)

    def no_verification(*args, **kwargs):
        pytest.fail("held history attempted current verification")

    monkeypatch.setattr(goal_iteration, "refresh_evidence", no_verification)
    monkeypatch.setattr(owner, "_binding_reason", no_verification)
    for operation in (
        lambda: owner.checkpoint(result),
        lambda: owner.db.goal_runs.checkpoint(result),
        lambda: owner.completion_check(goal.id),
        lambda: owner.review_result(
            goal.id,
            expected_revision=saved.revision,
            checkpoint_id=checkpoint.id,
            artifact_digest=checkpoint.artifact_digest,
            accepted=True,
        ),
    ):
        with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
            operation()
        assert _rows(owner.db) == before
    assert owner.checkpoint_run_id(goal.id, checkpoint.id) == result.native_run_id
    assert owner.evidence(goal.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("chain_held", [False, True])
async def test_held_goal_explicit_erasure_retains_tombstone_holds_and_receipt(
    stores, monkeypatch, tmp_path, chain_held
):
    from Tests.Agents.test_goal_progress import run_verifier

    coordinator, goal, result, _, _, calls = await run_verifier(
        stores, monkeypatch, tmp_path, human=True
    )
    owner = coordinator.service
    saved = owner.checkpoint(result)
    assert saved.status == "awaiting_result_review"
    with owner.db.transaction() as cursor:
        owner.db.hold_imported_authority(
            cursor,
            "fixture-only",
            (("goal", goal.id), ("wake_attempt", result.attempt_id)),
        )
        # An existing owner-row fixture tests retention, not importer success.
        cursor.execute(
            "INSERT INTO mobile_consolidation_batches VALUES (?, ?, ?, ?, ?)",
            ("fixture-only", "fixture", "fixture", "{}", "fixture"),
        )
    if chain_held:
        hold(owner.db, ("chain", goal.chain_id))
    before = _rows(owner.db)
    call_count = len(calls)
    if chain_held:
        with pytest.raises(ValueError, match="goal_not_settled"):
            owner.remove_payloads(goal.id)
        assert _rows(owner.db) == before
        refused = await coordinator.dispatch_once(goal.id)
        assert refused.reason_code == "goal_not_ready"
        assert len(calls) == call_count
        return
    removed = owner.remove_payloads(goal.id)
    assert removed.status == "removed" and removed.request is None
    assert removed.accounting == saved.accounting
    assert removed.id == saved.id and removed.chain_id == saved.chain_id
    after = _rows(owner.db)
    for table in (
        "mobile_import_authority_holds",
        "mobile_consolidation_batches",
        "automatic_work_chains",
        "automatic_work_reservations",
    ):
        assert after[table] == before[table]
    for table in (
        "goal_reports",
        "goal_checkpoints",
        "goal_evidence",
        "goal_waits",
        "goal_payload_reservations",
    ):
        assert after[table] == ()
    assert owner.get(goal.id).request is None
    assert owner.list_goals()[0].status == "removed"
    refused = await coordinator.dispatch_once(goal.id)
    assert refused.reason_code == "goal_not_ready"
    assert len(calls) == call_count


def test_held_goal_explicit_erasure_still_refuses_unsettled_evidence(db):
    owner = db.goal_runs
    goal = owner.create(request(), launch_id="retained")
    goal = owner.set_provisioning(goal, status="ready")
    attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="fixture")
    owner.control(goal.id)
    hold(db, ("goal", goal.id), ("wake_attempt", attempt.id))
    before = _rows(db)
    with pytest.raises(ValueError, match="goal_not_settled"):
        owner.remove_payloads(goal.id)
    assert _rows(db) == before


@pytest.mark.asyncio
async def test_recovery_projection_rechecks_holds_acquired_after_native_audit(stores):
    from Tests.Chat.test_goal_conversation_provisioning import service

    database, _, _, req = stores
    owner = service(stores)
    held = owner.create(req, launch_id="held")
    fresh = owner.create(req, launch_id="fresh")
    for goal in (held, fresh):
        database.automatic_work.prepare_goal_iteration(goal.id, owner_id="old-owner")
    database.automatic_work.recover(current_owner_id="replacement")
    audit = database.automatic_work.recovery_result
    assert {row[0] for row in audit.goals} == {held.id, fresh.id}
    hold(database, ("goal", held.id))
    before = owner.get(held.id)
    fresh_before = owner.get(fresh.id)
    projected = owner.project_recovery(audit)
    assert owner.get(held.id) == before
    assert next(row for row in projected if row.id == held.id) == before
    assert owner.get(fresh.id).revision == fresh_before.revision + 1
    assert not database.imported_authority_allowed("goal", held.id)
