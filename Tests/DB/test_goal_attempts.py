"""Goal authority cannot be minted twice or used as a survivor claim."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from Tests.Agents.test_goal_models import request
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture(autouse=True)
def enabled_goal_policy(monkeypatch):
    monkeypatch.setenv("TLDW_AGENTS_GOAL_RUNS_ENABLED", "true")


def ready_goal(db):
    goal = db.goal_runs.create(request(), launch_id="launch")
    return db.goal_runs.set_provisioning(goal, status="ready")


def test_concurrent_prepares_accept_exactly_one_generation(tmp_path):
    path = tmp_path / "runs.db"
    db = AgentRunsDB(path)
    goal = ready_goal(db)
    ledger = db.automatic_work
    assert callable(getattr(ledger, "prepare_goal_iteration", None)), (
        "goal admission missing"
    )

    def prepare(_):
        handle = AgentRunsDB(path)
        try:
            return handle.automatic_work.prepare_goal_iteration(
                goal.id, owner_id="owner"
            )
        finally:
            handle.close()

    with ThreadPoolExecutor(2) as pool:
        a, b = pool.map(prepare, range(2))
    assert a.id == b.id
    assert a.goal_id == goal.id and a.ordinal == 1
    with ThreadPoolExecutor(2) as pool:
        accepted = list(
            pool.map(
                lambda _: ledger.accept_goal_iteration(a.id, owner_id="owner"), range(2)
            )
        )
    assert sorted(accepted) == [False, True]
    assert ledger.snapshot(goal.chain_id).used["generation"] == 1
    assert ledger.snapshot(goal.chain_id).reserved["generation"] == 0
    db.close()


def test_kinds_owners_and_goal_link_are_checked_at_every_boundary(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db")
    goal = ready_goal(db)
    ledger = db.automatic_work
    assert callable(getattr(ledger, "prepare_goal_iteration", None)), (
        "goal admission missing"
    )
    attempt = ledger.prepare_goal_iteration(goal.id, owner_id="owner")
    for method in (
        ledger.read_attempt,
        ledger.accept_wake,
        ledger.abort_wake,
        ledger.complete_wake,
    ):
        with pytest.raises(ValueError, match="kind"):
            method(attempt.id, owner_id="owner")
    with pytest.raises(ValueError, match="owner"):
        ledger.accept_goal_iteration(attempt.id, owner_id="other")
    with pytest.raises(ValueError, match="1 to 256"):
        ledger.claim_wake(
            goal.chain_id,
            attempt_id="fake",
            owner_id="owner",
            session_id="s",
            run_ids=[],
        )
    assert ledger.abort_goal_iteration(attempt.id, owner_id="owner")
    assert not ledger.accept_goal_iteration(attempt.id, owner_id="owner")
    assert ledger.snapshot(goal.chain_id).available["generation"] == 3
    with db.transaction() as conn:
        conn.execute("DELETE FROM goal_iterations WHERE attempt_id=?", (attempt.id,))
    with pytest.raises(ValueError, match="link"):
        ledger.read_goal_attempt(attempt.id, owner_id="owner")
    db.close()


def test_goal_context_requires_current_accepted_attempt_and_independent_settings(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext

    db = AgentRunsDB(tmp_path / "runs.db")
    goal = ready_goal(db)
    ledger = db.automatic_work
    assert callable(getattr(ledger, "prepare_goal_iteration", None)), (
        "goal admission missing"
    )
    a = ledger.prepare_goal_iteration(goal.id, owner_id="owner")
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    monkeypatch.setenv("TLDW_AGENTS_GOAL_RUNS_ENABLED", "true")
    assert ledger.accept_goal_iteration(a.id, owner_id="owner")
    context = AutomaticWorkContext(
        ledger, goal.chain_id, "owner", a.id, attempt_kind="goal_iteration"
    )
    context.mark_accepted()
    assert context.output_cap(9000) == 8192
    monkeypatch.setenv("TLDW_AGENTS_MAX_GOAL_OUTPUT_TOKENS", "100")
    assert context.output_cap(9000) == 100
    monkeypatch.setenv("TLDW_AGENTS_GOAL_RUNS_ENABLED", "false")
    with pytest.raises(AutomaticWorkRefused, match="goal_runs_disabled"):
        context.check()
    with db.transaction() as conn:
        conn.execute(
            "UPDATE automatic_wake_attempts SET state='completed' WHERE id=?", (a.id,)
        )
    monkeypatch.setenv("TLDW_AGENTS_GOAL_RUNS_ENABLED", "true")
    with pytest.raises(AutomaticWorkRefused, match="review_required"):
        context.check()
    db.close()


def test_goal_api_cannot_consume_a_fleet_claim(tmp_path):
    from Tests.DB.test_automatic_wake_attempts import claim, survivor

    db = AgentRunsDB(tmp_path / "runs.db")
    ledger = db.automatic_work
    chain = ledger.create_chain("conversation", root_submission_id="manual")
    claim(db, chain, [survivor(db, chain)])
    for method in (
        ledger.read_goal_attempt,
        ledger.accept_goal_iteration,
        ledger.abort_goal_iteration,
    ):
        with pytest.raises(ValueError, match="kind"):
            method("attempt", owner_id="owner")
    assert ledger.accept_wake("attempt", owner_id="owner")
    assert ledger.complete_wake("attempt", owner_id="owner")
    assert not ledger.accept_wake("attempt", owner_id="owner")
    assert ledger.snapshot(chain).used["generation"] == 1
    db.close()


def test_refunded_preparations_do_not_count_as_accepted_iterations(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db")
    try:
        goal = ready_goal(db)
        for ordinal in range(1, 4):
            attempt = db.automatic_work.prepare_goal_iteration(
                goal.id, owner_id="owner"
            )
            assert attempt.ordinal == ordinal
            assert db.automatic_work.abort_goal_iteration(attempt.id, owner_id="owner")
            assert db.goal_runs.get(goal.id).iteration_count == 0
        attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="owner")
        assert attempt.ordinal == 4
        assert db.automatic_work.accept_goal_iteration(attempt.id, owner_id="owner")
        assert db.goal_runs.get(goal.id).iteration_count == 1
    finally:
        db.close()


def test_older_launch_without_optional_mcp_bindings_keeps_original_hash(
    tmp_path, monkeypatch
):
    import hashlib
    import json

    from tldw_chatbook.Agents.goal_models import GoalRequest, GoalToolScope

    path = tmp_path / "runs.db"
    db = AgentRunsDB(path)
    req = request()
    canonical = GoalRequest.canonical_json

    def old_canonical(self):
        payload = json.loads(canonical(self))
        payload["tool_scope"].pop("mcp_bindings")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    with monkeypatch.context() as patch:
        patch.setattr(GoalRequest, "canonical_json", old_canonical)
        original = db.goal_runs.create(req, launch_id="old-start")
        original_bytes = old_canonical(req)
    db.close()
    db = AgentRunsDB(path)
    try:
        restored = db.goal_runs.create(req, launch_id="old-start")
        assert restored.id == original.id
        assert (
            restored.payload_hash == hashlib.sha256(original_bytes.encode()).hexdigest()
        )
        with db.connection() as conn:
            assert (
                conn.execute("SELECT request_json FROM goal_runs").fetchone()[0]
                == original_bytes
            )
        changed = req.model_copy(
            update={"tool_scope": GoalToolScope(catalog_tools=("builtin:calculator",))}
        )
        with pytest.raises(ValueError, match="launch_payload_conflict"):
            db.goal_runs.create(changed, launch_id="old-start")
    finally:
        db.close()


@pytest.mark.parametrize("corruption", ["session", "goal"])
def test_goal_attempt_rejects_wrong_session_or_goal_link(tmp_path, corruption):
    db = AgentRunsDB(tmp_path / "runs.db")
    try:
        goal = ready_goal(db)
        attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="owner")
        other = db.goal_runs.create(request(), launch_id="other-goal")
        with db.transaction() as conn:
            if corruption == "session":
                conn.execute(
                    "UPDATE automatic_wake_attempts SET session_id='other' WHERE id=?",
                    (attempt.id,),
                )
            else:
                conn.execute(
                    "UPDATE goal_iterations SET goal_id=? WHERE attempt_id=?",
                    (other.id, attempt.id),
                )
        with pytest.raises(ValueError, match="link"):
            db.automatic_work.accept_goal_iteration(attempt.id, owner_id="owner")
        assert db.automatic_work.snapshot(goal.chain_id).used["generation"] == 0
    finally:
        db.close()


@pytest.mark.parametrize("setting", ["false", "invalid"])
def test_goal_acceptance_rechecks_live_enablement_inside_ledger(
    tmp_path, monkeypatch, setting
):
    db = AgentRunsDB(tmp_path / "runs.db")
    try:
        goal = ready_goal(db)
        attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="owner")
        monkeypatch.setenv("TLDW_AGENTS_GOAL_RUNS_ENABLED", setting)
        with pytest.raises(AutomaticWorkRefused, match="goal_runs_disabled"):
            db.automatic_work.accept_goal_iteration(attempt.id, owner_id="owner")
        assert db.automatic_work.snapshot(goal.chain_id).used["generation"] == 0
        assert (
            db.automatic_work.read_goal_attempt(attempt.id, owner_id="owner").state
            == "prepared"
        )
        assert db.automatic_work.abort_goal_iteration(attempt.id, owner_id="owner")
        assert db.automatic_work.snapshot(goal.chain_id).reserved["generation"] == 0
    finally:
        db.close()
