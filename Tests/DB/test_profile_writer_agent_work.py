"""Exercise stock AgentRuns workflow, goal, worktree and allowance mutations."""

import sys
from types import MappingProxyType

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows profile writer")
def test_stock_workflow_goal_and_worktree_use_checked_transactions(tmp_path, monkeypatch):
    from Tests.Agents.test_goal_models import request
    from Tests.DB.test_agent_worktree_recovery import _record
    from tldw_chatbook import config
    from tldw_chatbook.Agents.workflow_models import WorkflowAlreadyExists, WorkflowRequest
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.agent_worktrees import AgentWorktreeRepository

    submitted = []
    actual = profile_writer.call_bound_profile_writer

    def observe(db, profile, operation, arguments):
        submitted.append(operation)
        return actual(db, profile, operation, arguments)

    monkeypatch.setattr(profile_writer, "call_bound_profile_writer", observe)
    profile = config.get_user_data_dir()
    db = AgentRunsDB(profile / "agent_runs.db", reconcile_on_init=False)
    try:
        goal = db.goal_runs.create(request(), launch_id="goal-launch")
        assert isinstance(goal.accounting.used, MappingProxyType)
        assert db.goal_runs.control(goal.id, stop=True) == db.goal_runs.get(goal.id)
        assert isinstance(db.goal_runs.get(goal.id).accounting.used, MappingProxyType)
        with pytest.raises(ValueError, match="revision_conflict"):
            db.goal_runs.set_provisioning(goal, status="ready")

        # The existing passive workflow model uses a POSIX binding locator.
        # It records the reference without opening or authorizing that path.
        original = WorkflowRequest(conversation_id="workflow-test-chat", objective="Check the retained source", binding=goal.request.binding, targets=(goal.request.provider,))
        workflow = db.workflow_runs.create(original, launch_id="workflow-launch")
        assert isinstance(workflow.accounting.available, MappingProxyType)
        with pytest.raises(WorkflowAlreadyExists) as duplicate:
            db.workflow_runs.create(original, launch_id="independent-launch")
        assert duplicate.value.workflow_id == workflow.id
        assert db.workflow_runs.set_status(workflow.id, status="stopped", expected_control_revision=workflow.control_revision).status == "stopped"

        run = db.create_run(conversation_id=original.conversation_id, agent_kind="subagent")
        db.automatic_work.attach_run(run, workflow.chain_id)
        db.set_status(run, "done")
        trees = AgentWorktreeRepository(db)
        _record(trees, run, tmp_path)
        assert trees.mark_writer_finished(run, "exec-1", cleanup_proven=True)
        assert trees.claim(run, original.conversation_id, operation_id="first", action="apply")
        assert not trees.claim(run, original.conversation_id, operation_id="second", action="apply")
        assert trees.finish_operation(run, "first", state="applied")
        assert trees.get_for_conversation(run, original.conversation_id)["mutation_state"] == "applied"
        assert {
            "agent_runs.work.goal.create", "agent_runs.work.goal.control", "agent_runs.work.goal.set_provisioning",
            "agent_runs.work.workflow.create", "agent_runs.work.workflow.set_status", "agent_runs.work.ledger.attach_run",
            "agent_runs.work.worktree.record_created", "agent_runs.work.worktree.mark_writer_finished",
            "agent_runs.work.worktree.claim", "agent_runs.work.worktree.finish_operation",
        }.issubset(submitted)
    finally:
        db.close()
        profile_writer.drain_profile_writer(profile)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows profile writer")
def test_stock_and_explicit_hardlinked_agent_store_cannot_compete(tmp_path):
    import os
    from tldw_chatbook import config
    from tldw_chatbook.Agents.agent_models import AgentDefinition
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    profile = config.get_user_data_dir()
    canonical = profile / "agent_runs.db"
    foreign = tmp_path / "agent_runs.db"
    db = AgentRunsDB(canonical, reconcile_on_init=False)
    try:
        os.link(canonical, foreign)
        try:
            with pytest.raises(RecoveryRequired, match="profile_writer_database_changed"):
                db.create_agent_definition(AgentDefinition("alias", instructions="require one file identity"))
            with pytest.raises((OSError, RecoveryRequired)):
                AgentRunsDB(foreign, reconcile_on_init=False)
        finally:
            foreign.unlink()
        assert all(row["name"] != "alias" for row in db.list_agent_definitions())
    finally:
        db.close()
        profile_writer.drain_profile_writer(profile)


def test_unacknowledged_worktree_claim_is_never_completed_a_second_time(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from Tests.DB.test_agent_worktree_recovery import _record, _terminal_run
    from tldw_chatbook.Agents import agent_worktree_recovery as recovery
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.agent_worktrees import AgentWorktreeRepository
    from tldw_chatbook.DB.store_errors import StoreMutationOutcomeUnknown

    db = AgentRunsDB(":memory:")
    try:
        _terminal_run(db, "scope", "run")
        repository = AgentWorktreeRepository(db)
        _record(repository, "run", tmp_path)
        assert repository.mark_writer_finished("run", "exec-1", cleanup_proven=True)
        monkeypatch.setattr(recovery, "_snapshot", lambda *_: ("same", "one changed file"))
        monkeypatch.setattr(recovery, "run_git", lambda *_: b"changed.txt")
        actual = AgentWorktreeRepository.claim

        def claim_then_disconnect(*args, **kwargs):
            assert actual(*args, **kwargs)
            raise StoreMutationOutcomeUnknown("receipt unavailable")

        monkeypatch.setattr(AgentWorktreeRepository, "claim", claim_then_disconnect)
        monkeypatch.setattr(AgentWorktreeRepository, "finish_operation", lambda *_args, **_kwargs: pytest.fail("unacknowledged claim was mutated again"))
        refusal = recovery.recover_agent_worktree(db, authority=SimpleNamespace(root=tmp_path), conversation_id="scope", run_id="run", action="apply", request_confirmation=lambda _: {"allow": True}, should_cancel=lambda: False)
        assert refusal.reason_code == "persistence_failed"
        assert repository.get_for_conversation("run", "scope")["mutation_state"] == "applying"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_lost_goal_checkpoint_does_not_submit_a_second_mutation():
    import asyncio
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_goal_runs import ConsoleGoalCoordinator
    from tldw_chatbook.DB.store_errors import StoreMutationOutcomeUnknown

    def checkpoint(_):
        raise StoreMutationOutcomeUnknown("native checkpoint receipt was lost")

    async def invoke(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    async def dispatch(_):
        return SimpleNamespace(reason_code=None, attempt_id="accepted")

    coordinator = SimpleNamespace(
        _closed=False, _owner_id="owner", _wake=asyncio.Event(),
        ledger=SimpleNamespace(read_goal_attempt=lambda *_args, **_kwargs: SimpleNamespace(state="accepted")),
        service=SimpleNamespace(
            call=invoke, get=lambda _: SimpleNamespace(status="ready", retry_at=None),
            checkpoint=checkpoint, checkpoint_failed=lambda _: pytest.fail("unknown checkpoint was rewritten"),
        ),
        dispatch_once=dispatch, notify_goal_changed=lambda _: None,
    )
    with pytest.raises(StoreMutationOutcomeUnknown, match="receipt was lost"):
        await ConsoleGoalCoordinator._continue(coordinator, "goal")


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows profile writer")
def test_goal_service_uses_writer_for_wait_and_recovery_rows(monkeypatch):
    from types import SimpleNamespace

    from Tests.Agents.test_goal_models import request
    from tldw_chatbook import config
    from tldw_chatbook.Agents.goal_models import RecoveryResolution
    from tldw_chatbook.Agents.goal_run_service import GoalRunService
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    profile = config.get_user_data_dir()
    db = AgentRunsDB(profile / "agent_runs.db", reconcile_on_init=False)
    operations = []
    submit = profile_writer.call_bound_profile_writer

    def checked(db, profile, name, arguments):
        operations.append(name)
        return submit(db, profile, name, arguments)

    monkeypatch.setattr(profile_writer, "call_bound_profile_writer", checked)
    try:
        service = GoalRunService(db, SimpleNamespace())
        starting = db.goal_runs.create(request(), launch_id="service-wait")
        goal = db.goal_runs.set_provisioning(starting, status="ready")
        deferred = service.defer(goal.id, reason="primary_capacity", delay=1)
        assert deferred.retry_reason == "primary_capacity" and deferred.retry_at > 0
        service.clear_wait(goal.id)
        assert service.get(goal.id).retry_at is None
        fenced = service.checkpoint_failed(goal.id)
        assert fenced.status == "recovery_required"
        with pytest.raises(TypeError, match="typed recovery resolution required"):
            service.resolve_recovery(goal.id, expected_revision=fenced.revision, resolution="close_uncertain")
        closed = service.resolve_recovery(
            goal.id, expected_revision=fenced.revision, resolution=RecoveryResolution.CLOSE_UNCERTAIN,
        )
        assert closed.status == "closed" and closed.accounting.uncertain
        assert {
            "agent_runs.work.goal.defer_ready", "agent_runs.work.goal.clear_wait",
            "agent_runs.work.goal.checkpoint_failed", "agent_runs.work.goal.resolve_recovery",
        }.issubset(operations)
    finally:
        db.close()
        profile_writer.drain_profile_writer(profile)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows profile writer")
def test_unstarted_workflow_close_rechecks_the_claim_and_run_before_releasing_capacity(monkeypatch):
    from types import SimpleNamespace

    from Tests.DB.test_workflow_attempts import prepare
    from Tests.DB.test_workflow_runs import planned
    from tldw_chatbook import config
    from tldw_chatbook.DB import profile_writer
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    monkeypatch.setenv("TLDW_AGENTS_WORKFLOW_RUNS_ENABLED", "true")
    profile = config.get_user_data_dir()
    db = AgentRunsDB(profile / "agent_runs.db", reconcile_on_init=False)
    try:
        # Workflow history stores a POSIX locator. The storage fixture does
        # not launch a process or open the retained Workspace.
        location = SimpleNamespace(resolve=lambda: "/tmp/retained-workflow")
        db.automatic_work.recover(current_owner_id="owner")
        workflow = planned(db, location)
        phase = prepare(db, workflow)
        assert db.automatic_work.accept_workflow_phase(phase.id, owner_id="owner")
        claim = db.automatic_work.claim_workflow_task(phase.id, task_id="task-1", owner_id="owner")

        def rows():
            with db.connection() as connection:
                return tuple(connection.execute(
                    "SELECT t.status,t.retirement,p.state FROM workflow_task_attempts t "
                    "JOIN workflow_payload_reservations p ON p.attempt_id=t.id WHERE t.id=?",
                    (claim.id,),
                ).fetchone())

        before = rows()
        assert before[2] == "reserved"
        with pytest.raises(ValueError, match="workflow_native_launch_changed"):
            db.workflow_runs.close_unstarted_task(workflow.id, claim.id, launch_run_id="unrelated")
        assert rows() == before
        db.workflow_runs.close_unstarted_task(workflow.id, claim.id, launch_run_id=claim.launch_run_id)
        assert rows() == ("closed", "drained", "removed")
        db.create_run(
            conversation_id=workflow.request.conversation_id, work_chain_id=workflow.chain_id,
            agent_kind="subagent", run_id=claim.launch_run_id,
        )
        with pytest.raises(ValueError, match="workflow_native_launch_changed"):
            db.workflow_runs.close_unstarted_task(workflow.id, claim.id, launch_run_id=claim.launch_run_id)
        assert rows() == ("closed", "drained", "removed")
    finally:
        db.close()
        profile_writer.drain_profile_writer(profile)
