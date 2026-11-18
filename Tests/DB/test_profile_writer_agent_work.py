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
