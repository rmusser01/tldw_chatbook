"""Durable structural ownership for recoverable agent worktrees."""

import sqlite3
import threading

import pytest

from tldw_chatbook.Agents.agent_models import AgentDefinition, ToolResult
from tldw_chatbook.Agents.agent_service import _call_with_timeout
from tldw_chatbook.Agents.agent_worktree import _worktree_root_identity
from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
from tldw_chatbook.DB.agent_worktrees import AgentWorktreeRepository
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _record(repository, run_id, tmp_path, execution_id="exec-1", **overrides):
    repo = tmp_path / "repo"
    common = repo / ".git"
    child = tmp_path / "repo-child"
    common.mkdir(parents=True, exist_ok=True)
    child.mkdir(exist_ok=True)
    values = {
        "workspace_id": "workspace-1",
        "binding_id": "binding-1",
        "locator_fingerprint": "f" * 64,
        "repo_root": str(repo),
        "repo_identity": _worktree_root_identity(repo),
        "git_common_dir": str(common),
        "git_common_identity": _worktree_root_identity(common),
        "child_path": str(child),
        "child_identity": _worktree_root_identity(child),
        "branch": "agent/work",
        "base_sha": "a" * 40,
        "execution_id": execution_id,
    }
    values.update(overrides)
    repository.record_created(run_id=run_id, **values)


def _terminal_run(db, conversation_id, run_id):
    db.create_run(
        conversation_id=conversation_id,
        agent_kind="subagent",
        run_id=run_id,
    )
    db.set_status(run_id, status="done", result="done")


def test_v19_upgrade_preserves_definition_cap_and_structural_record_on_reopen(tmp_path):
    path = tmp_path / "runs.db"
    db = AgentRunsDB(path, client_id="seed")
    definition_id = db.create_agent_definition(
        AgentDefinition(
            name="capped",
            instructions="work",
            max_wall_seconds=12.5,
        )
    )
    _terminal_run(db, "conversation-1", "run-1")
    repository = AgentWorktreeRepository(db)
    _record(repository, "run-1", tmp_path)
    assert repository.mark_writer_finished("run-1", "exec-1", cleanup_proven=True)
    assert repository.claim(
        "run-1", "conversation-1", operation_id="operation-1", action="apply"
    )
    db.close()

    reopened = AgentRunsDB(path, client_id="reopen")
    try:
        row = AgentWorktreeRepository(reopened).get_for_conversation(
            "run-1", "conversation-1"
        )
        assert row is not None
        assert row["base_sha"] == "a" * 40
        assert row["repo_identity"] == _worktree_root_identity(tmp_path / "repo")
        assert row["git_common_identity"] == _worktree_root_identity(
            tmp_path / "repo/.git"
        )
        assert row["child_identity"] == _worktree_root_identity(tmp_path / "repo-child")
        assert row["writer_state"] == "drained"
        assert row["mutation_state"] == "applying"
        assert row["operation_id"] == "operation-1"
        assert reopened.get_agent_definition(definition_id)["max_wall_seconds"] == 12.5
        with reopened.connection() as connection:
            assert (
                connection.execute(
                    "SELECT MAX(version) FROM schema_version"
                ).fetchone()[0]
                == AgentRunsDB._CURRENT_SCHEMA_VERSION
                == 20
            )
    finally:
        reopened.close()


def test_unknown_duplicate_and_invalid_records_do_not_overwrite_ownership(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="validation")
    repository = AgentWorktreeRepository(db)
    with pytest.raises(ValueError, match="run"):
        _record(repository, "missing", tmp_path)
    _terminal_run(db, "conversation-1", "run-1")
    _record(repository, "run-1", tmp_path)
    with pytest.raises(ValueError, match="already"):
        _record(repository, "run-1", tmp_path, workspace_id="workspace-2")
    with pytest.raises(ValueError, match="base_sha"):
        _record(repository, "run-1", tmp_path, base_sha="not-a-sha")
    with pytest.raises(ValueError, match="repo_identity"):
        _record(repository, "run-1", tmp_path, repo_identity=(("relative", 1, 2, 3),))
    with pytest.raises(ValueError, match="branch"):
        _record(repository, "run-1", tmp_path, branch="agent/bad branch")
    assert (
        repository.get_for_conversation("run-1", "conversation-1")["workspace_id"]
        == "workspace-1"
    )
    db.close()


def test_conversation_scope_bounded_listing_and_foreign_ids_leave_rows_unchanged(
    tmp_path,
):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="scope")
    repository = AgentWorktreeRepository(db)
    for index, conversation in enumerate(
        ("conversation-1", "conversation-1", "conversation-2")
    ):
        run_id = f"run-{index}"
        _terminal_run(db, conversation, run_id)
        _record(repository, run_id, tmp_path, execution_id=f"exec-{index}")
        repository.mark_writer_finished(run_id, f"exec-{index}", cleanup_proven=True)

    assert repository.get_for_conversation("run-0", "conversation-2") is None
    rows = repository.list_for_conversation(
        "conversation-1", workspace_id="workspace-1", binding_id="binding-1", limit=1
    )
    assert [row["run_id"] for row in rows] == ["run-0"]
    assert "steps" not in rows[0]
    assert [
        row["run_id"]
        for row in repository.list_for_conversation(
            "conversation-1",
            workspace_id="workspace-1",
            binding_id="binding-1",
            after_run_id="run-0",
        )
    ] == ["run-1"]
    with pytest.raises(ValueError, match="limit"):
        repository.list_for_conversation(
            "conversation-1",
            workspace_id="workspace-1",
            binding_id="binding-1",
            limit=101,
        )

    before = repository.get_for_conversation("run-0", "conversation-1")
    assert not repository.mark_writer_finished("run-0", "foreign", cleanup_proven=True)
    assert not repository.claim(
        "run-0", "conversation-2", operation_id="foreign", action="apply"
    )
    assert not repository.finish_operation("run-0", "foreign", state="applied")
    assert repository.get_for_conversation("run-0", "conversation-1") == before
    db.close()


def test_competing_claims_and_stale_completion_are_transactional(tmp_path):
    path = tmp_path / "runs.db"
    seed = AgentRunsDB(path, client_id="seed")
    _terminal_run(seed, "conversation-1", "run-1")
    repository = AgentWorktreeRepository(seed)
    _record(repository, "run-1", tmp_path)
    repository.mark_writer_finished("run-1", "exec-1", cleanup_proven=True)
    seed.close()

    barrier = threading.Barrier(2)
    results = []

    def compete(operation_id):
        database = AgentRunsDB(path, client_id=operation_id)
        try:
            barrier.wait(3)
            results.append(
                AgentWorktreeRepository(database).claim(
                    "run-1",
                    "conversation-1",
                    operation_id=operation_id,
                    action="merge",
                )
            )
        finally:
            database.close()

    threads = [
        threading.Thread(target=compete, args=(f"operation-{i}",)) for i in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert sorted(results) == [False, True]

    verify = AgentRunsDB(path, client_id="verify")
    repository = AgentWorktreeRepository(verify)
    row = repository.get_for_conversation("run-1", "conversation-1")
    winner = row["operation_id"]
    loser = "operation-0" if winner == "operation-1" else "operation-1"
    assert not repository.finish_operation("run-1", loser, state="merged")
    assert repository.finish_operation("run-1", winner, state="merged")
    assert not repository.finish_operation("run-1", winner, state="merged")
    verify.close()


def test_uncertain_writer_state_is_sticky_and_never_claimable_after_reopen(tmp_path):
    path = tmp_path / "runs.db"
    db = AgentRunsDB(path, client_id="seed")
    _terminal_run(db, "conversation-1", "run-1")
    repository = AgentWorktreeRepository(db)
    _record(repository, "run-1", tmp_path)
    assert repository.mark_writer_finished("run-1", "exec-1", cleanup_proven=False)
    assert not repository.mark_writer_finished("run-1", "exec-1", cleanup_proven=True)
    db.close()

    reopened = AgentRunsDB(path, client_id="reopen")
    repository = AgentWorktreeRepository(reopened)
    assert (
        repository.get_for_conversation("run-1", "conversation-1")["writer_state"]
        == "uncertain"
    )
    assert not repository.claim(
        "run-1", "conversation-1", operation_id="operation-1", action="discard"
    )
    reopened.close()


@pytest.mark.parametrize("cleanup_proven", [True, False])
def test_physical_drain_callback_persists_only_after_delayed_worker_finishes(
    tmp_path, cleanup_proven
):
    path = tmp_path / "runs.db"
    db = AgentRunsDB(path, client_id="owner")
    _terminal_run(db, "conversation-1", "run-1")
    repository = AgentWorktreeRepository(db)
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="conversation-1"
    )
    _record(repository, "run-1", tmp_path, execution_id=owner.execution_id)
    gate = threading.Event()
    workers = []

    def delayed_tool():
        workers.append(threading.current_thread())
        assert gate.wait(5)
        return ToolResult(ok=True, content="done")

    owner.on_drained(
        lambda proven: repository.mark_writer_finished(
            "run-1", owner.execution_id, cleanup_proven=proven
        )
    )
    try:
        assert not _call_with_timeout(
            delayed_tool, 0.02, "delayed", lambda: False, owner=owner
        ).ok
        if not cleanup_proven:
            owner.mark_cleanup_unproven()
        owner.finish_root()
        assert (
            repository.get_for_conversation("run-1", "conversation-1")["writer_state"]
            == "held"
        )
    finally:
        gate.set()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
    expected = "drained" if cleanup_proven else "uncertain"
    assert (
        repository.get_for_conversation("run-1", "conversation-1")["writer_state"]
        == expected
    )
    db.close()

    reopened = AgentRunsDB(path, client_id="reopen")
    reopened_repository = AgentWorktreeRepository(reopened)
    assert (
        reopened_repository.get_for_conversation("run-1", "conversation-1")[
            "writer_state"
        ]
        == expected
    )
    assert (
        reopened_repository.claim(
            "run-1", "conversation-1", operation_id="operation-1", action="apply"
        )
        is cleanup_proven
    )
    reopened.close()


def test_reference_migration_upgrades_real_v19_shape(tmp_path):
    path = tmp_path / "runs.db"
    db = AgentRunsDB(path, client_id="seed")
    definition_id = db.create_agent_definition(
        AgentDefinition(
            name="migration-cap",
            instructions="work",
            max_wall_seconds=7.25,
        )
    )
    db.close()
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TABLE agent_worktrees")
        connection.execute("DELETE FROM schema_version WHERE version = 20")
        connection.commit()
        migration = (
            __import__("pathlib").Path(__file__).parents[2]
            / "tldw_chatbook/DB/migrations/agent_runs_v19_to_v20_worktree_recovery.sql"
        ).read_text(encoding="utf-8")
        connection.executescript(migration)
        assert (
            connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            == 20
        )
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name='agent_worktrees'"
        ).fetchone() == ("agent_worktrees",)
        assert connection.execute(
            "SELECT max_wall_seconds FROM agent_definitions WHERE id=?",
            (definition_id,),
        ).fetchone() == (7.25,)
    finally:
        connection.close()
