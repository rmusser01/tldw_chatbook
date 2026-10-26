"""Imported authority remains historical across native admission and recovery."""

import pytest

from Tests.Agents.test_goal_models import request
from Tests.DB.test_automatic_wake_attempts import claim, survivor
from Tests.DB.test_automatic_work_budget import chain
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.automatic_work import AutomaticWorkRefused
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


@pytest.fixture
def db():
    database = AgentRunsDB(":memory:")
    yield database
    database.close()


def hold(db, *authorities):
    with db.transaction() as cursor:
        db.hold_imported_authority(cursor, "import-operation", authorities)


def test_installed_heads_have_atomic_batch_receipts():
    for cls, version in ((AgentRunsDB, 26), (WorkspaceDB, 9), (CharactersRAGDB, 82)):
        database = cls(":memory:", client_id="import-test")
        try:
            assert cls._CURRENT_SCHEMA_VERSION == version
            with database.transaction() as cursor:
                assert (
                    cursor.execute(
                        "SELECT count(*) FROM mobile_consolidation_batches"
                    ).fetchone()[0]
                    == 0
                )
        finally:
            database.close()


@pytest.mark.parametrize("with_chain", [False, True])
def test_completed_undelivered_result_hold_blocks_all_native_projections(
    db, with_chain
):
    chain_id = chain(db) if with_chain else None
    run = survivor(db, chain_id)
    before = db.get_run(run)
    assert db.pending_wake_conversation_ids() == ("conversation",)
    assert db.pending_wake_results("conversation")[0]["id"] == run
    assert db.undelivered_wake_runs("conversation")[0]["id"] == run
    hold(db, ("run", run))
    assert not db.imported_authority_allowed("run", run)
    assert db.pending_wake_conversation_ids() == ()
    assert db.pending_wake_results("conversation") == []
    assert db.undelivered_wake_runs("conversation") == []
    assert db.mark_wake_delivered([run]) == 0
    assert db.get_run(run) == before


@pytest.mark.parametrize("kind", ["run", "chain", "wake_attempt"])
def test_hold_refuses_existing_claim_and_preserves_recovery_accounting(db, kind):
    chain_id = chain(db)
    run = survivor(db, chain_id)
    claim(db, chain_id, [run])
    hold(db, (kind, {"run": run, "chain": chain_id, "wake_attempt": "attempt"}[kind]))
    with db.connection() as conn:
        before = tuple(
            tuple(row)
            for row in conn.execute("SELECT * FROM automatic_work_reservations")
        )
        attempts = tuple(
            tuple(row) for row in conn.execute("SELECT * FROM automatic_wake_attempts")
        )
    with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
        claim(db, chain_id, [run])
    db.automatic_work.recover(current_owner_id="replacement")
    with db.connection() as conn:
        assert (
            tuple(
                tuple(row)
                for row in conn.execute("SELECT * FROM automatic_work_reservations")
            )
            == before
        )
        assert (
            tuple(
                tuple(row)
                for row in conn.execute("SELECT * FROM automatic_wake_attempts")
            )
            == attempts
        )
    assert db.get_run(run)["wake_delivered_at"] is None
    fresh = chain(db, submission="fresh-manual-turn")
    reservation = db.automatic_work.reserve(
        fresh,
        reservation_id="manual-generation",
        owner_id="replacement",
        kind="generation",
        amount=1,
    )
    assert reservation.chain_id == fresh
    assert not db.imported_authority_allowed(
        kind, {"run": run, "chain": chain_id, "wake_attempt": "attempt"}[kind]
    )


@pytest.mark.parametrize("kind", ["goal", "chain", "wake_attempt"])
def test_goal_existing_attempt_cannot_bypass_import_hold(db, kind):
    goal = db.goal_runs.create(request(), launch_id="goal-launch")
    goal = db.goal_runs.set_provisioning(goal, status="ready")
    attempt = db.automatic_work.prepare_goal_iteration(goal.id, owner_id="owner")
    hold(
        db,
        (
            kind,
            {"goal": goal.id, "chain": goal.chain_id, "wake_attempt": attempt.id}[kind],
        ),
    )
    with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
        db.automatic_work.prepare_goal_iteration(goal.id, owner_id="owner")


def test_binding_hold_survives_disk_readiness_and_save(tmp_path):
    database = WorkspaceDB(":memory:")
    registry = LocalWorkspaceRegistryService(database)
    workspace = registry.create_workspace(workspace_id="imported", name="Imported")
    binding = WorkspaceRuntimeBinding(
        binding_id="imported-binding",
        workspace_id=workspace.workspace_id,
        binding_kind="local-filesystem",
        label="Imported",
        locator=str(tmp_path),
        status="ready",
        metadata={"allow_write": True},
    )
    registry.save_runtime_binding(binding)
    with database.transaction() as cursor:
        database.hold_imported_bindings(
            cursor, "import-operation", [binding.binding_id]
        )
    registry.save_runtime_binding(binding)
    assert registry.get_runtime_binding(binding.binding_id).status.value == "blocked"
    assert (
        registry.list_runtime_bindings(workspace.workspace_id)[0].status.value
        == "blocked"
    )
    assert (
        registry.list_folder_bindings(workspace.workspace_id)[0].status.value
        == "blocked"
    )
    database.close()


def test_hold_and_imported_row_rollback_together(db):
    with pytest.raises(RuntimeError, match="crash"), db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO agent_runs (id,conversation_id,agent_kind,status,created_at,updated_at) VALUES ('atomic','conversation','primary','running','then','then')"
        )
        db.hold_imported_authority(cursor, "operation", [("run", "atomic")])
        cursor.execute(
            "INSERT INTO mobile_consolidation_batches VALUES ('operation','fingerprint','batch','{}','then')"
        )
        raise RuntimeError("crash")
    assert db.get_run("atomic") is None
    assert db.imported_authority_allowed("run", "atomic")
    with db.connection() as cursor:
        assert (
            cursor.execute(
                "SELECT count(*) FROM mobile_consolidation_batches"
            ).fetchone()[0]
            == 0
        )
        with pytest.raises(ValueError, match="owner_transaction"):
            db.hold_imported_authority(cursor, "operation", [("run", "atomic")])
    peer = AgentRunsDB(":memory:")
    try:
        with (
            peer.transaction() as cursor,
            pytest.raises(ValueError, match="owner_transaction"),
        ):
            db.hold_imported_authority(cursor, "operation", [("run", "atomic")])
    finally:
        peer.close()


@pytest.mark.parametrize("kind", ["chain", "chat_start_attempt", "run"])
def test_imported_chat_start_and_descendant_root_cannot_dispatch(db, kind):
    root = chain(db)
    source = db.create_run(
        conversation_id="conversation", agent_kind="primary", work_chain_id=root
    )
    kwargs = {
        "attempt_id": "chat-start",
        "source_run_id": source,
        "target_conversation_id": "target",
        "target_session_id": "target",
        "target_session_incarnation": "incarnation",
        "owner_id": "owner",
        "draft_revision": 0,
        "context_epoch": 0,
        "request_fingerprint": "a" * 64,
    }
    attempt = db.automatic_work.prepare_chat_start(**kwargs)
    hold(
        db,
        (kind, {"chain": root, "chat_start_attempt": attempt.id, "run": source}[kind]),
    )
    with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
        db.automatic_work.prepare_chat_start(**kwargs)
    with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
        db.automatic_work.accept_chat_start(attempt.id, owner_id="owner")
    with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
        db.automatic_work.check_active(attempt.chain_id, owner_id="owner")


@pytest.mark.parametrize("kind", ["run", "progress", "wake_attempt"])
def test_progress_and_final_acceptance_cannot_bypass_hold(db, kind):
    root = chain(db)
    run = survivor(db, root)
    attempt = db.automatic_work.claim_wake(
        root,
        attempt_id="progress-attempt",
        owner_id="owner",
        session_id="session",
        run_ids=(),
        progress_messages=(("message", run),),
    )
    hold(
        db,
        (kind, {"run": run, "progress": "message", "wake_attempt": attempt.id}[kind]),
    )
    assert (
        db.automatic_work.pending_progress_sources("conversation", [("message", run)])
        == ()
    )
    with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
        db.automatic_work.accept_wake(attempt.id, owner_id="owner")
    assert (
        db.automatic_work.read_attempt(attempt.id, owner_id="owner").state == "prepared"
    )


def test_existing_reservation_fast_path_cannot_reauthorize_import(db):
    root = chain(db)
    kwargs = {
        "reservation_id": "reserved",
        "owner_id": "owner",
        "kind": "generation",
        "amount": 1,
    }
    db.automatic_work.reserve(root, **kwargs)
    hold(db, ("chain", root))
    with pytest.raises(AutomaticWorkRefused, match="import_review_required"):
        db.automatic_work.reserve(root, **kwargs)


def test_native_binding_projection_is_never_file_authority(tmp_path):
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_chat_controller import (
        list_project_instruction_bindings,
    )
    from tldw_chatbook.Tools.workspace_file_roots import (
        _iter_valid_folder_bindings,
        workspace_context_note,
    )
    from tldw_chatbook.Workspaces.display_state import (
        _recompute_filesystem_binding_status,
    )

    database = WorkspaceDB(":memory:")
    registry = LocalWorkspaceRegistryService(database)
    registry.create_workspace(workspace_id="workspace", name="Workspace")
    registry.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="workspace",
            binding_id="held",
            binding_kind="local-filesystem",
            label="Held",
            locator=str(tmp_path),
            status="ready",
            metadata={"access": "rw"},
        )
    )
    assert tuple(
        _iter_valid_folder_bindings(registry.list_folder_bindings("workspace"))
    )
    assert list_project_instruction_bindings(
        SimpleNamespace(workspace_id="workspace"), registry
    )
    with database.transaction() as cursor:
        database.hold_imported_bindings(cursor, "operation", ["held"])
    binding = registry.list_folder_bindings("workspace")[0]
    assert binding.import_review_required
    assert _recompute_filesystem_binding_status(binding).import_review_required
    assert tuple(_iter_valid_folder_bindings([binding])) == ()
    assert (
        list_project_instruction_bindings(
            SimpleNamespace(workspace_id="workspace"), registry
        )
        == ()
    )
    assert str(tmp_path) not in workspace_context_note("workspace", registry=registry)
    database.close()


def test_fresh_process_open_preserves_held_running_and_terminal_records(tmp_path):
    import json
    import subprocess
    import sys

    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner

    path = tmp_path / "agent_runs.db"
    expected = {}
    with ConsoleExecutionOwner.acquire(path) as owner:
        native = AgentRunsDB(path, execution_owner=owner)
        try:
            for kind in ("primary", "subagent", "local_command"):
                run_id = native.create_run(
                    conversation_id="conversation", agent_kind=kind
                )
                hold(native, ("run", run_id))
                expected[run_id] = native.get_run(run_id)
            done = native.create_run(
                conversation_id="conversation", agent_kind="primary"
            )
            native.set_status(done, "done", "historical result")
            hold(native, ("run", done))
            expected[done] = native.get_run(done)
            control = native.create_run(conversation_id="control", agent_kind="primary")
        finally:
            native.close()
    program = """import json, sys
from Tests import real_profile_guard
real_profile_guard.install()
from pathlib import Path
from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
with ConsoleExecutionOwner.acquire(Path(sys.argv[1])) as owner:
    db=AgentRunsDB(Path(sys.argv[1]), execution_owner=owner)
    try:
        print(json.dumps({identity:db.get_run(identity) for identity in sys.argv[2:]}))
    finally:
        db.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path), *expected, control],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout.splitlines()[-1])
    assert {identity: actual[identity] for identity in expected} == expected
    assert actual[control]["status"] == "error"
