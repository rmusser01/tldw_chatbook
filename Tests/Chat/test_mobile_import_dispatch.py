"""Native Console startup and manual turns never approve imported authority."""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.DB.test_automatic_wake_attempts import survivor
from Tests.DB.test_automatic_work_budget import chain
from Tests.DB.test_mobile_import_holds import hold

# BASE6e83 reproduces the native fixture's config-admission refusal without this marker.
pytestmark = pytest.mark.bootstrap_profile


@pytest_asyncio.fixture
async def stores(tmp_path):
    from Tests.Agents.test_goal_models import request
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    with ConsoleExecutionOwner.acquire(tmp_path / "runs.db") as owner:
        runs = AgentRunsDB(tmp_path / "runs.db", execution_owner=owner)
        chat = CharactersRAGDB(tmp_path / "chat.db", "import-test")
        workspace = WorkspaceDB(tmp_path / "workspace.db")
        registry = LocalWorkspaceRegistryService(workspace)
        registry.create_workspace(workspace_id="workspace", name="Workspace")
        root = tmp_path / "fixture"
        root.mkdir()
        registry.save_runtime_binding(
            WorkspaceRuntimeBinding(
                workspace_id="workspace",
                binding_id="binding",
                binding_kind="local-filesystem",
                label="Fixture",
                locator=str(root),
                status="ready",
                metadata={"access": "rw"},
            )
        )
        persistence = ChatPersistenceService(chat, workspace_registry=registry)
        try:
            yield runs, persistence, registry, request(str(root))
        finally:
            for runtime in getattr(runs, "_goal_test_runtimes", ()):
                await runtime.dispose()
            runs.close()
            chat.close_connection()
            workspace.close()


@pytest.mark.asyncio
async def test_imported_goal_dispatch_startup_and_manual_send_preserve_history(
    stores, monkeypatch
):
    goal, _, _, controller, coordinator, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    runs = stores[0]
    hold(runs, ("goal", goal.id), ("chain", goal.chain_id))
    before = runs.goal_runs.get(goal.id)
    try:
        await controller.fleet_wake.recover()
        await coordinator.recover()
        coordinator.manual_send(goal.conversation_id)
        assert await coordinator.start(goal.id) == before
        result = await coordinator.dispatch_once(goal.id)
        assert result.reason_code == "import_review_required"
        assert calls == []
        assert runs.goal_runs.get(goal.id) == before
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_held_survivor_is_absent_from_fleet_seed(tmp_path, monkeypatch):
    from Tests.Chat import test_console_fleet_wake as wake_rig
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    owner = ConsoleExecutionOwner.acquire(tmp_path / "runs.db")
    monkeypatch.setattr(
        wake_rig,
        "owned_agent_runs_db",
        lambda path, **kwargs: AgentRunsDB(path, execution_owner=owner, **kwargs),
    )

    chat, _app, runs, _store, session, gateway, _bridge, controller = (
        wake_rig._controller_rig(tmp_path)
    )
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "true")
    root = chain(runs, conversation=session.id)
    run = survivor(runs, root, conversation=session.id)
    hold(runs, ("run", run))
    before = runs.get_run(run)
    try:
        await controller.fleet_wake.recover()
        assert controller.fleet_wake.seed_from_marks() == 0
        controller.fleet_wake.retry_soon()
        await asyncio.sleep(0.05)
        assert gateway.payloads == []
        result = await controller.submit_draft(
            "A fresh manual instruction", session_id=session.id
        )
        assert result.accepted
        assert len(gateway.payloads) == 1
        assert "saved result" not in str(gateway.payloads)
        assert runs.get_run(run) == before
    finally:
        controller.fleet_wake.dispose()
        runs.close()
        chat.close()
        owner.close()


def test_native_launch_discovery_reads_exact_hold_without_sweeping(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Chat import console_launch_wake as launch
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    path = tmp_path / "agent_runs.db"
    app = SimpleNamespace(chachanotes_db=SimpleNamespace(db_path=tmp_path / "chat.db"))
    monkeypatch.setattr(launch, "autowake_enabled", lambda: True)
    with ConsoleExecutionOwner.acquire(path) as owner:
        runs = AgentRunsDB(path, execution_owner=owner)
        try:
            root = chain(runs)
            run = survivor(runs, root)
            assert launch.pending_conversations_at_launch(app) == ("conversation",)
            hold(runs, ("run", run))
            assert launch.pending_conversations_at_launch(app) == ()
            assert runs.get_run(run)["wake_delivered_at"] is None
        finally:
            runs.close()
