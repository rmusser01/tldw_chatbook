"""Real three-store setup survives lost replies without adopting unrelated history."""

import importlib
from dataclasses import replace

import pytest

from Tests.Agents.test_goal_models import request
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding


@pytest.fixture
def stores(tmp_path):
    runs = AgentRunsDB(tmp_path / "runs.db")
    chat = CharactersRAGDB(tmp_path / "chat.db", "test")
    workspace_db = WorkspaceDB(tmp_path / "workspace.db")
    registry = LocalWorkspaceRegistryService(workspace_db)
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
    yield runs, persistence, registry, request(str(root))
    runs.close()
    chat.close_connection()
    workspace_db.close()


def service(stores):
    runs, persistence, _registry, _ = stores
    return importlib.import_module(
        "tldw_chatbook.Agents.goal_run_service"
    ).GoalRunService(runs, persistence)


@pytest.mark.parametrize(
    "boundary",
    [
        "before_chat",
        "after_chat",
        "before_workspace",
        "after_workspace",
        "before_ready",
    ],
)
def test_partial_provisioning_retries_exact_identity_without_duplicate_history(
    stores, monkeypatch, boundary
):
    runs, persistence, registry, req = stores
    owner = service(stores)
    if boundary in ("before_chat", "after_chat"):
        obj, method = persistence.db, "add_conversation"
    elif boundary in ("before_workspace", "after_workspace"):
        obj, method = registry, "link_membership"
    else:
        obj, method = runs.goal_runs, "set_provisioning"
    original = getattr(obj, method)

    def injected(*args, **kwargs):
        if boundary.startswith("after"):
            original(*args, **kwargs)
        raise RuntimeError("simulated crash at store boundary")

    monkeypatch.setattr(obj, method, injected)
    try:
        owner.create(req, launch_id="start")
    except RuntimeError:
        pass  # final run-store write can fail too; durable launch survives
    with runs.connection() as conn:
        first_id = conn.execute("SELECT id FROM goal_runs").fetchone()[0]
    first = owner.get(first_id)
    assert first.status == "starting"
    monkeypatch.setattr(obj, method, original)
    again = service(stores).create(req, launch_id="start")
    assert again.status == "ready"
    assert (again.id, again.conversation_id, again.chain_id) == (
        first.id,
        first.conversation_id,
        first.chain_id,
    )
    repeated = owner.create(req, launch_id="start")
    assert repeated.revision == again.revision
    memberships = registry.get_item_memberships("conversation", again.conversation_id)
    assert len(memberships) == 1
    with persistence.db.transaction() as conn:
        assert conn.execute("SELECT count(*) FROM conversations").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
    assert again.accounting.used["model_call"] == 0
    assert again.accounting.started_at is None


def test_preallocated_uuid_does_not_adopt_an_unrelated_conversation(stores):
    runs, persistence, registry, req = stores
    intent = runs.goal_runs.create(req, launch_id="start")
    persistence.db.add_conversation(
        {"id": intent.conversation_id, "title": "Unrelated"}
    )
    result = service(stores).create(req, launch_id="start")
    assert result.status == "paused"
    assert result.pause_reason == "conversation_identity_conflict"
    assert registry.get_item_memberships("conversation", intent.conversation_id) == ()
    assert (
        persistence.db.get_conversation_by_id(intent.conversation_id)["title"]
        == "Unrelated"
    )


@pytest.mark.parametrize(
    "change", ["missing", "retargeted", "access", "source_missing"]
)
def test_changed_binding_pauses_before_chat_creation(stores, change):
    _runs, persistence, registry, req = stores
    if change == "missing":
        registry.remove_runtime_binding("binding")
    elif change == "source_missing":
        req = req.model_copy(
            update={
                "source_bindings": (
                    req.binding.model_copy(update={"binding_id": "missing-source"}),
                )
            }
        )
    else:
        binding = registry.get_runtime_binding("binding")
        registry.save_runtime_binding(
            replace(
                binding,
                **(
                    {"locator": str(__import__("pathlib").Path(binding.locator).parent)}
                    if change == "retargeted"
                    else {"metadata": {"access": "ro"}}
                ),
            )
        )
    result = service(stores).create(req, launch_id="start")
    assert result.status == "paused"
    assert result.pause_reason in ("binding_missing", "binding_changed")
    assert persistence.db.get_conversation_by_id(result.conversation_id) is None


def test_zero_goal_policy_creates_no_chat_or_dispatch(stores):
    _, persistence, _, req = stores
    req = request(
        req.binding.locator, policy=req.policy.model_copy(update={"model_calls": 0})
    )
    result = service(stores).create(req, launch_id="zero")
    assert result.status == "paused"
    assert result.pause_reason == "goal_policy_disabled"
    assert persistence.db.get_conversation_by_id(result.conversation_id) is None


def test_reopen_all_stores_after_chat_commit_recovers_same_conversation(
    stores, monkeypatch
):
    runs, persistence, registry, req = stores
    with monkeypatch.context() as patch:
        patch.setattr(
            registry,
            "link_membership",
            lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("workspace unavailable")
            ),
        )
        first = service(stores).create(req, launch_id="start")
    paths = runs.db_path_str, persistence.db.db_path, registry.db.db_path_str
    runs.close()
    persistence.db.close_connection()
    registry.db.close()
    reopened_runs = AgentRunsDB(paths[0])
    reopened_chat = CharactersRAGDB(paths[1], "test")
    reopened_workspaces = WorkspaceDB(paths[2])
    try:
        registry2 = LocalWorkspaceRegistryService(reopened_workspaces)
        owner = importlib.import_module(
            "tldw_chatbook.Agents.goal_run_service"
        ).GoalRunService(
            reopened_runs,
            ChatPersistenceService(reopened_chat, workspace_registry=registry2),
        )
        result = owner.create(req, launch_id="start")
        assert result.status == "ready"
        assert result.conversation_id == first.conversation_id
        assert result.chain_id == first.chain_id
        assert (
            len(registry2.get_item_memberships("conversation", first.conversation_id))
            == 1
        )
    finally:
        reopened_runs.close()
        reopened_chat.close_connection()
        reopened_workspaces.close()


def test_deleted_goal_history_is_not_recreated_or_adopted(stores):
    _runs, persistence, _registry, req = stores
    owner = service(stores)
    first = owner.create(req, launch_id="start")
    row = persistence.db.get_conversation_by_id(first.conversation_id)
    persistence.db.soft_delete_conversation(
        first.conversation_id, expected_version=row["version"]
    )
    second = owner.create(req, launch_id="start")
    assert second.status == "paused"
    assert second.pause_reason == "conversation_identity_conflict"
    assert persistence.db.get_conversation_by_id(first.conversation_id) is None
