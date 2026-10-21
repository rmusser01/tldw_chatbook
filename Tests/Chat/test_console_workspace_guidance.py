"""Provider-bound Console evidence keeps workspace grounding available."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_automatic_library_preparation import (
    ConsoleChatStore,
    _PolicyCoordinator,
    _RagService,
    _StreamingFence,
    _row,
)
from Tests.Chat.test_console_dispatch_recovery import (
    _acceptance,
    _database,
    _insert,
    _restored_store,
)
from Tests.Chat.test_console_agent_bridge import _ChunkGateway
from Tests.console_provider_doubles import with_destination
from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools import workspace_file_roots
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

pytestmark = pytest.mark.bootstrap_profile


class _ProjectGateway(_ChunkGateway):
    """Control only external transport; retain the production agent/payload."""

    def __init__(self):
        super().__init__([["Project qualification complete."]])
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def resolve_for_send(self, _selection):
        self.entered.set()
        await self.release.wait()
        return with_destination(
            ConsoleProviderResolution(
                ready=True,
                provider="openai",
                model="test-model",
                base_url="http://127.0.0.1:9099/v1",
                readiness_key="openai",
                execution_key="openai",
                max_tokens=2048,
            )
        )


async def _project_case(tmp_path, monkeypatch, *, native, automatic=1, access=0):
    """Real durable policy, workspace authority, controller and agent bridge."""
    workspace_db = WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="guidance")
    registry = LocalWorkspaceRegistryService(workspace_db)
    aliases = {}
    for name in ("owned", "visible"):
        registry.create_workspace(workspace_id=name, name=name)
        folder = tmp_path / name
        folder.mkdir()
        (folder / "README.md").write_text(f"# {name} project\n", encoding="utf-8")
        binding = registry.add_folder_binding(name, folder, allow_write=False)
        aliases[name] = binding.binding_id
    monkeypatch.setattr(workspace_file_roots, "_default_registry_instance", None)
    monkeypatch.setattr(workspace_file_roots, "_registry_factory", lambda: registry)
    db, conversation_id, repository = _database(tmp_path / "chat.sqlite")
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE console_conversation_library_policy SET auto_retrieve_on_send = ?, "
            "assistant_library_access = ? WHERE conversation_id = ?",
            (automatic, access, conversation_id),
        )
    store, session_id = _restored_store(db, conversation_id)
    store.persistence.workspace_registry = registry
    store.replace_session_settings(
        session_id,
        ConsoleSessionSettings(
            provider="openai", model="test-model", base_url="http://127.0.0.1:9099/v1"
        ),
    )
    session = store._session_or_raise(session_id)
    session.workspace_id = "owned"
    # A folder binding is sufficient for fs_*; project instructions remain off.
    store.set_session_project_instruction_state(
        session_id, ProjectInstructionControlState.legacy_disabled()
    )
    await store.hydrate_session_library_policy(session_id)
    store.set_session_rag_scope(
        session_id,
        RagScope(
            (ScopeItem("note", "owned-note"), ScopeItem("media", "owned-media")),
            "owned",
        ),
    )
    gateway = _ProjectGateway()
    runs = AgentRunsDB(tmp_path / "runs.sqlite", client_id="guidance")
    bridge = ConsoleAgentBridge(
        agent_runs_db=runs, store=store, provider_gateway=gateway
    )
    service = _RagService(
        {"runtime_backend": "local", "results": [_row("owned Library evidence")]}
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="openai",
        model="test-model",
        base_url="http://127.0.0.1:9099/v1",
        agent_runtime_enabled=True,
        agent_bridge=bridge,
    )
    controller.app = SimpleNamespace(
        workspace_registry_service=registry,
        library_rag_search_service=service,
        unified_mcp_service=SimpleNamespace(
            get_kill_switch=lambda: False,
            gate_tool_test=lambda _hub: EffectiveToolState(
                state="allow", origin="global_default"
            ),
            is_session_approved=lambda *_args, **_kwargs: False,
            record_tool_decision=lambda *_args, **_kwargs: None,
        ),
    )
    capture = controller.resolve_runtime_turn_configuration_snapshot

    def configuration(owning_id):
        snapshot = capture(owning_id)
        return replace(
            snapshot,
            tool_configuration={
                **snapshot.tool_configuration,
                "agent_runtime_enabled": True,
                "local_tools_enabled": True,
                "native_tool_calls_enabled": native,
            },
        )

    controller._turn_context_provider = configuration
    return SimpleNamespace(
        db=db,
        conversation_id=conversation_id,
        repository=repository,
        store=store,
        session=session,
        gateway=gateway,
        controller=controller,
        service=service,
        aliases=aliases,
        registry=registry,
        runs=runs,
        workspace_db=workspace_db,
    )


def _policy_rows(case):
    return tuple(
        tuple(row)
        for row in case.db.get_connection().execute(
            "SELECT * FROM console_conversation_library_policy ORDER BY conversation_id"
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False], ids=["native", "fenced"])
@pytest.mark.parametrize("retry", [False, True], ids=["initial", "retry"])
@private_profile_test
async def test_pending_project_turn_stays_with_owning_tab(
    request, tmp_path, monkeypatch, native, retry
):
    """Using the viewed tab at preparation/dispatch would leak its root and scope."""
    case = await _project_case(tmp_path, monkeypatch, native=native)
    try:
        controller, store = case.controller, case.store
        if retry:
            configuration = controller.resolve_turn_configuration_snapshot(
                case.session.id
            )
            authority = await controller._capture_turn_library_authority(
                case.session.id, configuration
            )
            acceptance = replace(
                _acceptance(case.conversation_id),
                frozen_authority=authority,
                attempt_id=authority.attempt_id,
                user_content="Summarize this project",
                resolved_destination=with_destination(
                    ConsoleProviderResolution(
                        ready=True,
                        provider="openai",
                        model="test-model",
                        base_url="http://127.0.0.1:9099/v1",
                        readiness_key="openai",
                        execution_key="openai",
                        max_tokens=2048,
                    )
                ).resolved_destination,
            )
            _insert(case.db, case.repository, acceptance)
            store, session_id = _restored_store(case.db, case.conversation_id)
            store.persistence.workspace_registry = case.registry
            store.replace_session_settings(
                session_id,
                ConsoleSessionSettings(
                    provider="openai",
                    model="test-model",
                    base_url="http://127.0.0.1:9099/v1",
                ),
            )
            case.session = store._session_or_raise(session_id)
            case.session.workspace_id = "owned"
            store.set_session_project_instruction_state(
                session_id, ProjectInstructionControlState.legacy_disabled()
            )
            store.set_session_rag_scope(
                session_id,
                RagScope(
                    (
                        ScopeItem("note", "owned-note"),
                        ScopeItem("media", "owned-media"),
                    ),
                    "owned",
                ),
            )
            await store.hydrate_session_library_policy(session_id)
            controller.store = store
            controller._agent_bridge._store = store
            task = asyncio.create_task(
                controller.retry_dispatch_recovery(case.session.id)
            )
        else:
            task = asyncio.create_task(
                controller.submit_draft(
                    "Summarize this project", session_id=case.session.id
                )
            )
        await asyncio.wait_for(case.gateway.entered.wait(), 10)
        visible = store.create_session(workspace_id="visible", title="Other tab")
        store.set_session_rag_scope(
            visible.id, RagScope((ScopeItem("note", "visible-note"),), "visible")
        )
        case.registry.set_active_workspace("visible")
        case.gateway.release.set()
        result = await asyncio.wait_for(task, 30)

        assert result.accepted, result.visible_copy
        assert case.gateway.calls == 1
        payload = "\n".join(
            str(row.get("content", "")) for row in case.gateway.messages_seen[0]
        )
        assert case.aliases["owned"] in payload
        assert case.aliases["visible"] not in payload
        assert "owned Library evidence" in payload
        assert len(case.service.calls) == 1
        assert case.service.calls[0]["scope"].allowlist == {
            "notes": frozenset({"owned-note"}),
            "media": frozenset({"owned-media"}),
        }
        assert bool(case.gateway.tools_seen[0]) is native
        assert store.active_session_id == visible.id
    finally:
        case.gateway.release.set()
        await case.controller.shutdown()
        case.runs.close()
        case.db.close()
        case.workspace_db.close()


def _assert_mixed_grounding(messages) -> None:
    context = "\n".join(row["content"] for row in messages)
    assert "workspace file evidence" in context
    assert "[S#] labels only for the supplied Library references" in context
    assert "needle body" in context
    assert (
        "Use only available evidence references below to ground factual claims."
        not in context
    )


@pytest.mark.asyncio
async def test_automatic_evidence_provider_payload_allows_workspace_grounding():
    store = ConsoleChatStore()
    store.library_policy_coordinator = _PolicyCoordinator(ConsoleAutoRetrieve.AUTOMATIC)
    gateway = _StreamingFence()
    service = _RagService(
        {"runtime_backend": "local", "results": [_row("needle body")]}
    )
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.app = SimpleNamespace(library_rag_search_service=service)

    result = await controller.submit_draft("Explain this project")

    assert result.accepted
    assert gateway.provider_calls == 1
    _assert_mixed_grounding(gateway.messages)


@pytest.mark.asyncio
async def test_dispatch_retry_provider_payload_allows_workspace_grounding(tmp_path):
    db, conversation_id, repository = _database(tmp_path / "mixed-retry.sqlite")
    try:
        db.get_connection().execute(
            "UPDATE console_conversation_library_policy SET auto_retrieve_on_send = 1 WHERE conversation_id = ?",
            (conversation_id,),
        )
        db.get_connection().commit()
        initial_store, initial_session_id = _restored_store(db, conversation_id)
        await initial_store.hydrate_session_library_policy(initial_session_id)
        initial_controller = ConsoleChatController(
            store=initial_store,
            provider_gateway=_StreamingFence(),
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            agent_runtime_enabled=False,
        )
        configuration = initial_controller.resolve_turn_configuration_snapshot(
            initial_session_id
        )
        authority = await initial_controller._capture_turn_library_authority(
            initial_session_id, configuration
        )
        assert authority.policy.auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC
        acceptance = replace(
            _acceptance(conversation_id),
            frozen_authority=authority,
            attempt_id=authority.attempt_id,
        )
        _insert(db, repository, acceptance)
        store, session_id = _restored_store(db, conversation_id)
        await store.hydrate_session_library_policy(session_id)
        gateway = _StreamingFence()
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            agent_runtime_enabled=False,
        )
        controller.app = SimpleNamespace(
            library_rag_search_service=_RagService(
                {"runtime_backend": "local", "results": [_row("needle body")]},
            )
        )

        result = await controller.retry_dispatch_recovery(session_id)

        assert result.accepted, result.visible_copy
        assert gateway.provider_calls == 1
        _assert_mixed_grounding(gateway.messages)
    finally:
        db.close()
