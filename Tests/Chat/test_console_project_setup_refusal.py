"""Recovery decisions release a real Console send without provider dispatch."""

from types import SimpleNamespace

import pytest

from Tests.console_provider_doubles import persisted_console_store, with_destination
from tldw_chatbook.Agents.agent_models import RUN_DONE, RunOutcome
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["disable", "cancel", "unavailable"])
@pytest.mark.parametrize("ephemeral", [True, False])
async def test_setup_recovery_releases_run_and_excludes_unsent_echo(
    tmp_path, decision, ephemeral
):
    """Disabling unavailable instructions must not strand the Console run."""

    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.db", client_id="instruction-disable")
    )
    registry.create_workspace(workspace_id="w1", name="Workspace 1")
    store = persisted_console_store(
        db_path=tmp_path / "chat.db", workspace_registry=registry
    )
    session = store.create_session(workspace_id="w1", ephemeral=ephemeral)
    store.set_session_project_instruction_state(
        session.id,
        ProjectInstructionControlState(
            project_instructions_enabled=True,
            working_folder_binding_id="removed-binding",
            working_folder_locator_fingerprint="f" * 64,
        ),
    )
    provider_calls = []
    bridge_calls = []

    class Gateway:
        async def resolve_for_send(self, _selection):
            provider_calls.append(True)
            return with_destination(
                ConsoleProviderResolution(
                    provider="OpenAI",
                    base_url="http://127.0.0.1:18991/v1",
                    model="gpt-4o-mini",
                    ready=True,
                    readiness_key="openai",
                    execution_key="openai",
                    max_tokens=128,
                )
            )

    class Bridge:
        def run_reply(self, **_kwargs):
            bridge_calls.append(True)
            return "run-1", RunOutcome(status=RUN_DONE, steps=[], final_text="done")

    async def disable(_session_id, _options, _recovery_code):
        return decision, None

    controller = ConsoleChatController(
        store=store,
        provider_gateway=Gateway(),
        provider="openai",
        model="gpt-4o-mini",
        agent_bridge=Bridge(),
        agent_runtime_enabled=True,
        select_project_instruction_binding=None
        if decision == "unavailable"
        else disable,
    )
    controller.app = SimpleNamespace(
        workspace_registry_service=registry,
        call_from_thread=lambda callback: callback(),
    )

    first = await controller.submit_draft("first")

    assert controller.run_state_for(session.id).status is ConsoleRunStatus.BLOCKED
    assert bridge_calls == []

    messages = store.messages_for_session(session.id)
    assert not controller.activity_for(session.id).occupies_slot
    if decision == "disable":
        assert not session.project_instruction_state.project_instructions_enabled

    if ephemeral:
        assert (
            next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT).status
            == "failed"
        )
        assert not first.accepted
        assert not first.should_clear_draft
        assert (
            next(m for m in messages if m.role is ConsoleMessageRole.USER).status
            == "failed"
        )
        assert all(
            m["content"] != "first"
            for m in controller._provider_messages_for_session(session.id)
        )
    else:
        # Current dev retains durably accepted turns for recovery rather than
        # refunding them into a composer that could submit a duplicate.
        assert first.accepted
        assert store.dispatch_recovery_for_session(session.id) is not None
