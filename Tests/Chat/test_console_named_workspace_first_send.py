"""TASK-33621.23: the first send in a new named workspace reaches the provider.

Reproduced live on the merge-base 857b3dd7d0 and again on dafecd3a78: create a
named workspace from the New Workspace dialog (its default "Create a workspace
Agent" Persona), open its first chat, choose the project folder, send. The send
stopped at ``controller_submit status=blocked`` about 10 ms after
``durable_commit status=succeeded``. The transcript kept only the user row, the
status bar read "Agent blocked", and the log had nothing else.

The workspace's default Persona gives every new chat in it a system prompt.
The durable trace builder labelled that unsaved system row ACTIVE_REQUEST, the
request's ``system`` category rejects it, and the error was swallowed.
9fcf153421 (TASK-33940.3) fixed the labelling, but its test builds a plain
session with a hand-set system prompt. This test drives the creation path the
user takes instead:

- the real ``LocalWorkspaceRegistryService`` creates the workspace, and its real
  ``WorkspaceAgentProvisioner`` writes the "Workspace 1 Agent" Persona through
  the real ``LocalCharacterPersonaService``;
- the real store applies that default to the new chat through
  ``resolve_new_console_assistant``, as the Console runtime does;
- the folder is chosen through ``commit_project_instruction_setup_decision``,
  the Inspector picker's commit seam;
- the real controller sends with Capture On over a real SQLite file.

``_stream_assistant_response`` is replaced whole, from provider entry on: the
agent runtime, project-instruction resolution, the "Send project
instructions?" notice, the folder picker and the gateway do not run here. The
reported block happened before that call, at trace provenance, so this is
enough for it; the notice and Change Review variants rest on live checks. On
dafecd3a78 this test fails: the send ends BLOCKED, paused for trace
provenance, and the provider is never entered.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_first_send_atomicity import _CheckpointObservingGateway
from Tests.Chat.test_console_trace_first_send_atomicity import (
    _force_capture_on,
    _record_provider_entries,
)
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_assistant_defaults import (
    resolve_new_console_assistant,
)
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    commit_project_instruction_setup_decision,
    list_project_instruction_bindings,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunStatus,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_trace_provenance import ConsoleTraceCaptureMode
from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState
from tldw_chatbook.Chat.prompt_history import PromptHistory
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.Workspaces.agent_provisioning import WorkspaceAgentProvisioner
from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

pytestmark = pytest.mark.bootstrap_profile

_WORKSPACE_ID = "workspace-1"
_BINDING_ID = "workspace-1-folder"


def _new_named_workspace_console(
    tmp_path: Path,
) -> tuple[ConsoleChatStore, ConsoleChatController, LocalWorkspaceRegistryService]:
    """Create a named workspace the way the New Workspace dialog does."""
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "AGENTS.md").write_text("Always answer briefly.\n", encoding="utf-8")
    db = CharactersRAGDB(tmp_path / "chachanotes.sqlite", client_id="task-33621-23")
    personas = LocalCharacterPersonaService(
        db, persona_store_path=tmp_path / "personas.json"
    )
    provisioner = WorkspaceAgentProvisioner(
        personas, MCPPermissionStore(tmp_path / "mcp_permissions.json")
    )
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="task-33621-23"),
        agent_provisioner=provisioner.provision,
    )
    workspace = registry.create_workspace(
        workspace_id=_WORKSPACE_ID, name="Workspace 1"
    )
    # The dialog's default: the workspace gets its own Agent Persona.
    assert workspace.assistant_defaults is not None
    registry.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id=_WORKSPACE_ID,
            binding_id=_BINDING_ID,
            binding_kind="local-filesystem",
            label="project",
            locator=str(folder),
            status="ready",
            metadata={"access": "ro"},
        )
    )
    registry.set_active_workspace(_WORKSPACE_ID)
    app = SimpleNamespace(
        app_config={},
        workspace_registry_service=registry,
        local_character_persona_service=personas,
    )
    store = ConsoleChatStore(
        persistence=ChatPersistenceService(db, workspace_registry=registry),
        workspace_context=ConsoleWorkspaceContext(active_workspace_id=_WORKSPACE_ID),
        assistant_defaults_provider=lambda workspace_id, settings: (
            resolve_new_console_assistant(app, workspace_id, settings)
        ),
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=_CheckpointObservingGateway(db),
        provider="llama_cpp",
        model="test-model",
    )
    controller.prompt_history = PromptHistory(tmp_path / "history.jsonl")
    return store, controller, registry


def _choose_project_folder(
    store: ConsoleChatStore,
    registry: LocalWorkspaceRegistryService,
    session_id: str,
) -> None:
    """Apply the folder through the Inspector picker's commit seam."""
    session = next(item for item in store.sessions() if item.id == session_id)
    options = tuple(list_project_instruction_bindings(session, registry))
    action, chosen = commit_project_instruction_setup_decision(
        store=store,
        session_id=session_id,
        registry=registry,
        expected_state=session.project_instruction_state,
        expected_options=options,
        action="select",
        binding_id=_BINDING_ID,
    )
    assert action == "select" and chosen is not None


@pytest.mark.asyncio
async def test_first_send_in_a_new_named_workspace_reaches_the_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, controller, registry = _new_named_workspace_console(tmp_path)
    _force_capture_on(monkeypatch)
    entries = _record_provider_entries(controller, monkeypatch)

    session = controller.new_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", model="test-model")
    )
    # Setup check: the chat inherited the workspace Agent, as it does live.
    assert session.workspace_id == _WORKSPACE_ID
    assert session.assistant_kind == "persona"
    assert "Workspace 1" in (session.settings.system_prompt or "")
    _choose_project_folder(store, registry, session.id)

    result = await controller.submit_draft(
        "Reply with just the word pong.", session_id=session.id
    )

    preparation = store.preparation_for_session(session.id)
    assert (
        preparation is None
        or preparation.state is not ConsoleTurnPreparationState.PAUSED
    ), f"the first send paused: {preparation.pause_kind}"
    assert result.accepted is True
    assert result.terminal_status is not ConsoleRunStatus.BLOCKED, result.visible_copy
    assert controller.run_state_for(session.id).status is not ConsoleRunStatus.BLOCKED
    assert len(entries) == 1, "the first send must reach the provider exactly once"
    assert entries[0]["capture_mode_override"] is ConsoleTraceCaptureMode.CAPTURE_ON
    assert entries[0]["trace_request"] is not None
    sent_roles = [row.get("role") for row in entries[0]["provider_messages"]]
    assert sent_roles[0] == "system", "the workspace Agent's prompt must be sent"
