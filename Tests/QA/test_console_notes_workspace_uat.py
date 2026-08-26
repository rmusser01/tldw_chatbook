"""Joined-path UAT for Console agent access to a configured notes workspace."""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

from tldw_chatbook import config as config_module
from tldw_chatbook.Agents.agent_models import RUN_DONE
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.MCP.permission_store import (
    MCPPermissionStore,
    definition_hash,
    resolve_effective_state,
)


UAT_MESSAGE = "Hi from tldw_Chatbook!"
SEED_LINE = "UAT seed: this note belongs to the configured workspace."


def _fence(name: str, arguments: dict) -> str:
    return (
        "```tool_call\n" + json.dumps({"name": name, "arguments": arguments}) + "\n```"
    )


class ScriptedConsoleGateway:
    """Deterministic model turns over the real Console streaming adapter."""

    def __init__(self, scripts: list[list[str]]) -> None:
        self._scripts = scripts
        self.calls: list[list[dict]] = []

    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        script = self._scripts[len(self.calls) - 1]
        for chunk in script:
            yield chunk


class ScratchPermissionService:
    """Controller-facing service backed by a real scratch permission store."""

    def __init__(self, path: Path) -> None:
        self.permission_store = MCPPermissionStore(path)
        self.decisions: list[tuple[str, str, str]] = []
        self.session_approvals: set[tuple[str, str]] = set()

    def get_kill_switch(self) -> bool:
        return self.permission_store.get_kill_switch()

    def gate_tool_test(self, hub):
        return resolve_effective_state(self.permission_store.load(), hub)

    def is_session_approved(self, server_key: str, tool_name: str) -> bool:
        return (server_key, tool_name) in self.session_approvals

    def approve_for_session(self, server_key: str, tool_name: str) -> None:
        self.session_approvals.add((server_key, tool_name))

    def set_tool_state(self, server_key, tool_name, state, *, tool) -> None:
        self.permission_store.set_tool_state(
            server_key,
            tool_name,
            state,
            definition_hash=definition_hash(tool.description, tool.input_schema),
        )

    def record_tool_decision(
        self, server_key, tool_name, *, decision, initiator, error=None
    ) -> None:
        self.decisions.append((server_key, tool_name, decision))


def _controller_with(service: ScratchPermissionService) -> ConsoleChatController:
    controller = object.__new__(ConsoleChatController)
    controller.app = SimpleNamespace(unified_mcp_service=service)
    controller._pending_approval_event = None
    controller._pending_approval_decisions = None
    return controller


def test_console_agent_reads_then_updates_configured_workspace_note(
    tmp_path, monkeypatch
):
    """Config -> controller -> bridge -> provider -> real fs tools -> disk."""
    workspace = tmp_path / "workspace"
    notes_dir = workspace / "notes"
    notes_dir.mkdir(parents=True)
    note = notes_dir / "project.md"
    before = f"# Project Notes\n\n{SEED_LINE}\n"
    note.write_text(before, encoding="utf-8")

    config_path = tmp_path / "profile" / "config.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        "[console]\n"
        "local_tools_enabled = true\n"
        f"workspace_root = {json.dumps(str(workspace))}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    assert tomllib.loads(config_path.read_text(encoding="utf-8"))
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    settings = config_module.load_settings(force_reload=True)
    assert settings["console"]["local_tools_enabled"] is True
    assert settings["console"]["workspace_root"] == str(workspace)

    permission_path = tmp_path / "profile" / "mcp_permissions.json"
    service = ScratchPermissionService(permission_path)
    controller = _controller_with(service)
    local_provider, review_hook = controller._compose_local_provider()
    assert local_provider is not None
    assert local_provider._root == workspace.resolve()
    assert review_hook is not None

    for tool_name in ("fs_read", "fs_edit"):
        hub = local_provider.hub_tool_for(tool_name)
        service.permission_store.set_tool_state(
            hub.server_key,
            hub.name,
            "allow",
            definition_hash=definition_hash(hub.description, hub.input_schema),
        )

    gateway = ScriptedConsoleGateway(
        [
            [_fence("find_tools", {"query": "read and edit a workspace note"})],
            [_fence("load_tools", {"ids": ["local:fs_read", "local:fs_edit"]})],
            [_fence("fs_read", {"path": "notes/project.md"})],
            [
                _fence(
                    "fs_edit",
                    {
                        "path": "notes/project.md",
                        "old_string": SEED_LINE,
                        "new_string": f"{SEED_LINE}\n\n{UAT_MESSAGE}",
                    },
                )
            ],
            ["Read the existing note and added the requested message."],
        ]
    )
    store = ConsoleChatStore()
    session = store.ensure_session()
    prompt = f"Read notes/project.md, then add exactly '{UAT_MESSAGE}' to the note."
    store.append_message(session.id, role=ConsoleMessageRole.USER, content=prompt)
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "profile" / "runs.db", client_id="uat"),
        store=store,
        provider_gateway=gateway,
        native_tools_enabled=lambda: False,
    )

    _run_id, outcome = bridge.run_reply(
        conversation_id="uat-notes-round-trip",
        session_id=session.id,
        resolution=ConsoleProviderResolution(
            provider="scripted",
            base_url="",
            model="scripted-uat-model",
            ready=True,
            execution_key="scripted",
        ),
        assistant_message_id=assistant.id,
        model="scripted-uat-model",
        session_system_prompt="Use workspace tools to complete the request.",
        agent_messages=[{"role": "user", "content": prompt}],
        should_cancel=lambda: False,
        local_provider=local_provider,
        review_tool_calls=review_hook,
    )

    assert outcome.status == RUN_DONE
    assert (
        outcome.final_text == "Read the existing note and added the requested message."
    )
    calls = [step.tool_name for step in outcome.steps if step.kind == "tool_call"]
    assert calls == ["find_tools", "load_tools", "fs_read", "fs_edit"]
    read_result = next(
        step.result
        for step in outcome.steps
        if step.kind == "tool_result" and step.tool_name == "fs_read"
    )
    assert SEED_LINE in read_result
    assert any(
        SEED_LINE in str(message.get("content", "")) for message in gateway.calls[3]
    )

    after = note.read_text(encoding="utf-8")
    assert after == f"{before.rstrip()}\n\n{UAT_MESSAGE}\n"
    assert after.count(UAT_MESSAGE) == 1
    tool_messages = [
        message.content
        for message in store.messages_for_session(session.id)
        if message.role is ConsoleMessageRole.TOOL
    ]
    assert any("fs_read" in message for message in tool_messages)
    assert any("fs_edit" in message for message in tool_messages)
    assert service.decisions == []

    config_payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert config_payload["console"]["workspace_root"] == str(workspace)
    assert permission_path.exists()
    permission_payload = json.loads(permission_path.read_text(encoding="utf-8"))
    permission_profile = permission_payload["profiles"]["default"]
    assert permission_profile["global_default"] == "ask"
    permission_tools = permission_profile["servers"]["local:__local__"]["tools"]
    for tool_name in ("fs_read", "fs_edit"):
        assert permission_tools[tool_name]["state"] == "allow"
        assert permission_tools[tool_name]["definition_hash"]
