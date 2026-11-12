"""Ordinary log messages expose schema decisions without content bodies."""

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

from Tests.Agents.test_agent_service import (
    FakeBigProvider,
    ScriptedChat,
    _forced_discovery_plan,
    fence,
    native_call,
)
from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    AgentStep,
    RunBudget,
    RunOutcome,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("native", [False, True], ids=["fence", "native"])
@pytest.mark.parametrize("exhausted", [False, True], ids=["loaded", "budget-exhausted"])
def test_message_only_logs_report_schema_decisions_without_bodies(
    tmp_path, monkeypatch, native, exhausted
):
    class PrivateSchemaProvider(FakeBigProvider):
        def list_catalog(self):
            return [
                replace(
                    entry,
                    one_line_description="PRIVATE_SEARCH_QUERY PRIVATE_CATALOG_BODY",
                )
                for entry in super().list_catalog()
            ]

        def load_schema(self, tool_id):
            return replace(
                super().load_schema(tool_id), description="PRIVATE_SCHEMA_BODY"
            )

    def turn(name, args):
        return (
            {"content": None, "tool_calls": [native_call(name, args)]}
            if native
            else fence(name, args)
        )

    registry = ToolCatalogRegistry()
    registry.register_provider(PrivateSchemaProvider())
    replies = [
        turn("find_tools", {"query": "PRIVATE_SEARCH_QUERY"}),
        turn("load_tools", {"ids": ["t0", "PRIVATE_INVALID_ID"]}),
    ]
    if not exhausted:
        replies.append(turn("t0", {}))
    replies.append("done")
    db = AgentRunsDB(tmp_path / "runs.db", client_id="diagnostics")
    service = AgentService(db=db, registry=registry, chat_call=ScriptedChat(replies))
    if exhausted:
        monkeypatch.setattr(
            service, "_project_instruction_request_fits", lambda *_: False
        )
    messages = []
    sink = logger.add(messages.append, level="INFO", format="{message}")
    try:
        service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "PRIVATE_USER_PROMPT"}],
            config=AgentConfig(
                model="m",
                system_prompt="s",
                allowed_tools=("t0",),
                native_tools=native,
                budget=RunBudget(max_steps=20),
            ),
            api_endpoint="openai" if native else "llama_cpp",
            first_request_schema_plan=_forced_discovery_plan(),
        )
    finally:
        logger.remove(sink)
        db.close()
    diagnostics = [
        str(line).strip()
        for line in messages
        if str(line).startswith(
            ("agent schema request", "agent tool discovery", "agent schema load")
        )
    ]
    assert any(
        "active=0 runtime=2 find_tools=1 load_tools=1" in line for line in diagnostics
    )
    assert any("agent tool discovery matches=1" in line for line in diagnostics)
    expected = (
        "requested=2 accepted=0 omitted=1 invalid=1 details_omitted=1"
        if exhausted
        else "requested=2 accepted=1 omitted=0 invalid=1 details_omitted=0"
    )
    assert any(expected in line for line in diagnostics)
    if not exhausted:
        assert any("active=1 runtime=2" in line for line in diagnostics)
    rendered = "\n".join(diagnostics)
    for private in (
        "PRIVATE_SCHEMA_BODY",
        "PRIVATE_INVALID_ID",
        "PRIVATE_USER_PROMPT",
        "PRIVATE_SEARCH_QUERY",
        "PRIVATE_CATALOG_BODY",
    ):
        assert private not in rendered


def test_plain_step_log_is_bounded_and_excludes_content():
    store = MagicMock()
    store.messages_for_session.return_value = []
    identity = ConsoleChatStore()
    identity.create_session(session_id="s1")
    for name in (
        "progress_owner_scope",
        "prepare_progress_inbox",
        "session_workspace_id",
        "register_progress_message_store",
    ):
        setattr(store, name, getattr(identity, name))
    bridge = ConsoleAgentBridge(
        agent_runs_db=MagicMock(),
        store=store,
        provider_gateway=MagicMock(aclose_current_loop=AsyncMock()),
    )
    step = AgentStep(
        index=1,
        kind="tool_result",
        summary="PRIVATE_SUMMARY",
        tool_name="fs_\nread" + "x" * 500,
        args={"secret": "PRIVATE_ARGUMENT"},
        result="PRIVATE_RESULT",
        tool_outcome="blocked",
    )
    outcome = RunOutcome(status="done", steps=[step], final_text="done")
    messages = []
    sink = logger.add(messages.append, level="INFO", format="{message}")
    try:
        with patch.object(AgentService, "run_turn", return_value=("run-1", outcome)):
            bridge.run_reply(
                conversation_id="c1",
                session_id="s1",
                resolution=None,
                assistant_message_id="a1",
                model="gpt-4",
                session_system_prompt="sys",
                agent_messages=[{"role": "user", "content": "hi"}],
                should_cancel=lambda: False,
            )
    finally:
        logger.remove(sink)
    lines = [
        str(line).strip() for line in messages if str(line).startswith("agent run step")
    ]
    assert len(lines) == 1
    assert "kind=tool_result" in lines[0] and "outcome=blocked" in lines[0]
    assert "fs_\\nread" in lines[0]
    assert len(lines[0].splitlines()) == 1 and len(lines[0]) < 230
    assert all(
        value not in lines[0]
        for value in ("PRIVATE_SUMMARY", "PRIVATE_ARGUMENT", "PRIVATE_RESULT")
    )
