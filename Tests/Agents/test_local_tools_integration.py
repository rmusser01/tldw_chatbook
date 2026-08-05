# Tests/Agents/test_local_tools_integration.py
"""End-to-end local-tool integration (phase 1, ADR-032): a scripted model
emits a ```tool_call fence for fs_list; the run must flow fence -> registry
-> build_local_review_hook -> approval round trip -> LocalToolProvider.invoke
-> fs_list core -> result appended back into the model's next turn.

Harness pattern mirrors test_agent_service.py (ScriptedChat + real
AgentRunsDB, no network); provider/review-hook wiring mirrors
console_agent_bridge._compose_run_registry_and_allowed +
_combined_review_state_scope (registry with the local provider,
review_tool_calls=hook, review_state_scope=provider.stamp_scope).
"""

import json

import pytest

from tldw_chatbook.Agents.agent_models import RUN_DONE, AgentConfig
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_DENY_REFUSAL,
    LOCAL_SERVER_KEY,
    LocalToolProvider,
)
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.Chat.console_chat_controller import build_local_review_hook
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.MCP.permission_store import EffectiveToolState


def fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


class ScriptedChat:
    """Returns scripted replies; records every call's kwargs."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        message = item if isinstance(item, dict) else {"content": item}
        return {"choices": [{"message": message}]}


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


@pytest.fixture()
def workspace(tmp_path):
    """The confined workspace root: exactly one file, so its name MUST show
    up in a successful fs_list result."""
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "notes.txt").write_text("hello", encoding="utf-8")
    return root


def make_service(db, workspace, replies, approvals, approval_calls):
    """Assemble the run exactly as the bridge does: registry with builtins +
    the local provider, the build_local_review_hook batch hook, and the
    provider's stamp_scope as review_state_scope."""
    provider = LocalToolProvider(
        workspace_root=workspace,
        resolve_state=lambda hub: EffectiveToolState(
            state="ask", origin="global_default"
        ),
    )

    def request_approvals(pending):
        approval_calls.append(pending)
        return dict(approvals)

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(provider)
    chat = ScriptedChat(replies)
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        review_tool_calls=build_local_review_hook(provider, request_approvals),
        review_state_scope=provider.stamp_scope,
    )
    return service, chat


CFG = AgentConfig(
    model="test-model",
    system_prompt="You are helpful.",
    allowed_tools=("fs_list",),
)


def test_fs_list_fence_flow_executes_after_approve_once(db, workspace):
    approval_calls = []
    service, chat = make_service(
        db,
        workspace,
        [fence("fs_list", {"path": "."}), "The workspace has notes.txt."],
        {"fs_list": "approve_once"},
        approval_calls,
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "what files are here?"}],
        config=CFG,
        api_endpoint="llama_cpp",  # fence-protocol endpoint (harness pattern)
        should_cancel=lambda: False,
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "The workspace has notes.txt."
    assert len(chat.calls) == 2  # tool turn + final turn

    # 1. The tool step's result contains the filename (not an error).
    tool_results = [s for s in outcome.steps if s.kind == "tool_result"]
    assert [s.tool_name for s in tool_results] == ["fs_list"]
    assert "notes.txt" in tool_results[0].result
    assert not tool_results[0].result.startswith("ERROR")

    # 2. Exactly ONE approval round trip, gated on the local server key.
    assert len(approval_calls) == 1
    assert len(approval_calls[0]) == 1
    pending = approval_calls[0][0]
    assert isinstance(pending, MCPPendingCall)
    assert pending.server_key == LOCAL_SERVER_KEY == "local:__local__"
    assert pending.llm_name == "fs_list"
    assert pending.tool_name == "fs_list"
    assert pending.arguments == {"path": "."}

    # 3. The tool result went back to the model (fence convention: a
    # user-role "Tool result for {name}: ..." line in the second turn).
    second_payload = chat.calls[1]["messages_payload"]
    assert any(
        m["role"] == "user"
        and m["content"].startswith("Tool result for fs_list: ")
        and "notes.txt" in m["content"]
        for m in second_payload
    )


def test_fs_list_fence_flow_denied_still_completes(db, workspace):
    approval_calls = []
    service, chat = make_service(
        db,
        workspace,
        [fence("fs_list", {"path": "."}), "I could not list the files."],
        {"fs_list": "deny"},
        approval_calls,
    )

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "what files are here?"}],
        config=CFG,
        api_endpoint="llama_cpp",
        should_cancel=lambda: False,
    )

    # The run completes; the model's second turn still runs.
    assert outcome.status == RUN_DONE
    assert outcome.final_text == "I could not list the files."
    assert len(chat.calls) == 2

    # The denial surfaces as the pinned LOCAL_DENY_REFUSAL, never executed.
    tool_results = [s for s in outcome.steps if s.kind == "tool_result"]
    assert [s.tool_name for s in tool_results] == ["fs_list"]
    assert tool_results[0].result == f"ERROR: {LOCAL_DENY_REFUSAL}"
    assert "notes.txt" not in tool_results[0].result

    # The approval gate was still consulted exactly once.
    assert len(approval_calls) == 1
    assert approval_calls[0][0].server_key == LOCAL_SERVER_KEY
