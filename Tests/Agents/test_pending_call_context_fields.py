"""ADR-090: rationale + description ride the existing pending-call chain."""

from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Chat.console_chat_controller import _collect_mcp_pending
from tldw_chatbook.Agents.agent_models import ToolCall


class _StubProvider:
    """Minimal pending_gate_for stand-in: records kwargs, returns fixed rows."""

    def __init__(self):
        self.seen = []

    def pending_gate_for(self, llm_name, args, call_id="", rationale=""):
        self.seen.append({"llm_name": llm_name, "rationale": rationale})
        return MCPPendingCall(
            llm_name=llm_name,
            server_key="s",
            tool_name=llm_name,
            server_label="S",
            arguments=dict(args or {}),
            reason="ask",
            rationale=rationale,
        )


def test_pending_call_fields_default_empty():
    row = MCPPendingCall(
        llm_name="x", server_key="s", tool_name="x", server_label="S",
        arguments={}, reason="ask",
    )
    assert row.rationale == ""
    assert row.description == ""


def test_collect_mcp_pending_passes_rationale_through():
    provider = _StubProvider()
    calls = [ToolCall(name="fs_read", args={"path": "a"}, rationale="why")]
    rows = _collect_mcp_pending(provider, calls)
    assert rows and rows[0].rationale == "why"
    assert provider.seen[0]["rationale"] == "why"
