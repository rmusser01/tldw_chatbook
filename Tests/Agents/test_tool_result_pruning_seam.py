"""TASK-25911: the prune seam in the agent send path.

The pure pruner is covered in Tests/Chat/test_console_history_budget.py;
these pin the SERVICE wiring: settings reach the payload actually sent to
chat_call, the protocol-aware round boundary is used, and the default
(no settings, no config) is byte-identical to today (AC#6).
"""

from __future__ import annotations

from types import SimpleNamespace

from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat.console_history_budget import ToolResultPruneSettings


def _service(**kwargs):
    return AgentService(
        db=SimpleNamespace(), registry=ToolCatalogRegistry(), **kwargs
    )


def _history_with_stale_tool_rows():
    rows = []
    for index in range(1, 5):
        rows.append({"role": "user", "content": f"prompt {index}"})
        rows.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {"name": "reader", "arguments": "{}"},
                    }
                ],
            }
        )
        rows.append(
            {
                "role": "tool",
                "tool_call_id": f"call-{index}",
                "content": ("x" * 6000) if index <= 2 else "small",
            }
        )
        rows.append({"role": "assistant", "content": f"answer {index}"})
    return rows


def _call_model(service):
    config = AgentConfig(
        model="m",
        system_prompt="s",
        provider="openai",
        native_tools=True,
        budget=RunBudget(max_steps=10),
    )
    return service._make_call_model(config, "openai", [], run_id="run-1")


def test_configured_pruning_reaches_the_wire_payload() -> None:
    sent = {}

    def fake_chat(**kwargs):
        sent["payload"] = kwargs["messages_payload"]
        return {"choices": [{"message": {"content": "ok"}}]}

    service = _service(
        chat_call=fake_chat,
        tool_result_pruning=ToolResultPruneSettings(
            keep_recent_turns=2,
            min_result_chars=4000,
            head_chars=500,
            min_reclaim_chars=2000,
        ),
    )
    _call_model(service)(_history_with_stale_tool_rows(), ())

    tool_rows = [row for row in sent["payload"] if row.get("role") == "tool"]
    assert len(tool_rows) == 4
    assert len(tool_rows[0]["content"]) < 6000, "old big result not pruned"
    assert "pruned" in tool_rows[0]["content"]
    assert tool_rows[0]["tool_call_id"] == "call-1", "pairing must survive"
    assert tool_rows[3]["content"] == "small", "recent round touched"


def test_default_service_sends_the_payload_untouched() -> None:
    """AC#6: no settings and no config -> byte-identical send."""
    sent = {}

    def fake_chat(**kwargs):
        sent["payload"] = kwargs["messages_payload"]
        return {"choices": [{"message": {"content": "ok"}}]}

    service = _service(chat_call=fake_chat)
    history = _history_with_stale_tool_rows()
    _call_model(service)(history, ())

    tool_rows = [row for row in sent["payload"] if row.get("role") == "tool"]
    assert len(tool_rows[0]["content"]) == 6000
    assert "pruned" not in tool_rows[0]["content"]
