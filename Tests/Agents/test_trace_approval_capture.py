"""Real AgentService/approval seams produce durable causal Trace steps."""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Agents.agent_models import AgentConfig
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook
from tldw_chatbook.Chat.trajectory import derive_trajectory
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Tools.file_operation_tools import ReadFileTool


def _native_call(name: str, args: dict, call_id: str = "call-1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


@pytest.mark.parametrize(
    ("decision", "decision_kind", "terminal_kind"),
    (
        ("approve_once", "approval_approved", "tool_succeeded"),
        ("deny", "approval_denied", "tool_failed"),
    ),
)
def test_real_agent_service_and_approval_hook_capture_ordered_lifecycle(
    tmp_path, monkeypatch, decision: str, decision_kind: str, terminal_kind: str
) -> None:
    target = tmp_path / "evidence.txt"
    target.write_text("evidence", encoding="utf-8")

    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as workspace_roots

    monkeypatch.setattr(file_tools, "_tool_sandbox_root", lambda: tmp_path)
    monkeypatch.setattr(
        file_tools,
        "allowed_file_roots",
        lambda **_kwargs: (tmp_path.resolve(),),
    )
    monkeypatch.setattr(
        workspace_roots, "allowed_file_roots", lambda **_kwargs: (tmp_path.resolve(),)
    )

    gate = BuiltinToolGate(service=None)
    provider = BuiltinToolProvider(gate=gate)
    provider._tools["read_file"] = ReadFileTool()
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)

    approval_rounds = []

    def decide(pending):
        approval_rounds.append(tuple(row.llm_name for row in pending))
        return {row.call_id or row.llm_name: decision for row in pending}

    review_hook = build_tool_review_hook(gate, provider, None, decide)
    replies = [
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            _native_call("read_file", {"file_path": str(target)})
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "done"}}]},
    ]

    def fake_provider(**_kwargs):
        return replies.pop(0)

    db = AgentRunsDB(tmp_path / "agent-runs.db", client_id="test")
    try:
        service = AgentService(
            db,
            registry,
            chat_call=fake_provider,
            review_tool_calls=review_hook,
            review_state_scope=gate.stamp_scope,
        )
        run_id, outcome = service.run_turn(
            conversation_id="conv-1",
            messages=[{"role": "user", "content": "read it"}],
            config=AgentConfig(
                model="model",
                system_prompt="system",
                allowed_tools=("read_file",),
                native_tools=True,
            ),
            api_endpoint="openai",
        )

        assert outcome.status == "done"
        assert approval_rounds == [("read_file",)]
        durable = db.get_run(run_id)
        kinds = [step["kind"] for step in durable["steps"]]
        assert kinds.index("model_request_started") < kinds.index("tool_proposed")
        assert kinds.index("tool_proposed") < kinds.index("approval_requested")
        assert kinds.index("approval_requested") < kinds.index(decision_kind)
        assert kinds.index(decision_kind) < kinds.index(terminal_kind)

        snapshot = derive_trajectory(
            [],
            {},
            [],
            [],
            [],
            agent_runs=[durable],
            agent_steps=[
                {**step, "run_id": run_id, "conversation_id": "conv-1"}
                for step in durable["steps"]
            ],
        )
        records = [record for turn in snapshot.turns for record in turn.records]
        proposal = next(record for record in records if record.kind == "tool_proposed")
        assert proposal.parent_event_id == f"agent-run:{run_id}"
        assert proposal.field_states["args"] == "omitted"
        assert proposal.sensitivity == "tool_content"
        assert str(target) not in repr(proposal)
    finally:
        db.close()
