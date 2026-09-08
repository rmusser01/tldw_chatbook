"""Runtime contract for the editable bulk-reader preset."""

import importlib.util
import json
import threading
from dataclasses import replace

from Tests.Agents.conftest import join_fleet_children
from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    SPAWN_TOOL_NAME,
    AgentConfig,
    RunBudget,
)
from tldw_chatbook.Agents.agent_service import SUBAGENT_SYSTEM_PROMPT, AgentService
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider, _default_specs
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.MCP.permission_store import EffectiveToolState


def _fence(name: str, arguments: dict) -> str:
    payload = {"name": name, "arguments": arguments}
    return f"```tool_call\n{json.dumps(payload)}\n```"


def _reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class _RecordingChatProvider:
    """Thread-safe scripts addressed by parent/child model identity."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.parent_replies = [
            _fence(
                SPAWN_TOOL_NAME,
                {
                    "task": "Read source.txt, then try to replace it.",
                    "agent": "bulk-reader",
                },
            ),
            "parent done",
        ]
        self.child_replies = [
            _fence("fs_read", {"path": "source.txt"}),
            _fence("fs_write", {"path": "source.txt", "content": "changed"}),
            "child done",
        ]
        self.parent_calls: list[dict] = []
        self.child_calls: list[dict] = []

    def __call__(self, **kwargs) -> dict:
        messages = kwargs["messages_payload"]
        is_child = bool(
            messages
            and messages[0].get("role") == "system"
            and str(messages[0].get("content", "")).startswith(
                SUBAGENT_SYSTEM_PROMPT.split(".")[0]
            )
        )
        with self._lock:
            calls = self.child_calls if is_child else self.parent_calls
            replies = self.child_replies if is_child else self.parent_replies
            calls.append(kwargs)
            assert replies, "recording provider script exhausted"
            return _reply(replies.pop(0))


def test_bulk_reader_worker_model_and_reader_only_tools_reach_runtime(tmp_path):
    spec = importlib.util.find_spec("tldw_chatbook.Agents.agent_presets")
    assert spec is not None, "bulk-reader preset module is missing"
    from tldw_chatbook.Agents.agent_presets import BULK_READER_PRESET

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "source.txt"
    source.write_text("original evidence\n", encoding="utf-8")

    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    db.create_agent_definition(replace(BULK_READER_PRESET, model="budget-reader-model"))
    local = LocalToolProvider(
        workspace_root=workspace,
        specs=[
            item
            for item in _default_specs(workspace)
            if item.name in {"fs_read", "fs_write"}
        ],
        resolve_state=lambda _tool: EffectiveToolState(
            state="allow", origin="tool_override"
        ),
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(local)
    provider = _RecordingChatProvider()
    service = AgentService(db=db, registry=registry, chat_call=provider)
    config = AgentConfig(
        model="parent-model",
        system_prompt="You are the primary.",
        allowed_tools=("fs_read", "fs_write", SPAWN_TOOL_NAME),
        budget=RunBudget(max_steps=30, max_model_turns=12, max_subagents=1),
    )

    _run_id, outcome = service.run_turn(
        conversation_id="bulk-reader-runtime",
        messages=[{"role": "user", "content": "delegate"}],
        config=config,
        api_endpoint="recording-provider",
    )
    join_fleet_children(service)

    assert outcome.status == RUN_DONE
    assert source.read_text(encoding="utf-8") == "original evidence\n"
    assert len(provider.child_calls) == 3
    assert all(call["model"] == "budget-reader-model" for call in provider.child_calls)
    assert any(
        "1\toriginal evidence" in str(message.get("content", ""))
        for message in provider.child_calls[1]["messages_payload"]
    )
    assert any(
        "Tool not permitted: fs_write" in str(message.get("content", ""))
        for message in provider.child_calls[2]["messages_payload"]
    )
    assert all(call["model"] == "parent-model" for call in provider.parent_calls)
