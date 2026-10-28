"""Tool disclosure refusals explain recovery without granting execution."""

import pytest

from Tests.Agents.test_agent_service import (
    FakeBigProvider,
    ScriptedChat,
    _forced_discovery_plan,
    fence,
    native_call,
)
from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    RunBudget,
    ToolCall,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize(
    "allowed,disclosed,name,reloadable",
    [
        (("t0",), set(), "t0", True),
        ((), set(), "t0", False),
        ((), {"t0"}, "t0", False),
        (("t0",), set(), "unknown_tool", False),
        (("unknown_tool",), set(), "unknown_tool", False),
    ],
    ids=[
        "unloaded",
        "excluded",
        "disclosed-but-excluded",
        "unknown",
        "unknown-but-allowed",
    ],
)
def test_refusal_guidance_preserves_dispatch_and_approval_boundaries(
    tmp_path, allowed, disclosed, name, reloadable
):
    executions = []

    class RecordingProvider(FakeBigProvider):
        def invoke(self, tool_id, args):
            executions.append(tool_id)
            return super().invoke(tool_id, args)

    registry = ToolCatalogRegistry()
    registry.register_provider(RecordingProvider())
    db = AgentRunsDB(tmp_path / "runs.db", client_id="refusal")
    service = AgentService(db=db, registry=registry, chat_call=lambda **_: None)
    invoke = service._make_invoke_tool(
        AgentConfig(model="m", system_prompt="s", allowed_tools=allowed),
        disclosed,
        run_id="refused-run",
    )

    before_disclosed = frozenset(disclosed)
    result = invoke(ToolCall(name=name, args={}))

    assert not result.ok
    assert result.outcome == "blocked"
    assert result.dispatch_state == "not_started"
    assert result.approval_decision is None
    assert executions == []
    assert frozenset(disclosed) == before_disclosed
    if reloadable:
        assert "schema not loaded" in result.error
        assert "load_tools" in result.error
        assert 'ids=["t0"]' in result.error
        assert "retain" in result.error
    else:
        assert result.error == f"Tool not permitted: {name}"
        assert "load_tools" not in result.error


@pytest.mark.parametrize("native", [False, True], ids=["fence", "native"])
@pytest.mark.parametrize("replaced", [False, True], ids=["initial", "replaced"])
def test_allowed_tool_recovers_after_schema_reload(tmp_path, native, replaced):
    executions = []

    class RecordingProvider(FakeBigProvider):
        def invoke(self, tool_id, args):
            executions.append(tool_id)
            return super().invoke(tool_id, args)

    def tool_turn(name, args):
        if native:
            return {"content": None, "tool_calls": [native_call(name, args)]}
        return fence(name, args)

    replies = []
    if replaced:
        replies.extend(
            [
                tool_turn("load_tools", {"ids": ["fake:t0"]}),
                tool_turn("load_tools", {"ids": ["fake:t1"]}),
            ]
        )
    replies.extend(
        [
            tool_turn("t0", {}),
            tool_turn("load_tools", {"ids": ["t0"]}),
            tool_turn("t0", {}),
            "done",
        ]
    )
    chat = ScriptedChat(replies)
    registry = ToolCatalogRegistry()
    registry.register_provider(RecordingProvider())
    db = AgentRunsDB(tmp_path / "runs.db", client_id="reload")
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "Inspect the project"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("t0", "t1"),
            native_tools=native,
            budget=RunBudget(max_steps=20),
        ),
        api_endpoint="openai" if native else "llama_cpp",
        first_request_schema_plan=_forced_discovery_plan(),
    )

    assert outcome.status == "done"
    assert executions == ["fake:t0"]
    results = [
        step
        for step in db.get_run(run_id)["steps"]
        if step["kind"] == "tool_result" and step["tool_name"] == "t0"
    ]
    assert len(results) == 2
    assert results[0]["tool_outcome"] == "blocked"
    assert "schema not loaded" in results[0]["result"]
    assert "load_tools" in results[0]["result"]
    assert results[1]["tool_outcome"] == "success"
    assert results[1]["result"] == "invoked fake:t0"
    after_refusal = chat.calls[3 if replaced else 1]["messages_payload"]
    assert "schema not loaded" in after_refusal[-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["fence", "native"])
async def test_hook_preparation_preserves_schema_reload_guidance(tmp_path, native):
    import asyncio

    from tldw_chatbook.Agents.activation import worker_guard
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine

    def tool_turn(name, args):
        if native:
            return {"content": None, "tool_calls": [native_call(name, args)]}
        return fence(name, args)

    engine = HookEngine((), lambda *_: True, HookBudgetOwner())
    registry = ToolCatalogRegistry()
    registry.register_provider(FakeBigProvider())
    db = AgentRunsDB(tmp_path / "runs.db", client_id="hook-reload")
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=ScriptedChat(
            [
                tool_turn("t0", {}),
                tool_turn("load_tools", {"ids": ["t0"]}),
                tool_turn("t0", {}),
                "done",
            ]
        ),
        hooks_v2_engine=engine,
        hooks_v2_session_id="session",
        hooks_v2_turn_id="turn",
    )
    try:
        run_id, outcome = await asyncio.to_thread(
            worker_guard(service)(service.run_turn),
            conversation_id="c",
            messages=[{"role": "user", "content": "Inspect the project"}],
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
        results = [
            step
            for step in db.get_run(run_id)["steps"]
            if step["kind"] == "tool_result" and step["tool_name"] == "t0"
        ]
        assert outcome.status == "done"
        assert len(results) == 2
        assert "schema not loaded" in results[0]["result"]
        assert "load_tools" in results[0]["result"]
        assert results[0]["tool_outcome"] == "blocked"
        assert results[1]["result"] == "invoked fake:t0"
        assert results[1]["tool_outcome"] == "success"
    finally:
        await engine.close()
        db.close()
