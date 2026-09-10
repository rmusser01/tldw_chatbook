"""Search failures retain their outcome through the real agent loop."""

from tldw_chatbook.Agents.agent_models import (
    RUN_STUCK,
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolLoadSelection,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools import web_tool_impls


def test_unavailable_search_stops_after_three_different_queries(tmp_path, monkeypatch):
    queries = ("python release", "latest python version", "python release notes")
    backend_queries = []

    def unavailable(**kwargs):
        backend_queries.append(kwargs["search_query"])
        return {"processing_error": "DuckDuckGo returned an anti-bot challenge"}

    monkeypatch.setattr(
        "tldw_chatbook.Web_Scraping.WebSearch_APIs.perform_websearch", unavailable
    )
    provider = LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: EffectiveToolState(
            state="allow", origin="tool_override"
        ),
    )
    turns = iter(
        [
            ModelTurn(text="", tool_calls=(ToolCall("web_search", {"query": query}),))
            for query in queries
        ]
        + [ModelTurn(text="The unavailable searches did not stop the run.")]
    )
    traces = []
    records = []
    deps = LoopDeps(
        call_model=lambda _messages, _schemas: next(turns),
        invoke_tool=lambda call: provider.invoke(call.name, call.args),
        spawn=lambda _task: None,
        find_tools=lambda _query: [],
        load_schemas=lambda *_args: ToolLoadSelection(accepted=()),
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        on_trace_step=traces.append,
        on_record=lambda kind, payload: records.append((kind, payload)),
    )
    web_tool_impls._reset_state_for_tests()
    try:
        with use_run_id("search-failure-run"):
            outcome = run_agent_loop(
                AgentConfig(
                    model="test-model",
                    system_prompt="Search for the release.",
                    allowed_tools=("web_search",),
                    budget=RunBudget(max_steps=50, max_model_turns=10),
                ),
                [{"role": "user", "content": "Find the latest Python release."}],
                [provider.load_schema("web_search")],
                deps,
            )
    finally:
        web_tool_impls._reset_state_for_tests()

    assert outcome.status == RUN_STUCK
    assert backend_queries == list(queries)
    failures = [step for step in traces if step.kind == "tool_failed"]
    assert len(failures) == 3
    assert all(step.tool_outcome == "failed" for step in failures)
    assert not any(step.kind == "tool_succeeded" for step in traces)
    results = [step for step in outcome.steps if step.kind == "tool_result"]
    assert len(results) == 3
    assert all(
        "challenge" in step.result and "Stop repeating" in step.result
        for step in results
    )
    result_records = [payload for kind, payload in records if kind == "tool_result"]
    assert len(result_records) == 3
    assert all(payload["status"] == "error" for payload in result_records)
    stop_reason = next(
        step.summary for step in reversed(outcome.steps) if step.kind == "error"
    )
    assert "web_search" in stop_reason
    assert "fail" in stop_reason.lower()
