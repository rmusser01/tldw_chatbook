# Tests/Agents/test_run_log_prompt_integration.py
"""Truncation points at the log; the prompt mentions it only when real."""

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ModelTurn,
    RunBudget,
    STEP_TOOL_RESULT,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import _truncate_tool_result, run_agent_loop
from tldw_chatbook.Agents.agent_service import RUN_LOG_PROMPT_SECTION, AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

from Tests.Agents.test_agent_runtime import CALC, fence, make_deps


def test_trailer_points_at_the_record_when_one_exists():
    out = _truncate_tool_result("z" * 100, 10, "grep_files", record_number=412)
    assert "search_run_log" in out
    assert "412" in out


def test_trailer_keeps_the_old_wording_when_there_is_no_record():
    out = _truncate_tool_result("z" * 100, 10, "grep_files", record_number=None)
    assert "search_run_log" not in out
    assert "narrower query" in out


def test_untruncated_content_is_returned_unchanged():
    assert _truncate_tool_result("short", 100, "t", record_number=7) == "short"


def _service(tmp_path, capture):
    db = AgentRunsDB(tmp_path / "runs.db")
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    return AgentService(db, registry, chat_call=capture)


def _run(service):
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="BASE", budget=RunBudget()),
        api_endpoint="openai",
    )


def test_prompt_mentions_the_log_when_logging_is_active(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    prompts = []

    def capture(**kwargs):
        prompts.append(kwargs["messages_payload"][0]["content"])
        return {"choices": [{"message": {"content": "ok"}}]}

    _run(_service(tmp_path, capture))
    assert any("search_run_log" in p for p in prompts)
    assert all(p.startswith("BASE") for p in prompts)


def test_prompt_is_silent_about_the_log_when_it_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    prompts = []

    def capture(**kwargs):
        prompts.append(kwargs["messages_payload"][0]["content"])
        return {"choices": [{"message": {"content": "ok"}}]}

    _run(_service(tmp_path, capture))
    assert all("search_run_log" not in p for p in prompts)


def test_truncation_trailer_names_the_tool_result_record_not_the_tool_call_record():
    """Both `tool_call` and `tool_result` are captured at the same dispatch
    site (see _emit_record's two calls in run_agent_loop); the trailer must
    point at the SECOND one's record number -- the record holding the full,
    untruncated content -- not the first. A wrong pointer is worse than no
    pointer: the model follows it and confidently retrieves someone else's
    content. Distinct sentinel numbers per record kind make an off-by-one
    (capturing the tool_call number instead) fail loudly here."""
    huge = "y" * 5000

    def on_record(record_type, payload):
        return {"tool_call": 111, "tool_result": 999}.get(record_type)

    cfg = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_tool_result_chars=100),
    )
    deps = make_deps(
        [ModelTurn(text=fence("calculator", {"expression": "1"})), ModelTurn(text="done")],
        invoke=lambda c: ToolResult(ok=True, content=huge),
    )
    deps.on_record = on_record
    out = run_agent_loop(cfg, [{"role": "user", "content": "hi"}], [CALC], deps)

    result_steps = [s for s in out.steps if s.kind == STEP_TOOL_RESULT]
    assert result_steps, "expected a truncated tool_result step"
    trailer = result_steps[0].result
    assert "record 000999" in trailer
    assert "000111" not in trailer


def test_prompt_mentions_the_log_on_a_non_native_fence_protocol_endpoint(
    tmp_path, monkeypatch
):
    """test_prompt_mentions_the_log_when_logging_is_active above drives
    api_endpoint="openai", a NATIVE-tools provider -- _make_call_model's
    non-native `else` branch (where system_content gets reassigned from
    config.system_prompt to append the rendered fence protocol, the exact
    seam the brief's literal RUN_LOG_PROMPT_SECTION placement would have
    been clobbered by) never executes there, so that test alone does not
    prove the fix. This drives a genuine fence-protocol provider."""
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    prompts = []

    def capture(**kwargs):
        prompts.append(kwargs["messages_payload"][0]["content"])
        return {"choices": [{"message": {"content": "ok"}}]}

    service = _service(tmp_path, capture)
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="BASE", budget=RunBudget()),
        api_endpoint="llama_cpp",
    )
    assert prompts, "expected at least one call_model turn"
    assert any(RUN_LOG_PROMPT_SECTION in p for p in prompts)
    assert all(p.startswith("BASE") for p in prompts)
