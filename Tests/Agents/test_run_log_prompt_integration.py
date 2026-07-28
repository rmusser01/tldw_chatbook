# Tests/Agents/test_run_log_prompt_integration.py
"""Truncation points at the log; the prompt mentions it only when real."""

import pytest

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
from tldw_chatbook.Agents.agent_runtime import _truncate_tool_result
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


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
