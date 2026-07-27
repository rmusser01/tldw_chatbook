# Tests/Agents/test_run_log_service_wiring.py
"""The service owns the writer: one counter per run tree, every caller logged."""

from pathlib import Path

import pytest

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.run_log import RunLogWriter
from tldw_chatbook.Agents.run_log_format import iter_records
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db")
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    return db, registry, tmp_path


def chat_call_returning(text):
    def call(**kwargs):
        return {"choices": [{"message": {"content": text}}]}

    return call


def read_all(root: Path):
    run_dirs = list((root / "agent-runs").iterdir())
    run_dirs = [d for d in run_dirs if d.is_dir()]
    assert len(run_dirs) == 1
    records = []
    for segment in sorted(run_dirs[0].glob("logs.*.txt")):
        records.extend(iter_records(segment.read_bytes()))
    return records


def test_a_plain_run_writes_records_without_the_caller_wiring_anything(wired):
    db, registry, root = wired
    service = AgentService(db, registry, chat_call=chat_call_returning("hello"))
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    records = read_all(root)
    assert [r.type for r in records] == ["model"]
    assert records[0].content == "hello"
    assert records[0].kind == "primary"


def test_record_numbers_are_unique_across_the_whole_run_tree(wired):
    db, registry, root = wired
    writer = RunLogWriter()
    service = AgentService(
        db, registry, chat_call=chat_call_returning("x"), run_log_writer=writer
    )
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    # Simulate a child appending through the same shared writer.
    writer.append(run_id="child", kind="subagent", type="model", content="child work")
    numbers = [r.number for r in read_all(root)]
    assert numbers == sorted(set(numbers))


def test_disabled_writer_leaves_the_run_untouched(wired, monkeypatch):
    db, registry, root = wired
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    service = AgentService(db, registry, chat_call=chat_call_returning("hello"))
    _run_id, outcome = service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    assert outcome.final_text == "hello"
    assert not (root / "agent-runs").exists()
