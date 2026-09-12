"""Persist the runtime budget counter without relabelling it as raw usage."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import RUN_DONE, RUN_ERROR, AgentConfig
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.LLM_Calls import pricing_catalog


@pytest.mark.parametrize(
    "usage, expected",
    [
        ({"prompt_tokens": 100, "completion_tokens": 50}, 150),
        (
            {"input_tokens": 100, "cache_read_input_tokens": 100, "output_tokens": 50},
            160,
        ),
        (None, None),
    ],
)
def test_completed_service_run_persists_its_budget_counter(
    tmp_path, monkeypatch, usage, expected
):
    prices = SimpleNamespace(
        input_per_mtok=100, cache_read_per_mtok=10, cache_write_per_mtok=125
    )
    monkeypatch.setattr(
        pricing_catalog,
        "get_pricing_catalog",
        lambda: SimpleNamespace(get_pricing=lambda *args: prices),
    )
    response = {
        "choices": [{"message": {"content": "a completed answer"}}],
        "usage": usage,
    }
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    try:
        service = AgentService(
            db=db, registry=ToolCatalogRegistry(), chat_call=lambda **kwargs: response
        )
        run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "answer this"}],
            config=AgentConfig(model="budget-test", system_prompt="Be helpful."),
            api_endpoint="anthropic",
        )
        assert outcome.status == RUN_DONE
        if expected is None:
            assert outcome.total_tokens > 0  # Missing usage uses the runtime estimate.
        else:
            assert outcome.total_tokens == expected
        assert db.get_run(run_id)["budget_tokens"] == outcome.total_tokens
    finally:
        db.close()


def test_exception_escaping_runtime_leaves_budget_unknown(tmp_path, monkeypatch):
    def broken_loop(*args, **kwargs):
        raise RuntimeError("runtime escaped before returning its accounting")

    monkeypatch.setattr(agent_service, "run_agent_loop", broken_loop)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    try:
        service = AgentService(
            db=db, registry=ToolCatalogRegistry(), chat_call=lambda **kwargs: {}
        )
        run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[],
            config=AgentConfig(model="budget-test", system_prompt="Be helpful."),
            api_endpoint="anthropic",
        )
        assert outcome.status == RUN_ERROR
        assert db.get_run(run_id)["budget_tokens"] is None
    finally:
        db.close()


@pytest.mark.parametrize("first_budget, expected", [(None, 29), (17, 17)])
def test_measured_budget_survives_racing_terminal_observation(
    tmp_path, monkeypatch, first_budget, expected
):
    from tldw_chatbook.Agents.agent_models import RUN_CANCELLED

    db = AgentRunsDB(tmp_path / "runs.db", client_id="terminal-budget-race")
    try:
        service = AgentService(
            db, ToolCatalogRegistry(), chat_call=lambda **kwargs: {}
        )
        run_id = db.create_run(conversation_id="c", agent_kind="subagent")
        original_terminal = db.set_terminal_with_step

        def terminal_race(*args, **kwargs):
            # The caller already saw running. A real competing cancellation
            # now wins this same lifecycle index before the measured outcome.
            monkeypatch.setattr(db, "set_terminal_with_step", original_terminal)
            service._set_terminal_status(
                run_id, RUN_CANCELLED, result="first result",
                budget_tokens=first_budget,
            )
            return original_terminal(*args, **kwargs)

        monkeypatch.setattr(db, "set_terminal_with_step", terminal_race)
        assert not service._set_terminal_status(
            run_id, RUN_CANCELLED, result="late result", budget_tokens=29
        )
        row = db.get_run(run_id)
        assert row["budget_tokens"] == expected
        assert row["status"] == RUN_CANCELLED
        assert row["result"] == "first result"
        assert sum(step["kind"] == "agent_run_cancelled" for step in row["steps"]) == 1
    finally:
        db.close()
