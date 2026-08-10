"""AgentService reports steps live via on_step with agent_kind (Task 3)."""

import json

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY,
    AGENT_KIND_SUBAGENT,
    SPAWN_TOOL_NAME,
    STEP_SPAWN,
    STEP_TOOL_CALL,
    AgentConfig,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN


def _fence(name, args):
    return f"{FENCE_OPEN}\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _registry():
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    return reg


def test_on_step_receives_primary_steps_in_order(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry()
    script = [
        {
            "choices": [
                {"message": {"content": _fence("calculator", {"expression": "6*7"})}}
            ]
        },
        {"choices": [{"message": {"content": "It is 42."}}]},
    ]

    def chat_call(**kwargs):
        return script.pop(0)

    seen = []
    service = AgentService(
        db,
        reg,
        chat_call=chat_call,
        on_step=lambda step, kind, run_id: seen.append((kind, step.kind)),
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "6*7?"}],
        config=AgentConfig(
            model="m", system_prompt="s", allowed_tools=("calculator", SPAWN_TOOL_NAME)
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == "done"
    kinds = [k for (_who, k) in seen]
    assert STEP_TOOL_CALL in kinds
    assert all(who == AGENT_KIND_PRIMARY for (who, _k) in seen)


def test_on_step_distinguishes_subagent_steps(tmp_path, inline_spawns):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry()
    # Primary spawns a sub-agent; sub-agent answers directly.
    script = [
        {
            "choices": [
                {"message": {"content": _fence(SPAWN_TOOL_NAME, {"task": "add 1+1"})}}
            ]
        },
        {"choices": [{"message": {"content": "2"}}]},  # sub-agent's turn
        {"choices": [{"message": {"content": "The answer is 2."}}]},  # primary final
    ]

    def chat_call(**kwargs):
        return script.pop(0)

    seen = []
    service = AgentService(
        db,
        reg,
        chat_call=chat_call,
        on_step=lambda step, kind, run_id: seen.append((kind, step.kind)),
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "delegate"}],
        config=AgentConfig(
            model="m", system_prompt="s", allowed_tools=("calculator", SPAWN_TOOL_NAME)
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == "done"
    assert (AGENT_KIND_PRIMARY, STEP_SPAWN) in seen
    assert any(who == AGENT_KIND_SUBAGENT for (who, _k) in seen)


def test_on_step_exception_does_not_crash_run(tmp_path):
    """A raising on_step callback must be captured — never kill the run."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry()

    def chat_call(**kwargs):
        return {"choices": [{"message": {"content": "hello"}}]}

    def bad_on_step(step, kind, run_id):
        raise RuntimeError("boom")

    service = AgentService(db, reg, chat_call=chat_call, on_step=bad_on_step)
    _run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", allowed_tools=("calculator",)),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == "done"  # on_step raised — run still completes
    assert outcome.final_text == "hello"


def test_on_step_default_is_noop(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry()
    service = AgentService(
        db, reg, chat_call=lambda **k: {"choices": [{"message": {"content": "hello"}}]}
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", allowed_tools=("calculator",)),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == "done"  # no on_step wired → no crash


def test_on_step_receives_run_id_for_primary_and_child(tmp_path, inline_spawns):
    """Task 3 (fleet PR 2a): on_step's third argument is the run_id of the
    run that produced the step -- the PRIMARY run's own id for primary
    steps, and the CHILD run's own (distinct) id for a spawned sub-agent's
    steps. This is what lets a fleet of concurrent children be told apart
    downstream (PR 2b)."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry()
    script = [
        {
            "choices": [
                {
                    "message": {
                        "content": _fence(SPAWN_TOOL_NAME, {"task": "child work"})
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "child answer"}}]},
        {"choices": [{"message": {"content": "parent answer"}}]},
    ]

    def chat_call(**kwargs):
        return script.pop(0)

    seen = []
    service = AgentService(
        db,
        reg,
        chat_call=chat_call,
        on_step=lambda step, kind, run_id: seen.append((kind, run_id, step.kind)),
    )
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="m", system_prompt="s", allowed_tools=("calculator", SPAWN_TOOL_NAME)
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == "done"
    primary_ids = {r for k, r, _ in seen if k == AGENT_KIND_PRIMARY}
    child_ids = {r for k, r, _ in seen if k == AGENT_KIND_SUBAGENT}
    assert primary_ids == {run_id}
    assert len(child_ids) == 1 and child_ids.isdisjoint(primary_ids)
    # Every step carries a non-empty run id.
    assert all(r for _k, r, _s in seen)
