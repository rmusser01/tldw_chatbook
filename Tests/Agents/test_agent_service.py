# Tests/Agents/test_agent_service.py
"""Service tests: scripted chat_call (no network) + real AgentRunsDB."""
import json

import pytest

from tldw_chatbook.Agents.agent_models import (
    RUN_DONE, RUN_STUCK, SPAWN_TOOL_NAME, AgentConfig, RunBudget,
)
from tldw_chatbook.Agents.agent_service import (
    SUBAGENT_SYSTEM_PROMPT, AgentService,
)
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider, ToolCatalogRegistry,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def fence(name, args):
    return f'```tool_call\n{json.dumps({"name": name, "arguments": args})}\n```'


def provider_reply(text):
    return {"choices": [{"message": {"content": text}}]}


class ScriptedChat:
    """Returns scripted replies; records every call's kwargs."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return provider_reply(self.replies.pop(0))


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


def make_service(db, replies):
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = ScriptedChat(replies)
    return AgentService(db=db, registry=registry, chat_call=chat), chat


CFG = AgentConfig(model="test-model", system_prompt="You are helpful.",
                  allowed_tools=("calculator", "get_current_datetime",
                                 SPAWN_TOOL_NAME))


def test_plain_answer_persists_done_run(db):
    service, chat = make_service(db, ["Tokyo."])
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "capital of Japan?"}],
        config=CFG, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE and outcome.final_text == "Tokyo."
    run = db.get_run(run_id)
    assert run["status"] == "done" and run["result"] == "Tokyo."
    assert run["agent_kind"] == "primary"
    assert all(s["created_at"] for s in run["steps"])


def test_system_message_carries_protocol_and_user_prompt(db):
    service, chat = make_service(db, ["hi"])
    service.run_turn(conversation_id="c", messages=[
        {"role": "user", "content": "q"}], config=CFG,
        api_endpoint="llama_cpp")
    call = chat.calls[0]
    assert call["api_endpoint"] == "llama_cpp"
    assert call["streaming"] is False and call["model"] == "test-model"
    system = call["messages_payload"][0]
    assert system["role"] == "system"
    assert "You are helpful." in system["content"]
    assert "```tool_call" in system["content"]          # protocol rendered
    assert "calculator" in system["content"]            # direct-disclosed
    assert SPAWN_TOOL_NAME in system["content"]
    assert "find_tools" not in system["content"]        # small catalog


def test_real_tool_executes_through_gate(db):
    service, chat = make_service(
        db, [fence("calculator", {"expression": "6*7"}), "It is 42."])
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "6*7?"}],
        config=CFG, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE and outcome.final_text == "It is 42."
    # The tool result really came from CalculatorTool:
    followup = chat.calls[1]["messages_payload"]
    assert any("42" in m["content"] for m in followup
               if m["role"] == "user" and "Tool result" in m["content"])


def test_permission_gate_blocks_disallowed_tool(db):
    narrow = AgentConfig(model="m", system_prompt="s",
                         allowed_tools=("get_current_datetime",))
    service, _ = make_service(
        db, [fence("calculator", {"expression": "1"}), "gave up"])
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=narrow, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    run = db.get_run(run_id)
    results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    assert results and "not permitted" in results[0]["result"]


def test_spawn_creates_linked_child_with_clean_context(db):
    service, chat = make_service(db, [
        fence(SPAWN_TOOL_NAME, {"task": "compute 6*7"}),   # parent turn 1
        "sub answer: 42",                                  # CHILD turn 1
        "The sub-agent says 42.",                          # parent turn 2
    ])
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[
            {"role": "user", "content": "delegate this"}],
        config=CFG, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE and outcome.subagents_spawned == 1
    runs = db.list_runs("c")
    child = next(r for r in runs if r["agent_kind"] == "subagent")
    assert child["parent_run_id"] == run_id
    assert child["task"] == "compute 6*7"
    assert child["status"] == "done" and child["result"] == "sub answer: 42"
    # Clean context: the child's provider call saw ONLY its task + its own
    # system prompt — never the parent's transcript.
    child_call = chat.calls[1]["messages_payload"]
    assert child_call[0]["role"] == "system"
    assert SUBAGENT_SYSTEM_PROMPT.split(".")[0] in child_call[0]["content"]
    assert child_call[1] == {"role": "user", "content": "compute 6*7"}
    assert not any("delegate this" in m["content"] for m in child_call)
    assert db.count_subagent_runs("c") == 1


def test_subagent_result_is_capped(db):
    long_answer = "x" * 10000
    service, _ = make_service(db, [
        fence(SPAWN_TOOL_NAME, {"task": "t"}), long_answer, "done"])
    tight = AgentConfig(model="m", system_prompt="s",
                        allowed_tools=(SPAWN_TOOL_NAME,),
                        budget=RunBudget(max_subagent_result_chars=100))
    _, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=tight, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    run = db.list_runs("c", include_superseded=False)
    parent = next(r for r in run if r["agent_kind"] == "primary")
    capped = [s for s in parent["steps"] if s["kind"] == "tool_result"][0]
    assert "[truncated]" in capped["result"]
    assert len(capped["result"]) < 300


def test_child_cannot_spawn(db):
    service, _ = make_service(db, [
        fence(SPAWN_TOOL_NAME, {"task": "t"}),          # parent spawns
        fence(SPAWN_TOOL_NAME, {"task": "nested"}),     # CHILD tries to spawn
        "child recovered",                              # child answers
        "parent done",
    ])
    _, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=CFG, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c") == 1             # no grandchildren


def test_supersede_marks_old_tree_before_new_run(db):
    service, _ = make_service(db, ["first answer"])
    old_id, _ = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=CFG, api_endpoint="llama_cpp")
    service2, _ = make_service(db, ["second answer"])
    new_id, _ = service2.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=CFG, api_endpoint="llama_cpp", supersede_run_id=old_id)
    assert db.get_run(old_id)["status"] == "superseded"
    assert db.get_run(new_id)["status"] == "done"


def test_stuck_run_persists_stuck_status(db):
    replies = [fence("calculator", {"expression": "1"})] * 10
    service, _ = make_service(db, replies)
    tight = AgentConfig(model="m", system_prompt="s",
                        allowed_tools=("calculator",),
                        budget=RunBudget(max_steps=3))
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=tight, api_endpoint="llama_cpp")
    assert outcome.status == RUN_STUCK
    assert db.get_run(run_id)["status"] == "stuck"


def test_provider_exception_persists_error_status(db):
    def exploding_chat(**kwargs):
        raise RuntimeError("connection refused")

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    service = AgentService(db=db, registry=registry,
                           chat_call=exploding_chat)
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=CFG, api_endpoint="llama_cpp")
    assert outcome.status == "error"
    run = db.get_run(run_id)
    assert run["status"] == "error"
    assert any("connection refused" in (s.get("summary") or "")
               for s in run["steps"])
