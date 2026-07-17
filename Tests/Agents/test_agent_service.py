# Tests/Agents/test_agent_service.py
"""Service tests: scripted chat_call (no network) + real AgentRunsDB."""
import dataclasses
import json

import pytest

from tldw_chatbook.Agents.agent_models import (
    DIRECT_DISCLOSE_THRESHOLD, FIND_TOOLS_NAME, LOAD_TOOLS_NAME, RUN_DONE,
    RUN_STUCK, SPAWN_TOOL_NAME, AgentConfig, RunBudget, ToolCatalogEntry,
    ToolResult, ToolSchema,
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


def provider_reply(item):
    """str -> plain content reply; dict -> used as the full message."""
    if isinstance(item, dict):
        return {"choices": [{"message": item}]}
    return {"choices": [{"message": {"content": item}}]}


def native_call(name, args, call_id="c1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


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

def test_native_endpoint_sends_tools_and_suppresses_fence_protocol(db):
    service, chat = make_service(db, [
        {"content": None,
         "tool_calls": [native_call("calculator", {"expression": "2+2"})]},
        "4."])
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=CFG, api_endpoint="groq", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    first = chat.calls[0]
    names = [t["function"]["name"] for t in first["tools"]]
    assert "calculator" in names and "spawn_subagent" in names
    assert "tool_call" not in first["messages_payload"][0]["content"]  # no fence protocol
    # Second call's history carries the native pairing:
    second_payload = chat.calls[1]["messages_payload"]
    assistant = [m for m in second_payload if m["role"] == "assistant"][0]
    assert assistant["tool_calls"][0]["function"]["name"] == "calculator"
    tool_msg = [m for m in second_payload if m.get("role") == "tool"][0]
    assert tool_msg["tool_call_id"] == "c1" and "4" in tool_msg["content"]


def test_native_multi_call_reply_dispatches_both_tools_in_one_turn(db):
    service, chat = make_service(db, [
        {"content": None, "tool_calls": [
            native_call("calculator", {"expression": "2+2"}, "a"),
            native_call("get_current_datetime", {}, "b")]},
        "done"])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "go"}],
        config=CFG, api_endpoint="openai", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE
    tool_results = [s for s in outcome.steps if s.kind == "tool_result"]
    assert [s.tool_name for s in tool_results] == [
        "calculator", "get_current_datetime"]
    assert len(chat.calls) == 2  # one batch turn + one final turn


def test_fence_fallback_unchanged_for_llama_cpp(db):
    service, chat = make_service(db, [fence("calculator",
                                            {"expression": "2+2"}), "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=CFG, api_endpoint="llama_cpp", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE
    assert "tools" not in chat.calls[0]                     # no tools= kwarg at all
    assert "tool_call" in chat.calls[0]["messages_payload"][0]["content"]


def test_native_kill_switch_forces_fence(db):
    cfg = dataclasses.replace(CFG, native_tools=False)
    service, chat = make_service(db, [fence("calculator",
                                            {"expression": "2+2"}), "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=cfg, api_endpoint="groq", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE and "tools" not in chat.calls[0]


def test_native_subagent_turns_also_carry_tools(db):
    service, chat = make_service(db, [
        {"content": None,
         "tool_calls": [native_call("spawn_subagent",
                                    {"task": "say hi"}, "s1")]},
        "hi from child",   # child's (native-mode) only turn
        "done"])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "go"}],
        config=CFG, api_endpoint="groq", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE
    child_call = chat.calls[1]
    assert child_call["messages_payload"][0]["content"].startswith(
        SUBAGENT_SYSTEM_PROMPT)
    assert "tools" in child_call         # native_tools propagated to the child


def test_malformed_native_arguments_error_is_echoed_and_recoverable(db):
    bad = {"id": "m1", "type": "function",
           "function": {"name": "calculator", "arguments": "{broken"}}
    service, chat = make_service(db, [
        {"content": None, "tool_calls": [bad]},
        {"content": None,
         "tool_calls": [native_call("calculator", {"expression": "2+2"},
                                    "m2")]},
        "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=CFG, api_endpoint="groq", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    retry_payload = chat.calls[1]["messages_payload"]
    tool_msgs = [m for m in retry_payload if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "m1"
    assert "ERROR" in tool_msgs[0]["content"]  # empty-args invoke fails, echoed


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


class FakeBigProvider:
    """A provider with more tools than DIRECT_DISCLOSE_THRESHOLD.

    Mirrors Tests/Agents/test_tool_catalog.py's FakeBigProvider: a catalog
    this large forces the find/load path (initial_disclosure defers
    everything and offers find_tools/load_tools instead of disclosing
    directly), which is the only path that exercises load_schemas' own
    room-capping.
    """

    def list_catalog(self):
        return [ToolCatalogEntry(id=f"fake:t{i}", name=f"t{i}",
                                 one_line_description=f"tool {i}",
                                 source="fake")
                for i in range(DIRECT_DISCLOSE_THRESHOLD + 3)]

    def load_schema(self, tool_id):
        return ToolSchema(id=tool_id, name=tool_id.split(":")[1],
                          description="fake", parameters={"type": "object"})

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content=f"invoked {tool_id}")


def test_load_tools_gate_disclosure_mirrors_loop_active_cap(db):
    """F1 regression: disclosed_names must stay capped like the loop's
    active list. Room is only 2, so requesting t0/t1/t2 must leave t2
    ungated — the loop's own room-slicing keeps `active` at [t0, t1], and
    the gate must refuse anything the loop didn't actually admit.
    """
    registry = ToolCatalogRegistry()
    registry.register_provider(FakeBigProvider())
    allowed = tuple(f"t{i}" for i in range(DIRECT_DISCLOSE_THRESHOLD + 3))
    config = AgentConfig(
        model="m", system_prompt="s", allowed_tools=allowed,
        budget=RunBudget(max_active_tools=2, max_steps=20))
    chat = ScriptedChat([
        fence(LOAD_TOOLS_NAME, {"ids": ["fake:t0", "fake:t1", "fake:t2"]}),
        fence("t2", {}),   # beyond room — the loop refused it "no room"
        fence("t0", {}),   # within room — must remain callable
        "done",
    ])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=config, api_endpoint="llama_cpp")

    assert outcome.status == RUN_DONE and outcome.final_text == "done"
    run = db.get_run(run_id)
    tool_results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    assert len(tool_results) == 3

    load_result, t2_result, t0_result = tool_results
    # The loop's own room-slicing already only admits 2 into `active`.
    assert load_result["result"] == "loaded: t0, t1"
    # The gate must refuse t2: the loop never put it in `active`.
    assert "Tool not permitted: t2" in t2_result["result"]
    # t0 stayed in room, so it must remain genuinely callable through
    # the gate (a real provider invocation, not a permission error).
    assert t0_result["result"] == "invoked fake:t0"


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


# --- G3: reloading an already-active tool must not consume active-tool
# room or desync the gate's disclosed set from the loop's own `active`
# list. ---

def test_reload_already_disclosed_tool_does_not_desync_and_admits_next(db):
    registry = ToolCatalogRegistry()
    registry.register_provider(FakeBigProvider())
    allowed = tuple(f"t{i}" for i in range(DIRECT_DISCLOSE_THRESHOLD + 3))
    config = AgentConfig(
        model="m", system_prompt="s", allowed_tools=allowed,
        budget=RunBudget(max_active_tools=2, max_steps=30))
    chat = ScriptedChat([
        fence(LOAD_TOOLS_NAME, {"ids": ["fake:t0"]}),
        fence(LOAD_TOOLS_NAME, {"ids": ["fake:t0"]}),   # re-load: no room eaten
        fence(LOAD_TOOLS_NAME, {"ids": ["fake:t1"]}),   # must still be admitted
        fence("t1", {}),
        "done",
    ])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=config, api_endpoint="llama_cpp")

    assert outcome.status == RUN_DONE and outcome.final_text == "done"
    run = db.get_run(run_id)
    tool_results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    load1, load2, load3, t1_result = tool_results
    assert load1["result"] == "loaded: t0"
    # Re-loading t0 is filtered out before the room slice — the generic
    # "no valid tools" message is an acceptable, cap-integrity-preserving
    # trade-off per the review decision (it's indistinguishable from
    # "all ids invalid" from the loop's point of view).
    assert load2["result"] == "ERROR: No valid tools found to load"
    # t1 must be genuinely admitted — the loop's active list was never
    # polluted with a duplicate t0 entry, so room for t1 remains.
    assert load3["result"] == "loaded: t1"
    assert t1_result["result"] == "invoked fake:t1"


# --- Q7: disclosure (initial active set, find_tools, load_tools) must
# respect config.allowed_tools; the permission gate is a backstop, not
# the only checkpoint. ---

def test_initial_disclosure_excludes_disallowed_tools(db):
    narrow = AgentConfig(model="m", system_prompt="s",
                         allowed_tools=("calculator", SPAWN_TOOL_NAME))
    service, chat = make_service(db, ["ok"])
    service.run_turn(conversation_id="c", messages=[
        {"role": "user", "content": "q"}], config=narrow,
        api_endpoint="llama_cpp")
    system = chat.calls[0]["messages_payload"][0]["content"]
    assert "calculator" in system
    assert "get_current_datetime" not in system


def test_find_and_load_tools_respect_allowed_tools(db):
    registry = ToolCatalogRegistry()
    registry.register_provider(FakeBigProvider())
    # Catalog has t0..t10; only t0 is allowed even though t1 exists.
    config = AgentConfig(
        model="m", system_prompt="s", allowed_tools=("t0",),
        budget=RunBudget(max_active_tools=5, max_steps=20))
    chat = ScriptedChat([
        fence(FIND_TOOLS_NAME, {"query": "t"}),
        fence(LOAD_TOOLS_NAME, {"ids": ["fake:t0", "fake:t1"]}),
        fence("t1", {}),
        "done",
    ])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=config, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    run = db.get_run(run_id)
    tool_results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    find_result, load_result, t1_result = tool_results
    assert "t0" in find_result["result"]
    assert "t1" not in find_result["result"]
    assert load_result["result"] == "loaded: t0"
    assert "Tool not permitted: t1" in t1_result["result"]


# --- task-244 AC #3/#4: load_tools must fall back to resolve_name() when
# a model echoes a bare tool NAME (as seen in a find_tools result line)
# instead of the catalog id. ---

def test_load_tools_with_bare_name_loads_via_resolve_name_fallback(db):
    """AC #4: models echo the tool NAME from a find_tools result line, not
    the catalog id. load_tools(ids=["calculator"]) must load the tool."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(FakeBigProvider())  # catalog > threshold: forces find/load
    config = AgentConfig(
        model="m", system_prompt="s",
        allowed_tools=("calculator", "get_current_datetime"),
        budget=RunBudget(max_active_tools=8, max_steps=20))
    chat = ScriptedChat([
        fence(LOAD_TOOLS_NAME, {"ids": ["calculator"]}),
        fence("calculator", {"expression": "2+2"}),
        "4.",
    ])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=config, api_endpoint="llama_cpp")

    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    run = db.get_run(run_id)
    tool_results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    load_result, calc_result = tool_results
    # The bare name must actually LOAD -- not just avoid the error string.
    assert load_result["result"] == "loaded: calculator"
    calc_payload = json.loads(calc_result["result"])
    assert calc_payload["result"] == 4


def test_load_tools_bare_name_still_respects_allow_list(db):
    """A resolvable bare name OUTSIDE config.allowed_tools stays refused
    with the generic load error (Q7(c) gate unchanged)."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(FakeBigProvider())
    config = AgentConfig(
        model="m", system_prompt="s", allowed_tools=("calculator",),
        budget=RunBudget(max_active_tools=8, max_steps=20))
    chat = ScriptedChat([
        # "get_current_datetime" resolves via resolve_name(), but it is
        # not in config.allowed_tools -- Q7(c) must still refuse it.
        fence(LOAD_TOOLS_NAME, {"ids": ["get_current_datetime"]}),
        "done",
    ])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=config, api_endpoint="llama_cpp")

    assert outcome.status == RUN_DONE and outcome.final_text == "done"
    run = db.get_run(run_id)
    tool_results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    load_result, = tool_results
    assert load_result["result"] == "ERROR: No valid tools found to load"


def test_load_tools_unresolvable_junk_still_errors_generically(db):
    """ids=["definitely-not-a-tool"] -> 'No valid tools found to load'."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(FakeBigProvider())
    config = AgentConfig(
        model="m", system_prompt="s", allowed_tools=("calculator",),
        budget=RunBudget(max_active_tools=8, max_steps=20))
    chat = ScriptedChat([
        fence(LOAD_TOOLS_NAME, {"ids": ["definitely-not-a-tool"]}),
        "done",
    ])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "q"}],
        config=config, api_endpoint="llama_cpp")

    assert outcome.status == RUN_DONE and outcome.final_text == "done"
    run = db.get_run(run_id)
    tool_results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    load_result, = tool_results
    assert load_result["result"] == "ERROR: No valid tools found to load"


def test_idless_native_call_gets_synthesized_id_pairing_echo_and_result(db):
    """PR #648 review: some OpenAI-compatible servers omit tool-call ids. A
    synthesized id must appear identically in the assistant echo and the
    role='tool' reply, so history pairing never splits conventions."""
    idless = {"type": "function",
              "function": {"name": "calculator",
                           "arguments": json.dumps({"expression": "2+2"})}}
    service, chat = make_service(db, [
        {"content": None, "tool_calls": [idless]},
        "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=CFG, api_endpoint="groq", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    second_payload = chat.calls[1]["messages_payload"]
    assistant = [m for m in second_payload if m["role"] == "assistant"][0]
    tool_msg = [m for m in second_payload if m.get("role") == "tool"][0]
    assert assistant["tool_calls"][0]["id"] == "call_0"
    assert tool_msg["tool_call_id"] == "call_0"
    assert not any(m.get("role") == "user" and
                   str(m.get("content", "")).startswith("Tool result for")
                   for m in second_payload[1:])


def test_load_tools_same_batch_name_and_id_aliases_load_once(db):
    """PR #655 review (Gemini): one load_tools batch naming the SAME tool
    twice — bare name plus catalog id — must disclose it exactly once, so
    the loop's active list and this gate's disclosed set stay in lockstep
    (no duplicate schema, no phantom room-slot consumption)."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(FakeBigProvider())  # catalog > threshold: forces find/load
    config = AgentConfig(
        model="m", system_prompt="s",
        allowed_tools=("calculator", "get_current_datetime"),
        budget=RunBudget(max_active_tools=8, max_steps=20))
    chat = ScriptedChat([
        fence(LOAD_TOOLS_NAME, {"ids": ["calculator", "builtin:calculator"]}),
        fence("calculator", {"expression": "2+2"}),
        "4.",
    ])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=config, api_endpoint="llama_cpp")

    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    run = db.get_run(run_id)
    load_result = [s for s in run["steps"] if s["kind"] == "tool_result"][0]
    # Exactly one mention: "loaded: calculator" — not "calculator, calculator".
    assert load_result["result"] == "loaded: calculator"


# --- task-245: memoize the per-run fence-protocol render so an unchanged
# active tool set is rendered once, not once per model turn. ---

def test_protocol_render_memoized_across_unchanged_turns(db, monkeypatch):
    """AC #1: three fence turns with an unchanged active set must render the
    protocol exactly once; the payload text stays byte-identical per turn."""
    import tldw_chatbook.Agents.agent_service as svc
    real_render = svc.render_tool_protocol
    calls = []

    def counting_render(schemas):
        calls.append(tuple(s.name for s in schemas))
        return real_render(schemas)

    monkeypatch.setattr(svc, "render_tool_protocol", counting_render)
    # script: two calculator fence rounds + final answer = 3 model turns,
    # active set never changes (direct-disclose catalog, no load_tools).
    service, chat = make_service(db, [
        fence("calculator", {"expression": "1+1"}),
        fence("calculator", {"expression": "2+2"}),
        "done"])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "go"}],
        config=CFG, api_endpoint="llama_cpp", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE
    assert len(calls) == 1                       # rendered once, reused twice
    first_system = chat.calls[0]["messages_payload"][0]["content"]
    for later in chat.calls[1:]:
        assert later["messages_payload"][0]["content"] == first_system  # byte-stable


def test_protocol_rerenders_when_load_tools_admits_new_schema(db, monkeypatch):
    """AC #2: the cache invalidates the moment load_tools grows the active
    set — the very next turn's protocol includes the new tool."""
    import tldw_chatbook.Agents.agent_service as svc
    real_render = svc.render_tool_protocol
    calls = []

    def counting_render(schemas):
        calls.append(tuple(s.name for s in schemas))
        return real_render(schemas)

    monkeypatch.setattr(svc, "render_tool_protocol", counting_render)
    # Mirror the file's existing find/load test setup (FakeBigProvider forces
    # the load path). Script: load_tools fence -> calculator fence -> "done".
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    registry.register_provider(FakeBigProvider())  # catalog > threshold: forces find/load
    config = AgentConfig(
        model="m", system_prompt="s",
        allowed_tools=("calculator", "get_current_datetime"),
        budget=RunBudget(max_active_tools=8, max_steps=20))
    chat = ScriptedChat([
        fence(LOAD_TOOLS_NAME, {"ids": ["calculator"]}),
        fence("calculator", {"expression": "2+2"}),
        "done",
    ])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=config, api_endpoint="llama_cpp")

    assert outcome.status == RUN_DONE and outcome.final_text == "done"
    # Assert: counting_render was called exactly twice; the second recorded
    # name-tuple includes the newly loaded tool; the post-load turn's system
    # content contains the new tool's name while the pre-load turn's does not.
    assert len(calls) == 2
    assert "calculator" in calls[1]
    pre_load_system = chat.calls[0]["messages_payload"][0]["content"]
    post_load_system = chat.calls[1]["messages_payload"][0]["content"]
    assert "calculator" not in pre_load_system
    assert "calculator" in post_load_system


def test_native_endpoint_anthropic_sends_tools_and_suppresses_fence(db):
    """task-263: anthropic is native-capable — the service passes tools= and
    suppresses the fence protocol exactly as for the OpenAI-compatible set
    (the handler converts shapes internally; live-gated 2026-07-17)."""
    service, chat = make_service(db, [
        {"content": None,
         "tool_calls": [native_call("calculator", {"expression": "2+2"},
                                    "toolu_1")]},
        "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=CFG, api_endpoint="anthropic", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    first = chat.calls[0]
    assert "tools" in first
    assert "tool_call" not in first["messages_payload"][0]["content"]
    tool_msg = [m for m in chat.calls[1]["messages_payload"]
                if m.get("role") == "tool"][0]
    assert tool_msg["tool_call_id"] == "toolu_1"


def test_native_endpoint_google_sends_tools_and_suppresses_fence(db):
    """task-266: google is native-capable — tools= passed, fence suppressed
    (the handler converts shapes internally; live-gated 2026-07-17)."""
    service, chat = make_service(db, [
        {"content": None,
         "tool_calls": [native_call("calculator", {"expression": "2+2"},
                                    "call_g1")]},
        "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=CFG, api_endpoint="google", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    assert "tools" in chat.calls[0]
    assert "tool_call" not in chat.calls[0]["messages_payload"][0]["content"]


def test_native_endpoint_cohere_sends_tools_and_suppresses_fence(db):
    """task-267: cohere is a real NATIVE_TOOLS_PROVIDERS member (flipped
    after the 2026-07-17 live gate, Docs/superpowers/qa/cohere-native-
    2026-07/) — the service sends native tools and suppresses the fence
    protocol, mirroring the anthropic/google siblings above."""
    service, chat = make_service(db, [
        {"content": None,
         "tool_calls": [native_call("calculator", {"expression": "2+2"},
                                    "call_c1")]},
        "4."])
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=CFG, api_endpoint="cohere", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
    first = chat.calls[0]
    assert "tools" in first
    assert "tool_call" not in first["messages_payload"][0]["content"]
    tool_msg = [m for m in chat.calls[1]["messages_payload"]
                if m.get("role") == "tool"][0]
    assert tool_msg["tool_call_id"] == "call_c1"


def test_native_endpoint_with_no_schemas_omits_tools_kwarg(db):
    """Review minor m3 (task-243 final review): a native-capable endpoint
    whose run has NO disclosable schemas (empty allow-list, no sub-agents)
    must call the provider with no tools= kwarg at all — an empty tools
    list is rejected by several providers."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    config = AgentConfig(
        model="m", system_prompt="s", allowed_tools=(),
        budget=RunBudget(max_subagents=0))
    chat = ScriptedChat(["Just an answer."])
    service = AgentService(db=db, registry=registry, chat_call=chat)
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "hi"}],
        config=config, api_endpoint="groq", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE
    assert "tools" not in chat.calls[0]
