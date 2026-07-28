# Tests/Agents/test_run_log_eviction.py
"""TASK-1272 (Phase 3): evict older run-log-backed rounds from the SEND payload.

Two layers:

- Unit tests against `run_log_eviction.bound_history_for_send` and its
  round-boundary predicate directly, with explicit `window=`/`count_fn=`
  overrides for deterministic budgets (same style as
  `Tests/Chat/test_console_history_budget.py`).
- Integration tests through `AgentService.run_turn`, proving the flag/
  log-active gate and the end-to-end payload actually wire up.
"""

from __future__ import annotations

import json

import pytest

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.agent_models import (
    FENCE_TOOL_RESULT_PREFIX,
    SEARCH_RUN_LOG_TOOL_NAME,
    AgentConfig,
    RunBudget,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat import console_history_budget as budget_module
from tldw_chatbook.Chat.console_history_budget import bound_messages_to_window
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Agents.run_log_eviction import (
    RUN_LOG_EVICT_ENABLED_KEY,
    _make_round_boundary,
    bound_history_for_send,
)

from Tests.Agents.test_agent_service import ScriptedChat, native_call


# --------------------------------------------------------------------------
# Shared fixtures / helpers
# --------------------------------------------------------------------------


def _msg(role, content):
    return {"role": role, "content": content}


def _wordcount(messages, model):  # noqa: ARG001 -- matches bound_messages_to_window's count_fn shape
    total = 0
    for m in messages:
        content = m.get("content", "")
        total += len(str(content).split())
    return total


def fence_call(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _fence_result_row(tool, marker, padding_words=10):
    padding = " ".join(f"w{i}" for i in range(padding_words))
    return _msg("user", f"{FENCE_TOOL_RESULT_PREFIX}{tool}: {marker} {padding}")


def _fence_pairs_intact(messages: list[dict]) -> bool:
    """Every fence tool-result row must be immediately preceded by an
    assistant row in the SURVIVED sequence -- otherwise it is an orphaned
    reply with nothing explaining it (task-1272 requirement #3)."""
    for i, m in enumerate(messages):
        content = m.get("content")
        if (
            m.get("role") == "user"
            and isinstance(content, str)
            and content.startswith(FENCE_TOOL_RESULT_PREFIX)
        ):
            if i == 0 or messages[i - 1].get("role") != "assistant":
                return False
    return True


def _native_ids_paired(messages: list[dict]) -> bool:
    """The set of tool_call ids echoed by assistant messages must exactly
    equal the set of tool_call_ids answered by role="tool" replies in the
    SURVIVED sequence -- an orphan either way is a request strict
    providers reject (task-1272 requirement #3)."""
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id") if isinstance(tc, dict) else None
                if cid:
                    call_ids.add(cid)
        elif m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if tcid:
                result_ids.add(tcid)
    return call_ids == result_ids


def _build_fence_rounds(n: int):
    """[system, user(task), (assistant-call, user-result) x n] as a fence
    protocol payload -- exactly what `messages` looks like mid-run, just
    before the (n+1)-th `call_model` invocation."""
    payload = [_msg("system", "sys"), _msg("user", "start the task")]
    for i in range(1, n + 1):
        payload.append(_msg("assistant", fence_call("echo", {"i": i})))
        payload.append(_fence_result_row("echo", f"MARK{i}"))
    return payload


def _build_native_rounds(n: int):
    """[system, user(task), (assistant tool_calls, tool result) x n] as a
    native protocol payload."""
    payload = [_msg("system", "sys"), _msg("user", "start the task")]
    for i in range(1, n + 1):
        call_id = f"call_{i}"
        payload.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{}"},
                    }
                ],
            }
        )
        padding = " ".join(f"w{j}" for j in range(10))
        payload.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"MARK{i} {padding}",
            }
        )
    return payload


# --------------------------------------------------------------------------
# The required experiment: naive (Console) grouping vs the round-boundary fix
# --------------------------------------------------------------------------


def test_experiment_naive_grouping_orphans_a_fence_tool_result():
    """Force the OLD, un-fixed grouping (no `is_turn_boundary`, i.e. exactly
    what `_group_turns`/`bound_messages_to_window` did before this task) on
    a fence-protocol payload and show it drops a whole group that leaves a
    tool-result row with no assistant call in front of it. This is the
    concrete proof that requirement #3's fix is necessary, not cosmetic.
    """
    payload = _build_fence_rounds(4)
    naive = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=50, count_fn=_wordcount,  # tiny budget: several groups must drop
    )
    assert naive.dropped_turns > 0, "test is only meaningful if something dropped"
    assert not _fence_pairs_intact(naive.messages), (
        "naive role=='user' grouping was expected to orphan a fence "
        "tool-result row -- if this now passes, the primitive's DEFAULT "
        "boundary rule silently changed and this experiment needs revisiting"
    )


def test_fixed_round_boundary_never_orphans_a_fence_tool_result():
    """The SAME payload and budget as above, through the actual fix
    (`_make_round_boundary(native=False)`): no orphan, at any drop size."""
    payload = _build_fence_rounds(4)
    for window in (30, 50, 80, 120, 400):
        bound = bound_messages_to_window(
            payload, model="m", provider="p", response_reservation=0,
            window=window, count_fn=_wordcount,
            is_turn_boundary=_make_round_boundary(native=False),
        )
        assert _fence_pairs_intact(bound.messages), (
            f"orphaned fence tool-result at window={window}"
        )


def test_experiment_naive_grouping_never_trims_a_native_run_at_all():
    """A companion experiment for native: naive grouping doesn't orphan a
    native pair (tool results are role="tool", never "user"), but it also
    NEVER drops anything mid-run, because the only `role=="user"` row in
    a single dispatch is the original task -- so `current_turn` swallows
    the run's entire growth. This is *why* Phase 3 needs a round-aware
    boundary, not just a fence-protocol patch: reusing the primitive
    completely unmodified would leave native runs exactly as unbounded as
    before this task, defeating the whole point for long single-run
    growth.
    """
    payload = _build_native_rounds(6)
    naive = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=30, count_fn=_wordcount,  # budget far too small for 6 rounds
    )
    assert naive.dropped_turns == 0, (
        "expected naive grouping to find nothing droppable for a native "
        "run (single real user boundary at the very start) -- if this now "
        "drops something, re-check whether this experiment is still valid"
    )


def test_fixed_round_boundary_trims_a_native_run_without_orphaning():
    payload = _build_native_rounds(6)
    bound = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=30, count_fn=_wordcount,
        is_turn_boundary=_make_round_boundary(native=True),
    )
    assert bound.dropped_turns > 0, "expected the tiny window to force some drops"
    assert _native_ids_paired(bound.messages)
    # The most recent round (round 6) is the current turn -- always kept.
    assert any(
        m.get("role") == "tool" and "MARK6" in m.get("content", "")
        for m in bound.messages
    )


# --------------------------------------------------------------------------
# Live-verified 2026-07-28: without `pin_first_user`, the round-boundary fix
# above still lets the task instruction -- the payload's only REAL
# role="user" row -- fall into `kept_turns[0]`, the OLDEST group, and get
# dropped first. Reproduced live against llama.cpp gemma-4-26B: the agent
# lost track of its own task and started narrating about its log instead of
# finishing. These tests pin that down at both the primitive and the
# production seam, for both protocols.
# --------------------------------------------------------------------------

_TASK_TEXT = "start the task"


def test_pin_first_user_survives_fence_even_when_everything_else_drops():
    payload = _build_fence_rounds(6)
    starved = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=10, count_fn=_wordcount,  # far too small to keep anything else
        is_turn_boundary=_make_round_boundary(native=False),
        pin_first_user=True,
    )
    assert starved.dropped_turns > 0, "test only meaningful if something dropped"
    assert any(
        m.get("role") == "user" and m.get("content") == _TASK_TEXT
        for m in starved.messages
    ), "the task instruction must survive no matter how tight the window"


def test_without_the_pin_the_same_budget_drops_the_task_instruction():
    """The behavioural bar the coordinator asked for, expressed directly:
    the exact same payload and budget, with `pin_first_user` OFF, loses the
    task instruction -- proving the pin is load-bearing, not redundant with
    the round-boundary fix alone."""
    payload = _build_fence_rounds(6)
    unpinned = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=10, count_fn=_wordcount,
        is_turn_boundary=_make_round_boundary(native=False),
        pin_first_user=False,
    )
    assert not any(
        m.get("role") == "user" and m.get("content") == _TASK_TEXT
        for m in unpinned.messages
    ), (
        "expected the task instruction to be dropped WITHOUT the pin -- if "
        "this now passes, the round-boundary fix alone became sufficient "
        "and this regression test needs revisiting"
    )


def test_pin_first_user_survives_native_even_when_everything_else_drops():
    payload = _build_native_rounds(6)
    starved = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=10, count_fn=_wordcount,
        is_turn_boundary=_make_round_boundary(native=True),
        pin_first_user=True,
    )
    assert starved.dropped_turns > 0
    assert any(
        m.get("role") == "user" and m.get("content") == _TASK_TEXT
        for m in starved.messages
    )


def test_without_the_pin_the_same_budget_drops_the_task_instruction_native():
    payload = _build_native_rounds(6)
    unpinned = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=10, count_fn=_wordcount,
        is_turn_boundary=_make_round_boundary(native=True),
        pin_first_user=False,
    )
    assert not any(
        m.get("role") == "user" and m.get("content") == _TASK_TEXT
        for m in unpinned.messages
    )


def test_pin_does_not_change_console_default_behaviour():
    """`pin_first_user` defaults to False, so every existing Console call
    site (which never passes it) is completely unaffected."""
    payload = _build_fence_rounds(6)
    default = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=10, count_fn=_wordcount,
        is_turn_boundary=_make_round_boundary(native=False),
    )
    explicit_false = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=10, count_fn=_wordcount,
        is_turn_boundary=_make_round_boundary(native=False),
        pin_first_user=False,
    )
    assert default == explicit_false


# --------------------------------------------------------------------------
# bound_history_for_send: the actual production entry point
# --------------------------------------------------------------------------


def test_disabled_returns_the_exact_same_object():
    payload = _build_fence_rounds(20)  # huge -- would definitely need trimming
    result = bound_history_for_send(
        payload, model="m", provider="p", native=False, enabled=False,
    )
    assert result is payload, "enabled=False must be a true no-op, not a copy"


def test_enabled_but_fits_under_budget_leaves_payload_unchanged():
    payload = _build_fence_rounds(1)
    result = bound_history_for_send(
        payload, model="m", provider="p", native=False, enabled=True,
        window=100_000, response_reservation=0, count_fn=_wordcount,
    )
    assert result == payload
    assert not any(
        "Context note" in str(m.get("content", "")) for m in result
    )


def test_note_appears_only_when_something_was_actually_dropped():
    small = _build_fence_rounds(1)
    fits = bound_history_for_send(
        small, model="m", provider="p", native=False, enabled=True,
        window=100_000, response_reservation=0, count_fn=_wordcount,
    )
    assert not any("Context note" in str(m.get("content", "")) for m in fits)

    big = _build_fence_rounds(10)
    trimmed = bound_history_for_send(
        big, model="m", provider="p", native=False, enabled=True,
        window=30, response_reservation=0, count_fn=_wordcount,
    )
    assert any("Context note" in str(m.get("content", "")) for m in trimmed)
    assert any(SEARCH_RUN_LOG_TOOL_NAME in str(m.get("content", "")) for m in trimmed)


def test_system_prefix_and_current_round_are_always_preserved():
    payload = [
        _msg("system", "sys one"),
        _msg("system", "sys two"),
        _msg("user", "start the task with a lot of words padding it out here"),
    ]
    for i in range(1, 9):
        payload.append(_msg("assistant", fence_call("echo", {"i": i})))
        payload.append(_fence_result_row("echo", f"MARK{i}", padding_words=15))
    result = bound_history_for_send(
        payload, model="m", provider="p", native=False, enabled=True,
        window=20, response_reservation=0, count_fn=_wordcount,
    )
    assert result[0] == payload[0] and result[1] == payload[1]
    assert any(
        m.get("role") == "user" and "MARK8" in str(m.get("content", ""))
        for m in result
    ), "the most recent round must survive even under a starved budget"


def test_eviction_never_raises_and_degrades_to_full_history(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "tldw_chatbook.Agents.run_log_eviction.bound_messages_to_window", _boom
    )
    payload = _build_fence_rounds(5)
    result = bound_history_for_send(
        payload, model="m", provider="p", native=False, enabled=True,
    )
    assert result is payload


# --------------------------------------------------------------------------
# Integration: through AgentService.run_turn (real wiring, real gate)
# --------------------------------------------------------------------------


class _EchoProvider:
    """Fake ToolProvider: `echo` returns a fixed, caller-controlled string.

    Real tool content (e.g. `calculator`'s actual arithmetic) isn't
    controllable enough to deterministically overflow/underflow a chosen
    token window; this fixes the content so the round markers used to
    assert what did/didn't survive eviction are exact.
    """

    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self._calls = 0

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id="test:echo", name="echo", one_line_description="echo", source="test"
            )
        ]

    def load_schema(self, tool_id):
        return ToolSchema(id="test:echo", name="echo", description="echo", parameters={"type": "object"})

    def invoke(self, tool_id, args):
        content = self._contents[self._calls % len(self._contents)]
        self._calls += 1
        return ToolResult(ok=True, content=content)


def _round_marker(i: int) -> str:
    return f"MARK{i}_" + ("Z" * 800)


def _make_registry(n_rounds: int) -> ToolCatalogRegistry:
    registry = ToolCatalogRegistry()
    registry.register_provider(_EchoProvider([_round_marker(i) for i in range(1, n_rounds + 1)]))
    return registry


def _fence_replies(n_rounds: int) -> list:
    replies = [fence_call("echo", {"i": i}) for i in range(1, n_rounds + 1)]
    replies.append("done.")
    return replies


def _native_replies(n_rounds: int) -> list:
    replies = [
        {"content": None, "tool_calls": [native_call("echo", {"i": i}, f"call_{i}")]}
        for i in range(1, n_rounds + 1)
    ]
    replies.append("done.")
    return replies


BIG_BUDGET = RunBudget(max_steps=200, max_model_turns=200)


def _run_config(**kw):
    return AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("echo",),
        budget=BIG_BUDGET,
        **kw,
    )


#: Ties the tests to the production key name (`run_log_eviction.
#: RUN_LOG_EVICT_ENABLED_KEY`) rather than a re-typed copy of the env-var
#: string, mirroring `run_log._env_override`'s `TLDW_AGENTS_<KEY upper>`
#: convention.
_EVICT_ENV_VAR = f"TLDW_AGENTS_{RUN_LOG_EVICT_ENABLED_KEY.upper()}"


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


def test_flag_off_sends_full_history_regardless_of_window(db, tmp_path, monkeypatch):
    """Requirement #5: off by default. Even with a tiny window AND an
    active run log, no eviction happens unless the flag is explicitly on.
    """
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    monkeypatch.setattr(budget_module, "get_model_token_limit", lambda *a, **k: 50)
    n = 6
    chat = ScriptedChat(_fence_replies(n))
    service = AgentService(db=db, registry=_make_registry(n), chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "start"}],
        config=_run_config(native_tools=False),
        api_endpoint="llama_cpp",
    )
    last_payload = chat.calls[-1]["messages_payload"]
    assert not any(
        "Context note" in str(m.get("content", "")) for m in last_payload
    )
    assert any("MARK1_" in str(m.get("content", "")) for m in last_payload), (
        "the very first round's content must still be present -- nothing "
        "evicted with the flag off"
    )


def test_flag_on_and_log_unavailable_still_sends_full_history(db, tmp_path, monkeypatch):
    """Requirement #1, the hard gate: the flag alone is not enough. With
    no run log available (`resolve_log_root` -> None -> `log_active`
    False), eviction must never fire even under a starving window.
    """
    monkeypatch.setenv(_EVICT_ENV_VAR, "true")
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    monkeypatch.setattr(budget_module, "get_model_token_limit", lambda *a, **k: 50)
    n = 6
    chat = ScriptedChat(_fence_replies(n))
    service = AgentService(db=db, registry=_make_registry(n), chat_call=chat)
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "start"}],
        config=_run_config(native_tools=False),
        api_endpoint="llama_cpp",
    )
    last_payload = chat.calls[-1]["messages_payload"]
    assert any("MARK1_" in str(m.get("content", "")) for m in last_payload), (
        "logging unavailable must suppress eviction even with the flag on"
    )
    assert not any("Context note" in str(m.get("content", "")) for m in last_payload)


def test_flag_on_and_log_active_fence_protocol_drops_old_rounds_intact(
    db, tmp_path, monkeypatch
):
    """The headline scenario: fence protocol (the local-model path), flag
    on, log active, small window -- old rounds drop, the note appears, and
    no assistant-call/tool-result pair is ever split.
    """
    monkeypatch.setenv(_EVICT_ENV_VAR, "true")
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    monkeypatch.setattr(budget_module, "get_model_token_limit", lambda *a, **k: 3000)
    n = 10
    chat = ScriptedChat(_fence_replies(n))
    service = AgentService(db=db, registry=_make_registry(n), chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "start"}],
        config=_run_config(native_tools=False),
        api_endpoint="llama_cpp",
    )
    assert outcome.final_text == "done."
    last_payload = chat.calls[-1]["messages_payload"]
    assert not any("MARK1_" in str(m.get("content", "")) for m in last_payload), (
        "the earliest round should have been evicted under this tiny window"
    )
    assert any(f"MARK{n}_" in str(m.get("content", "")) for m in last_payload), (
        "the most recent round must always survive"
    )
    assert any("Context note" in str(m.get("content", "")) for m in last_payload)
    assert any(
        SEARCH_RUN_LOG_TOOL_NAME in str(m.get("content", "")) for m in last_payload
    )
    assert _fence_pairs_intact(last_payload)


def test_flag_on_and_log_active_native_protocol_drops_old_rounds_intact(
    db, tmp_path, monkeypatch
):
    """The same scenario on the native protocol: id pairing must survive."""
    monkeypatch.setenv(_EVICT_ENV_VAR, "true")
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    monkeypatch.setattr(budget_module, "get_model_token_limit", lambda *a, **k: 3000)
    n = 10
    chat = ScriptedChat(_native_replies(n))
    service = AgentService(db=db, registry=_make_registry(n), chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "start"}],
        config=_run_config(native_tools=True),
        api_endpoint="groq",
    )
    assert outcome.final_text == "done."
    last_payload = chat.calls[-1]["messages_payload"]
    assert not any("MARK1_" in str(m.get("content", "")) for m in last_payload)
    assert any(f"MARK{n}_" in str(m.get("content", "")) for m in last_payload)
    assert any("Context note" in str(m.get("content", "")) for m in last_payload)
    assert _native_ids_paired(last_payload)


def test_flag_on_fence_protocol_task_instruction_survives_a_starving_window(
    db, tmp_path, monkeypatch
):
    """Live-verified 2026-07-28 regression (runs C/D against llama.cpp
    gemma-4-26B, fence protocol): under a window so tight that eviction
    drops everything it can, the task instruction must still be present in
    EVERY payload actually sent -- not just the final one, since the live
    defect derailed the agent mid-run, not only at the end.
    """
    monkeypatch.setenv(_EVICT_ENV_VAR, "true")
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    monkeypatch.setattr(budget_module, "get_model_token_limit", lambda *a, **k: 50)
    n = 8
    chat = ScriptedChat(_fence_replies(n))
    service = AgentService(db=db, registry=_make_registry(n), chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": _TASK_TEXT}],
        config=_run_config(native_tools=False),
        api_endpoint="llama_cpp",
    )
    assert outcome.final_text == "done."
    assert len(chat.calls) > 1, "test needs multiple rounds to be meaningful"
    for call in chat.calls:
        payload = call["messages_payload"]
        assert any(
            m.get("role") == "user" and m.get("content") == _TASK_TEXT
            for m in payload
        ), "task instruction missing from a payload actually sent"


def test_flag_on_native_protocol_task_instruction_survives_a_starving_window(
    db, tmp_path, monkeypatch
):
    """The same regression, native protocol."""
    monkeypatch.setenv(_EVICT_ENV_VAR, "true")
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    monkeypatch.setattr(budget_module, "get_model_token_limit", lambda *a, **k: 50)
    n = 8
    chat = ScriptedChat(_native_replies(n))
    service = AgentService(db=db, registry=_make_registry(n), chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": _TASK_TEXT}],
        config=_run_config(native_tools=True),
        api_endpoint="groq",
    )
    assert outcome.final_text == "done."
    assert len(chat.calls) > 1, "test needs multiple rounds to be meaningful"
    for call in chat.calls:
        payload = call["messages_payload"]
        assert any(
            m.get("role") == "user" and m.get("content") == _TASK_TEXT
            for m in payload
        ), "task instruction missing from a payload actually sent"
