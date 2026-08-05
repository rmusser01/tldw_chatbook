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
    DEFAULT_MIN_RECENT_ROUNDS,
    RUN_LOG_EVICT_ENABLED_KEY,
    RUN_LOG_EVICT_MIN_RECENT_ROUNDS_KEY,
    _make_round_boundary,
    _task_row_index,
    bound_history_for_send,
    coerce_min_recent_rounds,
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
# Live-verified follow-up (2026-07-28, same day as the pin fix): even with
# the task instruction pinned, a tight enough window can still collapse
# "keep whatever fits" down to ONLY the current turn. The agent then cannot
# see the handful of rounds it just completed and repeats them -- live
# reproduction against llama.cpp gemma-4-26B showed byte-identical payloads
# across consecutive calls (a "fixed point": eviction removing exactly as
# many new rounds as are added), ending in the cycle detector firing and the
# run going `stuck`. `min_recent_turns` fixes this with a floor.
# --------------------------------------------------------------------------


def test_min_recent_turns_floor_keeps_the_last_N_rounds_under_a_starving_window():
    payload = _build_fence_rounds(8)
    bound = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=10, count_fn=_wordcount,  # would keep ONLY the current round without a floor
        is_turn_boundary=_make_round_boundary(native=False),
        pin_first_user=True,
        min_recent_turns=4,
    )
    # Verified by direct computation: kept_turns = [round1..round7] (7
    # entries; the task instruction is pinned separately, not one of
    # them), floor 4 permits dropping at most 7 - (4 - 1) = 4 of them, so
    # exactly rounds 5, 6, 7 survive from kept_turns plus round 8 as the
    # always-kept current turn -- 4 rounds total.
    assert bound.dropped_turns == 4
    for i in (5, 6, 7, 8):
        assert any(
            f"MARK{i}" in str(m.get("content", "")) for m in bound.messages
        ), f"round {i} should be within the floor"
    for i in (1, 2, 3, 4):
        assert not any(
            f"MARK{i} " in str(m.get("content", "")) for m in bound.messages
        ), f"round {i} is older than the floor and should have been dropped"


def test_without_a_floor_the_same_budget_keeps_only_the_current_round():
    """The behavioural bar: the exact same payload and budget, with
    `min_recent_turns` at its default (0), keeps ONLY round 8 -- proving
    the floor is load-bearing, not redundant with the pin or round-boundary
    fixes alone."""
    payload = _build_fence_rounds(8)
    unfloored = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=10, count_fn=_wordcount,
        is_turn_boundary=_make_round_boundary(native=False),
        pin_first_user=True,
    )
    for i in (5, 6, 7):
        assert not any(
            f"MARK{i} " in str(m.get("content", "")) for m in unfloored.messages
        ), (
            f"expected round {i} to be dropped WITHOUT a floor -- if this "
            f"now passes, the pin/round-boundary fixes alone became "
            f"sufficient and this regression test needs revisiting"
        )
    assert any(
        "MARK8" in str(m.get("content", "")) for m in unfloored.messages
    ), "the current round must still survive regardless"


def test_floor_degenerate_case_sends_over_budget_rather_than_shrinking_below_it():
    """If the pinned prefix plus the floor of recent rounds alone already
    exceed the window, the floor must never be reduced to make room --
    the payload is sent over budget instead, exactly as documented in
    `bound_messages_to_window`'s `min_recent_turns` docstring."""
    payload = _build_fence_rounds(8)
    # window=1 is smaller than even the system row alone can fit under any
    # positive margin -- the ultimate "nothing fits" case.
    bound = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=1, count_fn=_wordcount,
        is_turn_boundary=_make_round_boundary(native=False),
        pin_first_user=True,
        min_recent_turns=4,
    )
    # The floor (rounds 5-8) is still fully present despite being
    # impossible to fit in a window this tiny -- an over-budget send,
    # never a below-floor one.
    for i in (5, 6, 7, 8):
        assert any(f"MARK{i}" in str(m.get("content", "")) for m in bound.messages)
    assert bound.dropped_turns == 4


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
_MIN_RECENT_ROUNDS_ENV_VAR = (
    f"TLDW_AGENTS_{RUN_LOG_EVICT_MIN_RECENT_ROUNDS_KEY.upper()}"
)


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


# --------------------------------------------------------------------------
# Live-verified follow-up (2026-07-28): the minimum-recent-rounds floor at
# the actual production seam.
#
# Round content uses DISTINCT markers (`_fence_replies`/`_make_registry`,
# each round calling `echo` with different args) rather than identical
# repeated calls: `run_agent_loop`'s OWN cycle detector
# (`LOOP_DETECTION_N = 3`) operates on the loop's untouched `messages` list,
# independent of eviction, and fires after 3 IDENTICAL consecutive calls --
# which a scripted `chat_call` that always returns the same canned call
# trips well before enough rounds accumulate to observe the fixed point.
# A scripted chat_call also cannot reproduce a real model's confusion
# itself (it returns whatever is next in its list regardless of what it
# received) -- only the PAYLOAD SHAPE that live-verified as the cause of
# that confusion: without a floor, the number of DISTINCT rounds visible
# in any one payload is capped at 1 no matter how many rounds have
# happened, which is the structural "fixed point" the coordinator's byte-
# identical live payloads are a real model's symptom of.
# --------------------------------------------------------------------------


def _visible_round_marks(payload: list[dict], n: int) -> set[int]:
    return {
        i
        for i in range(1, n + 1)
        if any(f"MARK{i}_" in str(m.get("content", "")) for m in payload)
    }


def test_without_a_floor_the_production_payload_never_shows_more_than_one_round(
    db, tmp_path, monkeypatch
):
    """The floor forced to 0 (env override): once eviction is actually
    trimming, every payload shows AT MOST one round's worth of tool
    activity -- the structural signature behind the live-verified
    byte-identical-payload symptom."""
    monkeypatch.setenv(_EVICT_ENV_VAR, "true")
    monkeypatch.setenv(_MIN_RECENT_ROUNDS_ENV_VAR, "0")
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
    assert len(chat.calls) >= 4, "test needs enough rounds to be meaningful"
    for call in chat.calls[2:]:
        marks = _visible_round_marks(call["messages_payload"], n)
        assert len(marks) <= 1, (
            f"expected at most 1 round visible without a floor, saw {marks}"
        )


def test_with_the_default_floor_multiple_rounds_stay_visible_together(
    db, tmp_path, monkeypatch
):
    """The behavioural bar: the identical setup, WITHOUT overriding the
    floor (so the default of 4 applies), must show MORE than one round
    simultaneously once enough rounds have happened -- the floor prevents
    the single-round collapse the test above reproduces."""
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
    assert len(chat.calls) >= 4
    last_marks = _visible_round_marks(chat.calls[-1]["messages_payload"], n)
    assert len(last_marks) > 1, (
        "expected the default floor to keep more than one round visible -- "
        "if this now fails, the floor stopped taking effect at the "
        "production seam"
    )


def test_default_floor_keeps_distinct_recent_rounds_visible_together(
    db, tmp_path, monkeypatch
):
    """Positive check with DISTINCT round markers (not identical calls):
    under a starving window, the last `DEFAULT_MIN_RECENT_ROUNDS` rounds
    must all be simultaneously visible in the final payload, not just the
    single most recent one."""
    monkeypatch.setenv(_EVICT_ENV_VAR, "true")
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    monkeypatch.setattr(budget_module, "get_model_token_limit", lambda *a, **k: 50)
    n = 10
    chat = ScriptedChat(_fence_replies(n))
    service = AgentService(db=db, registry=_make_registry(n), chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": _TASK_TEXT}],
        config=_run_config(native_tools=False),
        api_endpoint="llama_cpp",
    )
    last_payload = chat.calls[-1]["messages_payload"]
    for i in range(n - DEFAULT_MIN_RECENT_ROUNDS + 1, n + 1):
        assert any(
            f"MARK{i}_" in str(m.get("content", "")) for m in last_payload
        ), f"round {i} should be within the default floor of {DEFAULT_MIN_RECENT_ROUNDS}"


def test_min_recent_rounds_config_key_is_honored(db, tmp_path, monkeypatch):
    """A caller can raise or lower the floor via `[agents]
    run_log_evict_min_recent_rounds` (env-var tier here, mirroring
    `run_log._setting`'s resolution order); confirms the wiring from config
    through `agent_service` to `bound_history_for_send` end to end, not
    just that the default happens to work."""
    monkeypatch.setenv(_EVICT_ENV_VAR, "true")
    monkeypatch.setenv(_MIN_RECENT_ROUNDS_ENV_VAR, "2")
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
    last_payload = chat.calls[-1]["messages_payload"]
    # Floor 2: only the last 2 rounds are guaranteed -- round n-2 (one
    # older than the floor) should be gone, unlike the default-floor test
    # above where a matching offset stays.
    assert not any(
        f"MARK{n - 2}_" in str(m.get("content", "")) for m in last_payload
    ), "floor=2 should not have kept a round two positions back"
    assert any(f"MARK{n}_" in str(m.get("content", "")) for m in last_payload)


# --------------------------------------------------------------------------
# task-1272 Phase 3 review finding A: the task instruction is not always the
# FIRST role=="user" row -- `AgentService.run_turn`'s own docstring says the
# primary-agent seed is "typically the conversation transcript plus any
# staged/RAG context", so a real payload can carry an older, unrelated
# Console turn ahead of the actual task. The original `pin_first_user=True`
# approach pinned whatever came first, which is provably wrong there. These
# tests build that exact shape and confirm the fix (`_task_row_index`, the
# LAST real user row, plus `bound_messages_to_window`'s `pin_row_index`)
# protects the actual task regardless of what precedes it.
# --------------------------------------------------------------------------


def _payload_with_older_turn_before_the_task(n_rounds: int) -> list[dict]:
    """[system, OLDER unrelated user/assistant turn, user(task), (fence
    rounds) x n] -- the production shape finding A is about, not the
    single-real-user-row shape every fixture above this section uses."""
    payload = [
        _msg("system", "sys"),
        _msg("user", "what is 2+2? (an older, unrelated Console turn)"),
        _msg("assistant", "4"),
        _msg("user", _TASK_TEXT),
    ]
    for i in range(1, n_rounds + 1):
        payload.append(_msg("assistant", fence_call("echo", {"i": i})))
        payload.append(_fence_result_row("echo", f"MARK{i}"))
    return payload


def test_task_row_index_finds_the_last_real_user_row_not_the_first():
    """Direct unit coverage of the new lookup: index 3 (the task), not
    index 1 (the older, unrelated turn)."""
    payload = _payload_with_older_turn_before_the_task(1)
    boundary = _make_round_boundary(native=False)
    assert _task_row_index(payload, boundary) == 3


def test_task_row_index_returns_none_with_no_real_user_row():
    """No row anywhere qualifies (every `role=="user"` row is a fence
    continuation, or there simply is none) -- must return `None`, not a
    wrong guess."""
    payload = [
        _msg("system", "sys"),
        _msg("assistant", "orphaned assistant turn, nothing preceding it"),
        _fence_result_row("echo", "MARK1"),
    ]
    boundary = _make_round_boundary(native=False)
    assert _task_row_index(payload, boundary) is None


def test_pin_first_user_would_protect_the_wrong_row_when_history_precedes_the_task():
    """The regression this fix targets, reproduced directly against the
    surviving `pin_first_user` primitive (left unchanged for its own
    existing callers/tests): given a payload seeded the way
    `AgentService.run_turn` actually seeds a primary agent -- a real,
    older conversation turn BEFORE the task -- `pin_first_user=True`
    protects that older turn instead and still lets the real task
    instruction be evicted under a tight window."""
    payload = _payload_with_older_turn_before_the_task(8)
    old_behavior = bound_messages_to_window(
        payload, model="m", provider="p", response_reservation=0,
        window=10, count_fn=_wordcount,
        is_turn_boundary=_make_round_boundary(native=False),
        pin_first_user=True,
    )
    assert not any(
        m.get("role") == "user" and m.get("content") == _TASK_TEXT
        for m in old_behavior.messages
    ), (
        "test only meaningful if the OLD forward-scan pin drops the task "
        "-- if this now passes, the regression this test guards against "
        "no longer reproduces and should be revisited"
    )


def test_bound_history_for_send_protects_the_actual_task_not_the_first_user_row():
    """The FIX, through the production entry point (`bound_history_for_
    send`, which no longer uses `pin_first_user` at all -- see
    `_task_row_index`): the same shape as above, same starving window --
    the task instruction now survives regardless of the older turn ahead
    of it."""
    payload = _payload_with_older_turn_before_the_task(8)
    result = bound_history_for_send(
        payload, model="m", provider="p", native=False, enabled=True,
        window=10, response_reservation=0, count_fn=_wordcount,
    )
    assert any(
        m.get("role") == "user" and m.get("content") == _TASK_TEXT
        for m in result
    ), "the actual task instruction must survive, not just some earlier user row"


def test_no_identifiable_task_row_declines_eviction_entirely():
    """The coordinator's ruling on finding A: when no row in the payload
    unambiguously qualifies as the task, eviction must decline outright --
    never guess a pin -- even under a starving window that would otherwise
    force heavy trimming. `bound_history_for_send` must return the exact
    same object, not even an attempted (and necessarily unprotected) trim.
    """
    payload = [
        _msg("system", "sys"),
        _msg("assistant", "orphaned assistant turn, no task preceding it"),
    ]
    for i in range(1, 4):
        payload.append(_msg("assistant", fence_call("echo", {"i": i})))
        payload.append(_fence_result_row("echo", f"MARK{i}"))
    result = bound_history_for_send(
        payload, model="m", provider="p", native=False, enabled=True,
        window=5, response_reservation=0, count_fn=_wordcount,
    )
    assert result is payload


# --------------------------------------------------------------------------
# task-1272 Phase 3 review finding B: eviction measures the payload with the
# Console token counter, which did not account for native-protocol
# `tool_calls` structures -- an assistant turn that only issues tool calls
# typically has empty `content`, so the (sometimes substantial) call itself
# was invisible to the counter, silently undercounting a native payload.
# Fixed at the root: `console_history_budget.count_console_messages_tokens`
# (the DEFAULT counter every caller here gets when `count_fn` is not
# overridden) now folds `tool_calls` into what it counts.
# --------------------------------------------------------------------------


def test_count_console_messages_tokens_counts_native_tool_calls():
    """Direct unit coverage of the counter itself: a message whose entire
    payload lives in `tool_calls` (empty `content`) must count for
    meaningfully more than the same message with `tool_calls` stripped."""
    big_args = json.dumps({"data": list(range(500))})
    plain = [{"role": "assistant", "content": ""}]
    with_calls = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": big_args},
                }
            ],
        }
    ]
    plain_tokens = budget_module.count_console_messages_tokens(plain, "gpt-4")
    with_calls_tokens = budget_module.count_console_messages_tokens(with_calls, "gpt-4")
    assert with_calls_tokens > plain_tokens + 100, (
        "a substantial tool_calls structure must move the count by more "
        "than incidental JSON-key overhead"
    )


def test_native_payload_with_substantial_tool_calls_is_bounded_correctly():
    """Integration proof through `bound_history_for_send`'s ACTUAL default
    counter (no `count_fn` override): a native round whose `tool_calls`
    carries a large arguments blob (empty `content`) must be evicted once
    it no longer fits, exactly like a round with equally large plain
    `content` would be -- not silently kept because the counter only ever
    looked at `content`.
    """
    big_args = json.dumps({"data": list(range(2000))})
    heavy_round = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": big_args},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    light_rounds: list[dict] = []
    for i in range(2, 6):
        call_id = f"call_{i}"
        light_rounds += [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "echo", "arguments": json.dumps({"i": i})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": f"MARK{i}"},
        ]
    payload = [_msg("system", "sys"), _msg("user", _TASK_TEXT), *heavy_round, *light_rounds]
    without_heavy_round = [_msg("system", "sys"), _msg("user", _TASK_TEXT), *light_rounds]

    # Calibrate against the REAL default counter this function actually
    # uses in production (no count_fn override): a window that comfortably
    # fits everything EXCEPT the heavy round's tool_calls, but not the full
    # payload -- so eviction is required, and only lands on the heavy
    # round if its tool_calls are actually counted. `window` stays well
    # under the primitive's win/50 margin threshold (25_600), so the
    # margin contribution is the constant 512 floor on both sides of this
    # calibration.
    light_tokens = budget_module.count_console_messages_tokens(without_heavy_round, "gpt-4")
    full_tokens = budget_module.count_console_messages_tokens(payload, "gpt-4")
    assert full_tokens > light_tokens + 500, (
        "test setup needs the heavy round's tool_calls to dominate the "
        "count -- if this now fails, make the arguments blob bigger"
    )
    window = light_tokens + 700

    result = bound_history_for_send(
        payload, model="gpt-4", provider="openai", native=True, enabled=True,
        window=window, response_reservation=0, min_recent_rounds=1,
    )

    def _has_call(messages, call_id):
        return any(
            call.get("id") == call_id
            for m in messages
            for call in (m.get("tool_calls") or [])
        )

    assert not _has_call(result, "call_1"), (
        "the oversized tool_calls round must be evicted once its "
        "arguments are actually counted toward the budget"
    )
    for i in range(2, 6):
        assert _has_call(result, f"call_{i}"), f"round {i} (small) should still fit"
    assert any(
        m.get("role") == "user" and m.get("content") == _TASK_TEXT for m in result
    )


# --------------------------------------------------------------------------
# task-1272 Phase 3 review finding D: `coerce_min_recent_rounds` had no
# direct unit tests of its own (only indirect coverage via env-var-string
# integration tests above). Finding C's OverflowError fix lives here too --
# the `float("inf")` case is the one that previously escaped uncaught.
# --------------------------------------------------------------------------


def test_coerce_min_recent_rounds_valid_value_passes_through():
    assert coerce_min_recent_rounds(7) == 7


def test_coerce_min_recent_rounds_negative_falls_back_to_default():
    assert coerce_min_recent_rounds(-3) == DEFAULT_MIN_RECENT_ROUNDS


def test_coerce_min_recent_rounds_zero_is_a_valid_deliberate_choice():
    assert coerce_min_recent_rounds(0) == 0


def test_coerce_min_recent_rounds_non_numeric_falls_back_to_default():
    assert coerce_min_recent_rounds("not-a-number") == DEFAULT_MIN_RECENT_ROUNDS


def test_coerce_min_recent_rounds_none_falls_back_to_default():
    assert coerce_min_recent_rounds(None) == DEFAULT_MIN_RECENT_ROUNDS


def test_coerce_min_recent_rounds_infinity_falls_back_to_default():
    """Finding C: `int(float('inf'))` raises `OverflowError`, which the
    original `except (TypeError, ValueError)` did not catch -- a config
    value of infinity used to raise straight out of run setup instead of
    degrading to the default."""
    assert coerce_min_recent_rounds(float("inf")) == DEFAULT_MIN_RECENT_ROUNDS
    assert coerce_min_recent_rounds(float("-inf")) == DEFAULT_MIN_RECENT_ROUNDS


def test_coerce_min_recent_rounds_nan_falls_back_to_default():
    assert coerce_min_recent_rounds(float("nan")) == DEFAULT_MIN_RECENT_ROUNDS


def test_coerce_min_recent_rounds_enormous_value_passes_through():
    """No documented upper bound: an absurdly large but finite floor is
    accepted as-is, same as any other non-negative int."""
    huge = 10**30
    assert coerce_min_recent_rounds(huge) == huge


def test_coerce_min_recent_rounds_truncating_float_rounds_toward_zero():
    assert coerce_min_recent_rounds(4.9) == 4
