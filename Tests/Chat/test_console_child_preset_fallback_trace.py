"""A spawned child's preset fallback (ADR-200) under Capture On.

TASK-42556 made the refusal a visible skip before any call; TASK-42564
(ADR-337 amendment 2026-11-26) sends and traces the fallback as a new chain
of the child. The real controller, store, agent bridge, gateway and trace
factory run over real SQLite; only the HTTP socket is a double
(``_spawn_turn``). The parent spawns the ``worker`` preset, whose own model
is unavailable and whose ``fallback_models`` names FALLBACK_MODEL.
"""


import pytest

from Tests.Chat.test_console_spawn_subagent_trace_capture import (
    FALLBACK_MODEL,
    MODEL,
    _spawn_turn,
)
from tldw_chatbook.Agents.agent_models import AgentConfig, ModelTurn, RunBudget, ToolResult
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.fallback_chain import FallbackCandidate, FallbackRuntime
from tldw_chatbook.Chat.Chat_Deps import ChatModelUnavailableError
from tldw_chatbook.Chat.console_agent_preparation import PRESET_FALLBACK_NOT_CAPTURED
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.owned_http_client]

SKIPPED = "Provider fallback skipped"


def _child(facts):
    """(models asked, switch steps, skip steps, the child's trace calls)."""
    models = [model for kind, model in facts.models if kind == "subagent"]
    steps = [summary or "" for _, summary in facts.child_steps]
    calls = [call[3:] for call in facts.owned_calls if not call[0]]
    return (
        models,
        [s for s in steps if s.startswith("Provider fallback:")],
        [s for s in steps if s.startswith(SKIPPED)],
        calls,
    )


def _primary_follow_up(facts):
    return [messages for kind, messages in facts.sent if kind == "primary"][-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", ["capture_on", "capture_off"])
async def test_a_spawned_childs_preset_fallback_is_sent_and_traced(
    tmp_path, monkeypatch, capture
):
    """The child falls back, its answer reaches the parent, and the parent's
    next message is sent. Under Capture On the fallback is a new chain of the
    child's actor, opening a child segment of its own; the Trace view shows it
    as the same sub-agent.

    On dev 6e41919880 the captured child switched, its call was reserved and
    refused at dispatch binding, and the child ended on "trace capture could
    not be saved"; since TASK-42556 it was a visible skip.
    """
    facts = await _spawn_turn(
        tmp_path,
        monkeypatch,
        "Say child done.",
        preset_fallback=True,
        capture=capture,
        follow_up=True,
        trace_view=True,
    )
    why = str(vars(facts))
    models, switched, skipped, calls = _child(facts)
    assert facts.result.accepted and facts.runs == [
        ("primary", "done"),
        ("subagent", "done"),
    ], why
    assert models == [MODEL, FALLBACK_MODEL] and len(switched) == 1, why
    assert skipped == [] and facts.child_errors == [], why
    # The fallback's answer reaches the parent, and the next message is sent.
    assert facts.follow_up.accepted and facts.follow_up_state == "completed", why
    assert "child done" in str(_primary_follow_up(facts)), why
    if capture == "capture_off":
        assert facts.calls == [], why
        return
    # Calls are created a second apart at most (CURRENT_TIMESTAMP), so the
    # harness's order is not theirs; each chain is found by its state.
    assert sorted(call[1:] for call in calls) == [
        ("agent_first", "complete"),
        ("agent_first", "error"),
    ], why
    run_of = {call[2]: call[0] for call in calls}
    failed, fallback = run_of["error"], run_of["complete"]
    assert failed.split(":")[0] == fallback.split(":")[0] and failed != fallback, why
    assert all(call[1] != "reserved" for call in facts.calls), why
    assert facts.boundary_errors == [] and facts.refusals == [], why
    # The parent's own calls, its next message's included, are all complete.
    assert [call[4:] for call in facts.owned_calls if call[0]] == [
        ("agent_first", "complete"),
        ("tool_loop", "complete"),
        ("agent_first", "complete"),
    ], why
    rows = [
        record.content_preview
        for turn in facts.snapshot.turns
        for record in turn.records
        if record.kind == "child_call"
    ]
    assert rows == [
        "sub-agent 1 · call 1 · agent first",
        "sub-agent 1 · call 2 · agent first",
    ], why


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", ["capture_on", "capture_off"])
async def test_an_alternate_that_keeps_the_target_is_a_visible_skip(
    tmp_path, monkeypatch, capture
):
    """The trace records a child's fallback only as a change of target.

    An alternate naming the child's own model is a retry, not a fallback:
    under Capture On it is skipped with the visible copy before any call or
    reservation, and the child ends on its own model's error.
    """
    facts = await _spawn_turn(
        tmp_path, monkeypatch, "Say child done.", preset_fallback="same", capture=capture
    )
    why = str(vars(facts))
    models, switched, skipped, calls = _child(facts)
    assert ("subagent", "error") in facts.runs, why
    if capture == "capture_off":
        assert models == [MODEL, MODEL] and len(switched) == 1, why
        return
    assert models == [MODEL] and switched == [], why
    assert skipped == [
        f"{SKIPPED}: custom-ep:spawntrace/{MODEL} ({PRESET_FALLBACK_NOT_CAPTURED})"
    ], why
    assert len(PRESET_FALLBACK_NOT_CAPTURED) <= 80
    assert "trace capture" not in facts.child_errors[0], why
    assert [call[1:] for call in calls] == [("agent_first", "error")], why


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["adapter", "reservation", "binding"])
async def test_a_child_fallback_the_rule_does_not_admit_is_never_sent(
    tmp_path, monkeypatch, where
):
    """Each check refuses before the fallback model is asked.

    - adapter: the runtime names a target the child never durably selected
      (the selection write is skipped); refused before preparation.
    - reservation: the ledger's link check reads the failed call as not
      fallen back from (it bore a response, or kept the target); refused in
      the reservation transaction, which rolls back. A settled call cannot
      be edited (a trigger), so the read is forged.
    - binding: the same, with the reservation's check disabled; refused at
      dispatch binding.
    """
    from tldw_chatbook.Chat import console_trace_child_fallback, console_trace_runtime

    if where == "adapter":
        monkeypatch.setattr(
            AgentRunsDB, "set_run_active_fallback", lambda *_args, **_kwargs: None
        )
    else:
        monkeypatch.setattr(
            console_trace_child_fallback, "falls_back_from", lambda *_args: False
        )
    if where == "binding":
        monkeypatch.setattr(
            console_trace_runtime, "child_fallback_admits", lambda *_a, **_k: None
        )
    facts = await _spawn_turn(
        tmp_path, monkeypatch, "Say child done.", preset_fallback=True
    )
    why = str(vars(facts))
    models, switched, _skipped, calls = _child(facts)
    assert models == [MODEL] and len(switched) == 1, why
    assert ("subagent", "error") in facts.runs, why
    states = sorted(call[2] for call in calls)
    if where == "adapter":
        assert "did not select this target" in facts.child_errors[0], why
        assert states == ["error"], why
    elif where == "reservation":
        assert states == ["error"], why
        assert facts.boundary_errors[-1][0] == "subagent", why
    else:
        assert states == ["error", "reserved"], why


def test_a_refusal_never_replaces_an_unready_alternates_own_reason():
    """Review minor (TASK-42556): the Capture refusal applies to ready
    alternates only, and an unready one keeps its own reason."""
    candidates = (
        FallbackCandidate("openai", True, False, "not configured", "alt-1", 1),
        FallbackCandidate("openai", True, True, model="alt-2", index=2),
        FallbackCandidate("openai", True, True, model="alt-3", index=3),
    )
    asked = []

    def refuse(candidate):
        asked.append(candidate.model)
        return "refused" if candidate.model == "alt-2" else ""

    def primary(*_args):
        raise ChatModelUnavailableError(provider="openai")

    runtime = FallbackRuntime(
        candidates,
        lambda _candidate: lambda *_args: ModelTurn(text="rescued"),
        pre_tool_only=True,
        refuse=refuse,
    )
    traced = []
    deps = LoopDeps(
        call_model=primary,
        on_trace_step=traced.append,
        invoke_tool=lambda call: ToolResult(ok=False, error="refused"),
        spawn=lambda task: ToolResult(ok=False),
        find_tools=lambda query: [],
        load_schemas=lambda *args: None,
        should_cancel=lambda: False,
        clock=lambda: 0,
        fallback=runtime,
        sleep=lambda _: None,
    )
    config = AgentConfig(
        model="primary",
        provider="openai",
        system_prompt="s",
        allowed_tools=(),
        budget=RunBudget(max_model_retries=0),
    )
    outcome = run_agent_loop(config, [{"role": "user", "content": "hi"}], [], deps)

    assert outcome.final_text == "rescued"
    assert asked == ["alt-2", "alt-3"]
    assert [s.summary for s in traced if (s.summary or "").startswith(SKIPPED)] == [
        f"{SKIPPED}: openai/alt-1 (not configured)",
        f"{SKIPPED}: openai/alt-2 (refused)",
    ]
