"""Choosing the next provider when the primary will not recover.

ADR-110 / TASK-25902. Retry (TASK-25901) handles a provider that will be back
in seconds; fallback handles one that will not. The chain is only consulted
after retries are exhausted or on a credit/quota-terminal class.

A candidate the readiness check calls unconfigured is skipped WITHOUT an
attempt, and the skip is reported -- a user who lists a provider they never set
up should learn that, not silently get a shorter chain than they think.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Agents.fallback_chain import (
    FallbackCandidate,
    is_credit_terminal,
    resolve_fallback_chain,
)
from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatProviderError,
    ChatRateLimitError,
)


def _ready(provider):
    return True


def _never_ready(provider):
    return False


def test_an_empty_chain_yields_nothing():
    """AC#10: unconfigured means no behaviour change at all."""
    assert resolve_fallback_chain([], "openai", _ready) == []
    assert resolve_fallback_chain(None, "openai", _ready) == []


def test_the_primary_is_never_a_fallback_for_itself():
    chain = resolve_fallback_chain(["openai", "groq"], "openai", _ready)

    assert [c.provider for c in chain] == ["groq"]


def test_order_is_preserved():
    chain = resolve_fallback_chain(["groq", "anthropic", "ollama"], "openai", _ready)

    assert [c.provider for c in chain] == ["groq", "anthropic", "ollama"]


def test_duplicates_are_collapsed():
    chain = resolve_fallback_chain(["groq", "groq", "anthropic"], "openai", _ready)

    assert [c.provider for c in chain] == ["groq", "anthropic"]


def test_an_unconfigured_candidate_is_skipped_and_says_so():
    """AC#3: a silently shorter chain is a surprise waiting to happen."""

    def only_groq_ready(provider):
        return provider == "groq"

    chain = resolve_fallback_chain(
        ["anthropic", "groq"], "openai", only_groq_ready
    )

    assert [c.provider for c in chain if c.ready] == ["groq"]
    skipped = [c for c in chain if not c.ready]
    assert [c.provider for c in skipped] == ["anthropic"]
    assert skipped[0].skip_reason


def test_a_chain_with_nothing_configured_is_reported_not_silent():
    chain = resolve_fallback_chain(["a", "b"], "openai", _never_ready)

    assert chain, "an all-unconfigured chain must still report its candidates"
    assert all(not c.ready for c in chain)


def test_blank_and_malformed_entries_are_dropped():
    chain = resolve_fallback_chain(
        ["", "  ", None, 42, "groq"], "openai", _ready
    )

    assert [c.provider for c in chain] == ["groq"]


def test_a_readiness_probe_that_raises_marks_the_candidate_unready():
    """A broken probe must not take the run down with it."""

    def explodes(provider):
        raise RuntimeError("SENTINEL")

    chain = resolve_fallback_chain(["groq"], "openai", explodes)

    assert chain and chain[0].ready is False


# --- which failures earn a fallback ---------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(ChatProviderError("no credit", status_code=402), id="402"),
        pytest.param(ChatProviderError("quota", status_code=403), id="403-quota"),
    ],
)
def test_credit_terminal_errors_trigger_fallback_immediately(exc):
    """No point retrying a provider that has told us the money is gone."""
    assert is_credit_terminal(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(ChatRateLimitError("slow down"), id="429"),
        pytest.param(ChatProviderError("bad gateway", status_code=502), id="502"),
        pytest.param(ChatAuthenticationError("bad key"), id="401"),
        pytest.param(ChatBadRequestError("malformed"), id="400"),
        pytest.param(ValueError("our bug"), id="programming-error"),
    ],
)
def test_other_errors_are_not_credit_terminal(exc):
    """429 is retry's job; 401/400 are the user's to fix, not another
    provider's to absorb."""
    assert is_credit_terminal(exc) is False


def test_candidate_carries_its_target_protocol():
    """The switch needs to know which way to project."""
    chain = resolve_fallback_chain(["groq"], "openai", _ready)

    assert isinstance(chain[0], FallbackCandidate)
    assert isinstance(chain[0].native, bool)


# --- the loop owns the switch (review C1/C2/C4, 2026-08-31) -----------------
#
# The first implementation was a wrapper around call_model. Review proved it
# inverted the retry composition (the wrapper absorbed transient errors before
# the loop's retry ever saw them), was per-call rather than sticky (violating
# accepted ADR-110 decision 4 and manufacturing the mixed-protocol history the
# ADR exists to prevent), and its switch report was dead code three ways over.
# These tests drive the REAL run_agent_loop, because the wrapper-only tests all
# passed while every one of those bugs shipped.

from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import (
    RUN_DONE,
    LoopDeps,
    run_agent_loop,
)
from tldw_chatbook.Agents.fallback_chain import FallbackRuntime


def _loop_deps(primary, runtime, *, trace_sink=None, slept=None):
    return LoopDeps(
        call_model=primary,
        invoke_tool=lambda c: ToolResult(ok=True, content="x"),
        spawn=lambda task: ToolResult(ok=True, content="x"),
        find_tools=lambda q: [],
        load_schemas=lambda _i, _m, _c: None,
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        on_trace_step=trace_sink if trace_sink is not None else (lambda s: None),
        sleep=(slept.append if slept is not None else (lambda s: None)),
        fallback=runtime,
    )


def _cfg(retries=2):
    return AgentConfig(
        model="m",
        system_prompt="s",
        provider="openai",
        budget=RunBudget(max_model_retries=retries),
    )


def _runtime(candidates, build):
    return FallbackRuntime(candidates=tuple(candidates), build=build)


def _ready_candidate(provider, native=True):
    return FallbackCandidate(provider=provider, native=native, ready=True)


def test_retry_is_exhausted_before_the_chain_is_consulted():
    """Review C1, proven inverted in the first implementation.

    ADR-110: "the chain is only consulted after retries are exhausted". The
    primary must be asked 1 + max_model_retries times, with backoff, before
    any candidate sees a request.
    """
    events = []
    slept = []

    def primary(messages, active, cont=None):
        events.append("primary")
        raise ChatRateLimitError("still down")

    def build(provider):
        def call(messages, active, cont=None):
            events.append(f"candidate:{provider}")
            return ModelTurn(text="rescued")

        return call

    outcome = run_agent_loop(
        _cfg(retries=2),
        [{"role": "user", "content": "hi"}],
        [],
        _loop_deps(primary, _runtime([_ready_candidate("groq")], build), slept=slept),
    )

    assert outcome.status == RUN_DONE
    assert events == ["primary", "primary", "primary", "candidate:groq"], events
    # Sliced sleeps: assert time, not call count.
    assert sum(slept) > 0, "the retries should have backed off first"


def test_the_switch_is_sticky():
    """Review C2 / ADR-110 decision 4: after a switch the run continues on the
    new provider. The first implementation re-failed the primary every turn."""
    events = []

    def primary(messages, active, cont=None):
        events.append("primary")
        raise ChatRateLimitError("dead")

    def build(provider):
        def call(messages, active, cont=None):
            events.append(f"candidate:{provider}")
            if len([e for e in events if e.startswith("candidate")]) == 1:
                return ModelTurn(
                    text='```tool_call\n{"name": "calculator", "arguments": {}}\n```'
                )
            return ModelTurn(text="finished")

        return call

    cfg = AgentConfig(
        model="m",
        system_prompt="s",
        provider="openai",
        allowed_tools=("calculator",),
        budget=RunBudget(max_model_retries=0),
    )
    outcome = run_agent_loop(
        cfg,
        [{"role": "user", "content": "hi"}],
        [],
        _loop_deps(primary, _runtime([_ready_candidate("ollama", native=False)], build)),
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "finished"
    assert events.count("primary") == 1, (
        f"primary was re-attempted after the switch: {events}"
    )


def test_credit_terminal_switches_without_burning_retries():
    """Out of money means out of money -- waiting cannot fix it."""
    events = []
    slept = []

    def primary(messages, active, cont=None):
        events.append("primary")
        raise ChatProviderError("payment required", status_code=402)

    def build(provider):
        return lambda m, a, c=None: ModelTurn(text="rescued")

    outcome = run_agent_loop(
        _cfg(retries=3),
        [{"role": "user", "content": "hi"}],
        [],
        _loop_deps(primary, _runtime([_ready_candidate("groq")], build), slept=slept),
    )

    assert outcome.status == RUN_DONE
    assert events == ["primary"], "a 402 should not be retried"
    assert slept == [], "no backoff for a credit-terminal error"


def test_an_auth_failure_is_never_absorbed_by_the_chain():
    """Handing a 401 to another provider hides what the user must fix."""

    def primary(messages, active, cont=None):
        raise ChatAuthenticationError("bad key")

    def build(provider):
        return lambda m, a, c=None: ModelTurn(text="should never run")

    with pytest.raises(ChatAuthenticationError):
        run_agent_loop(
            _cfg(retries=0),
            [{"role": "user", "content": "hi"}],
            [],
            _loop_deps(primary, _runtime([_ready_candidate("groq")], build)),
        )


def test_an_unready_candidate_is_skipped_and_the_skip_is_traced():
    traced = []

    def primary(messages, active, cont=None):
        raise ChatRateLimitError("down")

    attempted = []

    def build(provider):
        attempted.append(provider)
        return lambda m, a, c=None: ModelTurn(text="rescued")

    candidates = [
        FallbackCandidate(
            provider="anthropic", native=True, ready=False, skip_reason="not configured"
        ),
        _ready_candidate("groq"),
    ]
    outcome = run_agent_loop(
        _cfg(retries=0),
        [{"role": "user", "content": "hi"}],
        [],
        _loop_deps(primary, _runtime(candidates, build), trace_sink=traced.append),
    )

    assert outcome.status == RUN_DONE
    assert attempted == ["groq"]
    summaries = " | ".join(str(t.summary) for t in traced)
    assert "skipped" in summaries and "anthropic" in summaries


def test_the_switch_itself_is_visible_in_the_trace():
    """Review C4: the old report path was dead code three ways over, silently
    swallowed by a blanket except. The loop now reports through the same trace
    seam retries already use, which tests can actually observe."""
    traced = []

    def primary(messages, active, cont=None):
        raise ChatRateLimitError("down")

    def build(provider):
        return lambda m, a, c=None: ModelTurn(text="rescued")

    run_agent_loop(
        _cfg(retries=0),
        [{"role": "user", "content": "hi"}],
        [],
        _loop_deps(
            primary,
            _runtime([_ready_candidate("groq")], build),
            trace_sink=traced.append,
        ),
    )

    summaries = " | ".join(str(t.summary) for t in traced)
    assert "Provider fallback: openai -> groq" in summaries, summaries


def test_history_is_projected_for_a_fence_candidate():
    """The reason ADR-110 exists: native tool_calls must not reach a fence
    provider unprojected."""
    captured = {}

    def primary(messages, active, cont=None):
        raise ChatRateLimitError("down")

    def build(provider):
        def call(messages, active, cont=None):
            captured["messages"] = [dict(m) for m in messages]
            return ModelTurn(text="rescued")

        return call

    native_history = [
        {"role": "user", "content": "6*7?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "42"},
        {"role": "user", "content": "now answer"},
    ]

    outcome = run_agent_loop(
        _cfg(retries=0),
        native_history,
        [],
        _loop_deps(primary, _runtime([_ready_candidate("ollama", native=False)], build)),
    )

    assert outcome.status == RUN_DONE
    projected = captured["messages"]
    assert not any("tool_calls" in m for m in projected)
    assert not any(m.get("role") == "tool" for m in projected)


def test_no_runtime_means_byte_identical_behaviour():
    """AC#10: with no chain, a transient error follows the plain retry path."""
    calls = []

    def primary(messages, active, cont=None):
        calls.append(1)
        if len(calls) == 1:
            raise ChatRateLimitError("blip")
        return ModelTurn(text="fine")

    outcome = run_agent_loop(
        _cfg(retries=2),
        [{"role": "user", "content": "hi"}],
        [],
        _loop_deps(primary, None),
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "fine"
