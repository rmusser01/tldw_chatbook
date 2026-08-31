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


# --- the wrapper, not just the resolution ---------------------------------


def _service():
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    return AgentService(db=SimpleNamespace(), registry=ToolCatalogRegistry())


def _config(chain):
    from tldw_chatbook.Agents.agent_models import AgentConfig

    return AgentConfig(
        model="m", system_prompt="s", fallback_providers=tuple(chain)
    )


def test_no_chain_returns_the_primary_object_itself(monkeypatch):
    """AC#10: unconfigured must not even add a frame."""
    service = _service()

    def primary(messages, active, cont=None):
        return "primary"

    wrapped = service._wrap_with_fallback(
        primary,
        primary_endpoint="openai",
        build_for_provider=lambda p: None,
        config=_config([]),
        run_id="r1",
    )

    assert wrapped is primary


def test_a_transient_failure_switches_to_the_next_provider(monkeypatch):
    from tldw_chatbook.Agents import agent_service as svc

    service = _service()
    monkeypatch.setattr(service, "_provider_is_ready", lambda p: True)

    seen = {}

    def primary(messages, active, cont=None):
        raise ChatRateLimitError("primary down")

    def build(provider):
        def call(messages, active, cont=None):
            seen["provider"] = provider
            seen["messages"] = messages
            return "fallback-result"

        return call

    wrapped = service._wrap_with_fallback(
        primary,
        primary_endpoint="openai",
        build_for_provider=build,
        config=_config(["groq"]),
        run_id="r1",
    )

    assert wrapped([{"role": "user", "content": "hi"}], ()) == "fallback-result"
    assert seen["provider"] == "groq"


def test_an_auth_failure_is_not_absorbed_by_a_fallback(monkeypatch):
    """Handing 401 to another provider hides a problem the user must fix."""
    service = _service()
    monkeypatch.setattr(service, "_provider_is_ready", lambda p: True)

    def primary(messages, active, cont=None):
        raise ChatAuthenticationError("bad key")

    wrapped = service._wrap_with_fallback(
        primary,
        primary_endpoint="openai",
        build_for_provider=lambda p: (lambda *a, **k: "should not be reached"),
        config=_config(["groq"]),
        run_id="r1",
    )

    with pytest.raises(ChatAuthenticationError):
        wrapped([{"role": "user", "content": "hi"}], ())


def test_an_unready_candidate_is_skipped_without_an_attempt(monkeypatch):
    service = _service()
    monkeypatch.setattr(service, "_provider_is_ready", lambda p: p == "ollama")

    attempted = []

    def primary(messages, active, cont=None):
        raise ChatRateLimitError("down")

    def build(provider):
        attempted.append(provider)
        return lambda *a, **k: "ok"

    wrapped = service._wrap_with_fallback(
        primary,
        primary_endpoint="openai",
        build_for_provider=build,
        config=_config(["anthropic", "ollama"]),
        run_id="r1",
    )

    assert wrapped([{"role": "user", "content": "hi"}], ()) == "ok"
    assert attempted == ["ollama"], "the unready candidate was attempted anyway"


def test_history_is_projected_for_the_target_protocol(monkeypatch):
    """The whole reason ADR-110 exists."""
    service = _service()
    monkeypatch.setattr(service, "_provider_is_ready", lambda p: True)

    captured = {}

    def primary(messages, active, cont=None):
        raise ChatRateLimitError("down")

    def build(provider):
        def call(messages, active, cont=None):
            captured["messages"] = messages
            return "ok"

        return call

    native_history = [
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
    ]

    wrapped = service._wrap_with_fallback(
        primary,
        primary_endpoint="openai",
        build_for_provider=build,
        # ollama is a fence provider, so the native history must be projected
        config=_config(["ollama"]),
        run_id="r1",
    )
    wrapped(native_history, ())

    projected = captured["messages"]
    assert not any("tool_calls" in m for m in projected), (
        "native tool_calls reached a fence provider unprojected"
    )
    assert not any(m.get("role") == "tool" for m in projected)
