"""TASK-34100.3 (gap-02): continuations settle against the gateway's destination.

The durable dispatch row stores the provider destination exactly as
``resolve_console_destination`` computes it: a credential-free ORIGIN such as
``https://api.moonshot.ai`` for disclosure, plus (since dev's TASK-35582) the
continuation tuple -- execution provider, protocol and the full base URL
(``https://api.moonshot.ai/v1``). A provider continuation carries that full
base URL. The repository used to compare the continuation's base URL with the
ORIGIN, so every saved-chat continuation conflicted: a Moonshot reasoning
reply failed terminal settlement, and a tool batch failed its handoff. The
store tests passed only because they hand-built a destination whose identity
already carried the ``/v1`` path.

TASK-35582 fixed the comparison on dev (it matches the frozen continuation
tuple, not the origin); this branch's own fix was dropped in the rebase. These
tests stay as the regression guard the defect called for: every destination
here is built with the REAL resolver, never by hand. They pass on base
26b2e677dd, except ``test_a_settled_reasoning_reply_settles_its_trace_call``,
which guards this branch's trace-handoff drain and fails without it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from Tests.Chat.test_console_dispatch_continuation_handoff import (
    _DEEPSEEK_RUN_TARGET,
    _MOONSHOT_RUN_TARGET,
    _checkpoint_count,
    _continuation,
    _deepseek_acceptance,
    _event,
    _moonshot_acceptance,
    _portable_store,
)
from Tests.Chat.test_console_dispatch_recovery import _database, _insert, _start
from tldw_chatbook.Agents.agent_models import FinalContinuation
from tldw_chatbook.Chat.console_library_destination import resolve_console_destination
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationRound,
    ProviderContinuationCheckpoint,
    parse_provider_continuation_json,
)

#: The wire protocol each provider's continuation is recorded under.
_PROTOCOLS = {"moonshot": "chat_completions", "deepseek": "responses"}


def _real_destination(provider: str, model: str, base_url: str):
    return resolve_console_destination(
        ConsoleProviderResolution(
            provider=provider,
            base_url=base_url,
            model=model,
            ready=True,
            readiness_key=provider,
            execution_key=provider,
            continuation_protocol=_PROTOCOLS.get(provider),
        )
    )


def _with_real_destination(acceptance, provider: str, model: str, base_url: str):
    destination = _real_destination(provider, model, base_url)
    # The premise of the defect: the recorded identity is an origin, not the URL.
    assert destination.endpoint_identity != base_url
    return replace(acceptance, resolved_destination=destination)


def _kimi_final(base_url: str = "https://api.moonshot.ai/v1"):
    # Host-bound (schema 3): the Console adapter stamps the raw selection on
    # every candidate before the store sees it (TASK-35582).
    return ProviderContinuationCheckpoint(
        schema_version=3,
        selected_provider="moonshot",
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url=base_url,
        state="complete",
        rounds=(ContinuationRound("answer", ("private reasoning",), ()),),
    )


def _kimi_target(base_url: str = "https://api.moonshot.ai/v1"):
    """The host-captured run target for a Kimi send to ``base_url``."""
    return replace(_MOONSHOT_RUN_TARGET, api_base_url=base_url)


def _started_store(tmp_path: Path, acceptance_for):
    db, conversation_id, repository = _database(tmp_path / "chat.sqlite")
    checkpoint = _start(
        repository, _insert(db, repository, acceptance_for(conversation_id))
    )
    store, session_id, _, _ = _portable_store(db, conversation_id, tmp_path)
    store.publish_durable_dispatch_checkpoint(session_id, checkpoint, in_flight=True)
    return db, store, session_id, checkpoint


def test_reasoning_reply_settles_against_the_gateway_destination(tmp_path) -> None:
    db, store, session_id, _ = _started_store(
        tmp_path,
        lambda conversation_id: _with_real_destination(
            _moonshot_acceptance(conversation_id),
            "moonshot",
            "kimi-k3",
            "https://api.moonshot.ai/v1",
        ),
    )
    final = _kimi_final()

    store.persist_provider_continuation_event(
        FinalContinuation(_event().context, final, None, "answer"),
        continuation_target=_kimi_target(),
    )

    row = db.get_message_by_id("assistant-1")
    assert (row["assistant_generation_state"], row["content"]) == ("complete", "answer")
    assert parse_provider_continuation_json(row["provider_continuation_json"]) == final
    assert _checkpoint_count(db) == 0
    assert store.dispatch_recovery_for_session(session_id) is None
    db.close()


class _TraceCallHandoff:
    """The store's trace-settlement seam: one provider call awaiting its reply."""

    call_id = "kimi-trace-call"

    def __init__(self) -> None:
        self.settled_with: list[str | None] = []

    def settle(self, canonical_message_id: str | None) -> None:
        self.settled_with.append(canonical_message_id)


def test_a_settled_reasoning_reply_settles_its_trace_call(tmp_path) -> None:
    """The final checkpoint's dispatch settlement completes the reply, so it
    must also hand the reply to the trace call that produced it.

    TASK-34100.3 pass 3: no later terminal write runs for that reply, so every
    saved Kimi call stayed ``response_started`` and the Trace view listed none.
    """
    db, store, session_id, _ = _started_store(
        tmp_path,
        lambda conversation_id: _with_real_destination(
            _moonshot_acceptance(conversation_id),
            "moonshot",
            "kimi-k3",
            "https://api.moonshot.ai/v1",
        ),
    )
    reply = store._message_or_raise("assistant-1")  # noqa: SLF001
    reply.status = "streaming"  # the reply is still being produced
    handoff = _TraceCallHandoff()
    store.register_provider_trace_settlement(reply.id, handoff)
    assert handoff.settled_with == []
    assert store.pending_provider_trace_settlement_count(reply.id) == 1

    store.persist_provider_continuation_event(
        FinalContinuation(_event().context, _kimi_final(), None, "answer"),
        continuation_target=_kimi_target(),
    )

    assert db.get_message_by_id("assistant-1")["assistant_generation_state"] == (
        "complete"
    )
    assert handoff.settled_with == ["assistant-1"]
    assert store.pending_provider_trace_settlement_count(reply.id) == 0
    db.close()


def test_tool_batch_hands_off_against_the_gateway_destination(tmp_path) -> None:
    db, store, session_id, _ = _started_store(
        tmp_path,
        lambda conversation_id: _with_real_destination(
            _deepseek_acceptance(conversation_id),
            "deepseek",
            "deepseek-v4-flash",
            "https://api.deepseek.com/v1",
        ),
    )

    store.persist_provider_continuation_event(
        _event(), continuation_target=_DEEPSEEK_RUN_TARGET
    )

    row = db.get_message_by_id("assistant-1")
    assert row["assistant_generation_state"] == "continuation_active"
    assert parse_provider_continuation_json(row["provider_continuation_json"]) == (
        _continuation()
    )
    assert _checkpoint_count(db) == 0
    assert store.dispatch_recovery_for_session(session_id) is None
    db.close()


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.moonshot.cn/v1",  # same provider family, another origin
        "https://api.moonshot.ai:8443/v1",  # same host, another port
        "http://api.moonshot.ai/v1",  # same host, another scheme
    ],
)
def test_a_continuation_to_another_origin_still_conflicts(tmp_path, base_url) -> None:
    """The frozen destination still binds: a send whose run target moved to
    another origin, port or scheme cannot settle the accepted dispatch."""
    db, store, session_id, checkpoint = _started_store(
        tmp_path,
        lambda conversation_id: _with_real_destination(
            _moonshot_acceptance(conversation_id),
            "moonshot",
            "kimi-k3",
            "https://api.moonshot.ai/v1",
        ),
    )

    with pytest.raises(RuntimeError, match="destination mismatch"):
        store.persist_provider_continuation_event(
            FinalContinuation(_event().context, _kimi_final(base_url), None, "answer"),
            continuation_target=_kimi_target(base_url),
        )

    assert db.get_message_by_id("assistant-1")["provider_continuation_json"] is None
    assert _checkpoint_count(db) == 1
    db.close()


@pytest.mark.parametrize(
    ("acceptance_for", "provider", "model", "base_url"),
    [
        (_moonshot_acceptance, "moonshot", "kimi-k3", "https://api.moonshot.ai/v1"),
        (
            _deepseek_acceptance,
            "deepseek",
            "deepseek-v4-flash",
            "https://api.deepseek.com/v1",
        ),
    ],
    ids=["moonshot", "deepseek"],
)
def test_shared_dispatch_fixtures_hold_the_resolver_identity(
    acceptance_for, provider, model, base_url
) -> None:
    """Pin the shared hand-built rows to what the real resolver records."""
    built = acceptance_for("conversation-1").resolved_destination

    assert built == _real_destination(provider, model, base_url)
