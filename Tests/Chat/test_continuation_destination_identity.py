"""TASK-34100.3 (gap-02): continuations match the destination the gateway records.

The durable dispatch row stores the provider destination exactly as
``resolve_console_destination`` computes it: a credential-free ORIGIN such as
``https://api.moonshot.ai``. A provider continuation carries the full base URL
it was sent to (``https://api.moonshot.ai/v1``). The repository used to compare
the two for equality, so every saved-chat continuation conflicted: a Moonshot
reasoning reply failed terminal settlement, and a tool batch failed its
handoff. The existing store tests passed only because they hand-built a
destination whose identity already carried the ``/v1`` path.

Every test here builds the destination with the real resolver.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from Tests.Chat.test_console_dispatch_continuation_handoff import (
    _checkpoint_count,
    _continuation,
    _deepseek_acceptance,
    _event,
    _moonshot_acceptance,
    _portable_store,
)
from Tests.Chat.test_console_dispatch_recovery import _database, _insert, _start
from tldw_chatbook.Agents.agent_models import FinalContinuation
from tldw_chatbook.Chat.console_library_destination import (
    console_endpoint_identity,
    resolve_console_destination,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationRound,
    ProviderContinuationCheckpoint,
    parse_provider_continuation_json,
)


def _real_destination(provider: str, model: str, base_url: str):
    return resolve_console_destination(
        ConsoleProviderResolution(
            provider=provider,
            base_url=base_url,
            model=model,
            ready=True,
            readiness_key=provider,
            execution_key=provider,
        )
    )


def _with_real_destination(acceptance, provider: str, model: str, base_url: str):
    destination = _real_destination(provider, model, base_url)
    # The premise of the defect: the recorded identity is an origin, not the URL.
    assert destination.endpoint_identity != base_url
    return replace(acceptance, resolved_destination=destination)


def _kimi_final(base_url: str = "https://api.moonshot.ai/v1"):
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url=base_url,
        state="complete",
        rounds=(ContinuationRound("answer", ("private reasoning",), ()),),
    )


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
        FinalContinuation(_event().context, final, None, "answer")
    )

    row = db.get_message_by_id("assistant-1")
    assert (row["assistant_generation_state"], row["content"]) == ("complete", "answer")
    assert parse_provider_continuation_json(row["provider_continuation_json"]) == final
    assert _checkpoint_count(db) == 0
    assert store.dispatch_recovery_for_session(session_id) is None
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

    store.persist_provider_continuation_event(_event())

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
    db, store, session_id, checkpoint = _started_store(
        tmp_path,
        lambda conversation_id: _with_real_destination(
            _moonshot_acceptance(conversation_id),
            "moonshot",
            "kimi-k3",
            "https://api.moonshot.ai/v1",
        ),
    )

    with pytest.raises(RuntimeError, match="terminal settlement failed"):
        store.persist_provider_continuation_event(
            FinalContinuation(_event().context, _kimi_final(base_url), None, "answer")
        )

    assert db.get_message_by_id("assistant-1")["provider_continuation_json"] is None
    assert _checkpoint_count(db) == 1
    db.close()


def test_endpoint_identity_is_the_resolver_identity() -> None:
    """One function names the identity; the resolver and the match share it."""
    for base_url in (
        "https://api.moonshot.ai/v1",
        "https://API.Moonshot.ai/v1/",
        "http://127.0.0.1:9099/v1",
        "http://[::1]:8080/v1",
        "unix:///tmp/llama.sock",
    ):
        assert console_endpoint_identity(base_url) == (
            _real_destination("p", "m", base_url).endpoint_identity
        )
    assert console_endpoint_identity("not a url") is None
    assert console_endpoint_identity(None) is None
