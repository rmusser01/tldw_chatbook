"""Service-side redirect surface: mailbox + abort flag, one atomic unit.

TASK-26000. `redirect_primary` posts a STEERING_SOURCE_REDIRECT entry into the
SAME mailbox steering uses and sets the run's abort flag under the same lock;
the drain clears the flag when it consumes a redirect entry. The flag is what
the bridge composes into its STREAM-cancel predicate (aborting only the
in-flight model request) and what the loop's `has_pending_redirect` probe
reads -- one source of truth, no second mailbox to desync.
"""

from __future__ import annotations

from types import SimpleNamespace

from tldw_chatbook.Agents.agent_models import (
    MAX_STEERING_CHARS,
    STEERING_SOURCE_REDIRECT,
    STEERING_SOURCE_USER,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry


def _service(**kwargs):
    return AgentService(
        db=SimpleNamespace(), registry=ToolCatalogRegistry(), **kwargs
    )


def test_redirecting_an_unknown_run_is_refused_honestly():
    service = _service()

    refusal = service.redirect_primary("no-such-run", "do Y instead")

    assert refusal is not None
    assert "not running" in refusal.lower()


def test_redirecting_a_finished_run_is_refused():
    service = _service()
    service._register_primary_mailbox("run-1")
    service._unregister_primary_mailbox("run-1")

    assert service.redirect_primary("run-1", "too late") is not None


def test_empty_and_overlong_corrections_are_refused():
    service = _service()
    service._register_primary_mailbox("run-1")

    assert service.redirect_primary("run-1", "   ") is not None
    over = service.redirect_primary("run-1", "x" * (MAX_STEERING_CHARS + 1))
    assert over is not None
    assert str(MAX_STEERING_CHARS) in over


def test_accepted_redirect_posts_entry_and_raises_the_abort_flag():
    service = _service()
    drain = service._register_primary_mailbox("run-1")

    assert service.redirect_primary("run-1", "  no — the YAML parser  ") is None
    assert service._primary_redirect_pending("run-1") is True

    entries = drain()
    assert entries == [(STEERING_SOURCE_REDIRECT, "no — the YAML parser")]
    # consuming the redirect entry lowers the flag -- the next model call
    # must not be aborted by a redirect already delivered
    assert service._primary_redirect_pending("run-1") is False


def test_plain_steering_never_raises_the_abort_flag():
    service = _service()
    drain = service._register_primary_mailbox("run-1")

    assert service.steer_primary("run-1", "gentle nudge") is None
    assert service._primary_redirect_pending("run-1") is False

    entries = drain()
    assert entries == [(STEERING_SOURCE_USER, "gentle nudge")]


def test_mixed_drain_clears_the_flag_and_keeps_order():
    service = _service()
    drain = service._register_primary_mailbox("run-1")

    service.steer_primary("run-1", "first")
    service.redirect_primary("run-1", "second")

    assert service._primary_redirect_pending("run-1") is True
    assert drain() == [
        (STEERING_SOURCE_USER, "first"),
        (STEERING_SOURCE_REDIRECT, "second"),
    ]
    assert service._primary_redirect_pending("run-1") is False


def test_on_primary_redirect_ready_hands_working_callables():
    captured = {}

    def ready(redirect_fn, abort_probe):
        captured["redirect"] = redirect_fn
        captured["probe"] = abort_probe

    service = _service(on_primary_redirect_ready=ready)
    drain = service._register_primary_mailbox("run-1")

    assert set(captured) == {"redirect", "probe"}
    assert captured["probe"]() is False
    assert captured["redirect"]("switch to plan B") is None
    assert captured["probe"]() is True
    assert drain() == [(STEERING_SOURCE_REDIRECT, "switch to plan B")]
    assert captured["probe"]() is False

    service._unregister_primary_mailbox("run-1")
    assert captured["redirect"]("gone") is not None
    assert captured["probe"]() is False


def test_mid_continuation_calls_suppress_the_stream_cut():
    """Review F1: cutting a model call made WITH an in-flight continuation
    checkpoint truncates a chain the provider persists -- the loop then
    classifies it as a continuation-contract violation and errors the RUN.
    The abort probe must read False for exactly those calls (redirect
    degrades to steering there), and recover afterwards."""
    from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget
    from tldw_chatbook.Chat.provider_continuation import (
        ContinuationCall,
        ContinuationRound,
        ProviderContinuationCheckpoint,
    )

    probe_during_call = {}

    def fake_chat(**kwargs):
        probe_during_call["value"] = captured["probe"]()
        return {"choices": [{"message": {"content": "ok"}}]}

    captured = {}

    def ready(redirect_fn, abort_probe):
        captured["redirect"] = redirect_fn
        captured["probe"] = abort_probe

    service = _service(chat_call=fake_chat, on_primary_redirect_ready=ready)
    service._register_primary_mailbox("run-1")
    assert captured["redirect"]("go the other way") is None
    assert captured["probe"]() is True

    config = AgentConfig(
        model="m",
        system_prompt="s",
        provider="llama_cpp",
        budget=RunBudget(max_steps=10),
    )
    call_model = service._make_call_model(
        config,
        "llama_cpp",
        [],
        continuation_owner_key="native_id",
        continuation_owner_message_id="owner-1",
        run_id="run-1",
    )
    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="deepseek",
        protocol="responses",
        model="m",
        api_base_url="https://api.example.test/v1",
        state="active",
        rounds=(
            ContinuationRound(
                assistant_content="",
                reasoning_blocks=("private",),
                calls=(
                    ContinuationCall(
                        call_id="c1",
                        name="calculator",
                        arguments="{}",
                        state="pending",
                    ),
                ),
            ),
        ),
    )

    call_model([{"role": "user", "content": "x"}], (), checkpoint)
    assert probe_during_call["value"] is False, (
        "the stream cut was live during a mid-continuation call"
    )
    # suppression is scoped to the call -- afterwards the pending redirect
    # is visible again (it will be consumed by the next drain instead)
    assert captured["probe"]() is True

    call_model([{"role": "user", "content": "x"}], (), None)
    assert probe_during_call["value"] is True, (
        "a plain call (no checkpoint) must keep the cut armed"
    )
