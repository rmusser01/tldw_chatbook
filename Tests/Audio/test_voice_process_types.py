"""Validated child policy values and original-object parent compatibility."""

from __future__ import annotations

from dataclasses import fields
from importlib import import_module
from types import SimpleNamespace

import pytest


def _types():
    return import_module("tldw_chatbook.Audio.voice_process_types")


def _prepared(epoch=1, **overrides):
    types = _types()
    values = dict(
        attempt_epoch=epoch,
        decision=types.VoiceSpeculationDecision.PROVISIONAL,
        request_handle=types.VoiceRequestHandle("1" * 32),
        turn_context_handle=types.VoiceTurnContextHandle("2" * 32),
    )
    values.update(overrides)
    return types.AttemptDispatchPrepared(**values)


@pytest.mark.parametrize("epoch", [True, False, -1, 1.0, "1"])
def test_prepared_policy_rejects_invalid_epoch(epoch):
    types = _types()
    with pytest.raises((TypeError, ValueError)):
        _prepared(epoch)
    with pytest.raises((TypeError, ValueError)):
        types.AttemptToolPending(epoch)
    with pytest.raises((TypeError, ValueError)):
        types.ProviderAttemptFailed(epoch, "RuntimeError")


@pytest.mark.parametrize("value", [None, True, 1, "", "bad handle", "a" * 33, "G" * 32])
def test_handles_reject_malformed_values(value):
    types = _types()
    for handle in (types.VoiceRequestHandle, types.VoiceTurnContextHandle):
        with pytest.raises((TypeError, ValueError)):
            handle(value)


def test_prepared_policy_is_exact_typed_and_has_no_parent_objects():
    types = _types()
    event = _prepared()
    assert {field.name for field in fields(event)} == {
        "attempt_epoch",
        "decision",
        "request_handle",
        "turn_context_handle",
    }
    assert {field.name for field in fields(types.AttemptToolPending(1))} == {
        "attempt_epoch"
    }
    assert event.request_handle != event.turn_context_handle
    for bad in ("provisional", "unknown", True, SimpleNamespace(value="provisional")):
        with pytest.raises(TypeError):
            _prepared(decision=bad)
    with pytest.raises(TypeError):
        _prepared(request_handle=event.turn_context_handle)
    with pytest.raises(TypeError):
        _prepared(turn_context_handle=event.request_handle)


def test_public_enum_aliases_preserve_identity_and_settings_values():
    types = _types()
    from tldw_chatbook.Chat import console_voice_attempts as attempts
    from tldw_chatbook.Chat import console_voice_controls as controls
    from tldw_chatbook.Chat import console_voice_eligibility as eligibility
    from tldw_chatbook.Chat import console_voice_settings as settings

    assert attempts.AttemptCleanupOutcome is types.AttemptCleanupOutcome
    assert controls.ControlKind is types.ControlKind
    assert eligibility.VoiceSpeculationDecision is types.VoiceSpeculationDecision
    for key, value in (("DEFAULT", 700), ("MIN", 500), ("MAX", 3000)):
        name = f"RESPONSE_EAGERNESS_{key}_MS"
        assert getattr(settings, name) == getattr(types, name) == value


@pytest.mark.parametrize("deferred", [False, True])
def test_parent_policy_rejects_fabricated_context(deferred):
    from Tests.Chat.test_console_voice_eligibility import _prepared_request
    from tldw_chatbook.Chat.console_speculative_voice import (
        AttemptDispatchPrepared,
        SpeculativeTurnCoordinator,
    )

    fabricated = SimpleNamespace(library_authority=SimpleNamespace())
    with pytest.raises(TypeError):
        if deferred:
            AttemptDispatchPrepared(1, fabricated, _prepared_request())
        else:
            SpeculativeTurnCoordinator(
                effects=object(),
                frozen_session_context=fabricated,
                prepared_request=_prepared_request(),
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("governed", [False, True])
async def test_parent_classifies_originals_and_core_receives_only_handles(
    monkeypatch, deferred, governed
):
    from Tests.Chat.test_console_voice_effect_barrier import _Effects, _Scheduler
    from Tests.Chat.test_console_voice_eligibility import (
        _frozen_context,
        _prepared_request,
    )
    from Tests.Chat.test_console_speculative_voice import _revision, _speech
    from tldw_chatbook.Chat import console_speculative_voice as parent

    types = _types()
    core = import_module("tldw_chatbook.Audio.voice_turn_coordinator")
    context, request = _frozen_context(), _prepared_request()
    observed = []
    policies = []
    tools_seen = []
    original_classify = parent.classify_voice_speculation
    original_init = core.SpeculativeTurnCoordinator.__init__
    original_prepared = core.SpeculativeTurnCoordinator._on_attempt_dispatch_prepared
    original_tool = core.SpeculativeTurnCoordinator._on_tool_request

    def classify(**kwargs):
        assert kwargs["frozen_session_context"] is context
        assert kwargs["prepared_request"] is request
        observed.append(kwargs)
        return original_classify(**kwargs)

    def init(instance, **kwargs):
        assert "frozen_session_context" not in kwargs
        assert "prepared_request" not in kwargs
        if kwargs.get("prepared_policy") is not None:
            policies.append(kwargs["prepared_policy"])
        original_init(instance, **kwargs)

    async def prepared(instance, event):
        policies.append(event)
        await original_prepared(instance, event)

    def tool(instance, event):
        assert type(event) is types.AttemptToolPending
        assert not hasattr(event, "tool_calls")
        tools_seen.append(event)
        original_tool(instance, event)

    monkeypatch.setattr(parent, "classify_voice_speculation", classify)
    monkeypatch.setattr(core.SpeculativeTurnCoordinator, "__init__", init)
    monkeypatch.setattr(
        core.SpeculativeTurnCoordinator, "_on_attempt_dispatch_prepared", prepared
    )
    monkeypatch.setattr(core.SpeculativeTurnCoordinator, "_on_tool_request", tool)
    scheduler, effects = _Scheduler(), _Effects()
    kwargs = (
        {"deferred_attempt_preparation": True}
        if deferred
        else dict(
            frozen_session_context=context,
            prepared_request=request,
            requires_citation_creation=governed,
        )
    )
    coordinator = parent.SpeculativeTurnCoordinator(
        effects=effects, scheduler=scheduler, **kwargs
    )
    try:
        await coordinator.submit(_speech(1, started_ns=0))
        turn_id = coordinator.snapshot.turn_id
        await coordinator.submit(_revision(turn_id, 1, "hello", 10_000_000))
        scheduler.advance_ms(700)
        await coordinator.flush()
        if deferred:
            epoch = coordinator.snapshot.current_attempt_epoch
            await coordinator.submit(
                parent.AttemptDispatchPrepared(
                    epoch,
                    context,
                    request,
                    requires_citation_creation=governed,
                )
            )
        assert len(observed) == len(policies) == 1
        policy = policies[0]
        assert type(policy) is types.AttemptDispatchPrepared
        assert type(policy.request_handle) is types.VoiceRequestHandle
        assert type(policy.turn_context_handle) is types.VoiceTurnContextHandle
        if not governed:
            from tldw_chatbook.Chat.console_voice_attempts import (
                VoiceAttemptToolRequest,
            )

            await coordinator.submit(
                VoiceAttemptToolRequest(
                    coordinator.snapshot.current_attempt_epoch,
                    ({"private_tool_arguments": "must stay in parent"},),
                )
            )
        scheduler.advance_ms(1300)
        await coordinator.flush()
        assert effects.accepted_calls == [("hello", context)]
        assert effects.accepted_calls[0][1] is context
        assert len(tools_seen) == (0 if governed else 1)
        if deferred:
            assert coordinator._parent_effects._prepared is None
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_promotion_adapter_projects_exact_outcomes_without_hidden_commit():
    import asyncio
    from Tests.Chat.test_console_speculative_voice import _FakeEffects
    from tldw_chatbook.Chat import console_speculative_voice as parent
    from tldw_chatbook.Chat.console_voice_promotion import (
        VoicePromotionOutcome,
        VoicePromotionOutcomeStatus,
    )

    types = _types()
    effects = _FakeEffects()
    coordinator = parent.SpeculativeTurnCoordinator(effects=effects)
    kwargs = dict(
        turn_id="turn",
        attempt_epoch=1,
        transcript="hello",
        assistant_text="reply",
        terminal_boundary_ns=None,
    )
    for result, expected in (
        (None, types.VoiceTerminalDisposition.PROMOTED),
        (
            VoicePromotionOutcome(
                VoicePromotionOutcomeStatus.PROMOTED, "promotion", "session"
            ),
            types.VoiceTerminalDisposition.PROMOTED,
        ),
        (
            VoicePromotionOutcome(
                VoicePromotionOutcomeStatus.RECOVERY, "promotion", "session"
            ),
            types.VoiceTerminalDisposition.RECOVERY,
        ),
        (
            SimpleNamespace(status=VoicePromotionOutcomeStatus.PROMOTED),
            types.VoiceTerminalDisposition.FAILED,
        ),
        (True, types.VoiceTerminalDisposition.FAILED),
    ):
        effects.promotion_future = result
        assert coordinator._effects.promote(**kwargs) is expected
        future = asyncio.get_running_loop().create_future()
        effects.promotion_future = future
        mapped = coordinator._effects.promote(**kwargs)
        future.set_result(result)
        assert await mapped is expected
    await coordinator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["clean", True, object()])
async def test_unknown_cleanup_receipt_suspends_the_draft(outcome):
    from Tests.Chat.test_console_speculative_voice import (
        _coordinator,
        _speech,
        _start_attempt,
    )

    coordinator, scheduler, effects = await _coordinator(auto_cleanup=False)
    try:
        _, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(_speech(2, started_ns=800_000_000))
        effects.complete_cleanup(epoch, outcome)
        await coordinator.flush()
        assert coordinator.snapshot.failure_class == "voice_cleanup_stuck"
        assert coordinator.snapshot.speculation_suspended
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_parent_provider_failure_is_projected_before_core_reduction(monkeypatch):
    from Tests.Chat.test_console_speculative_voice import _coordinator, _start_attempt
    from tldw_chatbook.Chat.console_voice_attempts import ProviderAttemptFailed

    types = _types()
    core = import_module("tldw_chatbook.Audio.voice_turn_coordinator")
    original = core.SpeculativeTurnCoordinator._on_provider_failed
    projected = []

    async def failed(instance, event):
        assert type(event) is types.ProviderAttemptFailed
        projected.append(event)
        await original(instance, event)

    monkeypatch.setattr(core.SpeculativeTurnCoordinator, "_on_provider_failed", failed)
    coordinator, scheduler, effects = await _coordinator()
    try:
        _, epoch = await _start_attempt(coordinator, scheduler)
        event = ProviderAttemptFailed(epoch, "RuntimeError")
        await coordinator.submit(event)
        assert len(projected) == 1
        assert projected[0] is not event
        assert coordinator.snapshot.failure_class == "RuntimeError"
        assert effects.drafts[-1][-1] == "provider_failed"
    finally:
        await coordinator.close()
