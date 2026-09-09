"""Voice owners must cost nothing until used and retain one runtime identity."""

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


@pytest.mark.asyncio
async def test_unused_voice_owners_are_not_constructed_for_close_or_disposal(
    monkeypatch,
):
    import tldw_chatbook.Chat.console_voice_promotion as promotion
    import tldw_chatbook.Chat.console_voice_supervisor as supervisor

    def unexpected(*args, **kwargs):
        pytest.fail("unused voice owner was constructed")

    monkeypatch.setattr(promotion, "VoicePromotionOwner", unexpected)
    monkeypatch.setattr(supervisor, "VoiceDispatchSupervisor", unexpected)
    runtime = ConsoleRuntime(SimpleNamespace())
    assert runtime._voice_promotion_owner is None
    assert runtime._voice_dispatch_supervisor is None

    async def close(session_id, *, expected_revision, timeout_seconds):
        assert session_id == "ordinary-session" and expected_revision == 7
        return "closed"

    monkeypatch.setattr(runtime, "_close_session_after_voice_drain", close)
    assert (
        await runtime.close_session("ordinary-session", expected_revision=7) == "closed"
    )
    await runtime.dispose()
    for name in ("voice_promotion_owner", "voice_dispatch_supervisor"):
        with pytest.raises(RuntimeError, match="voice_runtime_disposed"):
            getattr(runtime, name)


@pytest.mark.asyncio
async def test_first_voice_access_constructs_each_owner_once_and_keeps_identity(
    monkeypatch,
):
    import tldw_chatbook.Chat.console_voice_promotion as promotion
    import tldw_chatbook.Chat.console_voice_supervisor as supervisor

    built = []
    for module, name in (
        (promotion, "VoicePromotionOwner"),
        (supervisor, "VoiceDispatchSupervisor"),
    ):
        original = getattr(module, name)

        def record(*args, _original=original, _name=name, **kwargs):
            built.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, record)

    runtime = ConsoleRuntime(SimpleNamespace())
    assert built == []
    owner = runtime.voice_promotion_owner
    dispatch = runtime.voice_dispatch_supervisor
    assert built == ["VoicePromotionOwner", "VoiceDispatchSupervisor"]
    assert runtime.voice_promotion_owner is owner
    assert runtime.voice_dispatch_supervisor is dispatch
    await runtime.dispose()
    assert runtime._voice_promotion_owner is owner
    assert runtime._voice_dispatch_supervisor is dispatch


@pytest.mark.asyncio
async def test_first_owner_inherits_pending_close_and_releases_its_exact_fence(
    monkeypatch,
):
    from Tests.Chat.test_console_speculative_voice_promotion_races import _voice_context
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_voice_promotion import VoicePromotionClaimStatus

    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    started, release = asyncio.Event(), asyncio.Event()

    async def close(*args, **kwargs):
        started.set()
        await release.wait()
        raise RuntimeError("close refused")

    monkeypatch.setattr(runtime, "_close_session_after_voice_drain", close)
    closing = asyncio.create_task(
        runtime.close_session(session.id, expected_revision=0)
    )
    await started.wait()
    try:
        owner = runtime.voice_promotion_owner
        assert (
            owner.try_claim(_voice_context(store, session.id)).status
            is VoicePromotionClaimStatus.CLOSE_FENCED
        )
    finally:
        release.set()
        with pytest.raises(RuntimeError, match="close refused"):
            await closing
    claim = owner.try_claim(_voice_context(store, session.id))
    assert claim.status is VoicePromotionClaimStatus.CLAIMED
    await owner.promote(claim)
    await runtime.dispose()


@pytest.mark.asyncio
async def test_gateway_registry_is_lazy_and_preserves_injected_identity():
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway

    gateway = ConsoleProviderGateway()
    assert gateway._provisional_trace_registry is None
    registry = gateway.provisional_trace_registry
    assert gateway.provisional_trace_registry is registry
    injected = ConsoleProviderGateway(provisional_trace_registry=registry)
    assert injected.provisional_trace_registry is registry
    await gateway.aclose()
    await injected.aclose()
