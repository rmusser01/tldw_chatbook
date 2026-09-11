"""Deterministic app-lifetime ownership races for speculative voice promotion."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_voice_promotion import (
    CompletedVoicePairCommit,
    ConsoleSessionBindingOrigin,
    VoicePromotionClaimStatus,
    VoicePromotionContext,
    VoicePromotionOutcomeStatus,
    VoicePromotionOwner,
    derive_voice_promotion_identities,
)


def _voice_context(
    store: ConsoleChatStore,
    session_id: str,
    *,
    promotion_id: str = "promotion-1",
) -> VoicePromotionContext:
    session = next(item for item in store.sessions() if item.id == session_id)
    return VoicePromotionContext(
        promotion_id=promotion_id,
        attempt_id="attempt-1",
        origin=ConsoleSessionBindingOrigin(
            session_id=session_id,
            session_incarnation=store._settings_session_incarnations[session_id],
            persisted_conversation_id=session.persisted_conversation_id,
            conversation_binding_revision=session.conversation_binding_revision,
        ),
        expected_native_leaf_id=store.active_leaf(session_id),
        expected_persisted_leaf_id=None,
        user_text="protected user text",
        assistant_text="protected assistant text",
        usage_json=None,
        terminal_boundary_id="terminal-1",
        capture_eligible_at_dispatch=False,
    )


def _durable_store() -> tuple[ConsoleChatStore, str]:
    store = ConsoleChatStore()
    session = store.create_session()
    session.persisted_conversation_id = "conversation-1"

    def commit_completed_voice_pair(*, destination, context):
        identities = derive_voice_promotion_identities(context.promotion_id)
        return CompletedVoicePairCommit(
            conversation_id=destination.persisted_conversation_id,
            user_message_id=identities.user_message_id,
            assistant_message_id=identities.assistant_message_id,
            terminal_receipt_id=identities.terminal_receipt_id,
            active_leaf_message_id=identities.assistant_message_id,
        )

    store.persistence = SimpleNamespace(
        commit_completed_voice_pair=commit_completed_voice_pair
    )
    return store, session.id


class _VoiceView:
    def console_view_hooks(self) -> dict:
        return {}


async def test_runtime_owner_claims_and_publishes_temporary_pair() -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    context = _voice_context(store, session.id)
    owner = VoicePromotionOwner(lambda: store)

    claim = owner.try_claim(context)
    outcome = await owner.promote(claim)

    assert claim.status is VoicePromotionClaimStatus.CLAIMED
    assert outcome.status is VoicePromotionOutcomeStatus.PROMOTED
    assert store.active_leaf(session.id) is not None
    assert await owner.wait_for_session(session.id, timeout=0) is True
    assert "protected user text" not in repr(claim)
    assert "protected assistant text" not in repr(outcome)


@pytest.mark.asyncio
async def test_quit_fence_wins_before_a_late_claim() -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    owner = VoicePromotionOwner(lambda: store)
    token = owner.begin_quit()

    claim = owner.try_claim(_voice_context(store, session.id))

    assert claim.status is VoicePromotionClaimStatus.QUIT_FENCED
    owner.abort_quit(token)
    accepted = owner.try_claim(
        replace(_voice_context(store, session.id), promotion_id="promotion-2")
    )
    assert accepted.status is VoicePromotionClaimStatus.CLAIMED
    await owner.promote(accepted)


def test_equal_but_nonidentical_quit_capabilities_cannot_release_or_consume() -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    owner = VoicePromotionOwner(lambda: store)
    token = owner.begin_quit()
    forged_token = replace(token)

    owner.abort_quit(forged_token)
    assert owner.try_claim(_voice_context(store, session.id)).status is (
        VoicePromotionClaimStatus.QUIT_FENCED
    )

    permit = owner.seal_quiescent(token)
    with pytest.raises(RuntimeError, match="stale"):
        owner.consume_quit_permit(replace(permit))
    owner.consume_quit_permit(permit)


@pytest.mark.asyncio
async def test_session_wait_registers_against_the_observed_owner_revision(
    monkeypatch,
) -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    owner = VoicePromotionOwner(lambda: store)
    claim = owner.try_claim(_voice_context(store, session.id))
    original_wait = owner._wait_for_change

    async def settle_between_observation_and_registration(timeout, *args, **kwargs):
        await owner.promote(claim)
        return await original_wait(timeout, *args, **kwargs)

    monkeypatch.setattr(
        owner, "_wait_for_change", settle_between_observation_and_registration
    )

    assert await owner.wait_for_session(session.id, timeout=0) is True


@pytest.mark.asyncio
async def test_quit_wait_registers_against_the_observed_owner_revision(
    monkeypatch,
) -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    owner = VoicePromotionOwner(lambda: store)
    claim = owner.try_claim(_voice_context(store, session.id))
    token = owner.begin_quit()
    original_wait = owner._wait_for_change

    async def settle_between_observation_and_registration(timeout, *args, **kwargs):
        await owner.promote(claim)
        return await original_wait(timeout, *args, **kwargs)

    monkeypatch.setattr(
        owner, "_wait_for_change", settle_between_observation_and_registration
    )

    assert await owner.wait_for_quiescence(token, timeout=0) is True


@pytest.mark.asyncio
async def test_cancelled_promote_caller_does_not_cancel_owner_rooted_task() -> None:
    store, session_id = _durable_store()
    runner_started = asyncio.Event()
    release_runner = asyncio.Event()

    async def blocked_runner(call):
        runner_started.set()
        await release_runner.wait()
        return call()

    owner = VoicePromotionOwner(lambda: store, sync_runner=blocked_runner)
    claim = owner.try_claim(_voice_context(store, session_id))
    caller = asyncio.create_task(owner.promote(claim))
    await runner_started.wait()

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert owner.status.active_claim_count == 1
    assert await owner.wait_for_session(session_id, timeout=0) is False
    release_runner.set()
    assert await owner.wait_for_session(session_id, timeout=1) is True


@pytest.mark.asyncio
async def test_claimed_blocked_runner_survives_navigation_close_and_quit_timeouts() -> (
    None
):
    store, session_id = _durable_store()
    runner_started = asyncio.Event()
    release_runner = asyncio.Event()

    async def blocked_runner(call):
        runner_started.set()
        await release_runner.wait()
        return call()

    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime._voice_promotion_owner = VoicePromotionOwner(
        lambda: store,
        sync_runner=blocked_runner,
    )
    owner = runtime.voice_promotion_owner
    claim = owner.try_claim(_voice_context(store, session_id))
    promotion = asyncio.create_task(owner.promote(claim))
    await runner_started.wait()

    view = _VoiceView()
    generation = runtime.attach_view(view)
    assert runtime.detach_view(view, generation) is True
    assert (
        await runtime.close_session(
            session_id,
            expected_revision=0,
            timeout_seconds=0,
        )
        is None
    )
    token = owner.begin_quit()
    assert await owner.wait_for_quiescence(token, timeout=0) is False
    assert promotion.done() is False
    assert owner._waiters == set()

    owner.abort_quit(token)
    release_runner.set()
    assert (await promotion).status is VoicePromotionOutcomeStatus.PROMOTED


@pytest.mark.asyncio
async def test_prepublication_failure_needs_no_private_store_recovery_seam() -> None:
    store, session_id = _durable_store()

    class PublicStoreOnly:
        def __getattr__(self, name):
            if name in {
                "_voice_promotion_lock",
                "_install_voice_promotion_recovery_for_live_lease",
            }:
                raise AssertionError(f"private store access: {name}")
            return getattr(store, name)

    async def failing_runner(_call):
        raise OSError("database unavailable")

    public_store = PublicStoreOnly()
    owner = VoicePromotionOwner(lambda: public_store, sync_runner=failing_runner)
    context = _voice_context(store, session_id)
    claim = owner.try_claim(context)

    outcome = await owner.promote(claim)

    assert outcome.status is VoicePromotionOutcomeStatus.RECOVERY
    assert owner.status.recovery_count == 1
    assert await owner.wait_for_session(session_id, timeout=0) is False
    assert "protected user text" not in repr(outcome)
    assert "protected assistant text" not in repr(owner._recoveries)
    assert (
        store.try_claim_voice_promotion(
            replace(context, promotion_id="promotion-2")
        ).status.value
        == "transient_contention"
    )


@pytest.mark.asyncio
async def test_owner_retry_reuses_exact_recovery_and_unblocks_close_and_quit() -> None:
    store, session_id = _durable_store()
    failing = True

    async def controlled_runner(call):
        if failing:
            raise OSError("database unavailable")
        return call()

    owner = VoicePromotionOwner(lambda: store, sync_runner=controlled_runner)
    claim = owner.try_claim(_voice_context(store, session_id))

    assert (await owner.promote(claim)).status is VoicePromotionOutcomeStatus.RECOVERY
    handle = owner.recovery_for_session(session_id)
    assert handle is not None
    with pytest.raises(RuntimeError, match="stale"):
        owner.discard_recovery(replace(handle))

    failing = False
    outcome = await owner.retry_recovery(handle)

    assert outcome.status is VoicePromotionOutcomeStatus.PROMOTED
    assert owner.status.recovery_count == 0
    assert owner.recovery_for_session(session_id) is None
    assert await owner.wait_for_session(session_id, timeout=0) is True
    token = owner.begin_quit()
    assert await owner.wait_for_quiescence(token, timeout=0) is True
    owner.abort_quit(token)
    with pytest.raises(RuntimeError, match="stale"):
        await owner.retry_recovery(handle)


@pytest.mark.asyncio
async def test_owner_rebranch_reuses_frozen_pair_on_selected_native_leaf(
    monkeypatch,
) -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    original_parent = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="original branch",
    )
    store.set_active_leaf(session.id, None)
    selected_parent = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="selected branch",
    )
    store.set_active_leaf(session.id, original_parent.id)
    context = _voice_context(store, session.id)
    original_publish = store.publish_temporary_voice_pair
    fail_first = True

    def controlled_publish(lease, frozen_context):
        nonlocal fail_first
        if fail_first:
            fail_first = False
            raise RuntimeError("native publication unavailable")
        return original_publish(lease, frozen_context)

    monkeypatch.setattr(store, "publish_temporary_voice_pair", controlled_publish)
    owner = VoicePromotionOwner(lambda: store)
    original_rebranch = store.rebranch_voice_promotion_recovery
    owner_lock_states: list[bool] = []

    def observe_rebranch_lock(lease):
        owner_lock_states.append(owner._lock._is_owned())
        return original_rebranch(lease)

    monkeypatch.setattr(
        store,
        "rebranch_voice_promotion_recovery",
        observe_rebranch_lock,
    )
    claim = owner.try_claim(context)
    assert (await owner.promote(claim)).status is VoicePromotionOutcomeStatus.RECOVERY
    handle = owner.recovery_for_session(session.id)
    assert handle is not None

    with pytest.raises(RuntimeError, match="voice promotion owns"):
        store.set_active_leaf(session.id, selected_parent.id)
    with pytest.raises(RuntimeError, match="stale"):
        owner.select_recovery_leaf(replace(handle), selected_parent.id)
    assert owner.select_recovery_leaf(handle, selected_parent.id) is True
    outcome = await owner.rebranch_recovery(handle)

    assert outcome.status is VoicePromotionOutcomeStatus.PROMOTED
    identities = derive_voice_promotion_identities(context.promotion_id)
    user = store._nodes_by_session[session.id][identities.user_message_id]
    assistant = store._nodes_by_session[session.id][identities.assistant_message_id]
    assert user.content == context.user_text
    assert assistant.content == context.assistant_text
    assert store._native_parent_by_message[user.id] == selected_parent.id
    assert store._native_parent_by_message[assistant.id] == user.id
    assert store.active_leaf(session.id) == assistant.id
    assert owner_lock_states == [False]
    assert owner.recovery_for_session(session.id) is None
    with pytest.raises(RuntimeError, match="stale"):
        await owner.rebranch_recovery(handle)
    with pytest.raises(RuntimeError, match="stale"):
        owner.select_recovery_leaf(handle, original_parent.id)


@pytest.mark.asyncio
async def test_owner_rebranch_uses_same_durable_transaction_and_stable_identities() -> (
    None
):
    store = ConsoleChatStore()
    session = store.create_session()
    first_parent = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="first durable branch",
    )
    store._nodes_by_session[session.id][
        first_parent.id
    ].persisted_message_id = "persisted-first"
    store.set_active_leaf(session.id, None)
    selected_parent = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="selected durable branch",
    )
    store._nodes_by_session[session.id][
        selected_parent.id
    ].persisted_message_id = "persisted-selected"
    session.persisted_conversation_id = "conversation-1"
    store.set_active_leaf(session.id, first_parent.id)
    context = _voice_context(store, session.id)
    context = replace(
        context,
        expected_persisted_leaf_id="persisted-first",
    )
    calls: list[tuple[object, VoicePromotionContext]] = []
    effects: list[str] = []

    def commit_completed_voice_pair(*, destination, context):
        effects.append("commit")
        calls.append((destination, context))
        identities = derive_voice_promotion_identities(context.promotion_id)
        return CompletedVoicePairCommit(
            conversation_id=destination.persisted_conversation_id,
            user_message_id=identities.user_message_id,
            assistant_message_id=identities.assistant_message_id,
            terminal_receipt_id=identities.terminal_receipt_id,
            active_leaf_message_id=identities.assistant_message_id,
        )

    store.persistence = SimpleNamespace(
        commit_completed_voice_pair=commit_completed_voice_pair
    )
    failures_before_commit = 2

    async def controlled_runner(call):
        nonlocal failures_before_commit
        if failures_before_commit:
            failures_before_commit -= 1
            raise OSError("database unavailable")
        return call()

    owner = VoicePromotionOwner(lambda: store, sync_runner=controlled_runner)
    claim = owner.try_claim(context)
    assert (await owner.promote(claim)).status is VoicePromotionOutcomeStatus.RECOVERY
    handle = owner.recovery_for_session(session.id)
    assert handle is not None

    assert owner.select_recovery_leaf(handle, selected_parent.id) is True
    outcome = await owner.rebranch_recovery(handle)

    assert outcome.status is VoicePromotionOutcomeStatus.PROMOTED
    assert effects == ["commit"]
    assert len(calls) == 1
    destination, committed_context = calls[0]
    assert destination.expected_persisted_leaf_id == "persisted-selected"
    assert committed_context.promotion_id == context.promotion_id
    assert committed_context.user_text == context.user_text
    assert committed_context.assistant_text == context.assistant_text
    identities = derive_voice_promotion_identities(context.promotion_id)
    user = store._nodes_by_session[session.id][identities.user_message_id]
    assert user.parent_message_id == "persisted-selected"
    assert store._native_parent_by_message[user.id] == selected_parent.id


@pytest.mark.asyncio
async def test_rebranch_scheduler_failure_retains_usable_exact_recovery(
    monkeypatch,
) -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    original_publish = store.publish_temporary_voice_pair
    fail_first = True

    def controlled_publish(lease, context):
        nonlocal fail_first
        if fail_first:
            fail_first = False
            raise RuntimeError("native publication unavailable")
        return original_publish(lease, context)

    monkeypatch.setattr(store, "publish_temporary_voice_pair", controlled_publish)
    owner = VoicePromotionOwner(lambda: store)
    claim = owner.try_claim(_voice_context(store, session.id))
    assert (await owner.promote(claim)).status is VoicePromotionOutcomeStatus.RECOVERY
    handle = owner.recovery_for_session(session.id)
    assert handle is not None

    class _FailingLoop:
        def create_task(self, _coroutine, *, name=None):
            raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_voice_promotion.asyncio.get_running_loop",
        lambda: _FailingLoop(),
    )
    with pytest.raises(RuntimeError, match="rebranch is unavailable"):
        await owner.rebranch_recovery(handle)
    monkeypatch.undo()

    assert owner.recovery_for_session(session.id) is handle
    assert owner.status.recovery_count == 1
    assert (await owner.retry_recovery(handle)).status is (
        VoicePromotionOutcomeStatus.PROMOTED
    )
    assert owner.recovery_for_session(session.id) is None


@pytest.mark.asyncio
async def test_owner_explicit_discard_releases_exact_recovery_and_quit_veto(
    monkeypatch,
) -> None:
    store, session_id = _durable_store()

    async def failing_runner(_call):
        raise OSError("database unavailable")

    owner = VoicePromotionOwner(lambda: store, sync_runner=failing_runner)
    claim = owner.try_claim(_voice_context(store, session_id))
    assert (await owner.promote(claim)).status is VoicePromotionOutcomeStatus.RECOVERY
    handle = owner.recovery_for_session(session_id)
    assert handle is not None
    original_recovery = store.voice_promotion_recovery
    owner_lock_states: list[bool] = []

    def observe_owner_lock(lease):
        owner_lock_states.append(owner._lock._is_owned())
        return original_recovery(lease)

    monkeypatch.setattr(store, "voice_promotion_recovery", observe_owner_lock)

    assert owner.discard_recovery(handle) is True

    assert owner_lock_states == [False]
    assert owner.status.recovery_count == 0
    assert await owner.wait_for_session(session_id, timeout=0) is True
    token = owner.begin_quit()
    assert await owner.wait_for_quiescence(token, timeout=0) is True
    owner.abort_quit(token)
    with pytest.raises(RuntimeError, match="stale"):
        owner.discard_recovery(handle)


@pytest.mark.asyncio
async def test_owner_recovery_decision_remains_fenced_through_final_bookkeeping(
    monkeypatch,
) -> None:
    store, session_id = _durable_store()

    async def failing_runner(_call):
        raise OSError("database unavailable")

    owner = VoicePromotionOwner(lambda: store, sync_runner=failing_runner)
    claim = owner.try_claim(_voice_context(store, session_id))
    assert (await owner.promote(claim)).status is VoicePromotionOutcomeStatus.RECOVERY
    handle = owner.recovery_for_session(session_id)
    assert handle is not None

    class PausingLock:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self._armed = False
            self.paused = threading.Event()
            self.resume = threading.Event()

        def arm_once(self) -> None:
            if not self.paused.is_set():
                self._armed = True

        def __enter__(self):
            self._lock.acquire()
            return self

        def __exit__(self, *_args) -> None:
            should_pause = self._armed
            self._armed = False
            self._lock.release()
            if should_pause:
                self.paused.set()
                self.resume.wait(timeout=1)

    lock = PausingLock()
    owner._lock = lock
    original_recovery = store.voice_promotion_recovery

    def arm_after_store_release(lease):
        result = original_recovery(lease)
        lock.arm_once()
        return result

    monkeypatch.setattr(store, "voice_promotion_recovery", arm_after_store_release)
    first = asyncio.create_task(asyncio.to_thread(owner.discard_recovery, handle))
    assert await asyncio.to_thread(lock.paused.wait, 1)
    competitor_was_rejected = False
    try:
        owner.discard_recovery(handle)
    except RuntimeError:
        competitor_was_rejected = True
    finally:
        lock.resume.set()

    assert await first is True
    assert competitor_was_rejected is True


def test_stale_capabilities_cannot_release_a_later_fence_and_permit_is_one_shot() -> (
    None
):
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    owner = VoicePromotionOwner(lambda: store)

    stale_token = owner.begin_quit()
    owner.abort_quit(stale_token)
    second_token = owner.begin_quit()
    owner.abort_quit(stale_token)
    assert owner.try_claim(_voice_context(store, session.id)).status is (
        VoicePromotionClaimStatus.QUIT_FENCED
    )

    stale_permit = owner.seal_quiescent(second_token)
    owner.abort_quit(stale_permit)
    final_token = owner.begin_quit()
    owner.abort_quit(stale_permit)
    final_permit = owner.seal_quiescent(final_token)
    owner.consume_quit_permit(final_permit)

    with pytest.raises(RuntimeError, match="stale"):
        owner.consume_quit_permit(final_permit)
    owner.abort_quit(final_permit)
    assert owner.try_claim(_voice_context(store, session.id)).status is (
        VoicePromotionClaimStatus.QUIT_FENCED
    )


@pytest.mark.asyncio
async def test_claim_binds_the_exact_store_that_issued_its_lease() -> None:
    original = ConsoleChatStore()
    original_session = original.create_session(ephemeral=True)
    replacement = ConsoleChatStore()
    replacement_session = replacement.create_session(ephemeral=True)
    selected = [original]
    owner = VoicePromotionOwner(lambda: selected[0])

    claim = owner.try_claim(_voice_context(original, original_session.id))
    selected[0] = replacement
    outcome = await owner.promote(claim)

    assert outcome.status is VoicePromotionOutcomeStatus.PROMOTED
    assert original.active_leaf(original_session.id) is not None
    assert replacement.active_leaf(replacement_session.id) is None
    assert owner.status.recovery_count == 0


@pytest.mark.asyncio
async def test_claim_is_rooted_before_return_even_when_caller_never_promotes() -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    owner = VoicePromotionOwner(lambda: store)
    claim_returned = asyncio.Event()
    never_resume = asyncio.Event()

    async def claim_then_abandon() -> None:
        claim = owner.try_claim(_voice_context(store, session.id))
        assert claim.status is VoicePromotionClaimStatus.CLAIMED
        claim_returned.set()
        await never_resume.wait()

    caller = asyncio.create_task(claim_then_abandon())
    await claim_returned.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert await owner.wait_for_session(session.id, timeout=1) is True
    assert store.active_leaf(session.id) is not None


@pytest.mark.asyncio
async def test_navigation_immediately_after_claim_cannot_orphan_rooted_task() -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    view = _VoiceView()
    generation = runtime.attach_view(view)

    claim = runtime.voice_promotion_owner.try_claim(_voice_context(store, session.id))
    assert claim.status is VoicePromotionClaimStatus.CLAIMED
    assert runtime.detach_view(view, generation) is True

    assert (
        await runtime.voice_promotion_owner.wait_for_session(
            session.id,
            timeout=1,
        )
        is True
    )
    assert store.active_leaf(session.id) is not None


def test_scheduler_failure_aborts_exact_store_lease(monkeypatch) -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    context = _voice_context(store, session.id)
    owner = VoicePromotionOwner(lambda: store)

    class _FailingLoop:
        def create_task(self, _coroutine, *, name=None):
            raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_voice_promotion.asyncio.get_running_loop",
        lambda: _FailingLoop(),
    )

    claim = owner.try_claim(context)

    assert claim.status is VoicePromotionClaimStatus.UNAVAILABLE
    assert owner.status.active_claim_count == 0
    assert store.try_claim_voice_promotion(context).status.value == "claimed"


@pytest.mark.asyncio
async def test_session_close_fence_serializes_both_claim_orders() -> None:
    first_store = ConsoleChatStore()
    first_session = first_store.create_session(ephemeral=True)
    other_session = first_store.create_session(ephemeral=True)
    first_owner = VoicePromotionOwner(lambda: first_store)

    close_first = first_owner.begin_session_close(first_session.id)
    assert (
        first_owner.try_claim(_voice_context(first_store, first_session.id)).status
        is VoicePromotionClaimStatus.CLOSE_FENCED
    )
    other_claim = first_owner.try_claim(_voice_context(first_store, other_session.id))
    assert other_claim.status is VoicePromotionClaimStatus.CLAIMED
    await first_owner.promote(other_claim)
    first_owner.abort_session_close(close_first)

    second_store, second_session_id = _durable_store()
    runner_started = asyncio.Event()
    release_runner = asyncio.Event()

    async def blocked_runner(call):
        runner_started.set()
        await release_runner.wait()
        return call()

    second_owner = VoicePromotionOwner(
        lambda: second_store,
        sync_runner=blocked_runner,
    )
    claim_first = second_owner.try_claim(
        _voice_context(second_store, second_session_id)
    )
    await runner_started.wait()
    close_after_claim = second_owner.begin_session_close(second_session_id)

    assert await second_owner.wait_for_session(second_session_id, timeout=0) is False
    assert (
        second_owner.try_claim(
            replace(
                _voice_context(second_store, second_session_id),
                promotion_id="promotion-2",
            )
        ).status
        is VoicePromotionClaimStatus.CLOSE_FENCED
    )
    release_runner.set()
    assert await second_owner.promote(claim_first)
    assert await second_owner.wait_for_session(second_session_id, timeout=1) is True
    second_owner.complete_session_close(close_after_claim)


def test_stale_session_close_token_cannot_release_a_later_fence() -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    owner = VoicePromotionOwner(lambda: store)

    stale = owner.begin_session_close(session.id)
    owner.abort_session_close(stale)
    current = owner.begin_session_close(session.id)
    owner.abort_session_close(stale)

    assert owner.try_claim(_voice_context(store, session.id)).status is (
        VoicePromotionClaimStatus.CLOSE_FENCED
    )
    owner.abort_session_close(current)


@pytest.mark.asyncio
async def test_begin_dispose_rejects_stale_permit_while_recovery_is_live() -> None:
    store, session_id = _durable_store()

    async def failing_runner(_call):
        raise OSError("database unavailable")

    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime._voice_promotion_owner = VoicePromotionOwner(
        lambda: store,
        sync_runner=failing_runner,
    )
    owner = runtime.voice_promotion_owner
    stale_token = owner.begin_quit()
    stale_permit = owner.seal_quiescent(stale_token)
    owner.abort_quit(stale_permit)
    claim = owner.try_claim(_voice_context(store, session_id))

    assert (await owner.promote(claim)).status is VoicePromotionOutcomeStatus.RECOVERY
    assert owner.status.recovery_count == 1
    recovery_token = owner.begin_quit()
    assert await owner.wait_for_quiescence(recovery_token, timeout=0) is False
    with pytest.raises(RuntimeError, match="not quiescent"):
        owner.seal_quiescent(recovery_token)
    owner.abort_quit(recovery_token)

    with pytest.raises(RuntimeError, match="stale"):
        runtime.begin_dispose(voice_promotion_permit=stale_permit)
    assert runtime._disposed is False
    assert owner.status.recovery_count == 1
