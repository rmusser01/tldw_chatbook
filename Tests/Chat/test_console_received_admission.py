"""Exact same-slot received admission and task-local promotion ownership."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import pytest

from Tests.Chat.test_console_automatic_library_preparation import _preparation
from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsolePreparationTransition,
    ConsoleTurnPreparationState,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def received_store():
    store = ConsoleChatStore()
    session = store.create_session(session_id="received-session", ephemeral=True)
    try:
        yield store, session
    finally:
        for live in store.sessions():
            store.close_session(live.id)
        store.end_app_runtime()


def _complete(session, *, preparation_id="received-preparation", **kwargs):
    return replace(
        _preparation(
            session_id=session.id,
            preparation_id=preparation_id,
            **kwargs,
        ),
        ephemeral=session.ephemeral,
    )


def test_received_claim_blocks_duplicate_and_direct_preparation(received_store):
    store, session = received_store
    claim = store.claim_received_turn(session.id, "request-one")
    preparation = _complete(session)

    assert claim is not None
    assert claim.session_id == session.id
    assert claim.request_id == "request-one"
    assert claim.sealed is False
    assert store.received_turn_for_session(session.id) is claim
    assert store.preparation_for_session(session.id) is None
    assert store.preparation_by_id(preparation.preparation_id) is None
    assert store.claim_received_turn(session.id, "request-two") is None
    assert store.begin_preparation(preparation) is None
    assert store.received_turn_for_session(session.id) is claim


@pytest.mark.parametrize(
    "state,pause_kind",
    [
        (ConsoleTurnPreparationState.PREPARING, None),
        (ConsoleTurnPreparationState.PAUSED, ConsolePreparationPauseKind.RETRIEVAL),
        (ConsoleTurnPreparationState.ACCEPTED, None),
    ],
)
def test_live_preparation_refuses_received_claim(received_store, state, pause_kind):
    store, session = received_store
    preparation = _complete(session, state=state, pause_kind=pause_kind)
    assert store.begin_preparation(preparation) is preparation

    assert store.claim_received_turn(session.id, "request-one") is None
    assert store.preparation_for_session(session.id) is preparation
    assert store.received_turn_for_session(session.id) is None


def test_received_claim_cannot_be_bypassed_by_old_id_idempotency(received_store):
    store, session = received_store
    old = _complete(session, state=ConsoleTurnPreparationState.SETTLED)
    assert store.begin_preparation(old) is old
    claim = store.claim_received_turn(session.id, "request-one")
    assert claim is not None

    assert store.begin_preparation(old) is None
    assert store.preparation_by_id(old.preparation_id) is old
    assert store.preparation_for_session(session.id) is None
    assert store.received_turn_for_session(session.id) is claim


def test_exact_promotion_preserves_slot_and_late_claim_release_is_noop(received_store):
    store, session = received_store
    claim = store.claim_received_turn(session.id, "request-one")
    preparation = _complete(session)
    assert claim is not None

    assert store.promote_received_turn(claim, preparation) is preparation
    assert store.received_turn_for_session(session.id) is None
    assert store.preparation_for_session(session.id) is preparation
    assert store.preparation_by_id(preparation.preparation_id) is preparation
    assert store.claim_received_turn(session.id, "request-two") is None
    assert store.release_received_turn(claim) is False
    assert store.seal_received_turn(claim) is False
    assert store.promote_received_turn(claim, preparation) is None
    assert store.preparation_for_session(session.id) is preparation


def test_sealed_claim_remains_occupied_until_exact_release(received_store):
    store, session = received_store
    claim = store.claim_received_turn(session.id, "request-one")
    assert claim is not None

    assert store.seal_received_turn(claim) is True
    assert claim.sealed is True
    assert store.received_turn_for_session(session.id) is claim
    assert store.promote_received_turn(claim, _complete(session)) is None
    assert store.begin_preparation(_complete(session)) is None
    assert store.claim_received_turn(session.id, "request-two") is None
    assert store.release_received_turn(claim) is True
    assert store.received_turn_for_session(session.id) is None
    assert store.release_received_turn(claim) is False


def test_late_claim_cleanup_cannot_release_same_request_successor(received_store):
    store, session = received_store
    old = store.claim_received_turn(session.id, "request-one")
    assert old is not None
    assert store.release_received_turn(old) is True
    successor = store.claim_received_turn(session.id, "request-one")
    assert successor is not None
    assert successor is not old
    assert successor.generation != old.generation

    assert store.release_received_turn(old) is False
    assert store.seal_received_turn(old) is False
    assert store.promote_received_turn(old, _complete(session)) is None
    assert store.received_turn_for_session(session.id) is successor


def test_preparation_cleanup_and_cas_do_not_remove_received_owner(received_store):
    store, session = received_store
    claim = store.claim_received_turn(session.id, "request-one")
    assert claim is not None
    transition = ConsolePreparationTransition(
        preparation_id="received-preparation",
        expected_state=ConsoleTurnPreparationState.PREPARING,
        new_state=ConsoleTurnPreparationState.READY,
        pause_kind=None,
        new_attempt_id=None,
    )

    assert store.compare_and_set_preparation(session.id, transition) is None
    assert (
        store.cancel_preparation(
            session.id,
            "received-preparation",
            expected_state=ConsoleTurnPreparationState.PREPARING,
        )
        is None
    )
    assert (
        store.remove_preparation(
            session.id,
            "received-preparation",
            expected_states=frozenset({ConsoleTurnPreparationState.PREPARING}),
        )
        is None
    )
    assert store.received_turn_for_session(session.id) is claim


@pytest.mark.parametrize("displacement", ["binding", "ephemeral", "incarnation"])
def test_promotion_refuses_changed_session_witness(received_store, displacement):
    store, session = received_store
    claim = store.claim_received_turn(session.id, "request-one")
    preparation = _complete(session)
    assert claim is not None
    if displacement == "binding":
        session.conversation_binding_revision += 1
    elif displacement == "ephemeral":
        session.ephemeral = False
    else:
        store.close_session(session.id)
        store.create_session(session_id=session.id, ephemeral=True)

    assert store.received_turn_matches_session(claim) is False
    assert store.promote_received_turn(claim, preparation) is None
    assert store.preparation_by_id(preparation.preparation_id) is None
    if displacement == "incarnation":
        successor = store.claim_received_turn(session.id, "request-two")
        assert successor is not None
        assert store.release_received_turn(claim) is False
        assert store.received_turn_for_session(session.id) is successor
    else:
        assert store.received_turn_for_session(session.id) is claim


def test_foreign_store_or_session_cannot_consume_claim(received_store):
    store, session = received_store
    claim = store.claim_received_turn(session.id, "request-one")
    assert claim is not None
    other_store = ConsoleChatStore()
    other_session = other_store.create_session(session_id=session.id, ephemeral=True)
    other_claim = other_store.claim_received_turn(other_session.id, "request-two")
    assert other_claim is not None
    second = store.create_session(session_id="other-session", ephemeral=True)
    try:
        assert other_store.received_turn_matches_session(claim) is False
        assert (
            other_store.promote_received_turn(claim, _complete(other_session)) is None
        )
        assert other_store.release_received_turn(claim) is False
        assert other_store.seal_received_turn(claim) is False
        assert other_store.received_turn_for_session(other_session.id) is other_claim
        assert store.promote_received_turn(claim, _complete(second)) is None
        assert store.received_turn_for_session(session.id) is claim
        assert store.preparation_for_session(second.id) is None
    finally:
        other_store.close_session(other_session.id)
        other_store.end_app_runtime()


def test_concurrent_received_and_direct_admission_have_one_winner(received_store):
    store, session = received_store
    preparation = _complete(session)
    barrier = threading.Barrier(2)

    def received():
        barrier.wait(timeout=5)
        return store.claim_received_turn(session.id, "request-one")

    def direct():
        barrier.wait(timeout=5)
        return store.begin_preparation(preparation)

    with ThreadPoolExecutor(max_workers=2) as executor:
        received_future = executor.submit(received)
        direct_future = executor.submit(direct)
        claim = received_future.result(timeout=5)
        admitted = direct_future.result(timeout=5)
    assert (claim is not None) + (admitted is not None) == 1
    if claim is not None:
        assert store.received_turn_for_session(session.id) is claim
        assert store.preparation_for_session(session.id) is None
    else:
        assert admitted is preparation
        assert store.preparation_for_session(session.id) is preparation


def test_concurrent_seal_and_promotion_have_one_lock_winner(received_store):
    store, session = received_store
    claim = store.claim_received_turn(session.id, "request-one")
    assert claim is not None
    preparation = _complete(session)
    barrier = threading.Barrier(2)

    def seal():
        barrier.wait(timeout=5)
        return store.seal_received_turn(claim)

    def promote():
        barrier.wait(timeout=5)
        return store.promote_received_turn(claim, preparation)

    with ThreadPoolExecutor(max_workers=2) as executor:
        seal_future = executor.submit(seal)
        promotion_future = executor.submit(promote)
        sealed = seal_future.result(timeout=5)
        promoted = promotion_future.result(timeout=5)
    assert sealed is (promoted is None)
    if sealed:
        assert claim.sealed is True
        assert store.received_turn_for_session(session.id) is claim
        assert store.preparation_for_session(session.id) is None
    else:
        assert promoted is preparation
        assert store.preparation_for_session(session.id) is preparation


@pytest.mark.asyncio
async def test_task_binding_preserves_stale_owner_without_authorizing_child(
    received_store,
):
    from tldw_chatbook.Chat.console_received_turn import (
        bind_received_turn_claim,
        received_turn_claim_for,
    )

    store, session = received_store
    claim = store.claim_received_turn(session.id, "request-one")
    assert claim is not None

    async def inherited_lookup():
        return received_turn_claim_for(store, session.id)

    assert received_turn_claim_for(store, session.id) is None
    with bind_received_turn_claim(store, claim):
        assert received_turn_claim_for(store, session.id) is claim
        with pytest.raises(RuntimeError, match="Received turn owner changed"):
            received_turn_claim_for(store, "other-session")
        assert await asyncio.create_task(inherited_lookup()) is None
        assert store.seal_received_turn(claim) is True
        assert received_turn_claim_for(store, session.id) is claim
        assert store.release_received_turn(claim) is True
        successor = store.claim_received_turn(session.id, "request-two")
        assert successor is not None
        assert received_turn_claim_for(store, session.id) is claim
        assert store.promote_received_turn(claim, _complete(session)) is None
        assert store.received_turn_for_session(session.id) is successor
    assert received_turn_claim_for(store, session.id) is None


@pytest.mark.asyncio
async def test_task_binding_restores_after_nested_scope_and_exception(received_store):
    from tldw_chatbook.Chat.console_received_turn import (
        bind_received_turn_claim,
        received_turn_claim_for,
    )

    store, session = received_store
    claim = store.claim_received_turn(session.id, "request-one")
    second = store.create_session(session_id="other-session", ephemeral=True)
    other = store.claim_received_turn(second.id, "request-two")
    assert claim is not None and other is not None

    with bind_received_turn_claim(store, claim):
        with pytest.raises(RuntimeError, match="scope failure"):
            with bind_received_turn_claim(store, other):
                assert received_turn_claim_for(store, second.id) is other
                with pytest.raises(RuntimeError, match="Received turn owner changed"):
                    received_turn_claim_for(store, session.id)
                raise RuntimeError("scope failure")
        assert received_turn_claim_for(store, session.id) is claim
    assert received_turn_claim_for(store, session.id) is None


@pytest.mark.parametrize("origin", list(ConsoleSubmissionOrigin))
def test_received_origin_is_presentation_data_only(received_store, origin):
    store, session = received_store
    claim = store.claim_received_turn(session.id, "request-one", origin=origin)
    assert claim is not None
    assert claim.origin is origin
    assert store.claim_received_turn(session.id, "request-two", origin=origin) is None
    assert store.begin_preparation(_complete(session)) is None


@pytest.mark.parametrize("origin", ["manual", None])
def test_received_origin_requires_existing_enum_without_consuming_slot(
    received_store, origin
):
    store, session = received_store
    with pytest.raises(TypeError):
        store.claim_received_turn(session.id, "request-one", origin=origin)
    assert store.received_turn_for_session(session.id) is None
    assert store.begin_preparation(_complete(session)) is not None


@pytest.mark.asyncio
async def test_task_binding_refuses_empty_replacement_store_before_ordinary_fallback(
    received_store,
):
    from tldw_chatbook.Chat.console_received_turn import (
        bind_received_turn_claim,
        received_turn_claim_for,
    )

    store, session = received_store
    claim = store.claim_received_turn(session.id, "request-one")
    assert claim is not None
    replacement_store = ConsoleChatStore()
    try:
        with bind_received_turn_claim(store, claim):
            with pytest.raises(RuntimeError, match="Received turn owner changed"):
                received_turn_claim_for(replacement_store, session.id)
            assert store.received_turn_for_session(session.id) is claim
        assert replacement_store.received_turn_for_session(session.id) is None
        assert replacement_store.preparation_for_session(session.id) is None
    finally:
        replacement_store.end_app_runtime()


def test_released_claim_retains_original_source_for_input_recovery(received_store):
    store, session = received_store
    claim = store.claim_received_turn(session.id, "request-one")
    assert claim is not None
    assert store.received_turn_matches_session(claim) is True
    assert store.seal_received_turn(claim) is True
    assert store.received_turn_matches_session(claim) is True
    assert store.received_turn_is_current(claim) is False
    assert store.release_received_turn(claim) is True

    assert store.received_turn_for_session(session.id) is None
    assert store.received_turn_is_current(claim) is False
    assert store.received_turn_matches_session(claim) is True
