"""A commit that fails before its reservation pauses -- unless another owns it.

TASK-33621.53. The store pauses a failed durable commit as PERSISTENCE only
once it has reserved the commit. A failure before that left the preparation
COMMITTING, which no surface could end, so the conversation refused every
later send. The controller now takes the same pause, but only while nothing
else owns the commit: a reservation still in flight or a commit already
recorded belongs to that owner, and pausing under it would let a second send
start beside a live one.

The mounted journeys (card, shelf, next send answered) are in
``Tests/UI/test_console_durable_commit_failure_recovery.py``; these pin the
guards on the real controller and store.
"""

from __future__ import annotations

import dataclasses

import pytest

from Tests.Chat.test_console_durable_commit_diagnostics import ReadyGateway
from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
    _execution_context_with_attempt,
)


class UnreachedPersistence:
    """The store's preflight fails first, so persistence is never called."""

    db = None

    def commit_durable_turn(self, *args, **kwargs):
        raise AssertionError("the commit must fail before persistence")


def _controller(before_failure):
    store = ConsoleChatStore(persistence=UnreachedPersistence())

    def failing_commit(acceptance):
        before_failure(store, acceptance.preparation_id)
        raise RuntimeError("Durable identity owner changed.")

    store.commit_durable_turn = failing_commit
    controller = ConsoleChatController(
        store=store,
        provider_gateway=ReadyGateway(),
        provider="llama_cpp",
        model="test-model",
        agent_runtime_enabled=False,
    )
    return controller, store


def _held(store):
    return store.preparation_for_session(store.active_session_id)


@pytest.mark.asyncio
@private_profile_test
async def test_a_commit_that_fails_before_its_reservation_pauses_for_persistence(
    request: pytest.FixtureRequest,
) -> None:
    """Before the fix the preparation stayed COMMITTING."""
    controller, store = _controller(lambda _store, _preparation_id: None)

    result = await controller.submit_draft("capital of Japan?")

    assert result.accepted is False
    assert "Couldn't save the prepared turn" in (result.visible_copy or "")
    held = _held(store)
    assert held is not None
    assert held.state is ConsoleTurnPreparationState.PAUSED, held.state
    assert held.pause_kind is ConsolePreparationPauseKind.PERSISTENCE


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("owner", ["reservation in flight", "commit recorded"])
async def test_a_commit_another_owner_holds_is_left_committing(
    request: pytest.FixtureRequest, owner: str
) -> None:
    """The pause never lands under a commit something else still owns."""

    def claim(store, preparation_id):
        owned = (
            store._durable_commit_in_flight
            if owner == "reservation in flight"
            else store._durable_commit_by_preparation
        )
        owned[preparation_id] = object()

    controller, store = _controller(claim)

    result = await controller.submit_draft("capital of Japan?")

    assert result.accepted is False
    held = _held(store)
    assert held is not None
    assert held.state is ConsoleTurnPreparationState.COMMITTING, held.state


@pytest.mark.asyncio
@private_profile_test
async def test_a_newer_attempt_of_the_failed_send_is_left_alone(
    request: pytest.FixtureRequest,
) -> None:
    """The pause is for the attempt that failed, never a newer one.

    The transition matrix gives a COMMITTING send no path to a new attempt,
    so this guard is defence in depth, as is the store's own attempt check
    on its reservation. The newer attempt is installed directly.
    """

    def supersede(store, preparation_id):
        with store.durable_preparation_lock:
            current = store.preparation_by_id(preparation_id)
            newer = dataclasses.replace(
                current,
                attempt_id="attempt-newer",
                execution_context=_execution_context_with_attempt(
                    current.execution_context, "attempt-newer"
                ),
            )
            store._preparations_by_session[current.session_id] = newer
            store._preparations_by_id[preparation_id] = newer

    controller, store = _controller(supersede)

    result = await controller.submit_draft("capital of Japan?")

    assert result.accepted is False
    held = _held(store)
    assert held is not None
    assert held.attempt_id == "attempt-newer"
    assert held.state is ConsoleTurnPreparationState.COMMITTING, held.state
