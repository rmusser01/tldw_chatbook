"""Closing a Console session mid-turn is not an error (TASK-22587).

Closing a session retires its durable preparation. An effect still in flight
then finds its fingerprint gone, and `_require_durable_fingerprint_locked`
raised `RuntimeError("Durable postcommit fingerprint changed.")` out of the
RELEASE path -- the one place whose whole job is to clean up after a failure.
Worse, that raise REPLACED the exception that sent us down the release path, so
the real cause was lost and the user saw a fingerprint message instead.

The guard's intent is right: an effect must not commit against a retired
preparation. What was missing is a defined outcome for "the preparation was
retired underneath me" (ordinary: the user closed a chat) as distinct from
"the fingerprint changed unexpectedly" (a bug). Those shared one raise.

`retire_durable_acceptance` leaves a tombstone carrying the SAME fingerprint,
so the two are decidable rather than guessed -- which is what
`_durable_retired_locked` reads.

The negative control below is the point of this file: narrowing a safety guard
is only safe if the guard still fires for the case it was written for.
"""

from __future__ import annotations

import pytest

from Tests.Chat.test_console_durable_turn_acceptance import _ready_store
from Tests.console_provider_doubles import provider_resolution
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleDurableAcceptanceRetired

EFFECT = "checkpoint_transition"


def _claimed_effect(tmp_path, *, claim: bool = True):
    """A committed durable turn, with one postcommit effect optionally claimed.

    The controller-level tests pass ``claim=False`` because
    ``_run_durable_postcommit_effect`` claims the effect itself; pre-claiming
    would make it refuse with "already in flight" before reaching the path
    under test.
    """

    db, _service, store, _preparation, acceptance = _ready_store(tmp_path)
    commit = store.commit_durable_turn(acceptance)
    prep = acceptance.preparation_id
    fingerprint = store.durable_acceptance_fingerprint_for(prep)
    assert fingerprint is not None, "harness precondition: a fingerprint exists"

    store.begin_durable_postcommit_effects(
        preparation_id=prep,
        session_id="session-1",
        assistant_message_id=commit.assistant_message_id,
        fingerprint=fingerprint,
    )
    if claim:
        assert store.claim_durable_postcommit_effect(
            prep, EFFECT, fingerprint=fingerprint
        ), "harness precondition: the effect must be claimable"
    return store, prep, fingerprint


def test_releasing_a_claim_after_the_session_closed_does_not_raise(tmp_path) -> None:
    """AC1/AC2: the release path survives the user closing the chat."""

    store, prep, fingerprint = _claimed_effect(tmp_path)

    store.close_session("session-1")

    store.abandon_durable_postcommit_effect(prep, EFFECT, fingerprint=fingerprint)


def test_retirement_has_already_released_the_in_flight_claim(tmp_path) -> None:
    """Why the no-op is safe rather than merely quiet.

    `retire_durable_acceptance` drops every in-flight key for the preparation,
    so by the time the release path runs there is nothing left to release. If
    that ever stops being true, a silent `return` would start leaking claims --
    so it is asserted here rather than assumed.
    """

    store, prep, fingerprint = _claimed_effect(tmp_path)
    assert (prep, EFFECT) in store._durable_effects_in_flight

    store.close_session("session-1")

    assert (prep, EFFECT) not in store._durable_effects_in_flight


def test_a_changed_fingerprint_still_raises(tmp_path) -> None:
    """NEGATIVE CONTROL: the guard still fires for the case it was written for.

    A preparation that was never retired, asked about with a fingerprint that
    does not match, must still raise the generic error -- and must NOT be
    reported as a retirement.
    """

    store, prep, fingerprint = _claimed_effect(tmp_path)
    from dataclasses import replace

    wrong = replace(fingerprint, assistant_message_id="wrong-assistant")

    with pytest.raises(RuntimeError, match="fingerprint changed") as caught:
        store.abandon_durable_postcommit_effect(prep, EFFECT, fingerprint=wrong)
    assert not isinstance(caught.value, ConsoleDurableAcceptanceRetired), (
        "an unexpected fingerprint change was misreported as an ordinary "
        "session close, which is exactly the conflation this task removed"
    )


def test_a_retired_preparation_reports_retirement_not_a_changed_fingerprint(
    tmp_path,
) -> None:
    """The two causes are now distinguishable by type, not by string."""

    store, prep, fingerprint = _claimed_effect(tmp_path)
    store.close_session("session-1")

    with pytest.raises(ConsoleDurableAcceptanceRetired):
        store.complete_durable_postcommit_effect(
            prep, EFFECT, fingerprint=fingerprint
        )


class _Gateway:
    """Minimal ready gateway; no send is performed in these tests."""

    async def resolve_for_send(self, selection):
        return provider_resolution()

    async def stream_chat(self, resolution, messages, **kwargs):
        yield "reply"


def _controller_over(store):
    return ConsoleChatController(
        store=store,
        provider_gateway=_Gateway(),
        provider="llama_cpp",
        model="test-model",
        agent_runtime_enabled=False,
    )


class _ClosedTheChat(RuntimeError):
    """The distinctive failure a real effect would raise."""


@pytest.mark.asyncio
async def test_an_effect_that_fails_while_the_chat_closes_reports_its_own_cause(
    tmp_path,
) -> None:
    """End-to-end shape of the original bug, on the realistic path.

    Before the fix the release path raised "Durable postcommit fingerprint
    changed." from inside `except BaseException:`, REPLACING the failure that
    sent us there. Note this passes on the store fix alone -- it does not
    exercise the controller's release guard, which
    `test_a_failing_release_never_replaces_the_original_failure` covers.
    """

    store, prep, fingerprint = _claimed_effect(tmp_path, claim=False)
    controller = _controller_over(store)

    def effect():
        store.close_session("session-1")
        raise _ClosedTheChat("the real cause")

    with pytest.raises(_ClosedTheChat, match="the real cause"):
        await controller._run_durable_postcommit_effect(
            prep, EFFECT, effect, fingerprint=fingerprint
        )


@pytest.mark.asyncio
async def test_closing_the_chat_during_a_successful_effect_does_not_raise(
    tmp_path,
) -> None:
    """AC1/AC3: close-during-collection on a DURABLE session is not an error.

    The effect's work succeeded; the session simply went away before it could
    be recorded. There is no ledger left to write to, and that is fine.
    """

    store, prep, fingerprint = _claimed_effect(tmp_path, claim=False)
    controller = _controller_over(store)

    def effect():
        store.close_session("session-1")
        return "done"

    result = await controller._run_durable_postcommit_effect(
        prep, EFFECT, effect, fingerprint=fingerprint
    )
    assert result == "done"


@pytest.mark.asyncio
async def test_a_failing_release_never_replaces_the_original_failure(
    tmp_path,
) -> None:
    """The release guard itself, pinned independently of WHY release fails.

    The store fix means retirement no longer makes `abandon` raise, so the
    realistic path above cannot exercise this. The invariant is still worth
    holding: code running inside `except BaseException:` must never turn one
    failure into a different, less informative one. Forcing `abandon` to raise
    is the only way to test that, and this test goes red without the guard.
    """

    store, prep, fingerprint = _claimed_effect(tmp_path, claim=False)
    controller = _controller_over(store)

    def exploding_release(*_args, **_kwargs):
        raise ValueError("bookkeeping blew up")

    store.abandon_durable_postcommit_effect = exploding_release

    def effect():
        raise _ClosedTheChat("the real cause")

    with pytest.raises(_ClosedTheChat, match="the real cause"):
        await controller._run_durable_postcommit_effect(
            prep, EFFECT, effect, fingerprint=fingerprint
        )
