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
from Tests.Chat.test_console_first_send_atomicity import _controller
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
        store.complete_durable_postcommit_effect(prep, EFFECT, fingerprint=fingerprint)


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


# ---------------------------------------------------------------------------
# The REAL path.
#
# Everything above drives `_run_durable_postcommit_effect` directly, which is
# not how a send reaches it. Qodo's review of #2123 made the point precisely:
# suppressing retirement inside one effect only moves the raise to the NEXT
# fingerprint-validated effect, or to the unconditional `retire` that ends the
# sequence -- and a helper-only test cannot see either. These drive
# `resume_durable_postcommit`, the enclosing orchestration real sends use.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_after_the_chat_was_closed_does_not_raise(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1 on the path production actually takes.

    A postcommit step fails, leaving a continuation to resume. The user closes
    the chat before it resumes -- so the whole sequence now runs against a
    retired preparation.
    """

    _db, store, controller, _gateway = _controller(tmp_path)
    original = store.publish_durable_turn_identity
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected identity publication")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "publish_durable_turn_identity", fail_once)
    first = await controller.submit_draft(
        "draft the user abandoned", session_id="session-1"
    )
    assert first.accepted is True, "harness precondition: the turn was accepted"
    assert first.preparation_id in controller._durable_postcommit_continuations

    store.close_session("session-1")

    resumed = await controller.resume_durable_postcommit(first.preparation_id)
    assert resumed is not None


@pytest.mark.asyncio
async def test_resume_after_close_leaves_no_dangling_continuation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing must not strand the continuation the resume was meant to clear.

    Without this, "does not raise" could be satisfied by bailing out early and
    leaking the very state the sequence exists to release.
    """

    _db, store, controller, _gateway = _controller(tmp_path)
    original = store.publish_durable_turn_identity
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected identity publication")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "publish_durable_turn_identity", fail_once)
    first = await controller.submit_draft("another draft", session_id="session-1")
    assert first.accepted is True

    store.close_session("session-1")
    await controller.resume_durable_postcommit(first.preparation_id)

    assert store.durable_content_retention_count() == 0, (
        "closing the chat left content-bearing durable state behind"
    )


@pytest.mark.asyncio
async def test_closing_midway_through_the_effect_sequence_does_not_raise(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close DURING the sequence -- Qodo's finding #2 on #2123.

    Closing before the resume is caught by the early lookup. Closing while the
    sequence runs is the harder case: the effect that notices retirement
    completes benignly, and then the NEXT fingerprint-validated effect raises.
    Suppressing retirement per-effect only moved the raise one effect along, so
    this is the test that distinguishes the two.

    The close happens inside effect #1 of eight, leaving the remaining seven to
    cope with a preparation that no longer exists.
    """

    _db, store, controller, _gateway = _controller(tmp_path)
    original = store.publish_durable_turn_identity
    calls = 0

    def fail_then_close(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Leaves a continuation for `resume_durable_postcommit` to pick up.
            raise RuntimeError("injected identity publication")
        # Second call is effect #1 of the resumed sequence: the user closes the
        # chat right here, mid-flight.
        result = original(*args, **kwargs)
        store.close_session("session-1")
        return result

    monkeypatch.setattr(store, "publish_durable_turn_identity", fail_then_close)
    first = await controller.submit_draft("mid-sequence draft", session_id="session-1")
    assert first.accepted is True, "harness precondition: the turn was accepted"

    resumed = await controller.resume_durable_postcommit(first.preparation_id)
    assert resumed is not None
    assert calls == 2, "harness precondition: the resume re-ran the first effect"


def test_retiring_an_already_retired_acceptance_is_a_no_op(tmp_path) -> None:
    """The sequence's own tail retire, after close already retired.

    Qodo's finding #2 on #2123 named this directly: the postcommit sequence
    ends with an unconditional `retire_durable_acceptance`, and closing the
    chat has already performed one. Retiring the SAME acceptance twice is a
    no-op -- the tombstone proves it is the same one -- where it used to raise
    "Durable acceptance fingerprint changed."
    """

    store, prep, fingerprint = _claimed_effect(tmp_path, claim=False)
    store.close_session("session-1")

    store.retire_durable_acceptance(prep, fingerprint)


def test_a_different_acceptance_on_a_retired_id_still_raises(tmp_path) -> None:
    """NEGATIVE CONTROL for the idempotent retire.

    Idempotency must key on THIS acceptance, not merely on the preparation id
    having a tombstone -- otherwise a genuinely different acceptance would be
    silently accepted as an already-done retirement.
    """

    from dataclasses import replace

    store, prep, fingerprint = _claimed_effect(tmp_path, claim=False)
    store.close_session("session-1")
    other = replace(fingerprint, assistant_message_id="a-different-turn")

    with pytest.raises(RuntimeError, match="fingerprint changed"):
        store.retire_durable_acceptance(prep, other)


def test_completed_effects_survive_the_close_that_retired_them(tmp_path) -> None:
    """Recovery must still be able to tell whether the provider had started.

    The failure handler asks which effects completed, and it asks AFTER a
    failure -- by which point the user may have closed the chat. Reading the
    live ledger raised there; the tombstone retains `completed` precisely so
    the answer survives, which is what keeps `provider_started` correct.
    """

    store, prep, fingerprint = _claimed_effect(tmp_path)
    store.complete_durable_postcommit_effect(prep, EFFECT, fingerprint=fingerprint)

    store.close_session("session-1")

    assert EFFECT in store.durable_completed_effects_for(prep, fingerprint=fingerprint)


def test_retirement_proof_survives_a_flood_of_unrelated_closes(tmp_path) -> None:
    """Qodo finding #4 on #2123: eviction must not lose retirement proof.

    Tombstones are FIFO-evicted past `DURABLE_TOMBSTONE_CAP`. A plain eviction
    could reclaim the tombstone of a preparation whose postcommit sequence was
    still running -- and that tombstone is the only proof its retirement was an
    ordinary close rather than a mutation. Losing it put the generic
    fingerprint-change error back on the in-flight effect, which made
    correctness depend on unrelated session-close volume.
    """

    from dataclasses import replace

    store, prep, fingerprint = _claimed_effect(tmp_path)
    store.close_session("session-1")
    assert store._durable_retired_locked(prep, fingerprint), (
        "harness precondition: the close left retirement proof"
    )

    # Far more unrelated retirements than the cap allows.
    for index in range(store.DURABLE_TOMBSTONE_CAP * 2):
        other_id = f"unrelated-{index}"
        other_fp = replace(fingerprint, assistant_message_id=other_id)
        store._durable_fingerprint_by_preparation[other_id] = other_fp
        store.retire_durable_acceptance(other_id, other_fp)

    assert store._durable_retired_locked(prep, fingerprint), (
        "the in-flight preparation's retirement proof was evicted, so its "
        "effect would see a fingerprint-change error instead of an ordinary "
        "close -- correctness must not depend on unrelated close volume"
    )


def test_protecting_tombstones_still_honours_the_cap(tmp_path) -> None:
    """The protection must not turn a bounded cache into a leak.

    A pinned entry is skipped, oldest-first -- but if everything is pinned the
    oldest is evicted anyway. Without that fallback, an unreleased protection
    would let the tombstone map grow without limit.
    """

    from dataclasses import replace

    store, prep, fingerprint = _claimed_effect(tmp_path)
    store.close_session("session-1")

    for index in range(store.DURABLE_TOMBSTONE_CAP * 2):
        other_id = f"pinned-{index}"
        other_fp = replace(fingerprint, assistant_message_id=other_id)
        store._durable_fingerprint_by_preparation[other_id] = other_fp
        # Pin every single one, which is the pathological case.
        store._durable_active_postcommit.add(other_id)
        store.retire_durable_acceptance(other_id, other_fp)

    assert len(store._durable_tombstones) <= store.DURABLE_TOMBSTONE_CAP, (
        f"tombstones grew past the cap ({len(store._durable_tombstones)} > "
        f"{store.DURABLE_TOMBSTONE_CAP}) -- protection became a leak"
    )


def test_releasing_activity_lets_the_tombstone_be_evicted_again(tmp_path) -> None:
    """Protection is scoped to the sequence, not permanent."""

    store, prep, _fingerprint = _claimed_effect(tmp_path)
    assert prep in store._durable_active_postcommit

    store.release_durable_postcommit_activity(prep)

    assert prep not in store._durable_active_postcommit
