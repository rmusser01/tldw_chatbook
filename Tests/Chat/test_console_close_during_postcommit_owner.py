"""Closing a chat must not read as an owner change (TASK-22690).

The tail of `resume_durable_postcommit` refuses to settle a continuation whose
OWNER changed underneath it. That guard is right, but it treated two different
events as one: an owner that genuinely changed (a bug) and an owner that is
legitimately GONE because the user closed the chat (ordinary). Closing mid-turn
therefore raised `RuntimeError("Durable continuation owner changed.")`.

This is the fourth site of the conflation TASK-22587 removed, and it is decided
the same way: `retire_durable_acceptance` leaves a tombstone carrying the SAME
fingerprint, so a matching tombstone proves the preparation was retired by a
close rather than replaced.

Measured at the raise before fixing anything: continuation absent, live
fingerprint gone, tombstone present AND matching -- so the close had gone
through `retire_durable_acceptance`, not the discard path.
"""

from __future__ import annotations

import pytest

from Tests.Chat.test_console_close_during_durable_postcommit import _claimed_effect


def test_a_retired_acceptance_is_reported_as_retired(tmp_path) -> None:
    """The public predicate the owner check now consults."""

    store, prep, fingerprint = _claimed_effect(tmp_path, claim=False)
    store.close_session("session-1")

    assert store.durable_acceptance_retired(prep, fingerprint) is True


def test_a_different_acceptance_is_not_reported_as_retired(tmp_path) -> None:
    """NEGATIVE CONTROL.

    Retirement must be keyed on THIS acceptance, not on the preparation id
    merely having a tombstone -- otherwise an owner that genuinely changed
    would be waved through as an ordinary close, which is exactly the failure
    the guard exists to prevent.
    """

    from dataclasses import replace

    store, prep, fingerprint = _claimed_effect(tmp_path, claim=False)
    store.close_session("session-1")
    other = replace(fingerprint, assistant_message_id="a-different-turn")

    assert store.durable_acceptance_retired(prep, other) is False


def test_an_unknown_preparation_is_not_reported_as_retired(tmp_path) -> None:
    """No tombstone at all is not evidence of a close."""

    store, _prep, fingerprint = _claimed_effect(tmp_path, claim=False)

    assert store.durable_acceptance_retired("never-existed", fingerprint) is False
