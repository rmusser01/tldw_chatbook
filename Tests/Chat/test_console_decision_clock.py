"""Decision budgets count time during which their owning card is answerable."""

from __future__ import annotations

import pytest

from Tests.Chat.test_console_interrupt_rounds import FakeSeamsFull
from tldw_chatbook.Chat.console_interrupt_rounds import (
    KIND_SETTER_ATTRS,
    InterruptRoundHost,
)


@pytest.mark.parametrize("kind", tuple(KIND_SETTER_ATTRS))
@pytest.mark.parametrize("away", ("navigation", "session"))
def test_decision_budget_pauses_while_its_card_cannot_be_answered(
    monkeypatch, kind, away
):
    """Wall-clock deadlines would expire at 102; ten visible seconds end at 110."""
    import tldw_chatbook.Chat.console_interrupt_rounds as rounds

    now = [0.0]
    monkeypatch.setattr(rounds.time, "monotonic", lambda: now[0])
    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    waits = iter((2.0, 102.0, 102.0, 109.0, 110.0))
    observed = []

    class Event:
        def wait(self, seconds):
            now[0] = next(waits)
            observed.append(now[0])
            if len(observed) == 1:
                if away == "navigation":
                    host.set_view_visible(False)
                else:
                    seams.store.active_session_id = "sess-B"
                    host.refresh_decision_clocks()
            elif len(observed) == 3:
                if away == "navigation":
                    host.set_view_visible(True)
                else:
                    seams.store.active_session_id = "sess-A"
                    host.refresh_decision_clocks()
            return False

    outcome = host.run_round(
        kind,
        "r1",
        {"round_id": "r1", "session_id": "sess-A", "timeout_seconds": 10.0},
        {"event": Event(), "session_id": "sess-A"},
        session_id="sess-A",
        owning_session_id="sess-A",
        deadline=10.0,
        is_parked=False,
    )
    assert outcome == "timeout"
    assert observed == [2.0, 102.0, 102.0, 109.0, 110.0]


@pytest.mark.parametrize("kind", tuple(KIND_SETTER_ATTRS))
def test_hidden_decision_still_observes_explicit_cancellation(kind):
    """Pausing a deadline must never pause Stop or application shutdown."""
    seams = FakeSeamsFull()
    host = InterruptRoundHost(seams)
    host.set_view_visible(False)
    seams.cancelled = True

    class Event:
        def wait(self, seconds):
            return False

    assert (
        host.run_round(
            kind,
            "r1",
            {"round_id": "r1", "session_id": "sess-A", "timeout_seconds": 10.0},
            {"event": Event(), "session_id": "sess-A"},
            session_id="sess-A",
            owning_session_id="sess-A",
            deadline=10.0,
            is_parked=False,
        )
        == "cancelled"
    )


@pytest.mark.parametrize("kind", tuple(KIND_SETTER_ATTRS))
def test_queued_decision_only_spends_budget_after_becoming_fifo_head(monkeypatch, kind):
    import tldw_chatbook.Chat.console_interrupt_rounds as rounds

    now = [0.0]
    monkeypatch.setattr(rounds.time, "monotonic", lambda: now[0])
    host = InterruptRoundHost(FakeSeamsFull())
    host.park_round_payload(kind, "first", {"session_id": "sess-A"})
    waits = iter((100.0, 109.0, 110.0))
    observed = []

    class Event:
        def wait(self, seconds):
            now[0] = next(waits)
            observed.append(now[0])
            if len(observed) == 1:
                host.unpark_round_payload(kind, "first")
                host.remount_head(kind, "sess-A")
            return False

    assert (
        host.run_round(
            kind,
            "second",
            {"round_id": "second", "session_id": "sess-A", "timeout_seconds": 10.0},
            {"event": Event(), "session_id": "sess-A"},
            session_id="sess-A",
            owning_session_id="sess-A",
            deadline=10.0,
            is_parked=False,
        )
        == "timeout"
    )
    assert observed == [100.0, 109.0, 110.0]


@pytest.mark.parametrize("kind", tuple(KIND_SETTER_ATTRS))
def test_legacy_unparked_decision_expires_while_its_card_is_visible(monkeypatch, kind):
    import tldw_chatbook.Chat.console_interrupt_rounds as rounds

    now = [0.0]
    monkeypatch.setattr(rounds.time, "monotonic", lambda: now[0])
    host = InterruptRoundHost(FakeSeamsFull())

    class Event:
        def wait(self, seconds):
            now[0] += 1
            assert now[0] <= 1, "Unparked legacy card never consumed its budget"
            return False

    assert (
        host.run_round(
            kind,
            "legacy",
            {"timeout_seconds": 1.0},
            {"event": Event(), "session_id": "sess-A"},
            session_id=None,
            owning_session_id="sess-A",
            deadline=1.0,
            is_parked=False,
        )
        == "timeout"
    )


@pytest.mark.parametrize("kind", tuple(KIND_SETTER_ATTRS))
def test_external_decision_projection_only_charges_supported_visible_kinds(
    monkeypatch, kind
):
    import tldw_chatbook.Chat.console_interrupt_rounds as rounds

    now = [0.0]
    monkeypatch.setattr(rounds.time, "monotonic", lambda: now[0])
    host = InterruptRoundHost(FakeSeamsFull())
    host.set_view_visible(False)
    host.set_decision_view("buddy", "sess-A", kinds=())
    observed = []

    class Event:
        def wait(self, _seconds):
            now[0] += 100 if not observed else 1
            observed.append(now[0])
            assert len(observed) <= 2
            if len(observed) == 1:
                host.set_decision_view("buddy", "sess-A", kinds=(kind,))
            return False

    assert (
        host.run_round(
            kind,
            "r",
            {"session_id": "sess-A"},
            {"event": Event(), "session_id": "sess-A"},
            session_id="sess-A",
            owning_session_id="sess-A",
            deadline=1.0,
            is_parked=False,
        )
        == "timeout"
    )
    host.set_decision_view("buddy", None)
    assert not host._decision_views
    assert observed == [100, 101]
