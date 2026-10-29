"""TASK-33620.1: an active emergency stop stays visible, and holds sends.

`/emergency-stop` holds every new run and scheduled dispatch and survives a
restart, but its only sign used to be a five-second toast: afterwards the
header read "Ready", the status strip was unchanged and Send looked live.
Worse, live on origin/dev 38c9d2edcf a send typed during the stop reached the
provider: the received-intent fast path (PR #3023) never consulted the stop,
so the refusal TASK-26004 put in ``send_refusal_copy`` stopped running for an
ordinary Enter.

These tests drive the real ChatScreen against a real ``emergency_stop.json``
(written with the production writer at the production path), with no stub of
the stop's reader.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.test_console_received_intent_feedback import (
    _received_console_case,
    _send,
)
from Tests.UI.test_console_received_intent_feedback import _until as _received_until
from tldw_chatbook.emergency_stop import (
    default_emergency_stop_path,
    read_emergency_stop,
    set_emergency_stop,
)

# Mounted-app tests that build app config in their body keep the
# collection-time profile (Tests/conftest.py ``bootstrap_profile``).
pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def persisted_stop():
    """An ACTIVE stop at the app's own path, as a restart finds it.

    The bootstrap profile's data directory is shared by every
    ``bootstrap_profile`` test in the process, so the file is removed after
    the test whatever happened: a leftover stop would refuse later sends.
    """
    path = default_emergency_stop_path()
    set_emergency_stop(path, reason="test hold")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _record_notices(monkeypatch, app) -> list[str]:
    notices: list[str] = []
    original = app.notify

    def record(message, *args, **kwargs):
        notices.append(str(message))
        return original(message, *args, **kwargs)

    monkeypatch.setattr(app, "notify", record)
    return notices


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["enter"])
async def test_a_send_during_a_persisted_stop_never_reaches_the_provider(
    monkeypatch, persisted_stop: Path, route: str
) -> None:
    """AC#7 + AC#4: the stock received-intent send is refused, naming the clear.

    ``_received_console_case`` is the harness whose Enter takes dev's
    received-intent path (the send-harness tests never enter it); on origin/dev
    38c9d2edcf this send reached the provider double.
    """
    async with _received_console_case(
        monkeypatch, f"estop-{route}", durable=True
    ) as case:
        notices = _record_notices(monkeypatch, case.console.app_instance)

        def refusals() -> list[str]:
            # The refusal reaches the user as a toast or as a transcript row,
            # depending on which gate on the dispatch path speaks first.
            rows = [
                message.content
                for message in case.store.messages_for_session(case.session.id)
            ]
            return [text for text in [*notices, *rows] if "Emergency stop" in text]

        _send(case, route)
        case.probe.release.set()
        refused = await _received_until(
            lambda: any("/emergency-stop clear" in text for text in refusals()), 5
        )
        # An accepted send would reach the provider double within this wait.
        await _received_until(lambda: bool(case.provider_calls), 2)
        assert case.provider_calls == [], (
            "a send during an active emergency stop reached the provider"
        )
        assert refused, f"no refusal named the clear command: {refusals()}"
        assert not any("Clear it to resume." in text for text in refusals())
        assert case.composer.draft_text() == case.draft, "the refused draft was lost"
    assert read_emergency_stop(persisted_stop).active


@pytest.mark.asyncio
async def test_a_stop_written_elsewhere_after_arrival_still_holds_the_fast_path(
    monkeypatch,
) -> None:
    """AC#7: the fast path reads the stop itself, not only the cached view.

    The stop is written after the Console arrived and without telling it (as
    another app instance on the same profile would), so the cached view still
    says "not held" and Send is live when Enter is pressed. Only the received
    path's own read can refuse this send.
    """
    async with _received_console_case(
        monkeypatch, "estop-elsewhere", durable=True
    ) as case:
        send = case.console.query_one("#console-send-message", Button)
        # Let the arrival read land first, so the cached view really says "not
        # held" when the stop appears (not asserted: the pre-fix code has no
        # arrival read, and must fail below, on the provider call).
        await _received_until(
            lambda: (
                getattr(case.console, "_console_emergency_stop_path", None) is not None
            ),
            5,
        )
        assert not send.disabled
        path = default_emergency_stop_path()
        set_emergency_stop(path, reason="written elsewhere")
        try:
            _send(case, "enter")
            case.probe.release.set()
            refused = await _received_until(
                lambda: any(
                    "/emergency-stop clear" in message.content
                    for message in case.store.messages_for_session(case.session.id)
                ),
                5,
            )
            await _received_until(lambda: bool(case.provider_calls), 2)
            assert case.provider_calls == [], (
                "a stop written after arrival did not hold the fast path"
            )
            assert refused, "the held send was not refused with the clear command"
        finally:
            path.unlink(missing_ok=True)
