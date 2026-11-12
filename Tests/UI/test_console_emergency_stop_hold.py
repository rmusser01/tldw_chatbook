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
from textual.widgets import Button, Static

from Tests.UI.test_console_received_intent_feedback import (
    _received_console_case,
    _send,
)
from Tests.UI.test_console_received_intent_feedback import _until as _received_until
from Tests.UI.test_console_visible_turn_attention import _console_app, _reach_console
from tldw_chatbook.emergency_stop import (
    default_emergency_stop_path,
    read_emergency_stop,
    set_emergency_stop,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleStatusChips,
    ConsoleStatusText,
)

# The copy is pinned literally (not imported) so these tests collect on the
# pre-fix code and fail there on what the user sees.
HEADER_HOLD = "Held · emergency stop"
STRIP_HOLD = "Emergency stop: runs and schedules held"
SEND_HOLD = "Send held — emergency stop is on"

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


async def _until(pilot, predicate, *, attempts: int = 100) -> bool:
    for _ in range(attempts):
        if predicate():
            return True
        await pilot.pause(0.05)
    return predicate()


async def _type_and_press_enter(chat, pilot, text: str) -> ConsoleComposerBar:
    composer = chat.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    await _until(pilot, lambda: chat.focused is not None)
    composer.load_draft(text)
    await pilot.pause(0.1)
    await pilot.press("enter")
    return composer


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["enter", "send-button"])
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

        send = case.console.query_one("#console-send-message", Button)
        held = await _received_until(lambda: send.disabled, 5)
        _send(case, route)
        case.probe.release.set()
        refused = await _received_until(
            lambda: any("/emergency-stop clear" in text for text in refusals()), 5
        )
        # An accepted send would reach the provider double within this wait.
        await _received_until(lambda: bool(case.provider_calls), 2)
        assert (
            case.provider_calls == []
        ), "a send during an active emergency stop reached the provider"
        # AC#3: Send is held before anything is pressed. A held button's press
        # is a no-op; Enter still dispatches and is refused with the copy.
        assert held, "Send was not held while the emergency stop was active"
        if route == "enter":
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
            assert (
                case.provider_calls == []
            ), "a stop written after arrival did not hold the fast path"
            assert refused, "the held send was not refused with the clear command"
            # The send's own read updated the view: the hold is now shown.
            assert send.disabled
        finally:
            path.unlink(missing_ok=True)


def _header_status(chat) -> str:
    return str(chat.query_one("#workbench-header-status", Static).content)


def _send_reason(chat) -> str:
    return str(chat.query_one("#console-send-disabled-reason", Static).content)


@pytest.mark.asyncio
async def test_a_persisted_stop_shows_its_hold_after_a_restart(
    tmp_path, monkeypatch, persisted_stop: Path
) -> None:
    """AC#1, AC#3, AC#4, AC#5 (Console at launch) and AC#6.

    The app starts with ``emergency_stop.json`` already active, as a restart
    finds it. Nothing tells the Console about it but its own arrival read.
    """
    app, gateway = _console_app(tmp_path)
    notices = _record_notices(monkeypatch, app)
    async with app.run_test(size=(160, 45), notifications=True) as pilot:
        chat = await _reach_console(app, pilot, "explore_home")
        assert await _until(
            pilot, lambda: "Held" in _header_status(chat)
        ), f"header still reads {_header_status(chat)!r} during an active stop"
        # AC#1: the header's hold state, in place of the readiness word.
        assert _header_status(chat) == HEADER_HOLD
        # AC#1: the strip's persistent chip and its Clear, on the visible
        # (unscrolled) start of the line.
        chips = chat.query_one("#console-status-chips", ConsoleStatusChips)
        assert chips.segment_text("estop") == STRIP_HOLD
        assert chips.segment_text("estop-clear") == "Clear"
        line = chat.query_one("#console-status-text", ConsoleStatusText)
        spans = dict((name, (start, end)) for name, start, end in line.segment_spans)
        assert spans["estop-clear"][1] <= line.scrollable_content_region.width
        # AC#5: the launch notice says scheduled dispatches are held.
        assert any(
            "scheduled dispatches are held" in notice
            and "/emergency-stop clear" in notice
            for notice in notices
        ), notices
        # AC#4: the empty transcript says the hold, never "Ready - type ...".
        # Its projection schedules refresh_messages; a painted header does
        # not mean the asynchronously reconciled empty row is painted yet.
        assert await _until(
            pilot,
            lambda: "Emergency stop is active"
            in str(chat.query_one("#console-empty-body", Static).content),
        )
        body = str(chat.query_one("#console-empty-body", Static).content)
        assert "Emergency stop is active" in body, body
        assert "type a message to begin" not in body
        # AC#3: a typed draft shows Send held, naming the stop, before Enter.
        composer = chat.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("say held")
        assert await _until(pilot, lambda: _send_reason(chat) != "")
        assert _send_reason(chat) == SEND_HOLD
        assert chat.query_one("#console-send-message", Button).disabled
        await pilot.pause(0.2)
        assert gateway.payloads == []
    assert read_emergency_stop(persisted_stop).active


@pytest.mark.asyncio
async def test_the_strip_clear_lifts_the_stop_and_the_next_send_gets_a_reply(
    tmp_path, persisted_stop: Path, record_property
) -> None:
    """AC#2: Clear lifts the stop; the header says Ready; a send gets a reply."""
    app, gateway = _console_app(tmp_path)
    gateway.reply = "cleared-reply"
    async with app.run_test(size=(160, 45), notifications=True) as pilot:
        chat = await _reach_console(app, pilot, "explore_home")
        assert await _until(pilot, lambda: "Held" in _header_status(chat))
        line = chat.query_one("#console-status-text", ConsoleStatusText)
        start, _end = next(
            (start, end)
            for name, start, end in line.segment_spans
            if name == "estop-clear"
        )
        await pilot.click("#console-status-text", offset=(start + 1, 0))
        assert await _until(
            pilot, lambda: "Held" not in _header_status(chat)
        ), "the header kept the hold after Clear"
        assert not read_emergency_stop(persisted_stop).active
        assert _header_status(chat).startswith("Ready")
        chips = chat.query_one("#console-status-chips", ConsoleStatusChips)
        assert not chips.segment_visible("estop")
        await _type_and_press_enter(chat, pilot, "say cleared")
        store = chat._console_chat_store
        # This checks completed dispatch after Clear, not a latency bound.
        # Cold native preparation can outlive the short projection poll.
        import time

        started = time.monotonic()
        replied = await _received_until(
            lambda: any(
                "cleared-reply" in message.content
                for message in store.messages_for_session(store.active_session_id)
            ),
            60,
        )
        record_property("post_clear_reply_seconds", time.monotonic() - started)
        assert replied, (
            "the send after Clear got no reply",
            len(gateway.payloads),
            [
                message.content
                for message in store.messages_for_session(store.active_session_id)
            ],
        )
        assert len(gateway.payloads) == 1


@pytest.mark.asyncio
async def test_the_command_holds_and_clears_the_console_visibly(tmp_path) -> None:
    """The original repro: ``/emergency-stop`` without a restart.

    Its toast fades in about five seconds; the hold must not fade with it.
    ``/emergency-stop clear`` then lifts it.
    """
    app, _gateway = _console_app(tmp_path)
    path = default_emergency_stop_path()
    try:
        async with app.run_test(size=(160, 45), notifications=True) as pilot:
            chat = await _reach_console(app, pilot, "explore_home")
            assert _header_status(chat).startswith("Ready")
            await _type_and_press_enter(chat, pilot, "/emergency-stop live test")
            assert await _until(pilot, lambda: "Held" in _header_status(chat))
            assert read_emergency_stop(path).active
            chips = chat.query_one("#console-status-chips", ConsoleStatusChips)
            assert chips.segment_visible("estop")
            await _type_and_press_enter(chat, pilot, "/emergency-stop clear")
            assert await _until(pilot, lambda: "Held" not in _header_status(chat))
            assert not read_emergency_stop(path).active
            assert not chips.segment_visible("estop")
    finally:
        path.unlink(missing_ok=True)


def _click_strip_clear(chat, pilot):
    line = chat.query_one("#console-status-text", ConsoleStatusText)
    start, _end = next(
        (start, end) for name, start, end in line.segment_spans if name == "estop-clear"
    )
    return pilot.click("#console-status-text", offset=(start + 1, 0))


@pytest.mark.asyncio
async def test_a_strip_clear_that_cannot_write_keeps_the_hold_and_the_app(
    tmp_path, monkeypatch, persisted_stop: Path
) -> None:
    """Review follow-up: a failed Clear is reported; it never closes the app.

    An unreadable stop file fails safe to active, which shows Clear; a
    read-only data directory then refuses the write. The ``OSError`` used to
    escape the strip's message handler and exit the app.
    """
    from tldw_chatbook.UI.Console_Modules import emergency_stop_hold

    def refuse(_path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(emergency_stop_hold, "clear_emergency_stop", refuse)
    app, gateway = _console_app(tmp_path)
    notices = _record_notices(monkeypatch, app)
    async with app.run_test(size=(160, 45), notifications=True) as pilot:
        chat = await _reach_console(app, pilot, "explore_home")
        assert await _until(pilot, lambda: "Held" in _header_status(chat))
        await _click_strip_clear(chat, pilot)
        assert await _until(
            pilot,
            lambda: any("Couldn't clear the emergency stop" in n for n in notices),
        ), f"the failed Clear was not reported: {notices}"
        assert any("PermissionError" in n and "still on" in n for n in notices)
        assert not any(n.startswith("Emergency stop cleared") for n in notices)
        assert _header_status(chat) == HEADER_HOLD
        chips = chat.query_one("#console-status-chips", ConsoleStatusChips)
        assert chips.segment_visible("estop")
        assert app.is_running
        assert gateway.payloads == []
    assert read_emergency_stop(persisted_stop).active


@pytest.mark.asyncio
async def test_a_stop_command_that_cannot_write_says_so_and_keeps_the_draft(
    tmp_path, monkeypatch
) -> None:
    """Review follow-up: ``/emergency-stop`` never claims a stop it did not set."""
    from tldw_chatbook.UI.Console_Modules import emergency_stop_hold

    def refuse(_path, *, reason=""):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(emergency_stop_hold, "set_emergency_stop", refuse)
    app, _gateway = _console_app(tmp_path)
    notices = _record_notices(monkeypatch, app)
    path = default_emergency_stop_path()
    async with app.run_test(size=(160, 45), notifications=True) as pilot:
        chat = await _reach_console(app, pilot, "explore_home")
        composer = await _type_and_press_enter(chat, pilot, "/emergency-stop freeze")
        assert await _until(
            pilot, lambda: any("Couldn't set the emergency stop" in n for n in notices)
        ), f"the failed stop was not reported: {notices}"
        assert any("OSError" in n and "It is off." in n for n in notices), notices
        assert not any(n.startswith("Emergency stop ACTIVE") for n in notices)
        assert _header_status(chat).startswith("Ready")
        assert composer.draft_text() == "/emergency-stop freeze"
        assert app.is_running
    assert not read_emergency_stop(path).active


@pytest.mark.asyncio
async def test_a_stop_cleared_elsewhere_lets_the_next_send_through(
    tmp_path, persisted_stop: Path
) -> None:
    """Review follow-up: the cached hold re-reads the file before refusing.

    Another app instance on the same profile clears the stop. This Console's
    cached view still says held; Enter must find the cleared file, send, and
    lift the hold everywhere.
    """
    from tldw_chatbook.emergency_stop import clear_emergency_stop

    app, gateway = _console_app(tmp_path)
    gateway.reply = "resumed-reply"
    async with app.run_test(size=(160, 45), notifications=True) as pilot:
        chat = await _reach_console(app, pilot, "explore_home")
        assert await _until(pilot, lambda: "Held" in _header_status(chat))
        clear_emergency_stop(persisted_stop)  # the other instance's Clear
        await _type_and_press_enter(chat, pilot, "say resumed")
        store = chat._console_chat_store
        assert await _until(
            pilot,
            lambda: any(
                "resumed-reply" in message.content
                for message in store.messages_for_session(store.active_session_id)
            ),
        ), "a send after the stop was cleared elsewhere was still refused"
        assert len(gateway.payloads) == 1
        assert await _until(pilot, lambda: "Held" not in _header_status(chat))
        chips = chat.query_one("#console-status-chips", ConsoleStatusChips)
        assert not chips.segment_visible("estop")
