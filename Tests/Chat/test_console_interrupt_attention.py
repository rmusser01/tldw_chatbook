"""task-31385: attention when a round blocks on the user off-screen."""

from __future__ import annotations

import threading

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_interrupt_rounds import InterruptRoundHost
from tldw_chatbook.UI.Navigation.main_navigation import CONSOLE_ATTENTION_ATTR


class _App:
    def __init__(self, *, headless: bool = False) -> None:
        self.is_headless = headless
        self.bells = 0

    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def bell(self) -> None:
        self.bells += 1


def _controller(app, *, attached: bool):
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    controller.app = app
    if attached:
        controller.set_pending_approval = lambda payload: None
        controller.park_pending_approval = lambda session_id: None
    else:
        controller.set_pending_approval = None
        controller.park_pending_approval = None
    return controller


def test_a_round_raised_while_console_is_away_rings_once_and_badges():
    app = _App()
    controller = _controller(app, attached=False)
    controller.on_pending_rounds_changed(1, "question", True)
    assert app.bells == 1 and getattr(app, CONSOLE_ATTENTION_ATTR) == 1
    # Teardown clears the badge and never rings.
    controller.on_pending_rounds_changed(0, "question", False)
    assert app.bells == 1 and getattr(app, CONSOLE_ATTENTION_ATTR) == 0


def test_a_round_raised_while_console_is_visible_badges_but_does_not_ring():
    app = _App()
    controller = _controller(app, attached=True)
    controller.on_pending_rounds_changed(2, "approval", True)
    assert app.bells == 0 and getattr(app, CONSOLE_ATTENTION_ATTR) == 2


def test_the_setting_silences_the_bell_and_headless_never_rings(monkeypatch):
    import tldw_chatbook.Chat.console_chat_controller as ccc

    monkeypatch.setattr(
        ccc,
        "get_cli_setting",
        lambda section, key, default=None: False
        if (section, key) == ("console", "interrupt_bell")
        else default,
    )
    app = _App()
    _controller(app, attached=False).on_pending_rounds_changed(1, "approval", True)
    assert app.bells == 0 and getattr(app, CONSOLE_ATTENTION_ATTR) == 1
    monkeypatch.undo()
    headless = _App(headless=True)
    _controller(headless, attached=False).on_pending_rounds_changed(1, "approval", True)
    assert headless.bells == 0


def test_no_app_is_a_no_op():
    controller = _controller(None, attached=False)
    controller.on_pending_rounds_changed(1, "approval", True)  # must not raise


def test_the_host_reports_arm_and_teardown_totals_through_the_seam():
    seen = []

    class _Seams:
        app = _App()
        store = ConsoleChatStore()
        set_pending_approval = None
        park_pending_approval = None

        @staticmethod
        def _is_session_cancelled(session_id, *, cancel_event=None, visit_event=None):
            return False

        @staticmethod
        def on_pending_rounds_changed(total, kind, raised):
            seen.append((total, kind, raised))

    host = InterruptRoundHost(_Seams())
    state = {"event": threading.Event(), "session_id": "s1"}
    state["event"].set()
    host.run_round(
        "question", "r1", {"round_id": "r1", "session_id": "s1"}, state,
        session_id="s1", owning_session_id="s1", deadline=None, is_parked=False,
    )
    assert seen == [(1, "question", True), (0, "question", False)]
