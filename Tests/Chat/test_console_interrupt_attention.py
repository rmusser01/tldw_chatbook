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


def _controller(app, *, attached: bool, pending: int = 0):
    controller = ConsoleChatController(store=ConsoleChatStore(), provider_gateway=None)
    controller.app = app
    # The badge re-reads the host's truth on the UI thread, so a test that
    # expects a count must have that many rounds registered.
    with controller._interrupt_host.lock:
        for index in range(pending):
            controller._interrupt_host.registries["question"][f"r{index}"] = {
                "event": threading.Event()
            }
    if attached:
        controller.set_pending_approval = lambda payload: None
        controller.park_pending_approval = lambda session_id: None
    else:
        controller.set_pending_approval = None
        controller.park_pending_approval = None
    return controller


def test_a_round_raised_while_console_is_away_rings_once_and_badges():
    app = _App()
    controller = _controller(app, attached=False, pending=1)
    controller.on_pending_rounds_changed(1, "question", True)
    assert app.bells == 1 and getattr(app, CONSOLE_ATTENTION_ATTR) == 1
    # Teardown clears the badge and never rings.
    with controller._interrupt_host.lock:
        controller._interrupt_host.registries["question"].clear()
    controller.on_pending_rounds_changed(0, "question", False)
    assert app.bells == 1 and getattr(app, CONSOLE_ATTENTION_ATTR) == 0


def test_a_round_raised_while_console_is_visible_badges_but_does_not_ring():
    app = _App()
    controller = _controller(app, attached=True, pending=2)
    controller.on_pending_rounds_changed(2, "approval", True)
    assert app.bells == 0 and getattr(app, CONSOLE_ATTENTION_ATTR) == 2


def _config_bell(monkeypatch, value):
    import tldw_chatbook.Chat.console_chat_controller as ccc

    monkeypatch.setattr(
        ccc,
        "get_cli_setting",
        lambda section, key, default=None: value
        if (section, key) == ("console", "interrupt_bell")
        else default,
    )


def _rings(monkeypatch, *, env, config) -> int:
    from tldw_chatbook.Chat.console_chat_controller import INTERRUPT_BELL_ENV_VAR

    if env is None:
        monkeypatch.delenv(INTERRUPT_BELL_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(INTERRUPT_BELL_ENV_VAR, env)
    _config_bell(monkeypatch, config)
    app = _App()
    _controller(app, attached=False, pending=1).on_pending_rounds_changed(1, "approval", True)
    return app.bells


def test_bell_setting_precedence_is_environment_then_config_then_default(monkeypatch):
    assert _rings(monkeypatch, env=None, config=None) == 1  # default on
    assert _rings(monkeypatch, env=None, config=False) == 0  # config wins over default
    assert _rings(monkeypatch, env="0", config=True) == 0  # env wins over config
    assert _rings(monkeypatch, env="true", config=False) == 1
    assert _rings(monkeypatch, env="   ", config=False) == 0  # blank env = unset
    assert _rings(monkeypatch, env="maybe", config=None) == 0  # app-wide bool rule: not truthy -> off
    assert _rings(monkeypatch, env="yes", config=None) == 1


def test_headless_never_rings_and_a_zero_total_raise_never_rings(monkeypatch):
    headless = _App(headless=True)
    _controller(headless, attached=False).on_pending_rounds_changed(1, "approval", True)
    assert headless.bells == 0
    app = _App()
    _controller(app, attached=False).on_pending_rounds_changed(0, "approval", True)
    assert app.bells == 0 and getattr(app, CONSOLE_ATTENTION_ATTR) == 0


def test_the_badge_shows_the_hosts_current_total_not_a_stale_dispatch():
    app = _App()
    controller = _controller(app, attached=True)
    with controller._interrupt_host.lock:
        controller._interrupt_host.registries["question"]["r1"] = {"event": threading.Event()}
    # An older teardown dispatch (total 0) landing after a newer arm must not
    # clear the badge while a round is still registered.
    controller.on_pending_rounds_changed(0, "approval", False)
    assert getattr(app, CONSOLE_ATTENTION_ATTR) == 1


def test_a_raising_marshal_never_escapes_the_hook():
    class _Broken(_App):
        def call_from_thread(self, fn, *args, **kwargs):
            raise RuntimeError("app is shutting down")

    app = _Broken()
    _controller(app, attached=False).on_pending_rounds_changed(1, "approval", True)  # must not raise


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


def _seams(seen, *, on_add=None):
    class _Seams:
        app = _App()
        store = ConsoleChatStore()
        set_pending_approval = None
        park_pending_approval = None

        @staticmethod
        def _is_session_cancelled(session_id, *, cancel_event=None, visit_event=None):
            return False

        @staticmethod
        def add_pending_round(session_id, round_id):
            if on_add is not None:
                on_add(round_id)

        @staticmethod
        def discard_pending_round(session_id, round_id):
            return None

        @staticmethod
        def on_pending_rounds_changed(total, kind, raised):
            seen.append((total, kind, raised))

    return _Seams()


def test_a_round_revoked_between_registration_and_announce_never_announces():
    seen = []
    host_box = {}
    state = {"event": threading.Event(), "session_id": "s1", "run_id": "run-1", "revoked": False}

    def _revoke(round_id):
        # The sweep lands right after registration, before the arm is announced.
        host_box["host"].revoke_for_run("run-1", {"question": lambda s: None})

    host = InterruptRoundHost(_seams(seen, on_add=_revoke))
    host_box["host"] = host
    outcome = host.run_round(
        "question", "r1", {"round_id": "r1", "session_id": "s1"}, state,
        session_id="s1", owning_session_id="s1", deadline=None, is_parked=False,
    )
    assert outcome == "revoked"
    assert all(raised is False for _total, _kind, raised in seen), seen
    assert seen[-1][0] == 0


def test_a_raising_attention_hook_never_skips_the_teardown():
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
            raise RuntimeError("badge painter exploded")

    host = InterruptRoundHost(_Seams())
    state = {"event": threading.Event(), "session_id": "s1"}
    state["event"].set()
    assert host.run_round(
        "question", "r1", {"round_id": "r1", "session_id": "s1"}, state,
        session_id="s1", owning_session_id="s1", deadline=None, is_parked=False,
    ) == "decided"
    assert host.registries["question"] == {} and host.payloads["question"] == {}
