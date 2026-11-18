"""The compatibility draft event consumes selected facts, never local owners."""

from types import SimpleNamespace

from Tests.Chat.test_console_execution_composition_values import _encode
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Chat.console_execution_composition_values import (
    decode_session_composition,
)
from tldw_chatbook.Chat.console_execution_projection_values import (
    decode_session_projection,
    encode_session_projection,
)
from tldw_chatbook.UI.Console_Modules import execution_state
from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
    ConsoleDraftSpendRefresh,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def test_selected_compatibility_draft_event_preserves_spend_hold_without_local_writes(
    monkeypatch,
):
    class Screen:
        _console_unknown_send_armed = object()

        def __getattr__(self, name):
            raise AssertionError(f"Selected draft event reached local owner: {name}")

    timers = []

    def schedule(delay, callback):
        timer = SimpleNamespace(stopped=False)
        timer.stop = lambda: setattr(timer, "stopped", True)
        timers.append((delay, callback, timer))
        return timer

    screen = Screen()
    screen._console_draft_spend_refresh = ConsoleDraftSpendRefresh(
        schedule_timer=schedule,
        sync_settings_summary=lambda: None,
        sync_cost_chip=lambda: None,
    )
    session = ConsoleChatSession()
    session.agent_handoff_state = "pending"
    row = decode_session_projection(encode_session_projection(session))

    def composition(active):
        return decode_session_composition(
            dict(_encode(session), draft_spend_run_active=active), row
        )

    client = SimpleNamespace(session_composition=composition(False))
    monkeypatch.setattr(execution_state, "selected_client", lambda _screen: client)
    ChatScreen._on_console_composer_draft_changed(
        screen, SimpleNamespace(value="draft")
    )
    assert screen._console_unknown_send_armed is None
    assert len(timers) == 1 and not timers[0][2].stopped
    client.session_composition = composition(True)
    ChatScreen._on_console_composer_draft_changed(screen, SimpleNamespace(value=""))
    assert timers[0][2].stopped and screen._console_draft_spend_refresh.timer is None
    assert len(timers) == 1
    client.session_composition = composition(False)
    ChatScreen._on_console_composer_draft_changed(screen, SimpleNamespace(value=""))
    assert len(timers) == 2
