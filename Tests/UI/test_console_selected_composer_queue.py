"""Selected queue facts feed the original composer action-state renderer."""

from dataclasses import replace
from types import SimpleNamespace

from Tests.Chat.test_console_execution_queue_values import _facts
from tldw_chatbook.Chat.console_display_state import ConsoleBlockedReason
from tldw_chatbook.UI.Console_Modules import composer_run_controls, execution_state
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def test_selected_original_composer_renders_queue_stop_and_each_existing_blocker(
    monkeypatch,
):
    _, facts = _facts(accepted=True)
    actions, shelves, queue_badges = [], [], []
    composer = SimpleNamespace(
        collapsed=False,
        has_sendable_payload=lambda: True,
        sync_action_state=lambda **state: actions.append(state),
        sync_prompt_queue_state=lambda **state: queue_badges.append(state),
        set_pending_attachment_label=lambda *args, **kwargs: None,
    )
    store = SimpleNamespace(
        active_session_id="session", pending_attachments=lambda _: []
    )
    controller = SimpleNamespace(store=store, is_stop_allowed=True)
    client = SimpleNamespace(
        _readiness_deadline=float("inf"),
        session_composition=SimpleNamespace(
            session_id="session", queue=facts, recovery=SimpleNamespace(dispatch=None)
        ),
    )
    regions = {
        "#console-native-composer": composer,
        "#console-prompt-queue": SimpleNamespace(
            sync_presentation=lambda *args: shelves.append(args)
        ),
        "#console-dispatch-recovery": SimpleNamespace(sync_recovery=lambda *_: None),
    }
    screen = SimpleNamespace(
        query_one=lambda name, *_: regions[name],
        _console_chat_controller=controller,
        _console_chat_store=store,
        _console_send_ack=None,
        _video=SimpleNamespace(_console_videogen_inflight_sessions=lambda: set()),
        _image=SimpleNamespace(
            _h3_image_edit_registry=lambda: SimpleNamespace(active=lambda _: None)
        ),
        _console_composer_send_blocker=lambda: ConsoleBlockedReason(""),
        _console_attachment_blocked_reason=lambda: "",
        _console_emergency_stop_copy="",
        _console_active_session_is_ephemeral=lambda: False,
    )
    monkeypatch.setattr(execution_state, "selected_client", lambda _: client)
    monkeypatch.setattr(composer_run_controls, "sync_stop_affordances", lambda _: None)
    monkeypatch.setattr(
        composer_run_controls, "sync_dictation_presentation", lambda *_: None
    )

    def sync():
        ChatScreen._sync_console_composer_action_state(screen, can_save_chatbook=True)
        return actions[-1]

    state = sync()
    assert state["send_label"] == "Queue" and not state["send_blocked"]
    assert state["run_active"] and state["stop_available"]
    assert shelves[-1][0] == "session" and queue_badges[-1] == {
        "count": 0,
        "paused": False,
    }
    client.session_composition.queue = replace(
        facts, dispatch_recovery_blocks_submission=True
    )
    assert sync()["send_blocked"] and actions[-1]["dispatch_recovery_blocked"]
    client.session_composition.queue = facts
    screen._console_composer_send_blocker = lambda: ConsoleBlockedReason("Choose model")
    assert (
        sync()["send_blocked"]
        and actions[-1]["setup_blocked_reason"].text == "Choose model"
    )
    screen._console_composer_send_blocker = lambda: ConsoleBlockedReason("")
    screen._console_emergency_stop_copy = "Stopped"
    assert sync()["send_blocked"] and actions[-1]["queue_blocked_reason"]


def test_selected_image_gate_reuses_only_captured_vision_and_keeps_unknown_fallback(
    monkeypatch,
):
    from tldw_chatbook.Chat import attachment_core
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    settings = ConsoleSessionSettings(
        provider="openai", model="actual-model", source="user"
    )
    client = SimpleNamespace(
        _readiness_deadline=float("inf"),
        session_composition=SimpleNamespace(vision_capable=False),
    )
    screen = SimpleNamespace(
        _console_pending_image_attachment=lambda: object(),
        _active_console_settings_readiness=lambda: (settings, None),
    )
    reads = []
    monkeypatch.setattr(execution_state, "selected_client", lambda _: client)
    blocked = ChatScreen._console_attachment_blocked_reason(screen)
    assert "actual-model can't accept images" in blocked and not reads
    client.session_composition.vision_capable = True
    assert ChatScreen._console_attachment_blocked_reason(screen) == "" and not reads
    client.session_composition.vision_capable = None
    monkeypatch.setattr(
        attachment_core, "is_vision_capable", lambda p, m: reads.append((p, m)) or True
    )
    assert ChatScreen._console_attachment_blocked_reason(screen) == ""
    assert reads == [("openai", "actual-model")]
    client.session_composition.vision_capable = True
    monkeypatch.setattr(attachment_core, "is_vision_capable", lambda *_: False)
    assert "can't accept images" in ChatScreen._console_attachment_blocked_reason(
        screen
    )
