"""Selected user rail inputs retain order without replaying stale effects."""

from types import SimpleNamespace

import pytest


def _surface(monkeypatch):
    from tldw_chatbook.UI.Console_Modules import execution_rail
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    clock = [2.0]
    row = SimpleNamespace(id="a", incarnation_id="a1")
    view = SimpleNamespace(active_session_id="a", session=lambda _: row)
    client = SimpleNamespace(_visit=3, _readiness_deadline=1.0, projection_view=view)
    state = SimpleNamespace(details_open=False, character_open=False, agent_open=True)
    effects = []

    def preference(*, section_updates, notify_on_failure, _reconcile=True):
        assert notify_on_failure is False
        assert _reconcile is False
        for section, opened in section_updates.items():
            effects.append(("preference", section, opened))
            setattr(state, f"{section}_open", opened)

    class RailSurface(SimpleNamespace):
        _apply_console_rail_section_toggle = ChatScreen._apply_console_rail_section_toggle
        _prepare_console_rail_section_toggle = ChatScreen._prepare_console_rail_section_toggle
        _write_console_rail_section_toggle = ChatScreen._write_console_rail_section_toggle
        _finish_console_rail_section_toggle = ChatScreen._finish_console_rail_section_toggle

    screen = RailSurface(
        app_instance=SimpleNamespace(console_execution_client=client),
        _current_console_rail_state=lambda: state,
        _set_console_rail_preference=preference,
        _sync_console_rail_visibility_if_changed=lambda _: None,
        query_one=lambda *_: SimpleNamespace(
            apply_section_open=lambda section, opened: effects.append(("widget", section, opened)),
        ),
        _request_console_context_allocation_reconcile=lambda: effects.append(("allocation",)),
        _character=SimpleNamespace(invalidate_refresh_scope=lambda: effects.append(("character",))),
        _pending_character_return_focus_id="character",
        _workspace=SimpleNamespace(restore_character_navigation_focus=lambda: effects.append(("focus",))),
        _agent=SimpleNamespace(_console_agent_fleet_summary_line=lambda: "active"),
    )
    client.require_view = lambda actual, visit: (
        None if actual is screen and visit == client._visit else pytest.fail("stale view applied")
    )
    monkeypatch.setattr(execution_rail, "selected_client", lambda actual: client if actual is screen else None)
    monkeypatch.setattr(execution_rail, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    refreshes = []
    monkeypatch.setattr(execution_rail, "refresh_selected_when_due", refreshes.append)
    return execution_rail, ChatScreen, screen, client, row, effects, refreshes


def test_expired_rail_toggles_replay_original_effects_in_order(monkeypatch):
    module, screen_type, screen, client, _, effects, refreshes = _surface(monkeypatch)
    for section, opened in (("details", None), ("details", None), ("character", True), ("agent", False)):
        screen_type._toggle_console_rail_section(screen, section, next_open=opened)
    assert effects == []
    assert refreshes == [screen] * 4
    client._readiness_deadline = 3.0
    module.replay_selected_rail_toggles(screen, client, 3)
    assert [item for item in effects if item[0] == "preference"] == [
        ("preference", "details", True), ("preference", "details", False),
        ("preference", "character", True), ("preference", "agent", False),
    ]
    assert ("character",) in effects and ("focus",) in effects
    assert screen._agent_section_user_dismissed_while_busy is True
    assert not screen._console_pending_process_rail_toggles
    count = len(effects)
    module.replay_selected_rail_toggles(screen, client, 3)
    assert len(effects) == count
    screen_type._toggle_console_rail_section(screen, "details", next_open=True)
    assert effects[count] == ("preference", "details", True)


@pytest.mark.parametrize("changed", ["visit", "session"])
def test_rail_toggle_never_applies_to_a_superseded_scope(monkeypatch, changed):
    module, screen_type, screen, client, row, effects, _ = _surface(monkeypatch)
    screen_type._toggle_console_rail_section(screen, "details", next_open=True)
    if changed == "visit":
        client._visit += 1
    else:
        row.id, row.incarnation_id = "b", "b1"
        client.projection_view.active_session_id = "b"
    client._readiness_deadline = 3.0
    module.replay_selected_rail_toggles(screen, client, client._visit)
    assert not effects
    assert not screen._console_pending_process_rail_toggles


def test_original_rail_effect_failure_is_not_replayed(monkeypatch):
    module, screen_type, screen, client, _, effects, _ = _surface(monkeypatch)
    client._readiness_deadline = 3.0
    error = RuntimeError("original preference failure")

    def fail(**_):
        effects.append("attempted")
        raise error

    screen._set_console_rail_preference = fail
    with pytest.raises(RuntimeError) as raised:
        screen_type._toggle_console_rail_section(screen, "details", next_open=True)
    assert raised.value is error
    module.replay_selected_rail_toggles(screen, client, 3)
    assert effects == ["attempted"]


def test_expiry_during_pre_effect_derivation_retains_the_actual_intent(monkeypatch):
    from tldw_chatbook.UI.Console_Modules.execution_state import ConsoleDisplayExpired

    module, screen_type, screen, client, _, effects, refreshes = _surface(monkeypatch)
    client._readiness_deadline = 3.0
    original = screen._current_console_rail_state

    def expired():
        raise ConsoleDisplayExpired("original display expired during derivation")

    screen._current_console_rail_state = expired
    screen_type._toggle_console_rail_section(screen, "details")
    assert effects == []
    assert refreshes == [screen]
    assert len(screen._console_pending_process_rail_toggles) == 1
    screen._current_console_rail_state = original
    module.replay_selected_rail_toggles(screen, client, 3)
    assert effects == [("preference", "details", True), ("widget", "details", True), ("allocation",)]
    assert not screen._console_pending_process_rail_toggles


def test_post_write_expiry_finishes_character_effects_without_resubmission(monkeypatch):
    from tldw_chatbook.UI.Console_Modules.execution_state import ConsoleDisplayExpired

    module, screen_type, screen, client, _, effects, refreshes = _surface(monkeypatch)
    client._readiness_deadline = 3.0
    original = screen._current_console_rail_state
    calls = []

    def current():
        calls.append(None)
        if len(calls) == 2:
            raise ConsoleDisplayExpired("original post-write display expired")
        return original()

    screen._current_console_rail_state = current
    screen_type._toggle_console_rail_section(screen, "character", next_open=True)
    assert effects == [("preference", "character", True)]
    assert refreshes == [screen]
    module.replay_selected_rail_toggles(screen, client, 3)
    assert effects == [
        ("preference", "character", True), ("widget", "character", True),
        ("allocation",), ("character",), ("focus",),
    ]
    assert len(calls) == 3  # Retry only the post-write derivation.
    module.replay_selected_rail_toggles(screen, client, 3)
    assert len(effects) == 5


def test_post_write_refresh_keeps_following_toggles_in_fifo_order(monkeypatch):
    from tldw_chatbook.UI.Console_Modules.execution_state import ConsoleDisplayExpired

    module, screen_type, screen, client, _, effects, _ = _surface(monkeypatch)
    client._readiness_deadline = 3.0
    original = screen._current_console_rail_state

    def current():
        if effects:
            raise ConsoleDisplayExpired("post-write display is still expired")
        return original()

    screen._current_console_rail_state = current
    screen_type._toggle_console_rail_section(screen, "details")
    screen_type._toggle_console_rail_section(screen, "details")
    assert effects == [("preference", "details", True)]
    screen._current_console_rail_state = original
    module.replay_selected_rail_toggles(screen, client, 3)
    assert effects == [
        ("preference", "details", True), ("widget", "details", True), ("allocation",),
        ("preference", "details", False), ("widget", "details", False), ("allocation",),
    ]


@pytest.mark.parametrize("changed", ["visit", "session"])
def test_post_write_expiry_never_finishes_on_a_superseded_scope(monkeypatch, changed):
    from tldw_chatbook.UI.Console_Modules.execution_state import ConsoleDisplayExpired

    module, screen_type, screen, client, row, effects, _ = _surface(monkeypatch)
    client._readiness_deadline = 3.0
    original = screen._current_console_rail_state

    def current():
        if effects:
            raise ConsoleDisplayExpired("original post-write display expired")
        return original()

    screen._current_console_rail_state = current
    screen_type._toggle_console_rail_section(screen, "character", next_open=True)
    assert effects == [("preference", "character", True)]
    if changed == "visit":
        client._visit += 1
    else:
        row.id, row.incarnation_id = "b", "b1"
        client.projection_view.active_session_id = "b"
    screen._current_console_rail_state = original
    module.replay_selected_rail_toggles(screen, client, client._visit)
    assert effects == [("preference", "character", True)]
    assert not screen._console_pending_process_rail_toggles
