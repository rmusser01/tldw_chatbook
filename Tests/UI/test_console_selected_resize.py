"""Resize retries retain the transition until current rail values are applied.

These are orchestration controls for the original resize body, provider TTL
consumer and resize deferral. Selection is isolated; no backend receipt or
native ownership is simulated.
"""

from types import SimpleNamespace

import pytest
from textual.events import Resize
from textual.geometry import Size

from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
from tldw_chatbook.Chat.console_rail_state import ConsoleRailState
from tldw_chatbook.UI.Console_Modules import execution_state
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def _resize_fixture(monkeypatch):
    clock = [0.0]
    scheduled, applied, headers = [], [], []
    client = SimpleNamespace(
        _visit=3, _readiness_deadline=1.0, provider_model_display=("provider", "model", None),
        _source_revision=4, projection_view=object(),
    )
    app = SimpleNamespace(console_execution_client=client, focused=None)
    widgets = {name: SimpleNamespace(display=True) for name in (
        "#console-workspace-grid", "#console-left-rail", "#console-right-rail",
        "#console-context-rail-handle", "#console-inspector-rail-handle",
    )}
    screen = SimpleNamespace(
        app=app, app_instance=app, _last_console_workspace_width_band="wide",
        query_one=lambda selector: widgets[selector],
        _is_descendant_or_self=lambda *_: False,
        _request_console_control_bar_sync=lambda: headers.append(True),
        _sync_console_rail_visibility_if_changed=applied.append,
        _notify_console_responsive_rail_collapse=lambda **_: None,
    )
    monkeypatch.setattr(execution_state, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(execution_state, "selected_client", lambda _: client)
    monkeypatch.setattr(execution_state, "_awaiting_resume_publication", lambda _: False)
    monkeypatch.setattr(execution_state, "selected_rail_publication", lambda _: None)
    monkeypatch.setattr(execution_state, "refresh_selected_when_due", scheduled.append)
    rail = ConsoleRailState(
        left_open=False, right_open=False, preferred_left_open=True,
        preferred_right_open=False, single_pane=True,
    )
    return screen, client, clock, scheduled, applied, headers, rail


def _resize(screen, width=80):
    event = Resize(Size(width, 40), Size(width, 40))
    ChatScreen._adapt_console_workspace_to_width.__wrapped__(screen, event)
    return event


@pytest.mark.parametrize("expiry", ["provider_read", "after_build"])
def test_selected_resize_expiry_keeps_latest_transition_for_fresh_retry(monkeypatch, expiry):
    screen, client, clock, scheduled, applied, headers, rail = _resize_fixture(monkeypatch)
    reads = []

    def derive(*, available_columns):
        reads.append(available_columns)
        if expiry == "provider_read":
            clock[0] = 1.0
        execution_state.provider_model_display(screen, client)
        clock[0] = 1.0
        return rail

    screen._current_console_rail_state = derive
    event = _resize(screen)
    assert screen._last_console_workspace_width_band == "wide"
    assert screen._console_pending_process_resize == (client, 3, event)
    assert applied == headers == []
    assert scheduled == [screen]
    newer = _resize(screen, 82)
    assert screen._console_pending_process_resize == (client, 3, newer)
    assert reads == [80]  # Expired ingress must not start another derivation.

    client._readiness_deadline = 2.0  # The next checked publication is current.
    ChatScreen._adapt_console_workspace_to_width.__wrapped__(screen, newer)
    assert reads == [80, 82]
    assert screen._last_console_workspace_width_band == "single-pane"
    assert screen._console_pending_process_resize is None
    assert applied == [rail]
    assert headers == [True]


@pytest.mark.parametrize("error", [
    RuntimeError("source changed"), ConsoleChannelLost("channel lost"),
    execution_state.ConsoleDisplayExpired("different display expired"),
])
def test_selected_resize_does_not_defer_source_or_channel_failures(monkeypatch, error):
    screen, _, _, scheduled, applied, headers, _ = _resize_fixture(monkeypatch)

    def derive(**_):
        raise error

    screen._current_console_rail_state = derive
    with pytest.raises(type(error)) as failed:
        _resize(screen)
    assert failed.value is error
    assert screen._last_console_workspace_width_band == "wide"
    assert applied == headers == scheduled == []


def test_selected_resize_failed_layout_does_not_commit_width_band(monkeypatch):
    screen, _, _, _, _, _, rail = _resize_fixture(monkeypatch)
    screen._current_console_rail_state = lambda **_: rail
    error = RuntimeError("actual layout failed")

    def apply(_):
        raise error

    screen._sync_console_rail_visibility_if_changed = apply
    with pytest.raises(RuntimeError) as failed:
        _resize(screen)
    assert failed.value is error
    assert screen._last_console_workspace_width_band == "wide"


@pytest.mark.parametrize("changed", ["source", "visit", "publication", "client"])
def test_selected_resize_refuses_scope_change_during_build(monkeypatch, changed):
    screen, client, _, scheduled, applied, headers, rail = _resize_fixture(monkeypatch)

    def derive(**_):
        if changed == "source":
            client._source_revision += 1
        elif changed == "visit":
            client._visit += 1
        elif changed == "publication":
            client.projection_view = object()
        else:
            screen.app_instance.console_execution_client = object()
        return rail

    screen._current_console_rail_state = derive
    with pytest.raises(RuntimeError, match="resize presentation changed"):
        _resize(screen)
    assert screen._last_console_workspace_width_band == "wide"
    assert applied == headers == scheduled == []


def test_local_resize_keeps_original_derivation_and_application(monkeypatch):
    screen, client, _, scheduled, applied, headers, rail = _resize_fixture(monkeypatch)
    screen.app_instance.console_execution_client = None
    client._readiness_deadline = -1
    screen._current_console_rail_state = lambda **_: rail
    _resize(screen)
    assert screen._last_console_workspace_width_band == "single-pane"
    assert applied == [rail]
    assert headers == [True]
    assert scheduled == []
