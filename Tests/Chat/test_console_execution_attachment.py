"""Selected parent visit revocation does not depend on channel success."""

import pytest


@pytest.mark.parametrize("entry", ["_sync_console_control_bar", "_sync_console_rail_and_controls"])
def test_coalesced_controls_use_current_selected_publication(monkeypatch, entry):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules import execution_state
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    updates = []
    client = SimpleNamespace(_readiness_deadline=12.0)
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(console_execution_client=client),
        is_attached=True, is_current=True, _pending_console_launch_context=None,
        _console_derivation_scope=nullcontext,
        _build_console_control_state=lambda launch: "control",
        _build_console_workbench_state=lambda control: "workbench",
        _push_console_control_state_if_changed=lambda c, w: updates.append((c, w)),
        _sync_console_settings_summary=lambda: updates.append("settings"),
        _sync_console_selected_recovery=lambda: updates.append("recovery"),
        _active_console_settings_readiness=lambda: "readiness",
        _apply_console_transcript_guidance=lambda value: updates.append(value),
        _sync_console_pending_delete_confirmation=lambda: updates.append("delete confirmation"),
        _sync_console_inspector_and_rail=lambda: updates.append("inspector"),
        _sync_console_cost_chip=lambda: updates.append("cost"),
        _sync_console_command_popup=lambda: updates.append("popup"),
    )
    monkeypatch.setattr(execution_state, "selected_client", lambda receiver: client)
    monkeypatch.setattr(execution_state, "_require_methods", lambda *args: None)
    clock = [10.0]
    monkeypatch.setattr(execution_state, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(execution_state, "refresh_selected_when_due", lambda _: updates.append("refresh"))

    sync = getattr(ChatScreen, entry)
    assert sync(screen)
    assert updates == [
        ("control", "workbench"), "settings", "recovery", "readiness",
        "delete confirmation", "inspector", "cost",
    ]
    updates.clear()
    ChatScreen._sync_console_workbench_actions_from_draft(screen)
    assert updates == [("control", "workbench"), "popup"]
    updates.clear()
    clock[0] = 12.0
    assert not sync(screen)
    assert updates == ["refresh"]
    updates.clear()
    ChatScreen._sync_console_workbench_actions_from_draft(screen)
    assert updates == ["refresh", "popup"]
    updates.clear()
    clock[0] = 10.0
    screen._build_console_workbench_state = lambda _: clock.__setitem__(0, 13.0)
    assert not sync(screen)
    assert updates == ["refresh"]
    updates.clear()
    screen._console_sync_maintenance_paused = True
    assert not sync(screen)
    assert not updates
    screen._console_sync_maintenance_paused = False
    screen.is_current = False
    monkeypatch.setattr(execution_state, "selected_client", lambda _: pytest.fail("covered view was used"))
    assert not sync(screen)


def test_selected_guidance_keeps_blocking_modal_on_original_readiness(monkeypatch):
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules import execution_state
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    values = []
    client = SimpleNamespace(_readiness_deadline=2.0)
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(console_execution_client=client),
        is_attached=True, is_current=True,
        _apply_console_transcript_guidance=values.append,
    )
    monkeypatch.setattr(execution_state, "selected_client", lambda _: client)
    monkeypatch.setattr(execution_state, "_require_methods", lambda *args: None)
    monkeypatch.setattr(execution_state, "settings_readiness", lambda *_: "current")
    monkeypatch.setattr(execution_state, "refresh_selected_when_due", lambda _: values.append("refresh"))
    ChatScreen._sync_console_transcript_guidance(screen)

    def expired(*args):
        raise execution_state.ConsoleDisplayExpired("expired")

    monkeypatch.setattr(execution_state, "settings_readiness", expired)
    ChatScreen._sync_console_transcript_guidance(screen)
    assert values == ["current", "refresh"]


def test_full_selected_inspector_drives_summary_composer_and_rail(monkeypatch):
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules import execution_state
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    client = SimpleNamespace(_readiness_deadline=float("inf"))
    state = SimpleNamespace(can_save_chatbook=True)
    rendered = []
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(console_execution_client=client),
        _pending_console_launch_context=None,
        _build_console_inspector_state=lambda _: state,
        query_one=lambda selector, _: SimpleNamespace(
            sync_state=lambda value: rendered.append((selector, value)),
        ),
        _console_chatbook_action_available=lambda: False,
        _sync_console_composer_action_state=lambda *, can_save_chatbook: rendered.append(("composer", can_save_chatbook)),
        _current_console_rail_state=lambda *, inspector_state: ("rail", inspector_state),
        _sync_console_rail_visibility_if_changed=rendered.append,
    )
    monkeypatch.setattr(execution_state, "selected_client", lambda _: client)
    ChatScreen._sync_console_inspector_and_rail(screen)
    assert rendered == [
        ("#console-run-inspector-state", state), ("#console-send-authority-summary", state),
        ("composer", False), ("rail", state),
    ]
    rendered.clear()
    client._readiness_deadline = 0.0
    with pytest.raises(execution_state.ConsoleDisplayExpired):
        ChatScreen._sync_console_inspector_and_rail(screen)
    assert not rendered


@pytest.mark.asyncio
async def test_due_refresh_is_accepted_by_textual_async_workers(monkeypatch):
    from types import SimpleNamespace

    from textual.worker import Worker
    from tldw_chatbook.UI.Console_Modules import execution_decisions, execution_state

    calls = []

    async def reconcile(screen):
        calls.append(screen)

    work = []
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(console_execution_client=SimpleNamespace(_visit=3)),
        is_attached=True, is_current=True, _console_attach_reconcile_running=False,
        run_worker=lambda callback, **_: work.append(callback) or SimpleNamespace(is_finished=False),
    )
    monkeypatch.setattr(execution_state, "reconcile_selected_view", reconcile)
    monkeypatch.setattr(execution_state, "selected_client", lambda _: SimpleNamespace(_readiness_deadline=0))
    monkeypatch.setattr(execution_decisions, "is_owned_project_modal", lambda *_: False)
    execution_state.refresh_selected_when_due(screen)
    execution_state.refresh_selected_when_due(screen)
    assert len(work) == 1
    await Worker._run_async(SimpleNamespace(_work=work[0]))
    assert calls == [screen]


def test_selected_resize_waits_for_the_current_visit_publication(monkeypatch):
    from types import SimpleNamespace

    from tldw_chatbook.UI.Console_Modules import execution_state

    client = SimpleNamespace(_visit=4, _readiness_deadline=10.0)
    screen = SimpleNamespace()
    monkeypatch.setattr(execution_state, "selected_client", lambda _screen: client)
    monkeypatch.setattr(execution_state, "time", SimpleNamespace(monotonic=lambda: 10.0))
    refreshes = []
    monkeypatch.setattr(execution_state, "refresh_selected_when_due", refreshes.append)
    assert execution_state.defer_selected_resize(screen, "first")
    assert execution_state.defer_selected_resize(screen, "latest")
    assert screen._console_pending_process_resize == (client, 4, "latest")
    assert refreshes == [screen, screen]
    client._readiness_deadline = 10.5
    assert not execution_state.defer_selected_resize(screen, "latest")
    assert screen._console_pending_process_resize is None


def test_detach_revokes_visit_before_missing_projection_row():
    from tldw_chatbook.Chat.console_execution_process import ConsoleExecutionClient
    from tldw_chatbook.Chat.console_execution_projection_view import ConsoleProjectionView

    client = ConsoleExecutionClient(None)
    old_view = object()
    client._view, client._visit = old_view, 7
    client.projection_view = ConsoleProjectionView(0, (), None, (), ())
    with pytest.raises(LookupError):
        client.detach_view(old_view, 7)
    assert client._view is None
    assert client._visit == 8
    assert client.detach_view(old_view, 7) is False
    assert not client._view_checkpoints


def test_projection_freshness_uses_the_actual_capture_in_this_request():
    from tldw_chatbook.Chat.console_execution_process import (
        ConsoleChannelLost,
        ConsoleReadinessExpired,
        _projection_readiness_deadline,
    )

    # A long queue precedes the checked capture. Transport ages it normally.
    assert _projection_readiness_deadline(10.0, 22.0, 0.75, 22.5) == 22.75
    with pytest.raises(ConsoleReadinessExpired, match="expired"):
        _projection_readiness_deadline(10.0, 22.0, 0.75, 22.75)
    for capture in (9.0, 23.0):
        with pytest.raises(ConsoleChannelLost, match="outside"):
            _projection_readiness_deadline(10.0, capture, 0.75, 22.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["fresh", "superseded", "channel_lost"])
async def test_expired_view_read_rechecks_the_visit_and_never_reissues_mount(outcome):
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_execution_process import (
        ConsoleChannelLost, ConsoleExecutionClient, ConsoleReadinessExpired,
    )

    client = ConsoleExecutionClient(None)
    screen = object()
    client.projection_view = SimpleNamespace(active_session_id="chat")
    client.require_view = lambda view, visit: (view is screen and visit == 7) or pytest.fail()
    checkpoints = []

    async def read(session_id, incarnation_id, *, _preparation, _checkpoint):
        assert (session_id, incarnation_id, _preparation) == ("chat", "inc", None)
        checkpoints.append(_checkpoint)
        if len(checkpoints) > 1:
            return "fresh"
        if outcome == "channel_lost":
            raise ConsoleChannelLost("unknown retirement")
        if outcome == "superseded":
            client._source_revision += 1
        raise ConsoleReadinessExpired("expired")

    client.read_projection = read
    if outcome == "fresh":
        assert await client.read_view_display(screen, 7, "chat", "inc", checkpoint="mount") == "fresh"
        assert checkpoints == ["mount", None]
    else:
        with pytest.raises(ConsoleChannelLost, match="superseded|unknown retirement"):
            await client.read_view_display(screen, 7, "chat", "inc", checkpoint="mount")
        assert checkpoints == ["mount"]
