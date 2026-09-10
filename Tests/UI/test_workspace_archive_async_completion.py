"""Deterministic workspace lifecycle and late-resume ownership regressions."""

import asyncio
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.UI.Console_Modules import workspace
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


def _host(registry):
    tasks, dialogs = [], []
    app = SimpleNamespace(workspace_registry_service=registry, notify=Mock())
    app.run_worker = lambda coroutine, **_: tasks.append(asyncio.create_task(coroutine))
    screen = SimpleNamespace(
        app=app,
        app_instance=app,
        is_mounted=True,
        _settings_selected_workspace_id="w",
        _settings_workspace_archive_receipt=None,
        _set_settings_workspaces_result=Mock(),
        _refresh_settings_workspaces_pane=Mock(),
        query_one=lambda *_: SimpleNamespace(value="Restored"),
    )
    app.screen = screen

    app.dialog_callbacks = {}

    def push_screen(dialog, **kwargs):
        dialogs.append(dialog)
        app.dialog_callbacks[id(dialog)] = kwargs.get("callback")
        mounted = asyncio.get_running_loop().create_future()
        mounted.set_result(None)
        return mounted

    app.push_screen = push_screen
    controller = SimpleNamespace(
        app_instance=app,
        _screen=screen,
        push_screen=app.push_screen,
        run_worker=app.run_worker,
        _invalidate_console_persisted_rows_cache=Mock(),
        _sync_console_chat_core_state=Mock(),
        _sync_native_console_chat_ui=AsyncMock(),
        _activate_console_session_for_workspace=Mock(),
    )
    return screen, controller, tasks, dialogs


async def _drain(tasks):
    for task in tasks:
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["console", "settings"])
async def test_workspace_archive_dispatches_read_and_write_offloop(surface):
    threads = []
    record = SimpleNamespace(workspace_id="w", active=False, name="Workspace")

    def storage(_):
        threads.append(threading.get_ident())
        return record

    screen, controller, tasks, dialogs = _host(
        SimpleNamespace(get_workspace=storage, archive_workspace=storage)
    )
    if surface == "console":
        ConsoleWorkspaceController._confirm_console_workspace_archive(controller, "w")
    else:
        # A newly mounted Settings screen has not published any receipt yet.
        del screen._settings_workspace_archive_receipt
        SettingsScreen.handle_workspace_archive(screen, SimpleNamespace(stop=Mock()))
    await _drain(tasks)
    await dialogs[0].confirm_callback()
    await _drain(tasks)
    if surface == "settings":
        assert screen._settings_workspace_archive_receipt is record
    assert len(threads) == 2
    assert threading.get_ident() not in threads


@pytest.mark.asyncio
async def test_settings_unarchive_dispatches_write_offloop():
    threads = []

    def restore(*_, **__):
        threads.append(threading.get_ident())
        return SimpleNamespace(workspace_id="w", name="Restored")

    screen, _, tasks, _ = _host(SimpleNamespace(unarchive_workspace=restore))
    SettingsScreen.handle_workspace_unarchive(screen, SimpleNamespace(stop=Mock()))
    await _drain(tasks)
    assert len(threads) == 1
    assert threading.get_ident() not in threads


@pytest.mark.asyncio
@pytest.mark.parametrize("superseded", [False, True])
async def test_late_existing_resume_keeps_authority_through_token_preparation(
    monkeypatch,
    superseded,
):
    session = SimpleNamespace(id="existing", persisted_conversation_id="saved")
    store = SimpleNamespace(active_session_id="prior", sessions=lambda: [session])
    chat = SimpleNamespace(store=store, switch_session=Mock())
    current = [True]

    async def prepare(*_):
        current[0] = not superseded
        await asyncio.sleep(0)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.conversation_token_preparation.prepare_conversation_tokens",
        prepare,
    )
    monkeypatch.setattr(
        workspace,
        "load_console_conversation_tree",
        AsyncMock(return_value={"conversation": {}}),
    )
    host = SimpleNamespace(
        app_instance=SimpleNamespace(notify=Mock()),
        _screen=object(),
        _ensure_console_chat_store=lambda: store,
        _capture_console_draft_switch_snapshot=Mock(),
        _console_session_id_for_workspace_conversation=lambda _: "existing",
        _ensure_chat_controller_fn=lambda: chat,
        _set_active_workspace_for_console_session=Mock(),
        _refresh_console_effective_scope_and_sync=AsyncMock(),
        _sync_console_chat_core_state=Mock(),
        _sync_native_console_chat_ui_fn=AsyncMock(),
        _sync_temporary_chip_fn=Mock(),
        _focus_composer_if_needed_fn=Mock(),
        _refresh_console_conversation_browser_after_selection=AsyncMock(),
        _restore_console_session_after_failed_open=AsyncMock(),
    )
    host.open_console_workspace_conversation = MethodType(
        ConsoleWorkspaceController.open_console_workspace_conversation, host
    )
    result = await ConsoleWorkspaceController._resume_console_workspace_conversation(
        host, "saved", preserve_persisted_scope=True, resume_if=lambda: current[0]
    )
    if superseded:
        assert result is None
        chat.switch_session.assert_not_called()
        host._set_active_workspace_for_console_session.assert_not_called()
        host._sync_native_console_chat_ui_fn.assert_not_called()
    else:
        assert result is True
        chat.switch_session.assert_called_once_with("existing")
        host._sync_native_console_chat_ui_fn.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["console", "settings"])
@pytest.mark.parametrize("leave_screen", [False, True])
async def test_workspace_archive_cancellation_keeps_reservation_until_commit(
    surface, leave_screen
):
    started, release = threading.Event(), threading.Event()
    record = SimpleNamespace(workspace_id="w", active=False, name="Workspace")
    ui_thread = threading.get_ident()

    def archive(_):
        assert threading.get_ident() != ui_thread
        started.set()
        assert release.wait(5)
        return record

    screen, controller, tasks, dialogs = _host(
        SimpleNamespace(get_workspace=lambda _: record, archive_workspace=archive)
    )
    session = SimpleNamespace(
        id="open", workspace_id="w", persisted_conversation_id="saved", draft=""
    )
    screen.app.console_runtime = SimpleNamespace(
        chat_store=SimpleNamespace(
            sessions=lambda: [session], messages_for_session=lambda _: []
        ),
        chat_controller=None,
    )
    screen.app._conversation_archive_inflight = {"unrelated"}
    if surface == "console":
        ConsoleWorkspaceController._confirm_console_workspace_archive(controller, "w")
    else:
        SettingsScreen.handle_workspace_archive(screen, SimpleNamespace(stop=Mock()))
    await _drain(tasks)
    original_tasks = asyncio.all_tasks()
    confirmation = asyncio.create_task(dialogs[0].confirm_callback())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert screen.app._conversation_archive_inflight == {"saved", "unrelated"}
        assert len(screen.app._workspace_lifecycle_operations) == 1
        if leave_screen:
            screen.app.screen = object()
        confirmation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await confirmation
        assert screen.app._conversation_archive_inflight == {"saved", "unrelated"}
    finally:
        release.set()
        remaining = asyncio.all_tasks() - original_tasks
        if remaining:
            await asyncio.wait_for(
                asyncio.gather(*remaining, return_exceptions=True), 3
            )
    assert screen.app._conversation_archive_inflight == {"unrelated"}
    assert not screen.app._workspace_lifecycle_operations
    if surface == "console":
        assert controller._console_workspace_archive_receipt is (
            record if leave_screen else None
        )
        controller._invalidate_console_persisted_rows_cache.assert_called_once()
        if leave_screen:
            controller._sync_console_chat_core_state.assert_not_called()
            screen.app.notify.assert_not_called()
            assert len(dialogs) == 1
        else:
            controller._sync_console_chat_core_state.assert_called_once()
            assert len(dialogs) == 2
            assert dialogs[1].__class__.__name__ == "WorkspaceArchiveReceiptModal"
    else:
        assert screen._settings_workspace_archive_receipt is record
        if leave_screen:
            assert screen._settings_selected_workspace_id == "w"
            screen._refresh_settings_workspaces_pane.assert_not_called()
        else:
            assert screen._settings_selected_workspace_id is None
            screen._refresh_settings_workspaces_pane.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("new_receipt", [False, True])
async def test_settings_restore_cancellation_retires_only_its_receipt(new_receipt):
    started, release = threading.Event(), threading.Event()
    receipt = SimpleNamespace(workspace_id="w")
    replacement = SimpleNamespace(workspace_id="w")

    def restore(*_, **__):
        started.set()
        assert release.wait(5)
        return SimpleNamespace(workspace_id="w", name="Restored")

    screen, _, tasks, _ = _host(SimpleNamespace(unarchive_workspace=restore))
    screen._settings_workspace_archive_receipt = receipt
    original_tasks = asyncio.all_tasks()
    SettingsScreen.handle_workspace_unarchive(screen, SimpleNamespace(stop=Mock()))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        if new_receipt:
            screen._settings_workspace_archive_receipt = replacement
        screen.app.screen = object()
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
    finally:
        release.set()
        remaining = asyncio.all_tasks() - original_tasks
        if remaining:
            await asyncio.wait_for(
                asyncio.gather(*remaining, return_exceptions=True), 3
            )
    assert screen._settings_workspace_archive_receipt is (
        replacement if new_receipt else None
    )
    screen._refresh_settings_workspaces_pane.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["console", "settings"])
async def test_workspace_archive_read_completion_cannot_reopen_after_navigation(
    surface,
):
    started, release = threading.Event(), threading.Event()

    def read(_):
        started.set()
        assert release.wait(5)
        return SimpleNamespace(workspace_id="w", active=False, name="Workspace")

    screen, controller, tasks, dialogs = _host(SimpleNamespace(get_workspace=read))
    if surface == "console":
        ConsoleWorkspaceController._confirm_console_workspace_archive(controller, "w")
    else:
        SettingsScreen.handle_workspace_archive(screen, SimpleNamespace(stop=Mock()))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        screen.app.screen = object()
    finally:
        release.set()
        await asyncio.wait_for(_drain(tasks), 3)
    assert not dialogs


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["console", "settings"])
@pytest.mark.parametrize("already_reserved", [False, True])
async def test_workspace_archive_failure_never_releases_another_reservation(
    surface, already_reserved
):
    from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError

    record = SimpleNamespace(workspace_id="w", active=False, name="Workspace")
    archive = Mock(side_effect=WorkspaceRegistryServiceError("locked"))
    screen, controller, tasks, dialogs = _host(
        SimpleNamespace(get_workspace=lambda _: record, archive_workspace=archive)
    )
    session = SimpleNamespace(
        id="open", workspace_id="w", persisted_conversation_id="saved", draft=""
    )
    screen.app.console_runtime = SimpleNamespace(
        chat_store=SimpleNamespace(
            sessions=lambda: [session], messages_for_session=lambda _: []
        ),
        chat_controller=None,
    )
    original = {"unrelated", "saved"} if already_reserved else {"unrelated"}
    screen.app._conversation_archive_inflight = original.copy()
    if surface == "console":
        ConsoleWorkspaceController._confirm_console_workspace_archive(controller, "w")
    else:
        SettingsScreen.handle_workspace_archive(screen, SimpleNamespace(stop=Mock()))
    await _drain(tasks)
    await dialogs[0].confirm_callback()
    assert screen.app._conversation_archive_inflight == original
    assert archive.call_count == (0 if already_reserved else 1)
    assert screen._settings_workspace_archive_receipt is None
    controller._invalidate_console_persisted_rows_cache.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["console", "settings"])
@pytest.mark.parametrize("stage", ["get_workspace", "archive_workspace"])
async def test_workspace_archive_unexpected_storage_failure_is_recoverable(
    surface, stage
):
    record = SimpleNamespace(workspace_id="w", active=False, name="Workspace")
    registry = SimpleNamespace(
        get_workspace=Mock(return_value=record),
        archive_workspace=Mock(return_value=record),
    )
    getattr(registry, stage).side_effect = OSError("unavailable")
    screen, controller, tasks, dialogs = _host(registry)
    if surface == "console":
        ConsoleWorkspaceController._confirm_console_workspace_archive(controller, "w")
    else:
        SettingsScreen.handle_workspace_archive(screen, SimpleNamespace(stop=Mock()))
    await _drain(tasks)
    if stage == "archive_workspace":
        await dialogs[0].confirm_callback()
    feedback = (
        screen.app.notify
        if surface == "console"
        else screen._set_settings_workspaces_result
    )
    assert "could not" in feedback.call_args.args[0].lower()
    assert not getattr(screen.app, "_workspace_lifecycle_operations", set())
    assert not getattr(screen.app, "_conversation_archive_inflight", set())


@pytest.mark.asyncio
async def test_settings_restore_unexpected_storage_failure_keeps_receipt():
    screen, _, tasks, _ = _host(
        SimpleNamespace(unarchive_workspace=Mock(side_effect=OSError("unavailable")))
    )
    receipt = screen._settings_workspace_archive_receipt = SimpleNamespace(
        workspace_id="w"
    )
    SettingsScreen.handle_workspace_unarchive(screen, SimpleNamespace(stop=Mock()))
    await _drain(tasks)
    assert screen._settings_workspace_archive_receipt is receipt
    assert "Retry Restore" in screen._set_settings_workspaces_result.call_args.args[0]


@pytest.mark.asyncio
async def test_cancelled_archive_receipt_appears_after_late_dialog_dismissal():
    started, release = threading.Event(), threading.Event()
    record = SimpleNamespace(workspace_id="w", active=False, name="Workspace")

    def archive(_):
        started.set()
        assert release.wait(5)
        return record

    screen, controller, tasks, dialogs = _host(
        SimpleNamespace(get_workspace=lambda _: record, archive_workspace=archive)
    )
    ConsoleWorkspaceController._confirm_console_workspace_archive(controller, "w")
    await _drain(tasks)
    dialog = dialogs[0]
    screen.app.screen = dialog
    confirmation = asyncio.create_task(dialog.confirm_callback())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        confirmation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await confirmation
    finally:
        release.set()
        await asyncio.gather(*screen.app._workspace_lifecycle_operations)
    assert controller._console_workspace_archive_receipt is record
    assert dialogs == [dialog]
    screen.app.screen = screen
    await screen.app.dialog_callbacks[id(dialog)](False)
    await _drain(tasks)
    assert len(dialogs) == 2
    assert dialogs[1].__class__.__name__ == "WorkspaceArchiveReceiptModal"
    assert controller._console_workspace_archive_receipt is None


@pytest.mark.asyncio
async def test_workspace_switcher_exposes_receipt_retained_while_away():
    record = SimpleNamespace(
        workspace_id="w", active=False, name="Workspace", archived=True
    )
    _, controller, tasks, dialogs = _host(
        SimpleNamespace(
            list_workspaces=lambda **_: [record], get_active_workspace=lambda: None
        )
    )
    controller._console_workspace_archive_receipt = record
    ConsoleWorkspaceController._open_console_workspace_switcher(controller)
    await _drain(tasks)
    assert dialogs[0]._show_archived is True
    assert controller._console_workspace_archive_receipt is None
