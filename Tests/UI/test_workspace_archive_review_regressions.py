"""Targeted archive lifecycle regressions from PR review."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


@pytest.mark.parametrize("matching_receipt", [True, False])
def test_settings_restore_clears_only_matching_archive_receipt(matching_receipt):
    receipt = SimpleNamespace(workspace_id="restored" if matching_receipt else "other")
    restored = SimpleNamespace(workspace_id="restored", name="Restored")
    screen = SimpleNamespace(
        _settings_selected_workspace_id="restored",
        _settings_workspace_archive_receipt=receipt,
        app_instance=SimpleNamespace(
            workspace_registry_service=SimpleNamespace(
                unarchive_workspace=Mock(return_value=restored)
            )
        ),
        query_one=lambda *_: SimpleNamespace(value="Restored"),
        _refresh_settings_workspaces_pane=Mock(),
    )
    SettingsScreen.handle_workspace_unarchive(screen, SimpleNamespace(stop=Mock()))
    assert screen._settings_workspace_archive_receipt is (
        None if matching_receipt else receipt
    )


@pytest.mark.asyncio
async def test_console_restore_invalidates_cached_rows_before_sync():
    events = []
    registry = SimpleNamespace(
        get_workspace=lambda _: SimpleNamespace(archived=True),
        unarchive_workspace=lambda *a, **k: SimpleNamespace(name="Restored"),
    )
    controller, tasks = _recovery_host(registry)
    controller._invalidate_console_persisted_rows_cache = lambda: events.append(
        "invalidate"
    )

    async def sync():
        events.append("sync")

    controller._sync_native_console_chat_ui = sync
    ConsoleWorkspaceController._restore_console_workspace(controller, "w")
    for task in tasks:
        await task
    assert events == ["invalidate", "sync"]


@pytest.mark.asyncio
async def test_console_archive_invalidates_cached_rows_before_sync():
    events = []
    dialogs = []
    record = SimpleNamespace(active=False, name="Archive")
    registry = SimpleNamespace(
        get_workspace=lambda _: record, archive_workspace=lambda _: record
    )
    app = SimpleNamespace(workspace_registry_service=registry, notify=Mock())
    screen = SimpleNamespace(
        _console_composer_or_none=lambda: None, _console_visible_draft_session_id=None
    )
    controller = SimpleNamespace(
        app_instance=app,
        _screen=screen,
        push_screen=lambda modal, **kwargs: dialogs.append(modal),
        _invalidate_console_persisted_rows_cache=lambda: events.append("invalidate"),
        _sync_console_chat_core_state=lambda: events.append("sync"),
    )
    ConsoleWorkspaceController._confirm_console_workspace_archive(controller, "w")
    await dialogs[0].confirm_callback()
    assert events == ["invalidate", "sync"]


@pytest.mark.asyncio
@pytest.mark.parametrize("draft_during_confirmation", [False, True])
async def test_settings_archive_checks_live_composer_before_and_after_confirmation(
    draft_during_confirmation,
):
    text = ["" if draft_during_confirmation else "Unsent text"]
    owner = SimpleNamespace(
        id="owner", persisted_conversation_id="c", workspace_id="w", draft=""
    )
    store = SimpleNamespace(
        sessions=lambda: [owner],
        messages_for_session=lambda _: [],
        set_session_draft=lambda _, draft: setattr(owner, "draft", draft),
    )
    console = SimpleNamespace(
        is_mounted=True,
        _console_visible_draft_session_id="owner",
        _console_composer_or_none=lambda: SimpleNamespace(draft_text=lambda: text[0]),
    )
    dialogs = []
    registry = SimpleNamespace(
        get_workspace=lambda _: SimpleNamespace(name="Workspace"),
        archive_workspace=Mock(),
    )
    app = SimpleNamespace(
        workspace_registry_service=registry,
        console_runtime=SimpleNamespace(chat_store=store, chat_controller=None),
        _reusable_screen_instances={"chat": (object(), console)},
        push_screen=lambda modal: dialogs.append(modal),
    )
    settings = SimpleNamespace(
        app_instance=app,
        app=app,
        _settings_selected_workspace_id="w",
        _set_settings_workspaces_result=Mock(),
        _refresh_settings_workspaces_pane=Mock(),
    )
    SettingsScreen.handle_workspace_archive(settings, SimpleNamespace(stop=Mock()))
    if draft_during_confirmation:
        assert len(dialogs) == 1
        text[0] = "Unsent text"
        await dialogs[0].confirm_callback()
    else:
        assert not dialogs
    registry.archive_workspace.assert_not_called()
    assert "draft" in settings._set_settings_workspaces_result.call_args.args[0]
    assert owner.draft == text[0]


def _recovery_host(registry):
    import asyncio

    tasks = []
    app = SimpleNamespace(workspace_registry_service=registry, notify=Mock())
    app.run_worker = lambda coroutine, **_: tasks.append(asyncio.create_task(coroutine))
    screen = SimpleNamespace(is_mounted=True, app=app)
    app.screen = screen
    controller = SimpleNamespace(
        app_instance=app,
        _screen=screen,
        run_worker=app.run_worker,
        _invalidate_console_persisted_rows_cache=Mock(),
        _sync_native_console_chat_ui=Mock(),
        _open_console_workspace_switcher=Mock(),
        push_screen=Mock(),
    )
    return controller, tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["get_workspace", "unarchive_workspace"])
async def test_console_restore_reports_storage_failure(failure_stage):
    from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError

    registry = SimpleNamespace(
        get_workspace=Mock(return_value=SimpleNamespace(archived=True)),
        unarchive_workspace=Mock(),
    )
    getattr(registry, failure_stage).side_effect = WorkspaceRegistryServiceError(
        "storage unavailable"
    )
    controller, tasks = _recovery_host(registry)
    ConsoleWorkspaceController._restore_console_workspace(controller, "w")
    for task in tasks:
        await task
    assert "retry" in controller.app_instance.notify.call_args.args[0].lower()
    controller._open_console_workspace_switcher.assert_called_once_with(
        show_archived=True
    )
    if failure_stage == "get_workspace":
        registry.unarchive_workspace.assert_not_called()


@pytest.mark.asyncio
async def test_console_restore_storage_does_not_block_ui_loop():
    import asyncio
    import threading
    import time

    ui_thread = threading.get_ident()
    threads = []

    def read(_):
        threads.append(threading.get_ident())
        time.sleep(0.1)
        return SimpleNamespace(archived=True, name="Workspace")

    def restore(*_, **__):
        threads.append(threading.get_ident())
        time.sleep(0.1)
        return SimpleNamespace(name="Workspace")

    controller, tasks = _recovery_host(
        SimpleNamespace(get_workspace=read, unarchive_workspace=restore)
    )

    async def sync():
        pass

    controller._sync_native_console_chat_ui = sync
    start = time.monotonic()
    ConsoleWorkspaceController._restore_console_workspace(controller, "w")
    await asyncio.sleep(0)
    assert time.monotonic() - start < 0.08
    for task in tasks:
        await task
    assert len(threads) == 2 and all(thread != ui_thread for thread in threads)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["read", "write", None])
async def test_settings_undo_storage_runs_offloop_and_keeps_receipt_on_failure(failure):
    import threading

    from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError

    ui_thread = threading.get_ident()
    threads = []
    receipt = SimpleNamespace(workspace_id="w", name="Workspace")

    def read(_):
        threads.append(threading.get_ident())
        if failure == "read":
            raise WorkspaceRegistryServiceError("storage unavailable")
        return receipt

    def restore(_):
        threads.append(threading.get_ident())
        if failure == "write":
            raise WorkspaceRegistryServiceError("storage unavailable")
        return receipt

    controller, tasks = _recovery_host(
        SimpleNamespace(get_workspace=read, unarchive_workspace=restore)
    )
    settings = controller._screen
    settings.app_instance = controller.app_instance
    settings._settings_workspace_archive_receipt = receipt
    settings._settings_selected_workspace_id = "w"
    settings._set_settings_workspaces_result = Mock()
    settings._refresh_settings_workspaces_pane = Mock()
    SettingsScreen.handle_workspace_archive_undo(settings, SimpleNamespace(stop=Mock()))
    for task in tasks:
        await task
    assert threads and all(thread != ui_thread for thread in threads)
    if failure:
        assert settings._settings_workspace_archive_receipt is receipt
        assert (
            "retry"
            in settings._set_settings_workspaces_result.call_args.args[0].lower()
        )
    else:
        assert settings._settings_workspace_archive_receipt is None


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_operation", ["read", "write"])
async def test_console_restore_fences_publication_after_navigation(blocked_operation):
    import asyncio
    import threading

    entered, release = threading.Event(), threading.Event()

    def block():
        entered.set()
        assert release.wait(2)

    def read(_):
        if blocked_operation == "read":
            block()
        return SimpleNamespace(archived=True)

    def write(*_, **__):
        if blocked_operation == "write":
            block()
        return SimpleNamespace(name="Restored")

    registry = SimpleNamespace(
        get_workspace=read, unarchive_workspace=Mock(side_effect=write)
    )
    controller, tasks = _recovery_host(registry)
    ConsoleWorkspaceController._restore_console_workspace(controller, "w")
    try:
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(0.005)
        controller._screen.app.screen = object()
    finally:
        release.set()
        for task in tasks:
            await task
    controller._sync_native_console_chat_ui.assert_not_called()
    controller.app_instance.notify.assert_not_called()
    if blocked_operation == "read":
        registry.unarchive_workspace.assert_not_called()
    else:
        controller._invalidate_console_persisted_rows_cache.assert_called_once()


@pytest.mark.asyncio
async def test_settings_undo_does_not_replace_newer_receipt_after_delayed_write():
    import asyncio
    import threading

    entered, release = threading.Event(), threading.Event()
    receipt = SimpleNamespace(workspace_id="w", name="Old")
    newer = SimpleNamespace(workspace_id="new", name="New")

    def restore(_):
        entered.set()
        assert release.wait(2)
        return receipt

    controller, tasks = _recovery_host(
        SimpleNamespace(get_workspace=lambda _: receipt, unarchive_workspace=restore)
    )
    settings = controller._screen
    settings.app_instance = controller.app_instance
    settings._settings_workspace_archive_receipt = receipt
    settings._settings_selected_workspace_id = "w"
    settings._set_settings_workspaces_result = Mock()
    settings._refresh_settings_workspaces_pane = Mock()
    SettingsScreen.handle_workspace_archive_undo(settings, SimpleNamespace(stop=Mock()))
    try:
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(0.005)
        settings._settings_workspace_archive_receipt = newer
        settings._settings_selected_workspace_id = "new"
    finally:
        release.set()
        for task in tasks:
            await task
    assert settings._settings_workspace_archive_receipt is newer
    assert settings._settings_selected_workspace_id == "new"
    settings._refresh_settings_workspaces_pane.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["console", "settings"])
async def test_cancelled_recovery_worker_completes_started_storage_write(surface):
    import asyncio
    import threading

    entered, release, committed = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    receipt = SimpleNamespace(workspace_id="w", name="Restored", archived=True)

    def restore(*_, **__):
        entered.set()
        assert release.wait(2)
        committed.set()
        return receipt

    controller, tasks = _recovery_host(
        SimpleNamespace(get_workspace=lambda _: receipt, unarchive_workspace=restore)
    )
    settings = controller._screen
    settings.app_instance = controller.app_instance
    settings._settings_workspace_archive_receipt = receipt
    settings._settings_selected_workspace_id = "w"
    settings._set_settings_workspaces_result = Mock()
    settings._refresh_settings_workspaces_pane = Mock()
    if surface == "console":

        async def sync():
            pass

        controller._sync_native_console_chat_ui = sync
        ConsoleWorkspaceController._restore_console_workspace(controller, "w")
    else:
        SettingsScreen.handle_workspace_archive_undo(
            settings, SimpleNamespace(stop=Mock())
        )
    try:
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(0.005)
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[0]
    finally:
        release.set()
    async with asyncio.timeout(1):
        while not committed.is_set():
            await asyncio.sleep(0.005)
    # Completion runs on the UI loop after the thread commits.
    async with asyncio.timeout(1):
        while (
            not controller._invalidate_console_persisted_rows_cache.called
            if surface == "console"
            else settings._settings_workspace_archive_receipt is not None
        ):
            await asyncio.sleep(0.005)
    if surface == "console":
        controller._invalidate_console_persisted_rows_cache.assert_called_once()
    else:
        assert settings._settings_workspace_archive_receipt is None
        settings._refresh_settings_workspaces_pane.assert_called_once()
