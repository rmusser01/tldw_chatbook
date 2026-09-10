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


def test_console_restore_invalidates_cached_rows_before_sync():
    events = []
    registry = SimpleNamespace(
        get_workspace=lambda _: SimpleNamespace(archived=True),
        unarchive_workspace=lambda *a, **k: SimpleNamespace(name="Restored"),
    )
    controller = SimpleNamespace(
        app_instance=SimpleNamespace(
            workspace_registry_service=registry, notify=Mock()
        ),
        _invalidate_console_persisted_rows_cache=lambda: events.append("invalidate"),
        _sync_native_console_chat_ui=lambda: events.append("sync"),
        run_worker=Mock(),
    )
    ConsoleWorkspaceController._restore_console_workspace(controller, "w")
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
