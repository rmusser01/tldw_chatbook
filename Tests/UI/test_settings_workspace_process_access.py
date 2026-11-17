"""Settings intent publication, separate from native backend qualification."""

import asyncio
from types import MethodType, SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("leave_workspace", [False, True])
async def test_selected_folder_toggle_keeps_one_setter_and_fences_late_feedback(
    request, monkeypatch, leave_workspace
):
    from tldw_chatbook.Chat import console_execution_workspace
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
    from tldw_chatbook.state.settings_config_models import SettingsCategoryId

    release = asyncio.Event()
    publications, refreshes, calls, workers = [], [], [], []

    async def retained_operation():
        await release.wait()

    operation = asyncio.create_task(retained_operation())

    def start(client, workspace_id, binding_id, *, allow_write):
        calls.append((client, workspace_id, binding_id, allow_write))
        return operation

    monkeypatch.setattr(
        console_execution_workspace, "set_workspace_binding_access", start
    )
    binding = SimpleNamespace(binding_id="binding", metadata={"access": "ro"})

    def unexpected_parent_setter(*args, **kwargs):
        raise AssertionError("Selected process called a second parent setter")

    registry = SimpleNamespace(
        list_folder_bindings=lambda workspace_id: [binding],
        set_folder_binding_access=unexpected_parent_setter,
    )
    client = object()
    app = SimpleNamespace(
        console_execution_client=client, workspace_registry_service=registry
    )

    def run_worker(coroutine, **kwargs):
        worker = asyncio.create_task(coroutine)
        workers.append(worker)
        return worker

    app.run_worker = run_worker
    screen = SimpleNamespace(
        app=app,
        app_instance=app,
        _task=object(),
        is_mounted=True,
        _category_swap_revision=1,
        _settings_selected_workspace_id="workspace",
        _settings_workspace_access_request=None,
        _active_category_id=lambda: SettingsCategoryId.WORKSPACES,
        _set_workspace_folder_result=lambda text, binding_id: publications.append(
            (text, binding_id)
        ),
        _refresh_settings_workspaces_pane=lambda: refreshes.append(True),
    )
    app.screen = screen
    screen._settings_workspace_process_folder_access = MethodType(
        SettingsScreen._settings_workspace_process_folder_access, screen
    )
    try:
        SettingsScreen._settings_workspace_toggle_folder_access(
            screen,
            SimpleNamespace(
                stop=lambda: None, button=SimpleNamespace(binding_id="binding")
            ),
        )
        assert calls == [(client, "workspace", "binding", True)]
        assert publications == [] and refreshes == []
        if leave_workspace:
            screen._settings_selected_workspace_id = "other"
        release.set()
        await asyncio.gather(*workers)
        if leave_workspace:
            assert publications == [] and refreshes == []
        else:
            assert publications == [("Folder access: read-write.", "binding")]
            assert refreshes == [True]
    finally:
        release.set()
        await asyncio.gather(operation, *workers, return_exceptions=True)
