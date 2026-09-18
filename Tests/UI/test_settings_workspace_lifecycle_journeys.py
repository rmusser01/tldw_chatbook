"""Settings lifecycle actions keep keyboard feedback and recovery visible."""

import asyncio

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_overview_search_journeys import _category, _painted
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness
from Tests.UI.test_settings_workspace_memory_confirmation import _press


@pytest.mark.parametrize(
    "action",
    ["rename_error", "rename_success", "restore_error", "archive", "create_error"],
)
@private_profile_test
async def test_workspace_lifecycle_feedback_is_visible(request, action):
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-review", name="Review workspace")
    if action == "restore_error":
        registry.archive_workspace("ws-review")
        registry.create_workspace(workspace_id="ws-collision", name="Review workspace")
    host = _StyledDestinationHarness(app, "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        await _category(host, pilot, "Workspaces")
        screen = host.screen
        if action == "create_error":
            await _press(host, pilot, "#settings-workspace-create")
            field = host.screen.query_one("#workspace-create-name", Input)
            field.focus()
            await pilot.press("home", "shift+end", *"Default", "enter")
            await _settle(host, pilot)
            result = host.screen.query_one("#workspace-create-error", Static)
            assert "already exists" in str(result.renderable)
            _assert_painted(host.screen, result)
            _assert_painted(host.screen, host.focused)
            return
        if action == "restore_error":
            checkbox = screen.query_one("#settings-workspaces-show-archived")
            checkbox.focus()
            await pilot.press("space")
            await _settle(host, pilot)
        await _press(host, pilot, "#settings-workspace-row-ws-review")
        if action.startswith("rename"):
            field = screen.query_one("#settings-workspace-rename-input", Input)
            field.focus()
            name = "Default" if action == "rename_error" else "Renamed"
            await pilot.press("home", "shift+end", *name)
            await _press(host, pilot, "#settings-workspace-rename-apply")
            if action == "rename_success":
                assert registry.get_workspace("ws-review").name == "Renamed"
                assert host.focused is screen.query_one(
                    "#settings-workspace-rename-apply"
                )
                _assert_painted(screen, host.focused)
                return
            assert field.value == "Default"
        elif action == "restore_error":
            await _press(host, pilot, "#settings-workspace-unarchive")
        else:
            await _press(host, pilot, "#settings-workspace-archive")
            async with asyncio.timeout(5):
                while host.screen is screen:
                    await pilot.pause(0.03)
            await _press(host, pilot, "#confirm-button")
            async with asyncio.timeout(5):
                while host.screen is not screen or not screen.query(
                    "#settings-workspace-archive-undo"
                ):
                    await pilot.pause(0.03)
            await _settle(host, pilot)
            assert registry.get_workspace("ws-review").archived
            _assert_painted(
                screen, screen.query_one("#settings-workspace-archive-undo")
            )
            _assert_painted(screen, host.focused)
            await _press(host, pilot, "#settings-workspace-archive-undo")
            await _settle(host, pilot)
            assert not registry.get_workspace("ws-review").archived
            assert registry.get_active_workspace().workspace_id != "ws-review"
            assert host.focused is screen.query_one("#settings-workspace-set-active")
            _assert_painted(screen, host.focused)
            return
        result = screen.query_one("#settings-workspace-lifecycle-result", Static)
        async with asyncio.timeout(5):
            while "already exists" not in str(result.renderable):
                await pilot.pause(0.03)
        await _settle(host, pilot)
        _assert_painted(screen, result)
        _assert_painted(screen, host.focused)
        assert " ".join(str(result.renderable).split()) in " ".join(
            _painted(host, result).split()
        )


@pytest.mark.parametrize("recovery", ["retry", "keep", "escape"])
@private_profile_test
async def test_partial_create_keeps_the_committed_workspace_clear(
    request, tmp_path, recovery
):
    app = _build_test_app()
    registry = app.workspace_registry_service
    host = _StyledDestinationHarness(app, "settings")
    folder = tmp_path / "project"
    folder.mkdir()
    async with host.run_test(size=(80, 24)) as pilot:
        await _category(host, pilot, "Workspaces")
        await _press(host, pilot, "#settings-workspace-create")
        modal = host.screen
        field = modal.query_one("#workspace-create-name", Input)
        field.focus()
        await pilot.press("home", "shift+end", *"Created once")
        modal.query_one("#workspace-default-persona", Select).value = "none"
        field = modal.query_one("#workspace-create-folder-path", Input)
        field.focus()
        await pilot.press(*str(folder))
        await _press(host, pilot, "#workspace-create-folder-add")
        assert modal._folders == [str(folder.resolve())]
        folder.rmdir()
        await _press(host, pilot, "#workspace-create-confirm")
        assert host.screen is modal
        created = [
            record
            for record in registry.list_workspaces()
            if record.name == "Created once"
        ]
        assert len(created) == 1
        workspace_id = created[0].workspace_id
        assert modal.query_one("#workspace-create-name", Input).disabled
        assert modal.query_one("#workspace-create-make-active").disabled
        assert (
            str(modal.query_one("#workspace-create-confirm", Button).label)
            == "Retry folders"
        )
        assert (
            str(modal.query_one("#workspace-create-cancel", Button).label)
            == "Keep workspace"
        )
        error = modal.query_one("#workspace-create-error", Static)
        _assert_painted(modal, error)
        _assert_painted(modal, host.focused)
        if recovery == "retry":
            folder.mkdir()
            await _press(host, pilot, "#workspace-create-confirm")
        elif recovery == "keep":
            await _press(host, pilot, "#workspace-create-cancel")
        else:
            await pilot.press("escape")
            await _settle(host, pilot)
        assert host.screen is not modal
        assert registry.get_workspace(workspace_id).name == "Created once"
        assert (
            len(
                [
                    record
                    for record in registry.list_workspaces()
                    if record.name == "Created once"
                ]
            )
            == 1
        )
        assert len(registry.list_folder_bindings(workspace_id)) == (
            1 if recovery == "retry" else 0
        )


@pytest.mark.parametrize("action", ["activate", "restore", "literal_archive"])
@private_profile_test
async def test_workspace_lifecycle_completion_keeps_context(request, action):
    app = _build_test_app()
    registry = app.workspace_registry_service
    name = "[red]Review[/red]" if action == "literal_archive" else "Review workspace"
    registry.create_workspace(workspace_id="ws-review", name=name)
    if action == "restore":
        registry.archive_workspace("ws-review")
    host = _StyledDestinationHarness(app, "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        await _category(host, pilot, "Workspaces")
        screen = host.screen
        if action == "restore":
            screen.query_one("#settings-workspaces-show-archived").focus()
            await pilot.press("space")
            await _settle(host, pilot)
        await _press(host, pilot, "#settings-workspace-row-ws-review")
        selector = {
            "activate": "#settings-workspace-set-active",
            "restore": "#settings-workspace-unarchive",
            "literal_archive": "#settings-workspace-archive",
        }[action]
        await _press(host, pilot, selector)
        if action == "literal_archive":
            async with asyncio.timeout(5):
                while host.screen is screen:
                    await pilot.pause(0.03)
            message = host.screen.query_one(".dialog-message")
            assert name in _painted(host, message)
            await pilot.press("escape")
            await _settle(host, pilot)
            assert not registry.get_workspace("ws-review").archived
            _assert_painted(screen, host.focused)
            return
        async with asyncio.timeout(5):
            while not screen.query("#settings-workspace-archive"):
                await pilot.pause(0.03)
        await _settle(host, pilot)
        if action == "activate":
            assert registry.get_active_workspace().workspace_id == "ws-review"
            assert host.focused is screen.query_one("#settings-workspace-archive")
        else:
            assert not registry.get_workspace("ws-review").archived
            assert registry.get_active_workspace().workspace_id != "ws-review"
            assert host.focused is screen.query_one("#settings-workspace-set-active")
        _assert_painted(screen, host.focused)


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_create_dialog_uses_the_current_theme_surface(request, theme):
    host = _StyledDestinationHarness(_build_test_app(), "settings")
    host.theme = theme
    async with host.run_test(size=(80, 24)) as pilot:
        await _category(host, pilot, "Workspaces")
        await _press(host, pilot, "#settings-workspace-create")
        from textual.color import Color

        content = host.screen.query_one("#workspace-create-modal")
        expected = Color.parse(host.get_css_variables()["surface"])
        assert content.styles.background == expected
