"""Settings ▸ Workspaces category registration (spec §4)."""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from Tests.UI.test_settings_configuration_hub import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _open_settings_category,
    _settle_settings_mount_storm,
    _visible_text,
    _wait_for_selector,
)


def _settings_without_splash(section, key=None, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


async def _wait_for_settings_screen(app, pilot) -> SettingsScreen:
    for _ in range(200):
        if isinstance(app.screen, SettingsScreen):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError("production app did not mount Settings")


@pytest.mark.asyncio
async def test_workspaces_category_registered_and_immediate() -> None:
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        text = _visible_text(screen)

        assert "Workspace management" in text
        # Immediate-apply category: the guided Save/Revert buttons are
        # suppressed exactly like Theme/Splash.
        assert not screen.query("#settings-save-category")


@pytest.mark.asyncio
async def test_create_rename_archive_unarchive_flow() -> None:
    from unittest.mock import patch

    from textual.widgets import Button, Checkbox, Input

    app = _build_test_app(configured_default="settings")
    registry = app.workspace_registry_service

    with patch(
        "tldw_chatbook.app.get_cli_setting",
        side_effect=_settings_without_splash,
    ):
        async with app.run_test(size=(180, 50)) as pilot:
            screen = await _wait_for_settings_screen(app, pilot)
            await _wait_for_selector(
                screen,
                pilot,
                "#settings-category-workspaces",
            )
            category = screen.query_one("#settings-category-workspaces", Button)
            category.scroll_visible(animate=False)
            category.press()
            await _wait_for_selector(
                screen,
                pilot,
                "#settings-workspace-create-name",
            )

            # Create with a free-form name.
            screen.query_one("#settings-workspace-create-name", Input).value = "Client X"
            screen.query_one("#settings-workspace-create", Button).press()
            created = []
            for _ in range(200):
                created = [
                    w for w in registry.list_workspaces() if w.name == "Client X"
                ]
                if created:
                    break
                await pilot.pause(0.01)
            assert len(created) == 1
            workspace_id = created[0].workspace_id
            await _wait_for_selector(
                screen,
                pilot,
                f"#settings-workspace-row-{workspace_id}",
            )

            # Select, rename.
            screen.query_one(f"#settings-workspace-row-{workspace_id}", Button).press()
            await _wait_for_selector(
                screen,
                pilot,
                "#settings-workspace-rename-input",
            )
            screen.query_one(
                "#settings-workspace-rename-input", Input
            ).value = "Client Y"
            screen.query_one("#settings-workspace-rename-apply", Button).press()
            await _wait_for_selector(
                screen,
                pilot,
                "#settings-workspace-set-active",
            )
            assert registry.get_workspace(workspace_id).name == "Client Y"

            # Duplicate rename surfaces inline, not as a crash.
            screen.query_one("#settings-workspace-rename-input", Input).value = "Default"
            screen.query_one("#settings-workspace-rename-apply", Button).press()
            for _ in range(200):
                if "already exists" in _visible_text(screen):
                    break
                await pilot.pause(0.01)
            assert "already exists" in _visible_text(screen)

            # Set active.
            screen.query_one("#settings-workspace-set-active", Button).press()
            await _wait_for_selector(
                screen,
                pilot,
                "#settings-workspace-archive",
            )
            assert registry.get_active_workspace().workspace_id == workspace_id

            # Archive (confirm dialog), falls back to Default.
            screen.query_one("#settings-workspace-archive", Button).press()
            for _ in range(200):
                if app.screen is not screen and app.screen.query("#confirm-button"):
                    break
                await pilot.pause(0.01)
            assert app.screen is not screen and bool(
                app.screen.query("#confirm-button")
            )
            confirm = app.screen
            confirm.query_one("#confirm-button", Button).press()
            for _ in range(200):
                if (
                    registry.get_workspace(workspace_id).archived
                    and registry.get_active_workspace().workspace_id
                    == "workspace-default"
                ):
                    break
                await pilot.pause(0.01)
            assert registry.get_workspace(workspace_id).archived is True
            assert registry.get_active_workspace().workspace_id == "workspace-default"

            # Hidden until Show archived; then unarchive (no auto-activate).
            await _wait_for_selector(
                screen,
                pilot,
                "#settings-workspaces-show-archived",
            )
            assert not screen.query(f"#settings-workspace-row-{workspace_id}")
            screen.query_one("#settings-workspaces-show-archived", Checkbox).value = True
            await _wait_for_selector(
                screen,
                pilot,
                f"#settings-workspace-row-{workspace_id}",
            )
            screen.query_one(f"#settings-workspace-row-{workspace_id}", Button).press()
            await _wait_for_selector(
                screen,
                pilot,
                "#settings-workspace-unarchive",
            )
            screen.query_one("#settings-workspace-unarchive", Button).press()
            for _ in range(200):
                record = registry.get_workspace(workspace_id)
                if not record.archived:
                    break
                await pilot.pause(0.01)
            record = registry.get_workspace(workspace_id)
            assert record.archived is False and record.active is False


@pytest.mark.asyncio
async def test_compact_overview_keeps_a_painted_recovery_action() -> None:
    """The full Settings app keeps one real action above the compact fold."""
    from unittest.mock import patch

    from textual.widgets import Button

    app = _build_test_app(configured_default="settings")
    with patch(
        "tldw_chatbook.app.get_cli_setting",
        side_effect=_settings_without_splash,
    ):
        async with app.run_test(size=(100, 32)) as pilot:
            screen = await _wait_for_settings_screen(app, pilot)
            await _wait_for_selector(
                screen,
                pilot,
                "#settings-open-appearance",
            )
            action = screen.query_one("#settings-open-appearance", Button)
            assert 0 <= action.region.y < action.region.bottom <= 32
            painted_row = "".join(
                segment.text
                for segment in screen._compositor.render_strips()[action.region.y]
            )
            assert "Theme" in painted_row


@pytest.mark.asyncio
async def test_default_workspace_card_is_protected() -> None:
    from textual.widgets import Button

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-workspace-default", Button).press()
        await pilot.pause(0.2)

        assert not screen.query("#settings-workspace-rename-apply")
        assert not screen.query("#settings-workspace-archive")
        assert "stays tool-less" in _visible_text(screen)


@pytest.mark.asyncio
async def test_archived_workspace_card_offers_only_unarchive() -> None:
    """Finding 3: an archived workspace's card must not offer controls that

    fail with a bare-id error against an archived (currently invisible)
    workspace -- rename/set-active/archive/folder-add all require the
    record to be reachable outside the Show-archived view.
    """
    from textual.widgets import Button, Checkbox

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-archived", name="Archived WS")
    registry.archive_workspace("ws-archived")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")

        screen.query_one("#settings-workspaces-show-archived", Checkbox).value = True
        await pilot.pause(0.3)
        screen.query_one("#settings-workspace-row-ws-archived", Button).press()
        await pilot.pause(0.2)

        assert not screen.query("#settings-workspace-rename-input")
        assert not screen.query("#settings-workspace-rename-apply")
        assert not screen.query("#settings-workspace-set-active")
        assert not screen.query("#settings-workspace-archive")
        assert not screen.query("#settings-workspace-folder-add")
        assert screen.query_one("#settings-workspace-unarchive", Button)
        assert (
            "Archived workspace. Unarchive it to rename, activate, "
            "or edit folders." in _visible_text(screen)
        )


@pytest.mark.asyncio
async def test_folder_bindings_add_toggle_remove_and_inline_errors(tmp_path) -> None:
    from textual.widgets import Button, Input

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-folders", name="Folder WS")
    project = tmp_path / "project"
    project.mkdir()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-ws-folders", Button).press()
        await pilot.pause(0.2)

        # Add (defaults to read-only).
        screen.query_one("#settings-workspace-folder-path", Input).value = str(project)
        screen.query_one("#settings-workspace-folder-add", Button).press()
        await pilot.pause(0.3)
        bindings = registry.list_folder_bindings("ws-folders")
        assert len(bindings) == 1
        binding = bindings[0]
        assert binding.metadata["access"] == "ro"
        assert "[ro]" in _visible_text(screen)

        # Toggle to rw.
        screen.query_one(
            f"#settings-workspace-folder-toggle-{binding.binding_id}", Button
        ).press()
        await pilot.pause(0.3)
        assert registry.list_folder_bindings("ws-folders")[0].metadata["access"] == "rw"

        # Invalid add explains inline.
        screen.query_one("#settings-workspace-folder-path", Input).value = str(
            tmp_path / "missing"
        )
        screen.query_one("#settings-workspace-folder-add", Button).press()
        await pilot.pause(0.3)
        assert "not a directory" in _visible_text(screen)

        # Remove.
        screen.query_one(
            f"#settings-workspace-folder-remove-{binding.binding_id}", Button
        ).press()
        await pilot.pause(0.3)
        assert registry.list_folder_bindings("ws-folders") == ()


@pytest.mark.asyncio
async def test_pane_refreshes_after_external_registry_change() -> None:
    from textual.widgets import Button

    app = _build_test_app()
    registry = app.workspace_registry_service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        assert not screen.query("#settings-workspace-row-ws-late")

        registry.create_workspace(workspace_id="ws-late", name="Late Arrival")
        screen._refresh_settings_workspaces_pane()
        await pilot.pause(0.3)
        assert screen.query_one("#settings-workspace-row-ws-late", Button)


@pytest.mark.asyncio
async def test_resume_refreshes_workspaces_pane_only_when_active(monkeypatch) -> None:
    refresh_calls = 0

    def fake_refresh(self):
        nonlocal refresh_calls
        refresh_calls += 1

    monkeypatch.setattr(SettingsScreen, "_refresh_settings_workspaces_pane", fake_refresh)

    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)

        # Still on Overview (the default category) -- resuming must not
        # touch the Workspaces pane.
        screen.on_screen_resume()
        await pilot.pause(0.2)
        assert refresh_calls == 0

        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.on_screen_resume()
        await pilot.pause(0.2)
        assert refresh_calls == 1


@pytest.mark.asyncio
async def test_scope_inspector_shows_immediate_actions_not_read_only() -> None:
    """Finding 1: Workspaces is an immediate-apply category, not read-only.

    The Scope Inspector impact pane (and the ``s`` Save-shortcut toast, both
    driven by ``_guided_action_message``) must not fall through to the
    generic "Guided edits: read-only." default -- Workspaces has its own
    message, mirroring how THEME/SPLASH_SCREEN get theirs.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        text = _visible_text(screen)

        assert "read-only" not in text
        assert (
            "Immediate actions: workspace changes apply as you make them; "
            "there is no draft to save or revert." in text
        )


@pytest.mark.asyncio
async def test_overview_pins_workspaces_recovery_copy() -> None:
    """Finding 2 (spec §6): Overview must point users at Settings > Workspaces.

    Cross-surface copy pin -- Settings Overview's "Where changes happen"
    recovery row is the guidance a user sees before ever opening the
    Workspaces category, so it must name the real owning surfaces.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert (
            "Sync status here is read-only - manage workspaces in "
            "Settings > Workspaces; switch in Console (Alt+W); run "
            "sync from the owning sync surfaces." in text
        )
