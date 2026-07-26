"""Settings ▸ Workspaces category registration (spec §4)."""

from __future__ import annotations

import pytest

from Tests.UI.test_settings_configuration_hub import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _open_settings_category,
    _visible_text,
)


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
    from textual.widgets import Button, Checkbox, Input

    app = _build_test_app()
    registry = app.workspace_registry_service
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")

        # Create with a free-form name.
        screen.query_one("#settings-workspace-create-name", Input).value = "Client X"
        screen.query_one("#settings-workspace-create", Button).press()
        await pilot.pause(0.3)
        created = [w for w in registry.list_workspaces() if w.name == "Client X"]
        assert len(created) == 1
        workspace_id = created[0].workspace_id

        # Select, rename.
        screen.query_one(f"#settings-workspace-row-{workspace_id}", Button).press()
        await pilot.pause(0.2)
        screen.query_one("#settings-workspace-rename-input", Input).value = "Client Y"
        screen.query_one("#settings-workspace-rename-apply", Button).press()
        await pilot.pause(0.3)
        assert registry.get_workspace(workspace_id).name == "Client Y"

        # Duplicate rename surfaces inline, not as a crash.
        screen.query_one("#settings-workspace-rename-input", Input).value = "Default"
        screen.query_one("#settings-workspace-rename-apply", Button).press()
        await pilot.pause(0.3)
        assert "already exists" in _visible_text(screen)

        # Set active.
        screen.query_one("#settings-workspace-set-active", Button).press()
        await pilot.pause(0.3)
        assert registry.get_active_workspace().workspace_id == workspace_id

        # Archive (confirm dialog), falls back to Default.
        screen.query_one("#settings-workspace-archive", Button).press()
        await pilot.pause(0.3)
        confirm = host.screen_stack[-1]
        confirm.query_one("#confirm-button", Button).press()
        await pilot.pause(0.4)
        assert registry.get_workspace(workspace_id).archived is True
        assert registry.get_active_workspace().workspace_id == "workspace-default"

        # Hidden until Show archived; then unarchive (no auto-activate).
        assert not screen.query(f"#settings-workspace-row-{workspace_id}")
        screen.query_one("#settings-workspaces-show-archived", Checkbox).value = True
        await pilot.pause(0.3)
        screen.query_one(f"#settings-workspace-row-{workspace_id}", Button).press()
        await pilot.pause(0.2)
        screen.query_one("#settings-workspace-unarchive", Button).press()
        await pilot.pause(0.3)
        record = registry.get_workspace(workspace_id)
        assert record.archived is False and record.active is False


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
