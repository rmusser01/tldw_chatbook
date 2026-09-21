"""Settings workspace-binding exclusion management (spec §3)."""

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
async def test_exclusion_add_remove_and_inline_errors(tmp_path) -> None:
    """Per-binding exclusion rows: add via input+button, remove via button,
    and registry validation failures surface inline without adding."""
    from textual.widgets import Button, Input

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-excl", name="Exclusions WS")
    project = tmp_path / "project"
    project.mkdir()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-ws-excl", Button).press()
        await pilot.pause(0.2)

        # Bind a folder first (exclusions are per folder binding).
        screen.query_one("#settings-workspace-folder-path", Input).value = str(project)
        screen.query_one("#settings-workspace-folder-add", Button).press()
        await pilot.pause(0.3)
        binding = registry.list_folder_bindings("ws-excl")[0]
        assert "Excluded (0)" in _visible_text(screen)

        # Add via the per-binding input + button.
        screen.query_one(
            f"#settings-workspace-excl-path-{binding.binding_id}", Input
        ).value = "secrets"
        screen.query_one(
            f"#settings-workspace-excl-add-{binding.binding_id}", Button
        ).press()
        await pilot.pause(0.3)
        exclusions = registry.list_binding_exclusions(binding.binding_id)
        assert [entry.path for entry in exclusions] == ["secrets"]
        assert "enforced on the next tool call" in _visible_text(screen)
        assert "Excluded (1)" in _visible_text(screen)

        # Invalid (absolute) path explains inline and adds nothing new.
        screen.query_one(
            f"#settings-workspace-excl-path-{binding.binding_id}", Input
        ).value = "/abs/path"
        screen.query_one(
            f"#settings-workspace-excl-add-{binding.binding_id}", Button
        ).press()
        await pilot.pause(0.3)
        assert "relative to the binding root" in _visible_text(screen)
        exclusions = registry.list_binding_exclusions(binding.binding_id)
        assert [entry.path for entry in exclusions] == ["secrets"]

        # Remove via the per-row Unexclude button.
        screen.query_one(
            f"#settings-workspace-excl-remove-{binding.binding_id}-0", Button
        ).press()
        await pilot.pause(0.3)
        assert registry.list_binding_exclusions(binding.binding_id) == ()
        assert "effective for new runs" in _visible_text(screen)
        assert "Excluded (0)" in _visible_text(screen)


@pytest.mark.asyncio
async def test_malformed_exclusion_metadata_still_renders(tmp_path) -> None:
    """Finding E (PR #2767): the pane never crashes on malformed metadata.

    ``metadata["exclusions"]`` is loosely typed; a non-list value (here the
    string ``"bogus"``) must render as "Excluded (0)" via the defensive
    reader rather than raising during pane composition.
    """
    from dataclasses import replace

    from textual.widgets import Button, Input

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-excl", name="Exclusions WS")
    project = tmp_path / "project"
    project.mkdir()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-ws-excl", Button).press()
        await pilot.pause(0.2)

        screen.query_one("#settings-workspace-folder-path", Input).value = str(project)
        screen.query_one("#settings-workspace-folder-add", Button).press()
        await pilot.pause(0.3)
        binding = registry.list_folder_bindings("ws-excl")[0]
        assert "Excluded (0)" in _visible_text(screen)

        # Corrupt the persisted metadata directly (simulating an older
        # writer or hand-edited state), then recompose the pane.
        registry.save_runtime_binding(
            replace(binding, metadata={**binding.metadata, "exclusions": "bogus"})
        )
        screen.query_one("#settings-workspace-row-ws-excl", Button).press()
        await pilot.pause(0.3)
        assert "Excluded (0)" in _visible_text(screen)
        assert screen.query_one(
            f"#settings-workspace-excl-path-{binding.binding_id}", Input
        )
