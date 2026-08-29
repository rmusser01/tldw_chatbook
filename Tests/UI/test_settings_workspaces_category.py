"""Settings ▸ Workspaces category registration (spec §4)."""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Workspaces import (
    ChangeReviewCapability,
    ChangeReviewConsent,
    ChangeReviewState,
    ChangeReviewStatus,
    RootReadiness,
    RootReadinessState,
)
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
                "#settings-workspace-create",
            )

            # Create with a free-form name via the shared create modal.
            screen.query_one("#settings-workspace-create", Button).press()
            await pilot.pause()
            modal = app.screen
            assert modal is not screen
            await _wait_for_selector(modal, pilot, "#workspace-create-name")
            modal.query_one("#workspace-create-name", Input).value = "Client X"
            # Leave inactive so the rename/duplicate-name/set-active steps
            # below still exercise the Settings-side "Set active" button.
            modal.query_one("#workspace-create-make-active", Checkbox).value = False
            modal.query_one("#workspace-create-confirm", Button).press()
            await pilot.pause()
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
async def test_activation_failure_surfaces_inline_but_creates_workspace() -> None:
    """TASK-17962: ``set_active_workspace`` raising after a successful
    create on the Settings surface must not crash the handler -- the
    failure shows up inline (``#settings-workspaces-result``) while the
    workspace itself still exists, mirroring the Console/Library seam
    tests for the same activation-failure path (Library:
    ``test_create_workspace_recomposes_after_activation_failure`` in
    ``Tests/UI/test_post_release_workspaces_library_depth.py``).
    """
    from unittest.mock import patch

    from textual.widgets import Button, Input

    from tldw_chatbook.Workspaces.registry_service import (
        WorkspaceRegistryServiceError,
    )

    app = _build_test_app(configured_default="settings")
    registry = app.workspace_registry_service

    with patch(
        "tldw_chatbook.app.get_cli_setting",
        side_effect=_settings_without_splash,
    ):
        async with app.run_test(size=(180, 50)) as pilot:
            screen = await _wait_for_settings_screen(app, pilot)
            await _wait_for_selector(screen, pilot, "#settings-category-workspaces")
            category = screen.query_one("#settings-category-workspaces", Button)
            category.scroll_visible(animate=False)
            category.press()
            await _wait_for_selector(screen, pilot, "#settings-workspace-create")

            def _boom(workspace_id: str) -> None:
                raise WorkspaceRegistryServiceError("boom")

            registry.set_active_workspace = _boom

            screen.query_one("#settings-workspace-create", Button).press()
            await pilot.pause()
            modal = app.screen
            assert modal is not screen
            await _wait_for_selector(modal, pilot, "#workspace-create-name")
            modal.query_one("#workspace-create-name", Input).value = "Client Z"
            # Leave "Switch to this workspace" checked (default) so Create
            # actually attempts activation.
            modal.query_one("#workspace-create-confirm", Button).press()
            await pilot.pause()

            created: list = []
            for _ in range(200):
                created = [
                    w for w in registry.list_workspaces() if w.name == "Client Z"
                ]
                if created and "boom" in _visible_text(screen):
                    break
                await pilot.pause(0.01)

            assert created, "workspace creation itself must have succeeded"
            assert "boom" in _visible_text(screen)


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
        visible = _visible_text(screen)
        assert "use private scratch" in visible
        assert "only to bind external folders" in visible
        assert "tool-less" not in visible


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


@pytest.mark.asyncio
async def test_change_review_toggle_flips_and_persists(tmp_path) -> None:
    """A new workspace is opt-in and discloses retained file contents."""
    from textual.widgets import Button

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-cr", name="CR WS")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-ws-cr", Button).press()
        await pilot.pause(0.2)

        text = _visible_text(screen)
        assert registry.change_review_enabled("ws-cr") is False
        assert "Tracking disabled" in text
        assert "shadow Git history" in text
        assert "application data" in text
        assert "file contents" in text
        assert "30 days" in text
        assert "does not erase existing history" in text
        screen.query_one(
            "#settings-workspace-change-review-toggle", Button
        ).press()
        await pilot.pause(0.3)
        assert registry.change_review_enabled("ws-cr") is True
        assert "Tracking enabled" in _visible_text(screen)
        screen.query_one(
            "#settings-workspace-change-review-toggle", Button
        ).press()
        await pilot.pause(0.3)
        assert registry.change_review_enabled("ws-cr") is False


@pytest.mark.asyncio
async def test_change_review_stale_toggle_reports_conflict_and_refreshes() -> None:
    """A stale rendered intent never inverts a newer external decision."""
    from textual.widgets import Button

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-stale", name="Stale WS")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-ws-stale", Button).press()
        await pilot.pause(0.2)

        stale_button = screen.query_one(
            "#settings-workspace-change-review-toggle", Button
        )
        registry.set_change_review_enabled("ws-stale", True)
        stale_button.press()
        await pilot.pause(0.3)

        assert registry.change_review_enabled("ws-stale") is True
        text = _visible_text(screen)
        assert "changed elsewhere" in text
        assert "Tracking enabled" in text


@pytest.mark.asyncio
async def test_change_review_unavailable_consent_has_no_toggle() -> None:
    """An unreadable registry state fails off with honest copy."""
    from textual.widgets import Button

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-unavailable", name="Unavailable WS")

    class UnavailableService:
        def status(self, _workspace_id):
            return ChangeReviewStatus(
                ChangeReviewCapability(ChangeReviewState.ENABLED),
                ChangeReviewConsent(ChangeReviewState.UNAVAILABLE),
            )

    app.change_review_consent_service = UnavailableService()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one(
            "#settings-workspace-row-ws-unavailable", Button
        ).press()
        await pilot.pause(0.2)

        assert "state could not be read" in _visible_text(screen)
        assert not screen.query("#settings-workspace-change-review-toggle")


@pytest.mark.asyncio
async def test_change_review_failed_root_offers_one_retry_without_paths() -> None:
    """Failed preparation stays non-blocking and exposes a bounded retry."""
    from textual.widgets import Button

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-retry", name="Retry WS")
    retries: list[str] = []

    class FailedService:
        def status(self, _workspace_id):
            return ChangeReviewStatus(
                ChangeReviewCapability(ChangeReviewState.ENABLED),
                ChangeReviewConsent(ChangeReviewState.ENABLED, "rev-1"),
                (
                    RootReadiness(
                        alias="folder-safe-alias",
                        state=RootReadinessState.FAILED,
                        reason="preparation failed at /private/secret/root",
                    ),
                ),
            )

        def retry_failed_roots(self, workspace_id):
            retries.append(workspace_id)
            return 1

    app.change_review_consent_service = FailedService()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-ws-retry", Button).press()
        await pilot.pause(0.2)

        text = _visible_text(screen)
        assert "chat and tools continue" in text
        assert "/private/secret/root" not in text
        screen.query_one("#settings-workspace-change-review-retry", Button).press()
        await pilot.pause(0.2)

        assert retries == ["ws-retry"]


@pytest.mark.asyncio
async def test_change_review_preparing_is_explicitly_non_blocking() -> None:
    """Preparation copy never implies that chat is waiting for the scan."""
    from textual.widgets import Button

    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-preparing", name="Preparing WS")

    class PreparingService:
        def status(self, _workspace_id):
            return ChangeReviewStatus(
                ChangeReviewCapability(ChangeReviewState.ENABLED),
                ChangeReviewConsent(ChangeReviewState.ENABLED, "rev-1"),
                (
                    RootReadiness(
                        alias="folder-safe-alias",
                        state=RootReadinessState.PREPARING,
                        reason="preparing",
                    ),
                ),
            )

    app.change_review_consent_service = PreparingService()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-ws-preparing", Button).press()
        await pilot.pause(0.2)

        text = _visible_text(screen)
        assert "Preparing change history" in text
        assert "background; chat and tools continue" in text
        assert not screen.query("#settings-workspace-change-review-retry")


@pytest.mark.asyncio
async def test_change_review_git_absent_shows_honest_copy(monkeypatch) -> None:
    """TASK-1979 AC#2: no git — the row states the reason, no dead toggle."""
    from textual.widgets import Button

    import tldw_chatbook.Workspaces.change_tracking as ct

    # Narrow seam: shutil is a shared module object — patching its `which`
    # breaks unrelated services (file-notes git calls it with path=).
    monkeypatch.setattr(
        ct.ShadowRepoService, "available", property(lambda self: False)
    )
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-nogit", name="NoGit WS")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-ws-nogit", Button).press()
        await pilot.pause(0.2)

        assert (
            "Change review needs git — install git to enable."
            in _visible_text(screen)
        )
        assert not screen.query("#settings-workspace-change-review-toggle")


@pytest.mark.asyncio
async def test_change_review_global_kill_disclosed_in_settings(
    monkeypatch,
) -> None:
    """Qodo #1264: with the global knob off, the card must say so instead
    of claiming per-workspace tracking is enabled."""
    from textual.widgets import Button

    monkeypatch.setenv("TLDW_CHANGE_REVIEW_ENABLED", "0")
    app = _build_test_app()
    registry = app.workspace_registry_service
    registry.create_workspace(workspace_id="ws-gk", name="GK WS")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _open_settings_category(pilot, "#settings-category-workspaces")
        screen.query_one("#settings-workspace-row-ws-gk", Button).press()
        await pilot.pause(0.2)

        text = _visible_text(screen)
        assert "disabled globally" in text, text
        assert "Tracking enabled" not in text
        assert not screen.query("#settings-workspace-change-review-toggle")
