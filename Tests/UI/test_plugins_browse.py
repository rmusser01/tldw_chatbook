"""Keyboard browsing preserves exact plugin identities and captured scope."""

import asyncio
from pathlib import Path

import pytest
from textual.widget import Widget
from textual.widgets import Button, Input, OptionList, Select, Static

from Tests.Plugins import platform_support
from Tests.Plugins.conftest import native_console as _native_console
from Tests.Plugins.conftest import native_package as _native_package
from Tests.Plugins.test_source_service import marketplace as _marketplace
from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Plugins.source_service import SourceService
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route

native_console = _native_console
native_package = _native_package
marketplace = _marketplace


def test_stale_search_cannot_retarget_selection():
    from tldw_chatbook.UI.Plugins_Modules.browse_state import accept_response

    assert not accept_response(3, 4)
    assert accept_response(4, 4)


def test_untrusted_display_is_literal_bounded_and_removes_terminal_controls():
    from tldw_chatbook.UI.Plugins_Modules.details import display_text

    text = display_text("[red]name\x1b]52;c;secret\x07\x1b[31m\u202e", limit=256)
    assert text == "[red]name"
    assert display_text("x" * 300, limit=256).endswith("[truncated]")
    assert len(display_text("x" * 300, limit=256)) <= 256


class BrowseApp(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET
    _SCREEN_OWNED_ROUTE_CSS = TldwCli._SCREEN_OWNED_ROUTE_CSS

    def __init__(self, rig):
        super().__init__()
        self.plugin_source_service = SourceService(rig.service)
        self.workspace_registry_service = rig.registry

    async def on_mount(self):
        TldwCli._ensure_screen_owned_css(self, "plugins")
        route = resolve_screen_route("plugins")
        await self.push_screen(
            TldwCli._create_navigation_screen(
                self, "plugins", route.load_screen_class()
            )
        )


async def settled(pilot, screen):
    for _ in range(100):
        await pilot.pause(0.02)
        if not screen._loading and not screen._action_busy:
            return
    raise AssertionError("Plugins worker did not settle")


class HeldPluginUnmount(Widget):
    """Keep real parent pruning open after its other children detach."""

    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def on_unmount(self):
        self.entered.set()
        await self.release.wait()


class DelayedPluginCall:
    """Gate timing only, retaining the original owner's result or refusal."""

    def __init__(self, original):
        self.original = original
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []
        self.result = None
        self.error = None

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.entered.set()
        await self.release.wait()
        try:
            self.result = await self.original(*args, **kwargs)
            return self.result
        except (OSError, ValueError, RuntimeError) as error:
            self.error = error
            raise


async def wait_for_plugin_prune(parent, control, hold):
    """Observe the installed Textual lifecycle rather than simulating flags."""
    await asyncio.wait_for(hold.entered.wait(), 3)
    async with asyncio.timeout(3):
        while control.is_attached or parent.is_running:
            await asyncio.sleep(0.01)
    assert not control.is_attached
    assert parent.is_attached
    assert parent.app.is_running
    assert not parent.is_running
    assert not hold.release.is_set()


def paint(screen):
    return "\n".join(
        "".join(segment.text for segment in strip)
        for strip in screen._compositor.render_strips()
    )


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_keyboard_details_back_resize_and_restore_keep_stable_selection(
    request, native_console, size
):
    rig = native_console
    review = await rig.install()
    rig.registry.set_active_workspace("workspace-a")
    app = BrowseApp(rig)
    async with app.run_test(size=size) as pilot:
        screen = app.screen
        await settled(pilot, screen)
        listing = screen.query_one("#plugins-list", OptionList)
        assert listing.option_count == 1
        generation = screen.browse.request_generation
        screen._check_scope_generation()
        assert screen.browse.request_generation == generation
        listing.focus()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert screen.browse.selected_installation_id == review.installation_id
        assert "1 of 1 selected components eligible by recorded metadata" in str(
            screen.query_one("#plugins-detail", Static).render()
        )
        assert "Activation in: A | Installation: this device" in paint(screen)
        if size[0] == 80:
            assert not screen.query_one("#plugins-list-pane").display
            assert screen.query_one("#plugins-detail-scroll").size.height >= 5
            assert screen.focused.id == "plugins-back"
            await pilot.press("enter")
            assert screen.query_one("#plugins-list-pane").display
            assert screen.focused.id == "plugins-list"
        await pilot.resize_terminal(120, 35)
        await pilot.pause()
        assert screen.browse.selected_installation_id == review.installation_id
        screen.query_one("#plugins-search", Input).value = "review"
        await settled(pilot, screen)
        screen.query_one("#plugins-read-readme").focus()
        await pilot.press("enter")
        await settled(pilot, screen)
        assert screen.focused is not None
        assert screen.focused.id == "plugins-read-readme"
        listing.focus()
        await pilot.pause()
        state = screen.save_state()
        assert state["browse"].query == "review"
        fresh = TldwCli._create_navigation_screen(app, "plugins", type(screen))
        fresh.restore_state(state)
        await app.switch_screen(fresh)
        await settled(pilot, fresh)
        assert fresh is not screen
        assert fresh.browse.selected_installation_id == review.installation_id
        assert fresh.query_one("#plugins-search", Input).value == "review"
        assert fresh.focused.id == "plugins-list"


@private_profile_test
@pytest.mark.asyncio
async def test_real_source_add_and_import_preview_are_keyboard_reachable(
    request, native_console, native_package
):
    app = BrowseApp(native_console)
    package = native_package()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = app.screen
        await settled(pilot, screen)
        screen.query_one("#plugins-add-source", Button).focus()
        await pilot.press("enter")
        screen.query_one("#plugins-location", Input).value = str(package)
        screen.query_one("#plugins-submit", Button).focus()
        await pilot.press("enter")
        await settled(pilot, screen)
        assert screen.browse.view == "marketplaces"
        sources = app.plugin_source_service.list_sources()
        assert len(sources) == 1 and not sources[0].cached
        screen.query_one("#plugins-import", Button).focus()
        await pilot.press("enter")
        screen.query_one("#plugins-import-app", Select).value = "codex"
        screen.query_one("#plugins-submit", Button).focus()
        await pilot.press("enter")
        await settled(pilot, screen)
        assert screen.browse.preview.candidates[0].kind == "package"
        assert not await native_console.service.management_snapshot(None)
        saved = screen.save_state()
        assert saved["browse"].preview is screen.browse.preview
        assert "No sources or packages imported" in str(
            screen.query_one("#plugins-form-result", Static).render()
        )


@private_profile_test
@pytest.mark.asyncio
async def test_changed_workspace_refuses_pending_readme_and_refreshes_scope(
    request, native_console, monkeypatch
):
    rig = native_console
    await rig.install()
    rig.registry.set_active_workspace("workspace-a")
    app = BrowseApp(rig)
    started, release = asyncio.Event(), asyncio.Event()
    original = rig.service.management_readme

    async def held(*args, **kwargs):
        result = await original(*args, **kwargs)
        started.set()
        await release.wait()
        return result + "old workspace content"

    monkeypatch.setattr(rig.service, "management_readme", held)
    async with app.run_test(size=(120, 35)) as pilot:
        screen = app.screen
        await settled(pilot, screen)
        screen.query_one("#plugins-list").focus()
        await pilot.press("down", "enter")
        screen.query_one("#plugins-read-readme").focus()
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), 5)
        rig.registry.set_active_workspace("workspace-b")
        release.set()
        await pilot.pause(0.7)
        await settled(pilot, screen)
        assert screen.browse.workspace_id == "workspace-b"
        assert "old workspace content" not in str(
            screen.query_one("#plugins-readme", Static).render()
        )
        assert "Disabled" in str(screen.query_one("#plugins-detail", Static).render())
        # TASK-41103: the scope line names the workspace, literally (a
        # markup-shaped name stays as typed), while its id keeps driving the
        # token, so a rename repaints the line and keeps the selection.
        assert "Activation in: B | Installation: this device" in paint(screen)
        selected = screen.browse.selected_installation_id
        assert selected is not None
        rig.registry.rename_workspace("workspace-b", "[b]Team[/b] B")
        screen._check_scope_generation()
        await settled(pilot, screen)
        assert "Activation in: [b]Team[/b] B | Installation: this device" in paint(
            screen
        )
        assert screen.browse.workspace_id == screen._token()[1] == "workspace-b"
        assert await screen._current(screen._token())
        assert screen.browse.selected_installation_id == selected
        assert "Disabled" in str(screen.query_one("#plugins-detail", Static).render())
        rig.registry.clear_active_workspace()
        screen._check_scope_generation()
        await settled(pilot, screen)
        assert "Activation in: Global default | Installation: this device" in paint(
            screen
        )


@private_profile_test
@pytest.mark.asyncio
async def test_reordered_rows_and_obsolete_search_keep_explicit_source_selection(
    request, native_console, native_package, monkeypatch
):
    from tldw_chatbook.Plugins.git_source import select_local_source

    app = BrowseApp(native_console)
    first = await app.plugin_source_service.add(select_local_source(native_package()))
    second = await app.plugin_source_service.add(select_local_source(native_package()))
    async with app.run_test(size=(120, 35)) as pilot:
        screen = app.screen
        await settled(pilot, screen)
        await pilot.press("3")
        await settled(pilot, screen)
        listing = screen.query_one("#plugins-list", OptionList)
        listing.focus()
        await pilot.press("down", "enter")
        selected = screen.browse.selected_source_id
        assert selected in {first, second}
        screen._publish(tuple(reversed(screen.rows)))
        assert screen._selected().source_id == selected
        assert listing.get_option_at_index(listing.highlighted).id == selected
        started, release = asyncio.Event(), asyncio.Event()
        original = app.plugin_source_service.load_cached

        async def held():
            result = await original()
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return result

        monkeypatch.setattr(app.plugin_source_service, "load_cached", held)
        screen.query_one("#plugins-search", Input).value = "old-query"
        await asyncio.wait_for(started.wait(), 5)
        screen.query_one("#plugins-search", Input).value = "no-matches"
        release.set()
        await settled(pilot, screen)
        assert screen.browse.query == "no-matches"
        assert screen.query_one("#plugins-list", OptionList).option_count == 0
        assert "No matches" in str(screen.query_one("#plugins-status", Static).render())
        assert screen.browse.selected_source_id == selected
        assert screen.query_one("#plugins-refresh", Button).disabled


@private_profile_test
@pytest.mark.asyncio
async def test_live_catalog_refresh_preview_and_stale_failure_preserve_source_identity(
    request, native_console, marketplace
):
    from tldw_chatbook.Plugins.git_source import select_local_source

    app = BrowseApp(native_console)
    source_id = await app.plugin_source_service.add(select_local_source(marketplace))
    async with app.run_test(size=(120, 35)) as pilot:
        screen = app.screen
        await settled(pilot, screen)
        await pilot.press("3")
        await settled(pilot, screen)
        screen.query_one("#plugins-list").focus()
        await pilot.press("down", "enter")
        screen.query_one("#plugins-refresh").focus()
        await pilot.press("enter")
        await settled(pilot, screen)
        assert app.plugin_source_service.list_sources()[0].cached
        await pilot.press("2")
        await settled(pilot, screen)
        screen.query_one("#plugins-list").focus()
        await pilot.press("down", "enter")
        screen.query_one("#plugins-preview-entry").focus()
        await pilot.press("enter")
        await settled(pilot, screen)
        assert screen.browse.preview.inspection.catalog_binding.source_id == source_id
        assert not await native_console.service.management_snapshot(None)
        (marketplace / ".agents/plugins/marketplace.json").write_text("invalid json")
        await pilot.press("3")
        await settled(pilot, screen)
        screen.query_one("#plugins-refresh").focus()
        await pilot.press("enter")
        await settled(pilot, screen)
        row = app.plugin_source_service.list_sources()[0]
        assert row.cached and row.refresh_error == "catalog_refresh_failed"
        assert "last-good (stale)" in str(
            screen.query_one("#plugins-detail", Static).render()
        )
        await pilot.press("2")
        await settled(pilot, screen)
        assert "stale cached results" in str(
            screen.query_one("#plugins-status", Static).render()
        )


@private_profile_test
@pytest.mark.asyncio
async def test_selected_repository_preview_captures_real_snapshot_without_install(
    request, native_console, native_package
):
    app = BrowseApp(native_console)
    package = native_package()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = app.screen
        await settled(pilot, screen)
        screen.query_one("#plugins-repository").focus()
        await pilot.press("enter")
        screen.query_one("#plugins-location", Input).value = str(package)
        screen.query_one("#plugins-submit").focus()
        await pilot.press("enter")
        await settled(pilot, screen)
        snapshot = screen.browse.preview
        assert snapshot.transport == "local"
        assert snapshot.repository == str(package)
        assert "skill:review" in snapshot.inspection.inventory
        assert Path(snapshot.materialized_identity).is_dir()
        assert not await native_console.service.management_snapshot(None)
        assert not app.plugin_source_service.list_sources()


@private_profile_test
@pytest.mark.asyncio
async def test_editing_import_location_refuses_pending_preview(
    request, native_console, native_package, monkeypatch
):
    import threading

    from tldw_chatbook.Plugins import imports

    app = BrowseApp(native_console)
    package = native_package()
    started, release = threading.Event(), threading.Event()
    original = imports.inspect_app_import

    def held(*args):
        result = original(*args)
        started.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(imports, "inspect_app_import", held)
    async with app.run_test(size=(80, 24)) as pilot:
        screen = app.screen
        await settled(pilot, screen)
        screen.query_one("#plugins-import").focus()
        await pilot.press("enter")
        screen.query_one("#plugins-location", Input).value = str(package)
        screen.query_one("#plugins-submit").focus()
        await pilot.press("enter")
        assert await asyncio.to_thread(started.wait, 5)
        screen.query_one("#plugins-location", Input).value = str(package / "other")
        await pilot.pause()
        release.set()
        await settled(pilot, screen)
        assert screen.browse.preview is None
        assert "earlier selection" in str(
            screen.query_one("#plugins-status", Static).render()
        )


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["back", "form"])
async def test_back_replaces_held_load_without_old_completion_clearing_new_owner(
    request, native_console, monkeypatch, transition
):
    rig = native_console
    await rig.install()
    original = rig.service.management_snapshot
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    calls = []

    async def held(*args, **kwargs):
        index = len(calls)
        calls.append(index)
        if index < 2:
            entered[index].set()
            try:
                await release[index].wait()
            except asyncio.CancelledError:
                await release[index].wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(rig.service, "management_snapshot", held)
    app = BrowseApp(rig)
    async with app.run_test(size=(80, 24)) as pilot:
        screen = app.screen
        await asyncio.wait_for(entered[0].wait(), 2)
        if transition == "form":
            screen.query_one("#plugins-import", Button).focus()
            await pilot.press("enter")
            await pilot.pause()
            assert not screen._loading
        await pilot.press("escape")
        await asyncio.wait_for(entered[1].wait(), 2)
        release[0].set()
        await pilot.pause()
        assert screen._loading, "old completion cleared a newer load owner"
        release[1].set()
        await settled(pilot, screen)
        assert screen.query_one("#plugins-list", OptionList).option_count == 1
        assert "Loading" not in str(
            screen.query_one("#plugins-status", Static).render()
        )


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["workspace-success", "snapshot-error"])
async def test_navigation_during_child_prune_settles_original_load(
    request, native_console, monkeypatch, boundary
):
    from textual.screen import Screen

    from tldw_chatbook.UI.Screens.plugins_screen import PluginsScreen

    rig = native_console
    installed = await rig.install(enable=False)
    rig.registry.set_active_workspace("workspace-a")
    app = BrowseApp(rig)
    if boundary == "workspace-success":
        original = PluginsScreen._workspace
        gate = DelayedPluginCall(original)

        async def held_workspace(screen):
            return await gate(screen)

        monkeypatch.setattr(PluginsScreen, "_workspace", held_workspace)
    else:
        gate = DelayedPluginCall(rig.service.management_snapshot)
        monkeypatch.setattr(rig.service, "management_snapshot", gate)
    async with app.run_test(size=(120, 35)):
        screen = app.screen
        hold = HeldPluginUnmount()
        try:
            await asyncio.wait_for(gate.entered.wait(), 3)
            await screen.mount(hold)
            scope = screen.query_one("#plugins-scope", Static)
            listing = screen.query_one("#plugins-list", OptionList)
            before = str(scope.render()), listing.option_count
            worker = next(w for w in screen.workers if w.group == "plugins-load")
            app.switch_screen(Screen())
            await wait_for_plugin_prune(screen, scope, hold)
            assert not listing.is_attached
            if boundary == "snapshot-error":
                await rig.service.aclose()
            gate.release.set()
            await asyncio.wait_for(worker.wait(), 10)
            assert (str(scope.render()), listing.option_count) == before
            assert not worker.is_cancelled
            if boundary == "workspace-success":
                assert gate.result.workspace_id == "workspace-a"
                assert gate.calls == [((screen,), {})]
                rows = await rig.service.management_snapshot("workspace-a")
                assert [row.installation_id for row in rows] == [
                    installed.installation_id
                ]
            else:
                assert isinstance(gate.error, PermissionError)
                assert str(gate.error) == "plugin_management_unavailable"
                assert str(gate.error.__cause__) == "plugin_service_closed"
                assert gate.calls[0][0] == ("workspace-a",)
        finally:
            gate.release.set()
            hold.release.set()


@private_profile_test
@pytest.mark.asyncio
async def test_authority_recovery_copy_is_truthful_without_a_reset_step(
    request, native_console
):
    """TASK-40132 (Q2, narrowed by D2): recovery_required names when it happens
    (a restore on another computer at the same paths, or a reset OS keyring) and
    that plugins cannot be recovered yet. It offers no Reset authority step,
    because setting authority up again refuses while the restored registry lists
    installed plugins; and no other-profile case, which AC#8 refuses."""
    rig = native_console

    async def recovery_required():
        return "recovery_required"

    rig.service.authority_status = recovery_required
    app = BrowseApp(rig)
    async with app.run_test(size=(120, 35)) as pilot:
        screen = app.screen
        await settled(pilot, screen)
        screen.query_one("#plugins-authority", Button).focus()
        await pilot.press("enter")
        for _ in range(100):
            await pilot.pause(0.02)
            status = str(screen.query_one("#plugins-status", Static).render())
            if "needs recovery" in status:
                break
        # D2: say what is true; no Reset authority step that bootstrap refuses.
        # TASK-40308: after a restore Authority… unlocks into recovery instead;
        # without one (this rig) nothing in-app recovers them.
        assert "OS keyring was reset" in status and "profile" not in status
        assert "cannot recover these plugins without a restored backup" in status
        assert "saved data stay on disk" in status
        assert "Reset authority" not in status


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (211, 44)])
async def test_import_preview_review_buttons_state_why_and_never_open(
    request, native_console, native_package, size
):
    """TASK-40712 AC#1: an Import… preview holds candidates, not a package. Its
    two review buttons are disabled with the reason on them, and a press that
    reaches the screen anyway opens nothing; the app keeps running."""
    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    app = BrowseApp(native_console)
    package = native_package()
    async with app.run_test(size=size) as pilot:
        screen = app.screen
        await settled(pilot, screen)
        screen.query_one("#plugins-import", Button).focus()
        await pilot.press("enter")
        screen.query_one("#plugins-location", Input).value = str(package)
        screen.query_one("#plugins-import-app", Select).value = "codex"
        screen.query_one("#plugins-submit", Button).focus()
        await pilot.press("enter")
        await settled(pilot, screen)
        assert screen.browse.preview.candidates
        status = str(screen.query_one("#plugins-status", Static).render())
        assert "Import previews only list what the folder holds" in status
        assert "Repository…" in status and "Add source…" in status
        for identity in ("plugins-review-install", "plugins-review-preview"):
            button = screen.query_one(f"#{identity}", Button)
            assert button.disabled
            assert str(button.label) == "No package to review"
            button.press()
            button.post_message(Button.Pressed(button))
            await pilot.pause()
            assert app.screen is screen
            assert not isinstance(app.screen, PluginReviewModal)
        # The whole label is painted in the form's button row at both widths.
        assert "No package to review" in paint(screen)
        assert app.is_running


@private_profile_test
@pytest.mark.asyncio
async def test_review_dialog_load_failure_stays_in_the_dialog(
    request, native_console, native_package
):
    """TASK-40712 AC#2: a review dialog whose load raises shows the failure in
    the dialog and leaves the app running. The forced failure is the crash
    the task found: an Import… preview handed to the dialog as its producer."""
    from tldw_chatbook.Plugins.imports import inspect_app_import
    from tldw_chatbook.Widgets.plugin_review_modal import PluginReviewModal

    app = BrowseApp(native_console)
    preview = inspect_app_import("codex", native_package())
    async with app.run_test(size=(120, 35)) as pilot:
        await settled(pilot, app.screen)
        await app.push_screen(
            PluginReviewModal(
                app.plugin_source_service, producer=preview, workspace_id=None
            )
        )
        modal = app.screen
        await modal.workers.wait_for_complete()
        status_widget = modal.query_one("#plugin-review-status", Static)
        for _ in range(100):
            await pilot.pause(0.02)
            if "could not load" in str(status_widget.render()):
                break
        await pilot.pause()
        status = str(status_widget.render())
        assert "Review could not load (AttributeError)" in status
        assert "Nothing was changed" in status
        assert app.is_running and app.screen is modal
        assert app.return_code is None
        assert modal.query_one("#plugin-review-prepare", Button).disabled


async def _until(predicate, what, seconds=120):
    from Tests.UI import pump_probe

    assert await pump_probe.until(predicate, seconds), f"{what} did not happen"


@platform_support.managed_runtime
@private_profile_test
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_restored_plugins_recover_from_the_plugins_screen(
    request, tmp_path, monkeypatch, native_package
):
    """TASK-40308 AC#1 and AC#2 from the Plugins screen, on real services: after
    a restore over recreated folders with plugins changed since the backup,
    Authority… with the backup's passphrase opens the recovery review. Confirming
    it, then rebinding the replaced download record (Candidates…) and the
    replaced saved-data folder (Manage plugin), each after its own review, runs
    the plugin again with its saved data."""
    from types import SimpleNamespace

    from Tests.Plugins.test_recovery import _restored_after_change
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Widgets.plugin_review_modal import (
        PluginAuthorityDialog,
        PluginRestoreAdoptionModal,
        PluginReviewModal,
        PluginStatusModal,
    )
    from tldw_chatbook.Workspaces.registry_service import (
        LocalWorkspaceRegistryService,
    )

    service, _, installation, root, approved = await _restored_after_change(
        tmp_path, monkeypatch, native_package
    )
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="plugin-tests")
    )
    registry.ensure_default_workspace()
    plugins = service()
    app = BrowseApp(SimpleNamespace(service=plugins, registry=registry))
    try:
        async with app.run_test(size=(120, 35)) as pilot:
            screen = app.screen
            await settled(pilot, screen)
            screen.query_one("#plugins-authority", Button).press()
            await _until(
                lambda: (
                    isinstance(app.screen, PluginAuthorityDialog)
                    and app.screen.query("#submit-button")
                ),
                "unlock dialog",
            )
            dialog = app.screen
            dialog.query_one("#password-input", Input).value = "test passphrase"
            dialog.query_one("#submit-button", Button).press()
            await _until(
                lambda: (
                    isinstance(app.screen, PluginRestoreAdoptionModal)
                    and app.screen.query("#plugin-adopt-close")
                ),
                "recovery review",
            )
            adoption = app.screen
            review = str(adoption.query_one("#plugin-adopt-review", Static).render())
            assert "This undoes 1 plugin change made after this backup" in review
            assert "Plugin: " in review and installation in review
            assert "Saved-data folders kept attached: 1" in review
            assert not approved
            adoption.query_one("#plugin-adopt-confirm", Button).press()
            status = adoption.query_one("#plugin-adopt-status", Static)
            await _until(lambda: str(status.render()), "recovery confirmation")
            # r14plg-rev1: hedged, since recovery_required has more than one cause.
            assert "recovery is not finished" in str(status.render())
            assert "folder replaced since last bound" in str(status.render())
            adoption.query_one("#plugin-adopt-close", Button).press()
            await _until(lambda: app.screen is screen, "recovery review closed")
            screen.query_one("#plugins-candidates", Button).press()
            await _until(
                lambda: (
                    isinstance(app.screen, PluginStatusModal)
                    and app.screen.rows
                    and app.screen.query("#plugin-status-close")
                ),
                "candidates",
            )
            candidates = app.screen
            listing = candidates.query_one(OptionList)
            listing.highlighted = next(
                index for index, row in enumerate(candidates.rows) if row["rebind"]
            )
            candidates.query_one("#plugin-status-rebind", Button).press()
            text = candidates.query_one("#plugin-review-status", Static)
            await _until(
                lambda: "Replaced since last bound" in str(text.render()),
                "rebind review",
            )
            shown = str(text.render())
            assert "Plugin: " in shown and installation in shown
            assert "Folder: " in shown and "Contents: " in shown
            candidates.query_one("#plugin-status-rebind-confirm", Button).press()
            await _until(
                lambda: "Plugin authority recovery completed" in str(text.render()),
                "rebind confirmation",
            )
            assert approved
            candidates.query_one("#plugin-status-close", Button).press()
            await _until(lambda: app.screen is screen, "candidates closed")
            await app.push_screen(
                PluginReviewModal(
                    app.plugin_source_service,
                    installation_id=installation,
                    workspace_id=None,
                )
            )
            manage = app.screen
            await manage.workers.wait_for_complete()
            await _until(lambda: not manage._loading, "manage dialog load")
            manage.query_one("#plugin-review-action", Select).value = "rebind-data"
            await pilot.pause()
            manage.query_one("#plugin-review-prepare", Button).press()
            await _until(
                lambda: not manage._busy and manage.draft.review is not None,
                "folder review",
            )
            body = str(manage.query_one("#plugin-review-body", Static).render())
            assert f"Folder: {root.path}" in body
            assert "Contents: 1 files, 18 bytes" in body
            assert "Replaced since last bound" in body
            confirm = manage.query_one("#plugin-review-confirm", Button)
            assert not confirm.disabled
            confirm.press()

            async def rebound():
                (row,) = await plugins.data_status(installation_id=installation)
                return row["recovery_reason"] is None

            for _ in range(600):
                if await rebound():
                    break
                await asyncio.sleep(0.1)
            assert await rebound(), "the saved-data folder was not rebound"
        assert (root.path / "notes.txt").read_bytes() == b"made by the plugin"
        await plugins._call(
            lambda: plugins._admission.capture(installation, None, "restored-run")
        )
    finally:
        await plugins.aclose()


@platform_support.managed_runtime
@private_profile_test
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_reset_after_restore_sets_plugins_aside_and_kept_data_reattaches(
    request, tmp_path, monkeypatch, native_package
):
    """TASK-40308 AC#1 fallback from the Plugins screen, on real services: with
    the backup's passphrase lost, Reset authority… says before and after the reset
    that setup sets the restored plugins aside and keeps their saved data;
    Authority… then sets up a new passphrase, and Manage plugin attaches the kept
    saved-data folder to the reinstalled plugin after its own review."""
    from types import SimpleNamespace

    from textual.widgets import Select

    from Tests.Plugins.test_recovery import _restored_after_change
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Widgets.plugin_review_modal import (
        PluginAuthorityDialog,
        PluginAuthorityResetModal,
        PluginReviewModal,
    )
    from tldw_chatbook.Workspaces.registry_service import (
        LocalWorkspaceRegistryService,
    )

    service, install, _, root, approved = await _restored_after_change(
        tmp_path, monkeypatch, native_package
    )
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="plugin-tests")
    )
    registry.ensure_default_workspace()
    plugins = service("elsewhere")
    app = BrowseApp(SimpleNamespace(service=plugins, registry=registry))
    try:
        async with app.run_test(size=(120, 35)) as pilot:
            screen = app.screen
            await settled(pilot, screen)
            screen.query_one("#plugins-reset-authority", Button).press()
            await _until(
                lambda: (
                    isinstance(app.screen, PluginAuthorityResetModal)
                    and app.screen.query("#plugin-reset-close")
                ),
                "reset dialog",
            )
            reset = app.screen
            status = reset.query_one("#plugin-reset-status", Static)
            reset.query_one("#plugin-reset-review", Button).press()
            await _until(lambda: "Namespace" in str(status.render()), "reset review")
            consequence = str(status.render())
            assert "sets the restored plugins aside" in consequence
            assert "keeps every saved-data folder where it is" in consequence
            assert "Nothing is deleted" in consequence
            # TASK-41107: package files move aside too, out of plugin storage.
            assert "move there too and stop counting against plugin storage" in consequence
            assert "unlock instead: that keeps every plugin" in consequence
            reset.query_one("#plugin-reset-confirm", Button).press()
            await _until(lambda: "reset." in str(status.render()), "reset")
            done = str(status.render())
            assert "Press Authority… to set it up again with a new passphrase" in done
            assert "Use Authority to set up again" not in done
            reset.query_one("#plugin-reset-close", Button).press()
            await _until(lambda: app.screen is screen, "reset closed")
            screen.query_one("#plugins-authority", Button).press()
            await _until(
                lambda: (
                    isinstance(app.screen, PluginAuthorityDialog)
                    and app.screen.query("#confirm-input")
                    and app.screen.query("#submit-button")
                ),
                "setup dialog",
            )
            setup = app.screen
            setup.query_one("#password-input", Input).value = "new passphrase"
            setup.query_one("#confirm-input", Input).value = "new passphrase"
            setup.query_one("#submit-button", Button).press()
            await _until(lambda: app.screen is screen, "setup")
            assert approved and await plugins.authority_status() == "ready"
            ((kept, former, _alias),) = await plugins.kept_data_folders()
            assert kept == root.root_id
            # r14plg-rev2: the set-aside registry is restored, untrusted text; a
            # markup-shaped alias must render literally, not crash the dialog.
            import sqlite3

            (aside,) = root.path.parent.parent.glob("reset-*")
            with sqlite3.connect(aside / "registry.sqlite3") as connection:
                connection.execute(
                    "UPDATE installations SET alias=? WHERE installation_id=?",
                    ("evil [/] [b]bold", former),
                )
            reinstalled = await install(plugins, "op-git-again")
            await app.push_screen(
                PluginReviewModal(
                    app.plugin_source_service,
                    installation_id=reinstalled,
                    workspace_id=None,
                )
            )
            manage = app.screen
            await manage.workers.wait_for_complete()
            await _until(lambda: not manage._loading, "manage dialog load")
            manage.query_one("#plugin-review-action", Select).value = "attach-data"
            await pilot.pause()
            # r14plg-rev1: the kept folders are offered with their unproven
            # former plugin, from the set-aside registry; nothing is typed.
            chooser = manage.query_one("#plugin-review-kept", Select)
            assert chooser.display
            assert not manage.query_one("#plugin-review-reference", Input).display
            ((label, value),) = chooser._options[1:]
            assert value == root.root_id
            assert former in str(label) and "(not proven)" in str(label)
            assert "evil [/] [b]bold" in str(label)
            chooser.value = root.root_id
            await pilot.pause()
            manage.query_one("#plugin-review-prepare", Button).press()
            await _until(
                lambda: not manage._busy and manage.draft.review is not None,
                "kept folder review",
            )
            body = str(manage.query_one("#plugin-review-body", Static).render())
            assert "Kept folder to attach:" in body
            assert f"Folder: {root.path}" in body
            assert "Contents: 1 files, 18 bytes" in body
            assert "Kept from before plugin authority was reset" in body
            assert f"Kept folder: {root.root_id} · was" in body
            assert "(not proven)" in body
            manage.query_one("#plugin-review-confirm", Button).press()

            async def attached():
                return await plugins.kept_data_folders() == ()

            for _ in range(600):
                if await attached():
                    break
                await asyncio.sleep(0.1)
            assert await attached(), "the kept folder was not attached"
        (status_row,) = await plugins.data_status(installation_id=reinstalled)
        assert status_row["root"].root_id == root.root_id
        assert status_row["recovery_reason"] is None
        assert (root.path / "notes.txt").read_bytes() == b"made by the plugin"
    finally:
        await plugins.aclose()
