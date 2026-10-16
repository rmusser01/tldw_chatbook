"""Keyboard browsing preserves exact plugin identities and captured scope."""

import asyncio
from pathlib import Path

import pytest
from textual.widgets import Button, Input, OptionList, Select, Static

from Tests.Plugins.conftest import native_console as _native_console
from Tests.Plugins.conftest import native_package as _native_package
from Tests.Plugins.test_source_service import marketplace as _marketplace
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

    def __init__(self, rig):
        super().__init__()
        self.plugin_source_service = SourceService(rig.service)
        self.workspace_registry_service = rig.registry

    async def on_mount(self):
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


def paint(screen):
    return "\n".join(
        "".join(segment.text for segment in strip)
        for strip in screen._compositor.render_strips()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_keyboard_details_back_resize_and_restore_keep_stable_selection(
    native_console, size
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
        assert "Activation in: workspace-a" in paint(screen)
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


@pytest.mark.asyncio
async def test_real_source_add_and_import_preview_are_keyboard_reachable(
    native_console, native_package
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


@pytest.mark.asyncio
async def test_changed_workspace_refuses_pending_readme_and_refreshes_scope(
    native_console, monkeypatch
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


@pytest.mark.asyncio
async def test_reordered_rows_and_obsolete_search_keep_explicit_source_selection(
    native_console, native_package, monkeypatch
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


@pytest.mark.asyncio
async def test_live_catalog_refresh_preview_and_stale_failure_preserve_source_identity(
    native_console, marketplace
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


@pytest.mark.asyncio
async def test_selected_repository_preview_captures_real_snapshot_without_install(
    native_console, native_package
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


@pytest.mark.asyncio
async def test_editing_import_location_refuses_pending_preview(
    native_console, native_package, monkeypatch
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


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["back", "form"])
async def test_back_replaces_held_load_without_old_completion_clearing_new_owner(
    native_console, monkeypatch, transition
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
