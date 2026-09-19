"""Refresh preserves Tool Profile keyboard context without overriding newer focus."""

import asyncio
from dataclasses import replace

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_configuration_hub import StyledSettingsDestinationHarness
from Tests.UI.test_settings_tool_profiles import (
    ToolProfileListing,
    _build_test_app,
    _open_settings_category,
    _profile,
    _WorkflowService,
)
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Settings_Widgets.tool_profiles_panel import ToolProfilesPanel


def _host():
    listing = ToolProfileListing(
        profiles=(
            _profile("alpha", origin="imported", binding_state="unbound"),
            _profile("beta", origin="imported", binding_state="unbound"),
        )
    )
    service = _WorkflowService(listing)
    app = _build_test_app()
    app.tool_pack_service = service
    return StyledSettingsDestinationHarness(app, "settings"), service


async def _open(pilot):
    await _open_settings_category(pilot, "#settings-category-tool-profiles")
    await pilot.app.workers.wait_for_complete()
    return pilot.app.screen.query_one(ToolProfilesPanel)


async def _focus(pilot, selector):
    button = pilot.app.screen.query_one(selector, Button)
    button.focus()
    await pilot.pause()
    button.scroll_visible(animate=False, immediate=True)
    await pilot.pause()
    assert pilot.app.focused is button
    return button


def _visible(host, control):
    region, clip = host.screen._compositor.visible_widgets[control]
    assert region.intersection(clip) == region


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["import", "export", "edit", "bind", "remove"])
@pytest.mark.parametrize("change", ["same", "reorder"])
@private_profile_test
async def test_profile_refresh_keeps_the_same_action_identity(request, action, change):
    host, service = _host()
    async with host.run_test(size=(80, 24)) as pilot:
        panel = await _open(pilot)
        selector = (
            "#tool-profiles-import"
            if action == "import"
            else f"#tool-profile-{action}-1"
        )
        old = await _focus(pilot, selector)
        listing = service.listing
        if change == "reorder":
            listing = ToolProfileListing(profiles=tuple(reversed(listing.profiles)))
        await panel.apply_listing(listing)
        await pilot.pause()
        focused = host.focused
        if change == "same":
            assert focused is old
        else:
            expected_id = (
                "tool-profiles-import"
                if action == "import"
                else f"tool-profile-{action}-0"
            )
            assert focused.id == expected_id
        if action != "import":
            assert panel._button_actions[focused][1].profile_id == "beta"
        _visible(host, focused)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["missing", "disabled", "unavailable"])
@private_profile_test
async def test_disappearing_action_has_a_safe_visible_continuation(request, change):
    host, service = _host()
    async with host.run_test(size=(80, 24)) as pilot:
        panel = await _open(pilot)
        await _focus(pilot, "#tool-profile-remove-1")
        if change == "missing":
            listing = ToolProfileListing(profiles=(service.listing.profiles[0],))
            expected = "tool-profiles-import"
        elif change == "disabled":
            listing = ToolProfileListing(
                profiles=(
                    service.listing.profiles[0],
                    replace(service.listing.profiles[1], removal_eligible=False),
                )
            )
            expected = "tool-profile-export-1"
        else:
            listing = ToolProfileListing(unavailable_category="store_invalid")
            expected = "settings-detail-pane-body"
        await panel.apply_listing(listing)
        await pilot.pause()
        assert host.focused.id == expected
        _visible(host, host.focused)


@pytest.mark.asyncio
@pytest.mark.parametrize("newer", ["rail", "modal"])
@private_profile_test
async def test_render_does_not_reclaim_newer_focus(request, monkeypatch, newer):
    host, service = _host()
    entered, release = asyncio.Event(), asyncio.Event()
    async with host.run_test(size=(80, 24)) as pilot:
        panel = await _open(pilot)
        await _focus(pilot, "#tool-profile-edit-1")
        original = panel.recompose

        async def held_recompose():
            entered.set()
            await release.wait()
            await original()

        monkeypatch.setattr(panel, "recompose", held_recompose)
        task = asyncio.create_task(
            panel.apply_listing(
                ToolProfileListing(profiles=tuple(reversed(service.listing.profiles)))
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if newer == "rail":
                expected = await _focus(pilot, "#settings-category-theme")
            else:
                await host.push_screen(
                    ConfirmationDialog(title="Newer dialog", message="Keep this focus")
                )
                await pilot.pause()
                expected = host.focused
            screen = host.screen
            release.set()
            await task
            await pilot.pause()
            assert host.screen is screen
            assert host.focused is expected
        finally:
            release.set()
            await task
            if newer == "modal":
                await pilot.press("escape")


@pytest.mark.asyncio
@private_profile_test
async def test_overlapping_renders_finish_teardown_and_keep_latest_listing(
    request, monkeypatch
):
    host, service = _host()
    entered, release = asyncio.Event(), asyncio.Event()
    async with host.run_test(size=(80, 24)) as pilot:
        panel = await _open(pilot)
        settings = host.screen
        await _focus(pilot, "#tool-profile-edit-1")
        original = panel.recompose
        calls = 0

        async def held_recompose():
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
            await original()

        monkeypatch.setattr(panel, "recompose", held_recompose)
        first = ToolProfileListing(profiles=tuple(reversed(service.listing.profiles)))
        latest = ToolProfileListing(
            profiles=(replace(service.listing.profiles[1], revision=4),)
        )
        try:
            settings._tool_profiles_listing_generation += 1
            settings._apply_tool_profiles_listing(
                settings._tool_profiles_listing_generation, first
            )
            await asyncio.wait_for(entered.wait(), 3)
            worker = next(
                w for w in host.workers if w.group == "settings-tool-profiles-render"
            )
            settings._tool_profiles_listing_generation += 1
            settings._apply_tool_profiles_listing(
                settings._tool_profiles_listing_generation, latest
            )
            await pilot.pause()
            assert not worker.is_cancelled
            release.set()
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert panel.profile_ids == ("beta",)
            assert panel.row("beta").revision == 4
            assert host.focused.id == "tool-profile-edit-0"
            _visible(host, host.focused)
        finally:
            release.set()
            await host.workers.wait_for_complete()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["import", "export", "remove"])
@private_profile_test
async def test_cancel_returns_to_the_invoking_profile_action(request, operation):
    host, service = _host()
    async with host.run_test(size=(80, 24)) as pilot:
        await _open(pilot)
        settings = host.screen
        selector = (
            "#tool-profiles-import"
            if operation == "import"
            else f"#tool-profile-{operation}-1"
        )
        old = await _focus(pilot, selector)
        await pilot.press("enter")
        async with asyncio.timeout(5):
            while host.screen is settings:
                await pilot.pause(0.025)
        await pilot.press("escape")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert host.screen is settings
        assert host.focused is old
        assert not any(
            call[0] in {"import", "publish", "remove"} for call in service.calls
        )
        _visible(host, old)
