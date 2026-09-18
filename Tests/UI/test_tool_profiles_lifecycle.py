"""Tool Profiles refresh must preserve action and initialization ownership."""

import asyncio
import threading
from dataclasses import replace

import pytest
from textual.widgets import Button
from textual.worker import WorkerState

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_tool_profiles import (
    DestinationHarness,
    ToolProfileListing,
    ToolProfilesPanel,
    _build_test_app,
    _open_settings_category,
    _PanelHarness,
    _profile,
    _WorkflowService,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


class _HeldPanel(ToolProfilesPanel):
    def __init__(self, listing):
        super().__init__(listing)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def on_button_pressed(self, event):
        self.entered.set()
        await self.release.wait()
        super().on_button_pressed(event)
        event.prevent_default()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["export", "edit", "bind", "remove"])
@pytest.mark.parametrize("change", ["replacement", "revision"])
async def test_queued_action_cannot_acquire_refreshed_profile(action, change):
    original = _profile("selected-A")
    current = replace(
        original,
        profile_id="replacement-B" if change == "replacement" else original.profile_id,
        revision=4,
        policy_digest="b" * 64,
    )
    panel = _HeldPanel(ToolProfileListing(profiles=(original,)))
    host = _PanelHarness(panel)
    async with host.run_test() as pilot:
        old_button = panel.query_one(f"#tool-profile-{action}-0", Button)
        old_button.press()
        await asyncio.wait_for(panel.entered.wait(), 2)
        await panel.apply_listing(ToolProfileListing(profiles=(current,)))
        assert not old_button.is_attached
        panel.release.set()
        await pilot.pause()
        assert host.events == []

        fresh = panel.query_one(f"#tool-profile-{action}-0", Button)
        fresh.press()
        await pilot.pause()
        assert len(host.events) == 1
        event = host.events[0]
        assert (event.profile_id, event.revision, event.policy_digest) == (
            current.profile_id,
            current.revision,
            current.policy_digest,
        )


class _LoadingHost(DestinationHarness):
    def __init__(self, app_instance):
        super().__init__(app_instance, "settings")
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.thread_release = threading.Event()

    def start_composition(self, threaded):
        self.composition = self.run_worker(
            self._compose_profiles_thread if threaded else self._compose_profiles(),
            thread=threaded,
            exit_on_error=False,
        )
        self.app_instance._tool_pack_composition_worker = self.composition

    async def _compose_profiles(self):
        self.started.set()
        await self.release.wait()
        self._publish_profiles()

    def _compose_profiles_thread(self):
        self.call_from_thread(self.started.set)
        if not self.thread_release.wait(10):
            raise TimeoutError("test composition was not released")
        self.call_from_thread(self._publish_profiles)

    def _publish_profiles(self):
        self.app_instance.tool_pack_service = _WorkflowService(
            ToolProfileListing(profiles=(_profile("ready-profile"),))
        )
        self.app_instance.tool_pack_service_unavailable_reason = None


@pytest.mark.asyncio
@pytest.mark.parametrize("departure", ["refresh", "remove"])
@pytest.mark.parametrize("threaded", [False, True])
@private_profile_test
async def test_settings_cancellation_preserves_shared_profile_initialization(
    request, departure, threaded
):
    app = _build_test_app()
    app.tool_pack_service = None
    app.tool_pack_service_unavailable_reason = "starting"
    host = _LoadingHost(app)
    async with host.run_test(size=(120, 35)) as pilot:
        await host.workers.wait_for_complete()
        host.start_composition(threaded)
        await host.started.wait()
        other_observer = host.run_worker(host.composition.wait(), exit_on_error=False)
        try:
            screen = host.screen
            screen._select_category("tool-profiles")
            await pilot.pause()
            if departure == "refresh":
                screen._request_tool_profiles_listing()
            else:
                await host.pop_screen()
            await pilot.pause()
            assert host.composition.state is WorkerState.RUNNING
            assert other_observer.state is WorkerState.RUNNING
        finally:
            host.release.set()
            host.thread_release.set()
        await host.composition.wait()
        await other_observer.wait()
        if departure == "remove":
            await host.push_screen(SettingsScreen(app))
            await _open_settings_category(pilot, "#settings-category-tool-profiles")
        await host.workers.wait_for_complete()
        await pilot.pause()
        panel = host.screen.query_one(
            "#settings-tool-profiles-panel", ToolProfilesPanel
        )
        assert panel.profile_ids == ("ready-profile",)
        assert host.composition.state is WorkerState.SUCCESS
