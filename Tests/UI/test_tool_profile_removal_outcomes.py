"""Removal outcomes remain truthful, current and visible beside their action."""

import threading
from dataclasses import replace

import pytest
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_configuration_hub import StyledSettingsDestinationHarness
from Tests.UI.test_settings_tool_profiles import (
    ToolProfileListing,
    _build_test_app,
    _open_settings_category,
    _profile,
    _WorkflowService,
)
from Tests.UI.test_tool_profile_review_lifetime import _activate, _wait
from tldw_chatbook.Tool_Packs.contracts import ToolPackError
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Settings_Widgets.tool_profiles_panel import ToolProfilesPanel


def _visible(host, control):
    region, clip = host.screen._compositor.visible_widgets[control]
    assert region.intersection(clip) == region


def _host(outcome, profile_id="research"):
    entered, release = threading.Event(), threading.Event()

    class Service(_WorkflowService):
        def remove_profile(self, profile_id, *, expected_revision):
            result = super().remove_profile(
                profile_id, expected_revision=expected_revision
            )
            entered.set()
            assert release.wait(20)
            profile = self.listing.profiles[-1]
            self.listing = ToolProfileListing(
                profiles=self.listing.profiles[:-1]
                + (
                    ()
                    if outcome in {"gone", "success"}
                    else (replace(profile, revision=4, policy_digest="b" * 64),)
                )
            )
            if outcome == "success":
                return result
            if outcome == "wrong_type":
                return object()
            if outcome == "exception":
                raise OSError("private storage detail")
            raise ToolPackError(
                "remove", "outcome_uncertain" if outcome == "gone" else outcome
            )

    service = Service(
        ToolProfileListing(
            profiles=tuple(
                _profile(name, origin="imported", binding_state="unbound")
                for name in ("alpha", "beta", "gamma", profile_id)
            )
        )
    )
    app = _build_test_app()
    app.tool_pack_service = service
    return StyledSettingsDestinationHarness(app, "settings"), service, entered, release


async def _begin(pilot, entered):
    await _open_settings_category(pilot, "#settings-category-tool-profiles")
    await pilot.app.workers.wait_for_complete()
    settings = pilot.app.screen
    await _activate(pilot, "#tool-profile-remove-3")
    await _wait(pilot, lambda: isinstance(pilot.app.screen, ConfirmationDialog))
    await _activate(pilot, "#confirm-button")
    await _wait(pilot, entered.is_set)
    assert pilot.app.screen is settings
    return settings


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["stale", "outcome_uncertain"])
@pytest.mark.parametrize("checkpoint", ["copy", "facts", "paint"])
@private_profile_test
async def test_removal_outcome_is_truthful_current_and_visible(
    request, outcome, checkpoint
):
    host, service, entered, release = _host(outcome)
    async with host.run_test(size=(80, 24)) as pilot:
        try:
            settings = await _begin(pilot, entered)
            release.set()
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert service.calls == [("remove", "research", 3)]
            if checkpoint == "copy":
                text = settings._tool_profiles_result.casefold()
                if outcome == "outcome_uncertain":
                    assert "uncertain" in text and "failed" not in text
                    assert "check" in text and "retry" in text
                else:
                    assert "changed" in text and "review" in text
            elif checkpoint == "facts":
                panel = settings.query_one(ToolProfilesPanel)
                assert panel.row("research").revision == 4
                assert panel.row("research").policy_digest == "b" * 64
            else:
                receipts = [
                    widget
                    for widget in settings.query(Static)
                    if widget.display
                    and str(widget.renderable) == settings._tool_profiles_result
                ]
                assert len(receipts) == 1
                _visible(host, receipts[0])
                focused = host.focused
                assert isinstance(focused, Button)
                assert focused.id == "tool-profile-remove-3"
                _visible(host, focused)
        finally:
            release.set()
            await host.workers.wait_for_complete()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        "in_use",
        "referenced",
        "non_removable",
        "exception",
        "wrong_type",
        "gone",
        "success",
    ],
)
@private_profile_test
async def test_removal_refusal_and_unknown_outcomes_offer_a_current_continuation(
    request, outcome
):
    profile_id = "[bold]research[/bold]"
    host, service, entered, release = _host(outcome, profile_id)
    async with host.run_test(size=(80, 24)) as pilot:
        try:
            settings = await _begin(pilot, entered)
            release.set()
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert service.calls == [("remove", profile_id, 3)]
            text = settings._tool_profiles_result
            assert "private storage detail" not in text
            if outcome in {"exception", "wrong_type", "gone"}:
                assert "uncertain" in text and "failed" not in text
            elif outcome == "success":
                assert profile_id in text and "permanently reserved" in text
            else:
                assert {
                    "in_use": "active run",
                    "referenced": "archived workspace",
                    "non_removable": "current details",
                }[outcome] in text
            gone = outcome in {"gone", "success"}
            panel = settings.query_one(ToolProfilesPanel)
            assert (profile_id not in panel.profile_ids) is gone
            assert host.focused.id == (
                "tool-profiles-import" if gone else "tool-profile-remove-3"
            )
            receipt = next(
                w
                for w in settings.query(Static)
                if w.display and str(w.renderable) == text
            )
            _visible(host, receipt)
            _visible(host, host.focused)
            assert not receipt._render_markup
        finally:
            release.set()
            await host.workers.wait_for_complete()


@pytest.mark.asyncio
@pytest.mark.parametrize("departure", ["category", "modal", "profile", "action"])
@private_profile_test
async def test_late_removal_outcome_does_not_move_newer_navigation(request, departure):
    host, _service, entered, release = _host("stale")
    async with host.run_test(size=(80, 24)) as pilot:
        try:
            settings = await _begin(pilot, entered)
            if departure == "category":
                await _activate(pilot, "#settings-category-theme")
                await _wait(
                    pilot,
                    lambda: (
                        settings.active_category == "theme"
                        and not settings._category_pane_swap_pending
                    ),
                )
            elif departure == "modal":
                await host.push_screen(
                    ConfirmationDialog(
                        title="Another dialog", message="Keep this focus"
                    )
                )
                await pilot.pause()
            else:
                button = settings.query_one(
                    "#tool-profile-edit-0"
                    if departure == "profile"
                    else "#tool-profile-edit-3",
                    Button,
                )
                button.focus()
                button.scroll_visible(animate=False, immediate=True)
                await pilot.pause()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            expected_screen, expected_id = host.screen, host.focused.id
            scroll = settings.query_one("#settings-detail-pane-body").scroll_y
            release.set()
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert host.screen is expected_screen
            assert host.focused.id == expected_id
            assert settings.query_one("#settings-detail-pane-body").scroll_y == scroll
            _visible(host, host.focused)
        finally:
            release.set()
            if isinstance(host.screen, ConfirmationDialog):
                await pilot.press("escape")
            await host.workers.wait_for_complete()


@pytest.mark.asyncio
@private_profile_test
async def test_refusal_after_confirmation_refresh_keeps_feedback_and_fallback_visible(
    request,
):
    host, service, entered, release = _host("in_use")
    async with host.run_test(size=(80, 24)) as pilot:
        try:
            await _open_settings_category(pilot, "#settings-category-tool-profiles")
            await host.workers.wait_for_complete()
            settings = host.screen
            await _activate(pilot, "#tool-profile-remove-3")
            await _wait(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
            profiles = service.listing.profiles
            service.listing = ToolProfileListing(
                profiles=profiles[:-1]
                + (
                    replace(
                        profiles[-1], removal_eligible=False, removal_blocker="in_use"
                    ),
                )
            )
            await _activate(pilot, "#confirm-button")
            await _wait(pilot, entered.is_set)
            release.set()
            await host.workers.wait_for_complete()
            await pilot.pause()
            assert host.focused.id == "tool-profile-export-3"
            assert settings.query_one("#tool-profile-remove-3", Button).disabled
            receipt = next(
                w
                for w in settings.query(Static)
                if w.display and str(w.renderable) == settings._tool_profiles_result
            )
            _visible(host, receipt)
            _visible(host, host.focused)
        finally:
            release.set()
            await host.workers.wait_for_complete()
