# test_settings_nav_active_scroll.py
# Description: task-4024 regression -- navigating to Settings must leave the
# nav bar scrolled so the active "Settings" destination (and its `is-active`
# highlight) is actually visible.
#
# The defect this pins: `SettingsScreen.on_mount` -> `_refresh_sync_rows()`
# sets screen-level `recompose=True` reactives ~300ms after mount, which
# recomposes the whole `BaseAppScreen` and mints a brand-new
# `MainNavigationBar` (discarding the first bar's already-successful
# scroll-to-active). The replacement bar's `_mark_mount_settled` marker (one
# `call_after_refresh` tick) fires BEFORE the post-recompose automatic focus
# placement lands on the bar's first button (`nav-home`) -- the opposite
# ordering from a first mount, where `AUTO_FOCUS` lands before the marker --
# so `on_descendant_focus` misrecorded that automatic landing as a
# DELIBERATE focus. From then on every settle pass (`_recenter_strip`, fed
# by the 0.5s interval and every resize) recentered on always-visible
# `nav-home` instead of the active `nav-settings`: `scroll_x` pinned at 0
# and even manual `scroll_to_widget(nav-settings)` calls were snapped back
# within one tick. The active highlight was permanently off-screen.

from __future__ import annotations

import pytest

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Constants import TAB_CHAT, TAB_SETTINGS
from tldw_chatbook.UI.Navigation.main_navigation import (
    MainNavigationBar,
    NavigateToScreen,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


async def _wait_for_screen(app, pilot, screen_type, tab: str):
    for _ in range(300):
        if app.current_tab == tab and isinstance(app.screen, screen_type):
            return app.screen
        await pilot.pause(0.01)
    raise AssertionError(f"did not finish routing to {screen_type.__name__}.")


def _current_nav_bar(screen) -> MainNavigationBar | None:
    """The screen's live nav bar, re-queried fresh.

    A screen-level recompose REPLACES the bar instance, so a held reference
    can go stale mid-test (it reads a dead widget's frozen geometry) -- every
    sample must re-query.
    """
    bars = list(screen.query(MainNavigationBar))
    return bars[-1] if bars else None


def _settings_button_fully_visible(screen) -> bool:
    """Whether `#nav-settings` is laid out fully inside the strip's viewport."""
    nav = _current_nav_bar(screen)
    if nav is None:
        return False
    try:
        strip = nav.query_one("#nav-destination-strip")
        button = nav.query_one("#nav-settings")
    except Exception:
        return False
    strip_region = strip.region
    button_region = button.region
    if strip_region.width <= 0 or button_region.width <= 0:
        return False
    return (
        button_region.x >= strip_region.x
        and button_region.right <= strip_region.right
        and button.has_class("is-active")
    )


@pytest.mark.asyncio
async def test_settings_nav_bar_scrolls_active_destination_into_view():
    """task-4024 AC#1/#3: after navigating to Settings (80 cols, where the
    strip must scroll to reach the Settings button), the active destination
    becomes fully visible within a bounded time -- and STAYS visible past
    the bar's own 0.5s recenter interval, surviving the sync-rows recompose
    that replaces the nav bar shortly after the screen mounts.
    """
    app = _build_test_app()
    app._initial_tab_value = TAB_CHAT
    async with app.run_test(size=(80, 24)) as pilot:
        await _wait_for_screen(app, pilot, ChatScreen, TAB_CHAT)
        await app.handle_screen_navigation(NavigateToScreen(TAB_SETTINGS))
        screen = await _wait_for_screen(app, pilot, SettingsScreen, TAB_SETTINGS)

        first_bar = _current_nav_bar(screen)
        assert first_bar is not None

        # Wait (bounded) for the sync-rows recompose to replace the bar --
        # the stuck state only exists on the REPLACEMENT bar, so a poll that
        # samples the first bar's successful scroll and returns early would
        # miss the defect entirely. If Settings ever stops recomposing, the
        # defect's precondition is gone and the visibility assertions below
        # still bind on the surviving bar.
        for _ in range(60):
            if _current_nav_bar(screen) is not first_bar:
                break
            await pilot.pause(0.05)

        # AC#1: bounded time to a visible, highlighted active destination.
        for _ in range(60):
            if _settings_button_fully_visible(screen):
                break
            await pilot.pause(0.05)
        else:
            nav = _current_nav_bar(screen)
            strip = nav.query_one("#nav-destination-strip") if nav else None
            button = nav.query_one("#nav-settings") if nav else None
            raise AssertionError(
                "active Settings destination never became visible: "
                f"replaced={nav is not first_bar} "
                f"scroll_x={float(strip.scroll_x) if strip else None} "
                f"strip.region={strip.region if strip else None} "
                f"button.region={button.region if button else None}"
            )

        # The stuck state's signature is a recenter loop that SNAPS BACK to
        # scroll 0 on its next 0.5s tick -- so visibility must survive one
        # full interval, not just be sampled once.
        await pilot.pause(0.7)
        assert _settings_button_fully_visible(screen), (
            "active Settings destination was scrolled into view but a later "
            "recenter tick snapped the strip away from it"
        )
