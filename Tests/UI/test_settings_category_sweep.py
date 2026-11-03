"""Regression sweep: every Settings category renders at UAT terminal sizes.

task-1338: UAT (headless Pilot, 120x35) found selecting the Theme or Splash
Screen category raised ``KeyError`` in ``SettingsScreen._inspector_guidance``
(the dict covered only 9 of 19 categories), blanking the whole screen, and
that focus was destroyed by the category recompose and never restored. These
tests visit ALL categories at both UAT sizes (120x35 and 80x24) and assert
the screen keeps rendering content and focus is restored after each switch.

task-1343: the eight read-only domain stub categories collapsed into the
single "Domain Ownership" category, so the sweep derives its list from the
sidebar summaries (12 categories) instead of the enum (which keeps the
retired stub ids as contract keys and legacy deep-link targets).

TASK-36525: the test builds app config in its body (``_build_test_app()``
runs ``load_settings()``), so it runs under ``@private_profile_test``; under
the per-test config redirect it raises ``RecoveryRequired`` (TASK-36381).
"""

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _visible_text,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId, SettingsScreen

ALL_CATEGORY_IDS = tuple(
    summary.category.value
    for summary in SettingsScreen(_build_test_app())._category_summaries()
)

#: Categories whose rail press lands focus somewhere other than their own
#: rail button. Theme lands in its list, where its c/t/e and j/k keys act
#: (TASK-33072, pinned by test_settings_theme_picker_screen.py's
#: test_rail_click_and_open_theme_focus_the_theme_list).
_FOCUS_AFTER_SELECT = {SettingsCategoryId.THEME.value: "settings-theme-list"}


async def _settle_settings(pilot) -> None:
    """Wait out mount-time/refresh workers plus the recompose they trigger."""
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    # Upstream's Domain Defaults rail starts collapsed; category sweeps need
    # every sidebar button visible/clickable, so pin it open.
    screen = getattr(pilot.app, "screen", None)
    if getattr(screen, "_domain_group_expanded", None) is False:
        screen._domain_group_expanded = True
        screen._apply_category_search_filter()
        await pilot.pause()


async def _click_settings_category(pilot, category_value: str) -> None:
    selector = f"#settings-category-{category_value}"
    screen = pilot.app.screen
    try:
        button = screen.query_one(selector)
        screen.query_one("#settings-category-list").scroll_to_widget(
            button, animate=False
        )
        await pilot.pause()
    except Exception:
        pass
    await pilot.click(selector)
    await _settle_settings(pilot)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 35), (80, 24)])
@private_profile_test
async def test_every_settings_category_renders_and_restores_focus(size, request):
    # Upstream keeps the per-domain rail slots (our task-1343 collapse was
    # superseded by the collapsible Domain Defaults rail), so pin a floor,
    # not an exact count.
    assert len(ALL_CATEGORY_IDS) >= 12
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=size) as pilot:
        await _settle_settings(pilot)
        screen = _active_destination_screen(host)

        for category_value in ALL_CATEGORY_IDS:
            await _click_settings_category(pilot, category_value)
            screen = _active_destination_screen(host)

            # No crash / no blank screen: the shell still renders content.
            assert screen.query("#settings-title"), (
                f"Settings title missing after selecting {category_value} at {size}"
            )
            visible_text = _visible_text(screen)
            assert "Settings" in visible_text, (
                f"Settings screen blank after selecting {category_value} at {size}"
            )

            # Focus is restored after the category recompose.
            focused = pilot.app.focused
            assert focused is not None, (
                f"Focus lost after selecting {category_value} at {size}"
            )
            expected = _FOCUS_AFTER_SELECT.get(
                category_value, f"settings-category-{category_value}"
            )
            assert focused.id == expected, (
                f"Expected focus on {expected}, got {focused!r} at {size}"
            )
