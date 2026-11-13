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
Each size's sweep is split across nodes over contiguous thirds of
``ALL_CATEGORY_IDS`` (the summary order the unsplit sweep used), and a pure
test checks that the thirds together are all of it.
"""

from types import SimpleNamespace
from typing import ClassVar

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _visible_text,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId, SettingsScreen

ALL_CATEGORY_IDS = tuple(
    summary.category.value
    # Enumerating the rail needs no running app or its databases at collection.
    for summary in SettingsScreen(SimpleNamespace(app_config={}))._category_summaries()
)

#: Categories whose rail press lands focus somewhere other than their own
#: rail button. Theme lands in its list, where its c/t/e and j/k keys act
#: (TASK-33072, pinned by test_settings_theme_picker_screen.py's
#: test_rail_click_and_open_theme_focus_the_theme_list).
_FOCUS_AFTER_SELECT = {SettingsCategoryId.THEME.value: "settings-theme-list"}

#: One sweep node per (size, group), each over a contiguous third of
#: ``ALL_CATEGORY_IDS`` (the summary order the unsplit sweep used), so a
#: census node stays well under the UI lane's 30 s per-entry budget: the
#: unsplit sweep took 28-53 s per size under local load (TASK-36525).
#: Contiguous, so every category-to-category switch but the group boundaries
#: is one the unsplit sweep made.
_GROUPS = (0, 1, 2)


def _category_group(group: int) -> tuple[str, ...]:
    """The categories sweep node ``group`` visits, in ``ALL_CATEGORY_IDS`` order."""
    share = -(-len(ALL_CATEGORY_IDS) // len(_GROUPS))
    return ALL_CATEGORY_IDS[group * share : (group + 1) * share]


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


async def _wait_for(pilot, predicate, timeout: float = 10.0) -> bool:
    """Poll ``predicate`` until it holds or ``timeout`` seconds pass."""
    for _ in range(int(timeout / 0.05)):
        if predicate():
            return True
        await pilot.pause(0.05)
    return bool(predicate())


async def _click_settings_category(pilot, category_value: str) -> None:
    """Press the category's real rail button, wait for the switch, settle.

    The name is kept for its many importers; since TASK-36525 this presses
    rather than clicks. ``Button.press()`` is the call a click and Enter both
    end in, and the coordinate click it replaces depended on where the row
    sat on screen. Textual drops a press on a hidden or disabled row
    silently, so a press that switches nothing fails here, naming the
    category.
    """
    screen = pilot.app.screen
    screen.query_one(f"#settings-category-{category_value}", Button).press()
    switched = await _wait_for(
        pilot,
        lambda: (
            screen.active_category == category_value
            and not screen._category_pane_swap_pending
        ),
    )
    assert switched, f"Pressing the {category_value} rail row did not open it"
    await _settle_settings(pilot)


def test_the_sweep_groups_cover_every_category_once():
    """Splitting the sweep loses no category: the groups, in order, are
    exactly ``ALL_CATEGORY_IDS`` (the summary order the unsplit sweep used),
    and none is empty."""
    # Upstream keeps the per-domain rail slots (our task-1343 collapse was
    # superseded by the collapsible Domain Defaults rail), so pin a floor,
    # not an exact count.
    assert len(ALL_CATEGORY_IDS) >= 12
    assert all(_category_group(group) for group in _GROUPS)
    swept = [value for group in _GROUPS for value in _category_group(group)]
    assert swept == list(ALL_CATEGORY_IDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 35), (80, 24)], ids=["120x35", "80x24"])
@pytest.mark.parametrize("group", _GROUPS)
@private_profile_test
async def test_every_settings_category_renders_and_restores_focus(size, group, request):
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=size) as pilot:
        await _settle_settings(pilot)
        screen = _active_destination_screen(host)
        assert (
            tuple(summary.category.value for summary in screen._category_summaries())
            == ALL_CATEGORY_IDS
        )

        for category_value in _category_group(group):
            await _click_settings_category(pilot, category_value)
            screen = _active_destination_screen(host)
            expected = _FOCUS_AFTER_SELECT.get(
                category_value, f"settings-category-{category_value}"
            )
            await _wait_for(
                pilot,
                lambda expected=expected: (
                    getattr(pilot.app.focused, "id", None) == expected
                ),
            )

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
            assert focused.id == expected, (
                f"Expected focus on {expected}, got {focused!r} at {size}"
            )


class _BundledSettingsHarness(DestinationHarness):
    """Settings with the app's own stylesheets, so its panes narrow as in the app."""

    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]


@pytest.mark.asyncio
@pytest.mark.parametrize("height", (20, 30))
@pytest.mark.parametrize("category", ("workspaces", "console-behavior", "theme"))
@private_profile_test
async def test_narrow_settings_categories_lay_out_at_every_width_60_to_40(
    category, height, request
):
    """TASK-38740: no Settings category raises while its detail pane narrows.

    Below about 60 columns the rail and inspector leave the detail pane a few
    cells. On the base build the Workspaces row ("Default (active) - 0
    folders") and Console behavior's "Conversation context and memory" button
    were laid out at two content cells, both of them Button line-pad, and the
    Theme list's options at zero. Folding a word at width 0 raised
    ``ValueError: range() arg 3 must not be zero`` and the app exited.
    """
    app = _build_test_app()
    host = _BundledSettingsHarness(app, "settings")

    async with host.run_test(size=(100, height)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, category)
        for width in range(60, 39, -1):
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            await pilot.pause()
            assert host._exception is None, (category, width, host._exception)
        assert pilot.app.screen.active_category == category


#: TASK-38953: the narrowest content an open category may get. The compact
#: layout's 32-cell rail left it 19 cells at 60 columns and 0 at 40.
_SINGLE_PANE_MIN_CONTENT = 30
#: TASK-38953 AC#3: rail, open category and Scope Inspector widths at the
#: full-screen sizes, measured on dev 2684e5453f before one-pane mode.
_FULL_SCREEN_PANE_WIDTHS = {(211, 44): (32, 126, 43), (235, 52): (32, 144, 49)}


@pytest.mark.asyncio
@pytest.mark.parametrize("height", (24, 30))
@pytest.mark.parametrize("category", ("overview", "providers-models"))
@private_profile_test
async def test_the_open_category_keeps_30_cells_from_60_down_to_40_columns(
    category, height, request
):
    """TASK-38953 AC#1: below the width where the rail beside the open
    category leaves it 30 cells, Settings shows one pane at a time: the open
    category, with a labelled way back to the category list."""
    app = _build_test_app()
    host = _BundledSettingsHarness(app, "settings")

    async with host.run_test(size=(100, height)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, category)
        screen = pilot.app.screen
        screen.set_focus(None)  # reading the open category, not the rail
        for width in range(60, 39, -1):
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            await pilot.pause()
            detail = screen.query_one("#settings-detail-pane")
            assert detail.content_region.width >= _SINGLE_PANE_MIN_CONTENT, (
                width,
                detail.content_region.width,
            )
            back = screen.query_one("#settings-single-pane-back", Button)
            assert back.display and back.region.width > 0, width
            assert "Categories" in str(back.label)
            assert not screen.query_one("#settings-category-pane").display, width
        assert screen.active_category == category
        # Narrowed with focus on a rail row, the rail is the pane shown.
        await pilot.resize_terminal(100, height)
        await pilot.pause()
        screen.query_one(f"#settings-category-{category}", Button).focus()
        await pilot.pause()
        await pilot.resize_terminal(50, height)
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one("#settings-category-pane").display
        assert not screen.query_one("#settings-detail-pane").display


async def _until_focused(pilot, widget, key: str, limit: int = 6) -> None:
    """Press ``key`` until ``widget`` holds focus, as a keyboard user would."""
    for _ in range(limit):
        if pilot.app.focused is widget:
            return
        await pilot.press(key)
        await pilot.pause()
    assert pilot.app.focused is widget, pilot.app.focused


@pytest.mark.asyncio
@pytest.mark.parametrize("width", (60, 40))
@pytest.mark.parametrize("group", _GROUPS)
@private_profile_test
async def test_every_category_is_reachable_by_keyboard_one_pane_at_a_time(
    width, group, request
):
    """TASK-38953 AC#2: with one pane at a time, the way back to the category
    list and every category in it are reached with keys alone. Each node
    walks one third of the rail, as the sweep above does."""
    app = _build_test_app()
    host = _BundledSettingsHarness(app, "settings")

    async with host.run_test(size=(width, 24)) as pilot:
        await _settle_settings(pilot)
        screen = pilot.app.screen
        rail = screen.query_one("#settings-category-pane")
        detail = screen.query_one("#settings-detail-pane")
        back = screen.query_one("#settings-single-pane-back", Button)
        screen.set_focus(None)
        await pilot.press("tab")
        await pilot.pause()
        assert pilot.app.focused is back
        categories = screen._filtered_category_values()
        assert sorted(categories) == sorted(ALL_CATEGORY_IDS)
        for category in _category_group(group):
            current = screen.active_category
            if category != current:
                await _until_focused(pilot, back, "shift+tab")
                await pilot.press("enter")
                await pilot.pause()
                assert rail.display and not detail.display, category
                assert pilot.app.focused is screen.query_one(
                    f"#settings-category-{current}", Button
                )
                steps = categories.index(category) - categories.index(current)
                await pilot.press(*(["down"] * steps or ["up"] * -steps))
                await pilot.press("enter")
                switched = await _wait_for(
                    pilot,
                    lambda category=category: (
                        screen.active_category == category
                        and not screen._category_pane_swap_pending
                    ),
                )
                assert switched, category
                await pilot.pause()
            assert detail.display and not rail.display, category
            assert back.display, category
            assert pilot.app.focused is not None, category


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((211, 44), (235, 52)), ids=("211x44", "235x52"))
@private_profile_test
async def test_full_screen_settings_keeps_all_three_panes(size, request):
    """TASK-38953 AC#3: at full-screen sizes the layout is the one dev had."""
    app = _build_test_app()
    host = _BundledSettingsHarness(app, "settings")

    async with host.run_test(size=size) as pilot:
        await _settle_settings(pilot)
        screen = pilot.app.screen
        widths = tuple(
            screen.query_one(pane_id).region.width
            for pane_id in (
                "#settings-category-pane",
                "#settings-detail-pane",
                "#settings-impact-pane",
            )
        )
        assert widths == _FULL_SCREEN_PANE_WIDTHS[size], widths
        assert not screen.query_one("#settings-single-pane-back", Button).display


@pytest.mark.asyncio
@private_profile_test
async def test_widening_the_terminal_never_narrows_the_detail_pane(request):
    """TASK-39134. Owner decision (2026-11-06): "Inspector waits for room: show
    the Scope Inspector only once the detail pane can keep at least the width
    it had just below the breakpoint, so widening the terminal never narrows
    the detail pane." On dev the pane (its region) went from 63 cells to 38 at
    100/101 columns, where the inspector arrived at its 21-cell minimum, and
    from 68 to 56 at 133/134, where it took its 36-cell floor."""
    app = _build_test_app()
    host = _BundledSettingsHarness(app, "settings")

    async with host.run_test(size=(96, 30)) as pilot:
        await _settle_settings(pilot)
        screen = pilot.app.screen
        detail = screen.query_one("#settings-detail-pane")
        inspector = screen.query_one("#settings-impact-pane")

        def read() -> tuple[int, int]:
            shown = inspector.display and screen.is_attached
            return detail.region.width, inspector.region.width if shown else 0

        widths: dict[int, tuple[int, int]] = {}
        for width in range(96, 146):  # the criterion's 96-140, and the floor
            await pilot.resize_terminal(width, 30)
            reading = None
            for _ in range(20):  # settled: the same reading across two pauses
                await pilot.pause()
                if (now := read()) == reading:
                    break
                reading = now
            widths[width] = reading

    narrowed = [
        (width, widths[width - 1][0], widths[width][0])
        for width in range(97, 146)
        if widths[width][0] < widths[width - 1][0]
    ]
    assert narrowed == [], str(widths)
    held = widths[100][0]
    early = [(width, d) for width, (d, i) in widths.items() if i and d < held]
    assert early == [], str(widths)
    assert widths[140][1] >= 21, str(widths)  # the inspector did come
    assert widths[145][1] == 36, str(widths)  # and reached its floor


@pytest.mark.asyncio
@private_profile_test
async def test_slash_in_one_pane_mode_opens_the_rail_with_its_filter(request):
    """TASK-38953 review r11g-rev1 I2: "/" focused the hidden category
    filter, so typing filtered a rail nobody could see."""
    app = _build_test_app()
    host = _BundledSettingsHarness(app, "settings")

    async with host.run_test(size=(60, 24)) as pilot:
        await _settle_settings(pilot)
        screen = pilot.app.screen
        rail = screen.query_one("#settings-category-pane")
        search = screen.query_one("#settings-category-search")
        screen.set_focus(None)
        await pilot.pause()
        assert not rail.display  # the open category is the pane shown
        await pilot.press("slash")
        await pilot.pause()
        await pilot.pause()
        assert rail.display
        assert not screen.query_one("#settings-detail-pane").display
        assert pilot.app.focused is search
        assert search.region.width > 0
