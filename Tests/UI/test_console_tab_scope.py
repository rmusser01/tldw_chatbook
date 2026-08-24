"""TASK-2154.11 (AC-02): Console Tab/Shift+Tab cycle within the focused pane.

The app-level focus chain drags a Console Tab tour through all 15 nav buttons
between the composer cluster and the Console control bar, stranding keyboard
users in app chrome (``p5-focus-tour.txt``). Tab is now scoped to the focused
widget's Console region (``CONSOLE_TAB_REGIONS``); F6/Shift+F6 remain the way
to move between panes. These tests pin the new contract end to end: a scripted
tour reaches the transcript, the status chips, and the Inspector pane within a
handful of stops, and no stop ever lands on a ``nav-*`` button.
"""

from __future__ import annotations

import pytest

import tldw_chatbook.app as app_module
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from tldw_chatbook.UI.Screens.chat_screen import CONSOLE_TAB_REGIONS


@pytest.fixture(autouse=True)
def _disable_full_app_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


def _mark_console_onboarding_complete(app) -> None:
    app.app_config = getattr(app, "app_config", {}) or {}
    console_config = app.app_config.setdefault("console", {})
    onboarding = console_config.setdefault("onboarding", {})
    onboarding["first_send_completed"] = True


async def _wait_for_focused_id(app, pilot, widget_id: str) -> None:
    for _ in range(40):
        if getattr(app.focused, "id", None) == widget_id:
            return
        await pilot.pause(0.05)
    raise AssertionError(
        f"Expected focus on {widget_id!r}, found {getattr(app.focused, 'id', None)!r}"
    )


def _focused_region_roots(widget) -> tuple[str, ...] | None:
    """Return the CONSOLE_TAB_REGIONS roots owning ``widget`` (or None)."""
    current = widget
    while current is not None:
        current_id = getattr(current, "id", None)
        for roots in CONSOLE_TAB_REGIONS:
            if current_id in roots:
                return roots
        current = getattr(current, "parent", None)
    return None


def _focus_chain_ids_within(screen, roots: tuple[str, ...]) -> list[str]:
    """Ids of focus-chain widgets living under any of ``roots`` (DOM order)."""
    root_widgets = [screen.query_one(f"#{root}") for root in roots]
    ids: list[str] = []
    for widget in screen.focus_chain:
        current = widget
        while current is not None:
            if current in root_widgets:
                ids.append(getattr(widget, "id", None) or type(widget).__name__)
                break
            current = getattr(current, "parent", None)
    return ids


async def _ready_console(app, pilot):
    console = app.screen
    await _wait_for_selector(console, pilot, "#console-native-composer")
    console._set_console_rail_preference(
        left_open=True,
        right_open=True,
        notify_on_failure=False,
    )
    await pilot.pause()
    console.query_one("#console-native-composer").focus()
    await _wait_for_focused_id(app, pilot, "console-native-composer")
    return console


@pytest.mark.asyncio
async def test_console_tab_from_composer_stays_in_region_and_wraps():
    """Tab from the composer cycles composer+control-bar only -- never nav-*."""
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "chat"
    _mark_console_onboarding_complete(app)

    async with app.run_test(size=(160, 48)) as pilot:
        console = await _ready_console(app, pilot)
        composer_region = (
            "console-workbench-header",
            "console-control-bar",
            "console-native-composer",
        )
        assert CONSOLE_TAB_REGIONS[0] == composer_region
        expected_cycle = _focus_chain_ids_within(console, composer_region)
        assert len(expected_cycle) >= 6, (
            "composer region should hold the composer cluster + control bar "
            f"buttons, got {expected_cycle}"
        )

        visited: list[str] = []
        for _ in range(len(expected_cycle) + 2):
            await pilot.press("tab")
            await pilot.pause(0.05)
            focused = app.focused
            focused_id = getattr(focused, "id", None) or "<none>"
            assert not focused_id.startswith("nav-"), (
                f"Tab escaped into app chrome: {focused_id}"
            )
            assert _focused_region_roots(focused) == composer_region, (
                f"Tab left the composer region: {focused_id}"
            )
            visited.append(focused_id)

        assert "console-auto-speak" in visited
        assert "console-hands-free-switch" in visited

        # The cycle wraps: after a full loop the sequence repeats itself.
        cycle_len = len(expected_cycle)
        assert visited[cycle_len] == visited[0], (
            f"Tab did not wrap within the composer region: {visited}"
        )


@pytest.mark.asyncio
async def test_console_shift_tab_reverses_tab_within_region():
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "chat"
    _mark_console_onboarding_complete(app)

    async with app.run_test(size=(160, 48)) as pilot:
        await _ready_console(app, pilot)

        await pilot.press("tab")
        await pilot.pause(0.05)
        first = getattr(app.focused, "id", None)
        await pilot.press("tab")
        await pilot.pause(0.05)
        second = getattr(app.focused, "id", None)
        assert first != second

        await pilot.press("shift+tab")
        await pilot.pause(0.05)
        assert getattr(app.focused, "id", None) == first

        await pilot.press("shift+tab")
        await pilot.pause(0.05)
        assert getattr(app.focused, "id", None) == "console-native-composer"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "switch_id",
    ["console-auto-speak", "console-hands-free-switch"],
)
async def test_console_f6_from_header_switch_advances_from_composer_pane(
    switch_id: str,
) -> None:
    """Header switches participate in the composer pane's F6 tour."""
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "chat"
    _mark_console_onboarding_complete(app)

    async with app.run_test(size=(160, 48)) as pilot:
        console = await _ready_console(app, pilot)
        console.query_one(f"#{switch_id}").focus()
        await _wait_for_focused_id(app, pilot, switch_id)

        await pilot.press("f6")
        await _wait_for_focused_id(app, pilot, "console-context-rail-collapse")


@pytest.mark.asyncio
async def test_console_tab_cycles_within_left_rail_region():
    """Tab inside the left rail never crosses into nav chrome or other panes."""
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "chat"
    _mark_console_onboarding_complete(app)

    async with app.run_test(size=(160, 48)) as pilot:
        console = await _ready_console(app, pilot)
        console.query_one("#console-context-rail-collapse").focus()
        await _wait_for_focused_id(app, pilot, "console-context-rail-collapse")
        rail_region = ("console-context-rail-handle", "console-left-rail")

        for _ in range(8):
            await pilot.press("tab")
            await pilot.pause(0.05)
            focused = app.focused
            focused_id = getattr(focused, "id", None) or "<none>"
            assert not focused_id.startswith("nav-"), (
                f"Tab escaped into app chrome: {focused_id}"
            )
            assert _focused_region_roots(focused) == rail_region, (
                f"Tab left the rail region: {focused_id}"
            )


@pytest.mark.asyncio
async def test_console_tab_from_nav_button_keeps_default_chain():
    """Focus already in app chrome (a clicked nav button) keeps the old chain.

    Scoping applies to the Console workbench regions; once focus IS in the
    nav bar (mouse entry), Tab must remain able to traverse it -- otherwise
    app-level navigation loses keyboard access entirely.
    """
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "chat"
    _mark_console_onboarding_complete(app)

    async with app.run_test(size=(160, 48)) as pilot:
        console = await _ready_console(app, pilot)
        console.query_one("#nav-home").focus()
        await _wait_for_focused_id(app, pilot, "nav-home")

        await pilot.press("tab")
        await pilot.pause(0.05)
        focused_id = getattr(app.focused, "id", None) or ""
        assert focused_id.startswith("nav-"), (
            f"Tab from a nav button should stay in the app-level chain, got {focused_id}"
        )


@pytest.mark.asyncio
async def test_console_focus_tour_reaches_transcript_chips_inspector_under_ten_stops():
    """AC#5: scripted F6+Tab tour reaches transcript, chips, Inspector fast."""
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = "chat"
    _mark_console_onboarding_complete(app)

    async with app.run_test(size=(160, 48)) as pilot:
        await _ready_console(app, pilot)

        stops: list[tuple[str, str]] = []

        async def tour(key: str) -> None:
            await pilot.press(key)
            await pilot.pause(0.05)
            stops.append((key, getattr(app.focused, "id", None) or "<none>"))

        # F6 out of the composer: rail pane, then transcript pane.
        await tour("f6")
        await tour("f6")
        assert stops[-1][1] == "console-native-transcript", f"tour: {stops}"

        # Optional transcript actions may precede the status chips. Traverse
        # only this region, leaving one of the ten stops for the final F6.
        transcript_region = ("console-transcript-region", "console-status-chips")
        for _ in range(7):
            await tour("tab")
            focused = app.focused
            focused_id = getattr(focused, "id", None) or "<none>"
            assert not focused_id.startswith("nav-"), f"tour: {stops}"
            assert _focused_region_roots(focused) == transcript_region, (
                f"Tab left the transcript/status-chips region: {stops}"
            )

            node = focused
            while node is not None:
                if getattr(node, "id", None) == "console-status-chips":
                    break
                node = getattr(node, "parent", None)
            if node is not None:
                break
        else:
            pytest.fail(f"expected a status chip within the stop budget: {stops}")

        # F6 again: Inspector pane (rail open -> its collapse button).
        await tour("f6")
        assert stops[-1][1] == "console-inspector-rail-collapse", f"tour: {stops}"

        assert len(stops) <= 10, f"tour took too long: {stops}"
        assert not any(focused_id.startswith("nav-") for _, focused_id in stops)
