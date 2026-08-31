"""The conversation row action menu, driven through the real Console.

TASK-23200. The rail's conversation rows carried a star button that shipped
disabled on a fresh install, stretched to the full height of a multi-line row,
and was explained by "Local stars unavailable" printed beside it. This suite
pins the replacement: a one-row asterisk that opens an anchored, keyboard
operable menu.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_console_left_rail import make_console_pilot
from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
    ConsoleConversationActionMenu,
)


def _opener(screen) -> Button:
    return screen.query_one("#console-conversation-actions-0", Button)


@pytest.mark.asyncio
async def test_row_carries_a_one_row_asterisk_not_a_full_height_star() -> None:
    """The control must not reserve the row's whole height any more."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        opener = _opener(screen)

        assert str(opener.label).strip() == "*"
        assert opener.disabled is False
        assert opener.region.height == 1, (
            "the action opener is still reserving full row height"
        )
        assert not screen.query(".console-conversation-star"), (
            "the retired star control is still being composed"
        )


@pytest.mark.asyncio
async def test_local_stars_unavailable_jargon_is_gone() -> None:
    """The developer-facing line must not appear in the rail at all."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        rail = screen.query_one("#console-left-rail")
        text = " ".join(
            str(getattr(widget, "renderable", ""))
            for widget in rail.query("*")
            if widget.display
        )
        assert "Local stars unavailable" not in text
        assert not screen.query("#console-conversation-browser-marks-unavailable")


@pytest.mark.asyncio
async def test_asterisk_opens_the_menu_with_the_expected_entries() -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)

        menu = screen.query_one(ConsoleConversationActionMenu)
        labels = [str(button.label).strip() for button in menu.query(Button)]
        assert labels == [
            "Favourite",
            "Change status ▸",
            "Archive",
            "Rename…",
            "More ▸",
        ]


@pytest.mark.asyncio
async def test_every_disabled_entry_states_its_precondition() -> None:
    """A greyed control with no explanation is the defect being removed."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)

        menu = screen.query_one(ConsoleConversationActionMenu)
        for button in menu.query(Button):
            if button.disabled:
                assert button.tooltip, (
                    f"{button.id} is disabled with no stated reason"
                )


@pytest.mark.asyncio
async def test_more_opens_delete_and_back_returns() -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        menu = screen.query_one(ConsoleConversationActionMenu)

        more = next(
            button
            for button in menu.query(Button)
            if getattr(button, "console_action_id", "") == "page:more"
        )
        more.press()
        await pilot.pause(0.5)
        assert menu.page == "more"
        assert [
            getattr(button, "console_action_id", "") for button in menu.query(Button)
        ] == ["page:root", "delete"]

        back = next(iter(menu.query(Button)))
        back.press()
        await pilot.pause(0.5)
        assert menu.page == "root"


@pytest.mark.asyncio
async def test_escape_steps_out_of_a_submenu_before_closing() -> None:
    """Escape in a submenu returns to the root rather than dropping the row."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        menu = screen.query_one(ConsoleConversationActionMenu)

        next(
            button
            for button in menu.query(Button)
            if getattr(button, "console_action_id", "") == "page:more"
        ).press()
        await pilot.pause(0.5)
        assert menu.page == "more"

        await pilot.press("escape")
        await pilot.pause(0.5)
        assert menu.page == "root", "escape closed the menu instead of stepping back"
        assert screen.query(ConsoleConversationActionMenu)

        await pilot.press("escape")
        await pilot.pause(0.5)
        assert not screen.query(ConsoleConversationActionMenu), (
            "escape at the root did not close the menu"
        )


@pytest.mark.asyncio
async def test_menu_focuses_its_first_actionable_entry_on_open() -> None:
    """Keyboard users must land on something they can actually choose."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.4)

        menu = screen.query_one(ConsoleConversationActionMenu)
        focused = pilot.app.focused
        assert focused is not None
        assert focused in list(menu.query(Button))
        assert not focused.disabled


@pytest.mark.asyncio
async def test_click_outside_closes_the_menu_without_dispatching(monkeypatch) -> None:
    """ADR-068 dismiss contract: a click elsewhere folds the menu, no actions.

    Clicking the composer is the canonical stranding path: Textual moves
    focus to the clicked widget before the press bubbles to the screen, so
    the dismissal must also leave focus exactly where the click put it.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        assert screen.query(ConsoleConversationActionMenu)

        dispatched: list[object] = []
        monkeypatch.setattr(
            screen,
            "on_conversation_action_chosen",
            lambda event: dispatched.append(event),
        )

        assert await pilot.click("#console-native-composer")
        await pilot.pause(0.3)

        assert not screen.query(ConsoleConversationActionMenu), (
            "a click outside the menu left it open"
        )
        assert dispatched == [], "an outside click dispatched a menu action"
        assert pilot.app.focused is not None
        assert pilot.app.focused is not _opener(screen), (
            "outside-click dismissal pulled focus back to the opener"
        )


@pytest.mark.asyncio
async def test_click_on_menu_chrome_keeps_the_menu_open() -> None:
    """A click on the menu's border must not fold it mid-inspection.

    Targets the top border row (offset y=0) -- menu chrome, not a button --
    through the same screen-level mouse path a real terminal press takes.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        menu = screen.query_one(ConsoleConversationActionMenu)

        await pilot.click(ConsoleConversationActionMenu, offset=(2, 0))
        await pilot.pause(0.3)

        assert screen.query_one(ConsoleConversationActionMenu), (
            "a click on the menu itself dismissed it"
        )
        assert menu.page == "root"


@pytest.mark.asyncio
async def test_escape_with_focus_outside_the_menu_closes_it() -> None:
    """Escape must reach a stranded menu even after focus moved elsewhere.

    Focus is moved to the composer without a mouse press (the screen seam
    directly), which is the state a user reaches via keyboard pane cycling
    once click-outside dismissal exists.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)
        assert screen.query(ConsoleConversationActionMenu)

        composer = screen.query_one("#console-native-composer")
        screen.set_focus(composer)
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause(0.3)

        assert not screen.query(ConsoleConversationActionMenu), (
            "escape from outside the menu left it stranded"
        )
        assert pilot.app.focused is composer, (
            "escape-from-elsewhere moved focus instead of only closing the menu"
        )


@pytest.mark.asyncio
async def test_pressing_the_asterisk_again_replaces_rather_than_stacks() -> None:
    """The opener's press path still ends with exactly one menu mounted."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        _opener(screen).press()
        await pilot.pause(0.3)

        await pilot.click("#console-conversation-actions-0")
        await pilot.pause(0.3)

        mounted = screen.query(ConsoleConversationActionMenu)
        assert len(mounted) == 1, (
            f"expected one replaced menu, found {len(mounted)}"
        )


@pytest.mark.unit
def test_menu_width_constant_and_stylesheet_cannot_drift() -> None:
    """The two encodings of the menu's width must agree.

    Qodo review, PR #2233: anchoring clamps against `MENU_WIDTH` while
    rendering uses the CSS `width`, so if one changes alone the menu is
    positioned for a size it is not drawn at.

    Qodo's suggested fix -- interpolate the constant into the stylesheet --
    is not available here: `css/build_css.py` lifts `BUNDLED_CSS` into the
    built stylesheet statically and rejects anything that is not a plain
    string literal, so an f-string breaks the CSS bundle build outright
    (observed: "BUNDLED_CSS is not a plain string literal"). Pinning them
    together in a test gives the same protection within that constraint.
    """
    import re

    from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
        ConsoleConversationActionMenu,
    )

    declared = re.search(
        r"ConsoleConversationActionMenu\s*\{[^}]*?\bwidth:\s*(\d+)\s*;",
        ConsoleConversationActionMenu.BUNDLED_CSS,
        re.S,
    )
    assert declared, "the menu stylesheet no longer declares an explicit width"
    assert int(declared.group(1)) == ConsoleConversationActionMenu.MENU_WIDTH, (
        f"stylesheet width {declared.group(1)} != MENU_WIDTH "
        f"{ConsoleConversationActionMenu.MENU_WIDTH}; anchoring and rendering "
        "have drifted apart"
    )
