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
