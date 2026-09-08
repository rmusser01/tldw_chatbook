"""Production-styled geometry and focus contracts for Ctrl+K Character chats."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import pytest
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Input

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationPage,
    CharacterConversationRow,
    LocalCharacterConversationTarget,
    ResolvedLocalCharacterKey,
)
from tldw_chatbook.Chat.console_switcher_state import SwitcherMode
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
)


def _rows(count: int = 8) -> tuple[CharacterConversationRow, ...]:
    authority = ResolvedLocalCharacterKey("authority-a", 1)
    return tuple(
        CharacterConversationRow.resolved(
            LocalCharacterConversationTarget(authority, f"conversation-{index}"),
            character_label="Ada",
            title=f"Character conversation {index}",
            last_modified=f"2026-09-{20 - index:02d}T12:00:00+00:00",
            created_at="2026-09-01T00:00:00Z",
            selected_excerpt=f"Excerpt {index}",
        )
        for index in range(count)
    )


class _GeometryApp(ConsolidatedCSSApp):
    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]

    async def on_mount(self) -> None:
        async def character_loader(**_kwargs):
            return CharacterConversationPage(_rows(), 58, None, 5)

        await self.push_screen(
            ConsoleSessionSwitcherModal(
                character_loader=character_loader,
                initial_mode=SwitcherMode.CHARACTER_CHATS,
                profile_authority="profile-a",
                authority_token="runtime-a",
            )
        )


@pytest.mark.asyncio
async def test_exact_52_by_20_budget_and_four_two_line_results() -> None:
    app = _GeometryApp()
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        screen = app.screen
        modal = screen.query_one("#console-switcher-modal", Vertical)
        results = screen.query_one("#console-switcher-results", VerticalScroll)
        detail = screen.query_one("#console-switcher-selected-detail")
        actions = screen.query_one("#console-switcher-page-controls")
        footer = screen.query_one("#console-switcher-footer")

        assert modal.region == (0, 0, 52, 20)
        assert modal.content_region.width >= 48
        assert results.region.height == 8
        assert detail.region.height == 2
        assert actions.region.height == 1
        assert footer.region.height == 1
        assert len(screen.query(".console-switcher-result")) == 8
        assert "Meaning" not in str(screen.render())
        first = screen.query_one(".console-switcher-result", Button)
        assert first.region.height == 2
        assert app.focused is screen.query_one("#console-switcher-query", Input)
        assert screen.query_one("#console-switcher-cancel", Button).region.bottom <= 19
        app.export_screenshot()
        frame = "\n".join(strip.text for strip in screen._compositor.render_strips())
        for label in ("Character chats", "History", "Cancel", "Search local"):
            assert label in frame


@pytest.mark.asyncio
async def test_focus_order_is_modes_search_results_actions_cancel() -> None:
    app = _GeometryApp()
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        ids = []
        for _ in range(15):
            await pilot.press("tab")
            ids.append(getattr(app.focused, "id", ""))

        assert ids[0] == "console-switcher-results"
        assert all(
            widget_id.startswith("console-switcher-result-") for widget_id in ids[1:9]
        )
        assert "console-switcher-next-page" in ids
        assert "console-switcher-cancel" in ids
        cancel_index = ids.index("console-switcher-cancel")
        next_index = ids.index("console-switcher-next-page")
        assert next_index < cancel_index
        assert ids[cancel_index + 1 : cancel_index + 4] == [
            "console-switcher-active-mode",
            "console-switcher-history-mode",
            "console-switcher-character-mode",
        ]
        await pilot.press("shift+tab")
        assert getattr(app.focused, "id", "") != "console-switcher-selected-detail"


@pytest.mark.asyncio
async def test_wide_layout_retains_bounds_and_selected_detail_only() -> None:
    app = _GeometryApp()
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        modal = app.screen.query_one("#console-switcher-modal", Vertical)
        labels = [
            str(button.label) for button in app.screen.query(".console-switcher-result")
        ]
        detail = str(app.screen.query_one("#console-switcher-selected-detail").render())

        assert modal.region.width == 76
        assert modal.region.height <= 35
        assert all("Excerpt" not in label for label in labels)
        assert "Excerpt 0" in detail


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(52, 20), (120, 50)])
@pytest.mark.parametrize(
    "excerpt",
    ["Long ASCII excerpt " * 20, "研究🙂界隈 " * 35],
    ids=["ascii", "wide-unicode"],
)
async def test_long_selected_detail_paints_timestamp_on_second_row_after_resize(
    size, excerpt
):
    from Tests.UI.test_console_character_switcher import _CharacterSwitcherApp

    async def loader(**_kwargs):
        return CharacterConversationPage(
            (replace(_rows(1)[0], selected_excerpt=excerpt),), 1, None, 5
        )

    app = _CharacterSwitcherApp(
        character_loader=loader, initial_mode=SwitcherMode.CHARACTER_CHATS
    )
    async with app.run_test(size=size) as pilot:
        other_size = (120, 50) if size == (52, 20) else (52, 20)
        for current_size in (size, other_size, size):
            await pilot.resize_terminal(*current_size)
            await pilot.pause()
            screen = app.screen
            detail = screen.query_one("#console-switcher-selected-detail")
            assert detail.content_region.height == 2
            strips = screen._compositor.render_strips()
            painted = [
                strips[y]
                .crop(detail.content_region.x, detail.content_region.right)
                .text
                for y in range(detail.content_region.y, detail.content_region.bottom)
            ]
            assert "Long ASCII" in painted[0] or "研究" in painted[0]
            assert screen._entries[0].absolute_time in painted[1]
            assert "RESUME CHAT" in painted[1]


@pytest.mark.asyncio
async def test_tab_focus_moves_the_painted_activation_marker() -> None:
    app = _GeometryApp()
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        await pilot.press("tab", "tab", "tab")
        buttons = list(app.screen.query(".console-switcher-result"))
        assert app.focused is buttons[1]
        assert buttons[1].has_class("console-switcher-result-candidate")
        assert not buttons[0].has_class("console-switcher-result-candidate")


@pytest.mark.asyncio
async def test_context_handoff_is_contained_and_transfers_exact_query():
    from Tests.UI.test_console_character_context import _controller
    from tldw_chatbook.UI.Console_Modules.character_context import (
        ConsoleCharacterQueryHandoffCapability,
    )
    from tldw_chatbook.Widgets.Console.console_character_context import (
        ConsoleCharacterContext,
    )

    handoffs = []
    controller = _controller(
        query_handoff_capability=ConsoleCharacterQueryHandoffCapability(True),
        query_handoff=handoffs.append,
    )

    class HandoffApp(ConsolidatedCSSApp):
        CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]

        def compose(self):
            widget = ConsoleCharacterContext(controller)
            widget.styles.width = 30
            controller._state_changed = widget.sync_state
            yield widget

    app = HandoffApp()
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        owner = app.screen.query_one(ConsoleCharacterContext)
        if owner._task is not None:
            await owner._task
        await controller.search("needle")
        await pilot.pause()
        button = app.screen.query_one("#console-character-query-handoff", Button)
        assert owner.content_region.contains_region(button.region)
        button.press()
        await pilot.pause()
        assert [handoff.query for handoff in handoffs] == ["needle"]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(52, 20), (120, 50)])
async def test_unavailable_unicode_rows_keep_metadata_on_their_second_painted_line(
    size,
):
    from Tests.UI.test_console_character_switcher import (
        _CharacterSwitcherApp,
        _unavailable_row,
    )
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        UnavailableCharacterReason,
    )

    rows = tuple(
        _unavailable_row(
            str(index),
            "研究🙂 Long unavailable conversation " * 4,
            UnavailableCharacterReason.MISSING_CARD,
            "2026-09-03T12:00:00Z",
        )
        for index in range(4)
    )

    async def loader(**_kwargs):
        return CharacterConversationPage(rows, 4, None, 4)

    app = _CharacterSwitcherApp(
        character_loader=loader, initial_mode=SwitcherMode.CHARACTER_CHATS
    )
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        screen = app.screen
        strips = screen._compositor.render_strips()
        buttons = list(screen.query(".console-switcher-result"))
        assert len(buttons) == 4
        for button in buttons:
            assert button.region.height == 2
            second = (
                strips[button.region.y + 1]
                .crop(button.region.x, button.region.right)
                .text
            )
            assert "Historical Ada" in second
            assert "Local" in second
            assert all(
                line.cell_length
                <= button.content_size.width - 2 * button.styles.line_pad
                for line in button.label.split("\n")
            )
        frame = "\n".join(strip.text for strip in strips)
        assert "Cancel" in frame
