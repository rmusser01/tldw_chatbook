"""Appearance controls fit compact terminals with the real app stylesheets."""

from typing import ClassVar

import pytest
from textual.widgets import Button, Input

import tldw_chatbook.Widgets.Console.console_appearance_picker_modal as picker
from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_console_appearance_picker import PickerHarness
from tldw_chatbook.Chat.console_appearance import ConsoleConversationAppearance
from tldw_chatbook.Widgets.emoji_picker import EmojiButton


class StyledPickerHarness(PickerHarness):
    CSS_PATH: ClassVar = list(APP_STYLESHEETS)


@pytest.fixture
def catalog(monkeypatch):
    # Two full rows expose inherited Button widths that a three-icon fixture
    # misses; the native check separately uses the real catalog.
    chars = "🚀🎨🧪🌍🌙⭐🔥💡📚🎵🎯✅🐱🐶🌲🌻🍎🍋⚽🏆📷🔑💎🎁"
    entries = [
        {
            "char": char,
            "name": "rocket" if i == 0 else f"icon {i}",
            "category": "objects",
            "aliases": [],
        }
        for i, char in enumerate(chars)
    ]
    monkeypatch.setattr(picker, "get_emoji_data", lambda: (entries, {}, []))
    monkeypatch.setattr(picker, "load_recent_emojis", list)
    saved = []
    monkeypatch.setattr(picker, "save_recent_emoji", saved.append)
    return saved


def _visible(modal, widget):
    region, clip = modal._compositor.visible_widgets[widget]
    assert region.intersection(clip) == region, (widget.id, region, clip)


def _paint(modal, widget):
    region = widget.content_region
    return "\n".join(
        strip.crop(region.x, region.right).text
        for strip in modal._compositor.render_strips()[region.y : region.bottom]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("cancel", ["keyboard", "pointer"])
@private_profile_test
async def test_compact_actions_remain_visible_through_resize_and_cancel(
    request, catalog, theme, cancel
):
    app = StyledPickerHarness(
        conversation_id="fixture", conversation_title="Research", icon="🚀"
    )
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.screen
        for size in ((80, 24), (170, 48), (80, 24)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            for action in (picker.APPLY_ID, picker.CLEAR_ID, picker.CANCEL_ID):
                button = modal.query_one(f"#{action}", Button)
                _visible(modal, button)
                assert str(button.label) in _paint(modal, button)

        field = modal.query_one(f"#{picker.HEX_INPUT_ID}", Input)
        field.value = "c0ffee"
        await pilot.pause()
        assert modal._selected_color == "#c0ffee"
        modal.query_one(f"#{picker.FILTER_INPUT_ID}", Input).focus()
        await pilot.pause()
        if cancel == "keyboard":
            await pilot.press("shift+tab")
            assert app.focused.id == picker.CANCEL_ID
            _visible(modal, app.focused)
            await pilot.press("enter")
        else:
            assert await pilot.click(f"#{picker.CANCEL_ID}")
        await pilot.pause()
        assert app.result is None
        assert modal not in app.screen_stack
        assert catalog == []


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_palette_labels_icon_rows_and_selected_result_fit(
    request, catalog, theme, size
):
    app = StyledPickerHarness(conversation_id="fixture", icon="🚀", color="#c0ffee")
    app.theme = theme
    async with app.run_test(size=size) as pilot:
        modal = app.screen
        await pilot.pause()
        none = modal.query_one(f"#{picker.SWATCH_NONE_ID}", Button)
        _visible(modal, none)
        assert "none" in _paint(modal, none), (
            _paint(modal, none),
            none.content_region,
            none.styles,
        )
        # Every icon in the first full row is reachable without horizontal
        # scrolling or a test-side scroll repair.
        icons = list(modal.query(EmojiButton))
        assert len(icons) == 24
        for icon in icons[:12]:
            _visible(modal, icon)
        assert await pilot.click(icons[1])
        await pilot.pause()
        assert modal._selected_icon == "🎨"

        swatches = list(modal.query(f".{picker.SWATCH_CLASS}"))
        last = swatches[-1]
        last.focus()
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        _visible(modal, last)
        assert "■" in _paint(modal, last)
        assert await pilot.click(last)
        await pilot.pause()
        assert modal._selected_color == last.hex_color
        assert modal.query_one(f"#{picker.HEX_INPUT_ID}", Input).value == last.hex_color
        none.focus()
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        assert await pilot.click(none)
        await pilot.pause()
        assert modal._selected_color is None
        assert modal.query_one(f"#{picker.HEX_INPUT_ID}", Input).value == ""
        assert await pilot.click(f"#{picker.APPLY_ID}")
        await pilot.pause()
        assert app.result == ConsoleConversationAppearance(icon="🎨")
        assert catalog == ["🎨"]


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_filter_typing_survives_debounced_grid_focus_refresh(
    request, catalog, theme
):
    app = StyledPickerHarness(conversation_id="fixture", icon="🎨")
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.screen
        field = modal.query_one(f"#{picker.FILTER_INPUT_ID}", Input)
        for length, char in enumerate("rocket", start=1):
            await pilot.press(char)
            await pilot.pause(picker.FILTER_DEBOUNCE_SECONDS + 0.15)
            assert app.focused is field
            assert field.value == "rocket"[:length]
            assert field.selection.is_empty
            assert field.cursor_position == length
        await pilot.press("enter")
        await pilot.pause()
        assert modal._selected_icon == "🚀"
        await pilot.press("escape")
        await pilot.pause()
        assert app.result is None
        assert catalog == []
