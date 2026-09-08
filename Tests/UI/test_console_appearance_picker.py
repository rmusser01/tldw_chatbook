"""Console conversation appearance picker contracts (task-31207)."""

from __future__ import annotations

import pytest
from textual.app import App

from Tests.UI.consolidated_css import ConsolidatedCSSApp

import tldw_chatbook.Widgets.Console.console_appearance_picker_modal as modal_module
from tldw_chatbook.Chat.console_appearance import ConsoleConversationAppearance
from tldw_chatbook.Widgets.Console.console_appearance_picker_modal import (
    EMOJI_SELECTED_CLASS,
    SWATCH_SELECTED_CLASS,
    ConsoleAppearancePickerModal,
    filter_appearance_emojis,
)
from tldw_chatbook.Widgets.emoji_picker import EmojiButton

_FAKE_CATALOG = (
    {"char": "🧪", "name": "test tube", "category": "objects", "aliases": ["lab"]},
    {"char": "🎨", "name": "art", "category": "objects", "aliases": ["paint"]},
    {"char": "🚀", "name": "rocket", "category": "travel", "aliases": ["ship"]},
)


@pytest.fixture(autouse=True)
def _sandbox_emoji_sources(monkeypatch):
    """Pin the catalog and keep the picker off the real recents file."""
    monkeypatch.setattr(
        modal_module, "get_emoji_data", lambda: (_FAKE_CATALOG, {}, [])
    )
    monkeypatch.setattr(modal_module, "load_recent_emojis", lambda: [])
    saved: list[str] = []
    monkeypatch.setattr(modal_module, "save_recent_emoji", saved.append)
    return saved


def test_filter_matches_name_and_alias() -> None:
    assert [e["char"] for e in filter_appearance_emojis(_FAKE_CATALOG, "")] == [
        "🧪",
        "🎨",
        "🚀",
    ]
    assert [e["char"] for e in filter_appearance_emojis(_FAKE_CATALOG, "rocket")] == [
        "🚀"
    ]
    assert [e["char"] for e in filter_appearance_emojis(_FAKE_CATALOG, "LAB")] == ["🧪"]
    assert filter_appearance_emojis(_FAKE_CATALOG, "nope") == []


def test_constructor_rejects_foreign_values() -> None:
    modal = ConsoleAppearancePickerModal(
        conversation_id="conv-1",
        conversation_title="T",
        icon="not one glyph",
        color="cerulean",
    )
    assert modal._selected_icon is None
    assert modal._selected_color is None


class PickerHarness(ConsolidatedCSSApp):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self.result: object = "not-dismissed"

    def on_mount(self) -> None:
        self.push_screen(
            ConsoleAppearancePickerModal(**self._kwargs),
            callback=self._record_result,
        )

    def _record_result(self, result: object) -> None:
        self.result = result


async def _current_appearance(modal: ConsoleAppearancePickerModal):
    return ConsoleConversationAppearance(
        icon=modal._selected_icon, color=modal._selected_color
    )


@pytest.mark.asyncio
async def test_apply_commits_icon_and_color_and_saves_recents(
    _sandbox_emoji_sources,
) -> None:
    harness = PickerHarness(
        conversation_id="conv-1", conversation_title="Lab chat"
    )
    async with harness.run_test(size=(80, 30)) as pilot:
        modal = harness.screen_stack[-1]
        assert isinstance(modal, ConsoleAppearancePickerModal)
        await pilot.pause()

        # Carried-in unset state: nothing highlighted, no selection.
        assert await _current_appearance(modal) == ConsoleConversationAppearance()

        # Pick the first emoji, then a color swatch, then Apply.
        first_emoji = modal.query(EmojiButton).first()
        first_emoji.press()
        await pilot.pause()
        swatches = [
            button
            for button in modal.query(".console-appearance-swatch")
            if getattr(button, "hex_color", None) is not None
        ]
        swatches[0].press()
        await pilot.pause()

        assert first_emoji.has_class(EMOJI_SELECTED_CLASS)
        assert swatches[0].has_class(SWATCH_SELECTED_CLASS)
        assert await _current_appearance(modal) == ConsoleConversationAppearance(
            icon="🧪", color=swatches[0].hex_color
        )

        modal.query_one("#console-appearance-picker-apply").press()
        await pilot.pause()
        assert harness.result == ConsoleConversationAppearance(
            icon="🧪", color=swatches[0].hex_color
        )
        assert _sandbox_emoji_sources == ["🧪"]


@pytest.mark.asyncio
async def test_carried_in_appearance_is_preselected(_sandbox_emoji_sources) -> None:
    harness = PickerHarness(
        conversation_id="conv-1",
        conversation_title="T",
        icon="🚀",
        color="#22d3ee",
    )
    async with harness.run_test(size=(80, 30)) as pilot:
        modal = harness.screen_stack[-1]
        assert isinstance(modal, ConsoleAppearancePickerModal)
        await pilot.pause()

        assert await _current_appearance(modal) == ConsoleConversationAppearance(
            icon="🚀", color="#22d3ee"
        )
        highlighted = [
            button
            for button in modal.query(EmojiButton)
            if button.has_class(EMOJI_SELECTED_CLASS)
        ]
        assert [button.emoji_data["char"] for button in highlighted] == ["🚀"]
        selected_swatch = [
            button
            for button in modal.query(".console-appearance-swatch")
            if button.has_class(SWATCH_SELECTED_CLASS)
        ]
        assert [button.hex_color for button in selected_swatch] == ["#22d3ee"]


@pytest.mark.asyncio
async def test_clear_resets_to_unset(_sandbox_emoji_sources) -> None:
    harness = PickerHarness(
        conversation_id="conv-1",
        conversation_title="T",
        icon="🚀",
        color="#22d3ee",
    )
    async with harness.run_test(size=(80, 30)) as pilot:
        modal = harness.screen_stack[-1]
        assert isinstance(modal, ConsoleAppearancePickerModal)
        await pilot.pause()

        modal.query_one("#console-appearance-picker-clear").press()
        await pilot.pause()
        assert harness.result == ConsoleConversationAppearance()
        assert _sandbox_emoji_sources == []


@pytest.mark.asyncio
async def test_filter_narrows_the_grid(_sandbox_emoji_sources) -> None:
    harness = PickerHarness(conversation_id="conv-1", conversation_title="T")
    async with harness.run_test(size=(80, 30)) as pilot:
        modal = harness.screen_stack[-1]
        assert isinstance(modal, ConsoleAppearancePickerModal)
        await pilot.pause()

        filter_input = modal.query_one("#console-appearance-picker-filter")
        filter_input.value = "rocket"
        # Drive the debounced filter synchronously (the timer is 0.2s).
        modal._cancel_filter_timer()
        modal._apply_filter("rocket")
        await pilot.pause()

        chars = [
            button.emoji_data["char"] for button in modal.query(EmojiButton)
        ]
        assert chars == ["🚀"]


class SwitcherHarness(App[None]):
    def __init__(self, rows) -> None:
        super().__init__()
        self._rows = rows

    def on_mount(self) -> None:
        from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
            ConsoleSessionSwitcherModal,
        )

        self.push_screen(ConsoleSessionSwitcherModal(rows=self._rows))


@pytest.mark.asyncio
async def test_switcher_row_shows_colored_icon_left_of_title() -> None:
    """task-31208: the Ctrl+K switcher renders the appearance icon left of
    the title, and rows without one are unchanged."""
    from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
        _switcher_icon_prefix,
    )
    from tldw_chatbook.Workspaces.conversation_browser_state import (
        ConsoleConversationBrowserInputRow,
    )

    def _switcher_row(key, title, *, icon="", color=""):
        return ConsoleConversationBrowserInputRow(
            row_key=key,
            conversation_id=key,
            native_session_id=None,
            title=title,
            scope_type="global",
            workspace_id=None,
            workspace_label="",
            icon=icon,
            color=color,
        )

    rows = (
        _switcher_row("conv-icon", "Lab chat", icon="🧪", color="#f87171"),
        _switcher_row("conv-plain", "Plain chat"),
    )
    harness = SwitcherHarness(rows)
    async with harness.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        modal = harness.screen_stack[-1]
        labels = {
            button.id: str(button.label)
            for button in modal.query(".console-switcher-result")
        }
        ordered = list(labels.values())
        # dev's label shape is "<caret> <state column> <title>\n  <metadata>";
        # the icon sits between the state column and the title.
        lab = next(label for label in ordered if "🧪" in label)
        plain = next(label for label in ordered if "Plain chat" in label)
        assert "Lab chat" in lab
        assert lab.index("🧪") < lab.index("Lab chat")
        # The plain row carries no icon glyph at all.
        assert "🧪" not in plain

    # Pure prefix helper: unset rows contribute nothing.
    from tldw_chatbook.Chat.console_switcher_state import ConsoleSwitcherEntry

    plain_entry = ConsoleSwitcherEntry(
        row_key="k",
        title="t",
        subtitle="",
        native_session_id=None,
        conversation_id="k",
        scope_type="global",
        workspace_id=None,
        is_active=False,
    )
    assert _switcher_icon_prefix(plain_entry) == ""


@pytest.mark.asyncio
async def test_custom_hex_entry_applies_valid_color(_sandbox_emoji_sources) -> None:
    """task-31209: typing a canonical hex (with or without '#') selects it;
    junk input leaves the current selection untouched."""
    harness = PickerHarness(
        conversation_id="conv-1",
        conversation_title="T",
        color="#22d3ee",
    )
    async with harness.run_test(size=(80, 34)) as pilot:
        modal = harness.screen_stack[-1]
        assert isinstance(modal, ConsoleAppearancePickerModal)
        await pilot.pause()

        hex_input = modal.query_one("#console-appearance-picker-hex")
        # Carried-in color is prefilled.
        assert hex_input.value == "#22d3ee"

        hex_input.value = "c0ffee"
        await pilot.pause()
        assert modal._selected_color == "#c0ffee"

        # Junk must not clobber the selection mid-typing.
        hex_input.value = "not a color"
        await pilot.pause()
        assert modal._selected_color == "#c0ffee"

        modal.query_one("#console-appearance-picker-apply").press()
        await pilot.pause()
        assert harness.result == ConsoleConversationAppearance(
            icon=None, color="#c0ffee"
        )


@pytest.mark.asyncio
async def test_unstorable_emoji_sequence_is_rejected_at_selection(
    _sandbox_emoji_sources,
) -> None:
    """PR #2480 review (#5): a ZWJ-family catalog entry must be rejected at
    selection time -- Apply must never raise or record a stray recent."""
    harness = PickerHarness(conversation_id="conv-1", conversation_title="T")
    async with harness.run_test(size=(80, 34)) as pilot:
        modal = harness.screen_stack[-1]
        assert isinstance(modal, ConsoleAppearancePickerModal)
        await pilot.pause()
        notes: list[str] = []
        modal.app.notify = lambda message, **kwargs: notes.append(str(message))

        modal._select_icon("👨‍👩‍👧")
        await pilot.pause()
        assert modal._selected_icon is None
        assert notes, "the rejection must be surfaced, not silent"

        modal.query_one("#console-appearance-picker-apply").press()
        await pilot.pause()
        assert harness.result == ConsoleConversationAppearance()
        assert _sandbox_emoji_sources == []


@pytest.mark.asyncio
async def test_preview_renders_color_not_markup_tags(_sandbox_emoji_sources) -> None:
    """PR #2480 review (#6): the preview is markup=False, so the color
    sample must arrive as styled content -- literal '[#...]' tags are the
    bug."""
    harness = PickerHarness(
        conversation_id="conv-1",
        conversation_title="[b]Bold[/b] chat",
        icon="🧪",
        color="#f87171",
    )
    async with harness.run_test(size=(80, 34)) as pilot:
        modal = harness.screen_stack[-1]
        assert isinstance(modal, ConsoleAppearancePickerModal)
        await pilot.pause()
        preview_text = str(modal.query_one("#console-appearance-picker-preview").render())
        assert "#f87171" in preview_text
        # Only the color tag must not leak literally; a markup-bearing TITLE
        # legitimately shows its literal brackets in this markup=False view.
        assert "[#" not in preview_text


@pytest.mark.asyncio
async def test_swatch_press_updates_custom_color_field(
    _sandbox_emoji_sources,
) -> None:
    """PR #2480 review (#9): the custom-color input must show what Apply
    will persist after a palette/no-color swatch press."""
    harness = PickerHarness(conversation_id="conv-1", conversation_title="T")
    async with harness.run_test(size=(80, 34)) as pilot:
        modal = harness.screen_stack[-1]
        assert isinstance(modal, ConsoleAppearancePickerModal)
        await pilot.pause()
        hex_input = modal.query_one("#console-appearance-picker-hex")

        swatches = [
            button
            for button in modal.query(".console-appearance-swatch")
            if getattr(button, "hex_color", None) is not None
        ]
        swatches[0].press()
        await pilot.pause()
        assert hex_input.value == swatches[0].hex_color

        modal.query_one("#console-appearance-swatch-none").press()
        await pilot.pause()
        assert hex_input.value == ""
        assert modal._selected_color is None


def test_switcher_prefix_renders_placeholder_for_color_only() -> None:
    """PR #2480 review (#8): a color-only appearance keeps a visible marker
    in the switcher via the placeholder glyph, matching the rail."""
    from tldw_chatbook.Chat.console_glyphs import GLYPH_APPEARANCE_PLACEHOLDER
    from tldw_chatbook.Chat.console_switcher_state import ConsoleSwitcherEntry
    from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
        _switcher_icon_prefix,
    )

    color_only = ConsoleSwitcherEntry(
        row_key="k",
        title="t",
        subtitle="",
        native_session_id=None,
        conversation_id="k",
        scope_type="global",
        workspace_id=None,
        is_active=False,
        icon="",
        color="#22d3ee",
    )
    prefix = _switcher_icon_prefix(color_only)
    assert GLYPH_APPEARANCE_PLACEHOLDER in prefix
    assert prefix.startswith("[#22d3ee]")


@pytest.mark.asyncio
async def test_escape_cancels_with_none_without_crashing(_sandbox_emoji_sources) -> None:
    """Escape must run the one-shot safe cancel and dismiss with None.

    Regression: _perform_safe_cancel passed the SYNCHRONOUS
    cancel_filter_timer to run_cancel_effect_once, which awaits the effect --
    every Escape press raised TypeError before dismissing.
    """
    harness = PickerHarness(conversation_id="conv-1", conversation_title="Lab chat")
    async with harness.run_test(size=(80, 30)) as pilot:
        modal = harness.screen_stack[-1]
        assert isinstance(modal, ConsoleAppearancePickerModal)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert harness.result is None
