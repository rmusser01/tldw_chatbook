"""Console conversation appearance picker: icon + color (task-31207).

Cursor-style customization: one modal picks a conversation's emoji icon
(reusing the dormant ``Widgets/emoji_picker`` grid + recents store) and a
palette color. The modal is result-driven like the workspace rename modal —
``dismiss(ConsoleConversationAppearance)`` on Apply, an all-unset appearance
on Clear, ``dismiss(None)`` on cancel — so the pushing controller owns the
write path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, HorizontalScroll, Vertical
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_appearance import (
    CONSOLE_APPEARANCE_PALETTE,
    ConsoleConversationAppearance,
    is_valid_console_appearance_color,
    is_valid_console_appearance_icon,
    normalize_console_appearance_hex_input,
)
from tldw_chatbook.Widgets.emoji_picker import (
    EmojiButton,
    EmojiGrid,
    ProcessedEmoji,
    get_emoji_data,
    load_recent_emojis,
    save_recent_emoji,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

FILTER_INPUT_ID = "console-appearance-picker-filter"
HEX_INPUT_ID = "console-appearance-picker-hex"
GRID_ID = "console-appearance-picker-emoji"
PREVIEW_ID = "console-appearance-picker-preview"
COLORS_ID = "console-appearance-picker-colors"
SWATCH_CLASS = "console-appearance-swatch"
SWATCH_NONE_ID = "console-appearance-swatch-none"
APPLY_ID = "console-appearance-picker-apply"
CLEAR_ID = "console-appearance-picker-clear"
CANCEL_ID = "console-appearance-picker-cancel"
EMOJI_SELECTED_CLASS = "console-appearance-emoji-selected"
SWATCH_SELECTED_CLASS = "console-appearance-swatch-selected"

FILTER_DEBOUNCE_SECONDS = 0.2


def filter_appearance_emojis(
    emojis: Sequence[ProcessedEmoji], query: str
) -> list[ProcessedEmoji]:
    """Return emojis whose name or aliases contain the query.

    Args:
        emojis: Catalog entries to filter, in display order.
        query: Case-insensitive substring; an empty or blank query returns
            the input order unchanged (the grid's default population).

    Returns:
        The matching entries, in input order. Input is never mutated.
    """
    needle = str(query or "").strip().casefold()
    if not needle:
        return list(emojis)
    matches: list[ProcessedEmoji] = []
    for emoji in emojis:
        if needle in str(emoji.get("name", "")).casefold():
            matches.append(emoji)
            continue
        for alias in emoji.get("aliases", []) or []:
            if needle in str(alias).casefold():
                matches.append(emoji)
                break
    return matches


def _default_emoji_sequence() -> list[ProcessedEmoji]:
    """Recents first, then the full catalog, deduplicated."""
    all_emojis, _, _ = get_emoji_data()
    by_char = {str(emoji["char"]): emoji for emoji in all_emojis}
    ordered: list[ProcessedEmoji] = []
    for char in load_recent_emojis():
        emoji = by_char.pop(char, None)
        if emoji is not None:
            ordered.append(emoji)
    ordered.extend(by_char.values())
    return ordered


class ConsoleAppearancePickerModal(
    SafeModalDismissMixin, ModalScreen[ConsoleConversationAppearance | None]
):
    """Pick one conversation's icon glyph and palette color."""

    BUNDLED_CSS = """
    ConsoleAppearancePickerModal {
        align: center middle;
    }

    #console-appearance-picker-modal {
        width: 64;
        max-width: 100%;
        height: 29;
        max-height: 100%;
        border: tall $accent;
        background: $panel;
        padding: 1 2;
    }

    #console-appearance-picker-filter {
        width: 100%;
        height: 3;
    }

    #console-appearance-picker-preview {
        width: 100%;
        height: 1;
        color: $text-muted;
    }

    #console-appearance-picker-emoji {
        width: 100%;
        height: 1fr;
        min-height: 6;
        background: $surface-darken-1;
    }

    #console-appearance-picker-colors {
        /* task-31209: a scrollable strip -- the 22-swatch palette (plus
           "none") no longer fits one modal row, and shrinking swatches to
           two cells left no content width under a selection border. */
        width: 100%;
        height: 3;
        margin-top: 1;
    }

    #console-appearance-picker-hex {
        width: 1fr;
        height: 3;
        margin-top: 1;
    }

    .console-appearance-swatch {
        width: 3;
        min-width: 3;
        max-width: 3;
        height: 3;
        min-height: 3;
        margin: 0;
        padding: 0;
        border: none;
        text-align: center;
        content-align: center middle;
    }

    .console-appearance-swatch-selected {
        /* Background highlight, not a border: the 2-cell swatch has no room
           for a 2-cell border (zero content width crashed cell chopping). */
        background: $accent 25%;
        text-style: bold;
    }

    #console-appearance-swatch-none {
        width: 5;
        min-width: 5;
        max-width: 5;
    }

    .console-appearance-emoji-selected {
        background: $accent 25%;
        text-style: bold;
    }

    #console-appearance-picker-actions {
        width: 100%;
        height: 3;
        margin-top: 1;
    }

    #console-appearance-picker-actions Button.console-appearance-action {
        width: 1fr;
        height: 3;
        border: none;
        margin: 0 1 0 0;
    }

    #console-appearance-picker-apply {
        background: $primary;
        color: $text;
    }

    #console-appearance-picker-clear,
    #console-appearance-picker-cancel {
        background: $surface;
        color: $text;
    }
    """

    BINDINGS: ClassVar = [("escape", "request_safe_cancel", "Cancel")]
    SAFE_MODAL_CONTENT = "#console-appearance-picker-modal"

    def __init__(
        self,
        *,
        conversation_id: str,
        conversation_title: str = "",
        icon: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.conversation_id = str(conversation_id or "")
        self._conversation_title = str(conversation_title or "")
        # Defensive re-validation: the row state is sanitized already, but a
        # stale/foreign caller must not smuggle markup into the preview.
        self._selected_icon: str | None = (
            icon if icon and is_valid_console_appearance_icon(icon) else None
        )
        self._selected_color: str | None = (
            color if color and is_valid_console_appearance_color(color) else None
        )
        self._all_emojis = _default_emoji_sequence()
        self._filter_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="console-appearance-picker-modal"):
            yield Static("Customize conversation", classes="console-modal-header")
            yield Static("", id=PREVIEW_ID, markup=False)
            yield Input(
                placeholder="Search icons by name…",
                id=FILTER_INPUT_ID,
            )
            yield EmojiGrid(
                self._all_emojis, id=GRID_ID, can_focus=False
            )
            with HorizontalScroll(id=COLORS_ID):
                yield Button(
                    "none",
                    id=SWATCH_NONE_ID,
                    classes=SWATCH_CLASS,
                    compact=True,
                )
                for label, hex_color in CONSOLE_APPEARANCE_PALETTE:
                    swatch = Button(
                        _swatch_label(hex_color),
                        id=f"console-appearance-swatch-"
                        f"{label.lower().replace(' ', '-')}",
                        classes=SWATCH_CLASS,
                        compact=True,
                        tooltip=label,
                    )
                    swatch.hex_color = hex_color  # type: ignore[attr-defined]
                    yield swatch
            yield Input(
                placeholder="Or type a custom color, e.g. #c0ffee or c0ffee…",
                id=HEX_INPUT_ID,
            )
            with Horizontal(id="console-appearance-picker-actions"):
                yield Button(
                    "Apply",
                    id=APPLY_ID,
                    compact=True,
                    classes="console-appearance-action",
                )
                yield Button(
                    "Clear",
                    id=CLEAR_ID,
                    compact=True,
                    classes="console-appearance-action",
                )
                yield Button(
                    "Cancel",
                    id=CANCEL_ID,
                    compact=True,
                    classes="console-appearance-action",
                )

    async def on_mount(self) -> None:  # type: ignore[override]
        # EmojiGrid populates itself from its constructor list on mount;
        # highlight the carried-in selection on top of it.
        self._sync_emoji_highlight()
        self._sync_preview()
        self._sync_swatch_highlight()
        if self._selected_color is not None:
            self.query_one(f"#{HEX_INPUT_ID}", Input).value = self._selected_color
        self._focus_filter()

    def on_unmount(self) -> None:
        self._cancel_filter_timer()

    @on(Input.Changed, f"#{FILTER_INPUT_ID}")
    def _filter_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._cancel_filter_timer()
        query = event.value
        self._filter_timer = self.set_timer(
            FILTER_DEBOUNCE_SECONDS,
            lambda: self._settle_filter(query),
        )

    @on(Input.Submitted, f"#{FILTER_INPUT_ID}")
    def _filter_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._cancel_filter_timer()
        self._apply_filter(event.value)
        # Enter commits the first match, the reaction picker's muscle memory.
        grid = self._grid()
        if grid is None:
            return
        try:
            first = grid.query(EmojiButton).first()
        except NoMatches:
            return
        char = str(
            getattr(first, "emoji_data", {}).get("char", "") or ""
        )
        if char:
            self._select_icon(char)

    @on(Button.Pressed, ".emoji_button")
    def _emoji_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        char = str(getattr(event.button, "emoji_data", {}).get("char", "") or "")
        if char:
            self._select_icon(char)

    @on(Input.Changed, f"#{HEX_INPUT_ID}")
    def _hex_changed(self, event: Input.Changed) -> None:
        """Apply a valid custom hex color live; ignore anything else.

        task-31209: a leading '#' is optional ("c0ffee" == "#c0ffee"); an
        invalid or partial value never disturbs the current selection — the
        "none" swatch is the explicit way to clear, so a mid-typing state
        must not silently drop a chosen color.
        """
        event.stop()
        normalized = normalize_console_appearance_hex_input(event.value)
        if normalized is None:
            return
        self._selected_color = normalized
        self._sync_preview()
        self._sync_swatch_highlight()

    @on(Button.Pressed, f".{SWATCH_CLASS}")
    def _swatch_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button = event.button
        if button.id == SWATCH_NONE_ID:
            self._selected_color = None
        else:
            hex_color = getattr(button, "hex_color", None)
            self._selected_color = (
                hex_color if hex_color and is_valid_console_appearance_color(hex_color) else None
            )
        self._sync_preview()
        self._sync_swatch_highlight()
        # PR #2480 review (#9): keep the custom-color field in step with the
        # swatch selection so what it displays is what Apply persists. (The
        # programmatic assignment re-fires `_hex_changed`, which is a no-op
        # here: the normalized value is exactly `_selected_color`.)
        try:
            hex_input = self.query_one(f"#{HEX_INPUT_ID}", Input)
        except (NoMatches, QueryError):
            hex_input = None
        if hex_input is not None:
            hex_input.value = self._selected_color or ""
        self._focus_filter()

    @on(Button.Pressed, f"#{APPLY_ID}")
    def _apply_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        icon = self._selected_icon
        # PR #2480 review (#5): build the validating result BEFORE recording
        # the recent-emoji side effect, so a rejected icon cannot leave a
        # stray recents entry (selection-time rejection makes this belt and
        # braces for foreign callers of `_select_icon`).
        result = ConsoleConversationAppearance(
            icon=icon, color=self._selected_color
        )
        if icon is not None:
            save_recent_emoji(icon)
        self.dismiss(result)

    @on(Button.Pressed, f"#{CLEAR_ID}")
    def _clear_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(ConsoleConversationAppearance())

    @on(Button.Pressed, f"#{CANCEL_ID}")
    async def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source

        async def cancel_filter_timer() -> None:
            self._cancel_filter_timer()

        await self.run_cancel_effect_once(cancel_filter_timer)
        self.dismiss_safe_once(None)

    def _grid(self) -> EmojiGrid | None:
        try:
            return self.query_one(f"#{GRID_ID}", EmojiGrid)
        except (NoMatches, QueryError):
            return None

    def _apply_filter(self, query: str) -> None:
        grid = self._grid()
        if grid is None:
            return
        matches = filter_appearance_emojis(self._all_emojis, query)
        grid.populate_grid(matches)
        self._sync_emoji_highlight()
        self._focus_filter()

    def _settle_filter(self, query: str) -> None:
        self._filter_timer = None
        if not self.is_mounted:
            return
        self._apply_filter(query)

    def _cancel_filter_timer(self) -> None:
        if self._filter_timer is not None:
            self._filter_timer.stop()
            self._filter_timer = None

    def _select_icon(self, char: str) -> None:
        """Select one icon glyph, rejecting unstorable sequences (#5)."""
        if not is_valid_console_appearance_icon(char):
            # PR #2480 review (#5): the shared catalog includes ZWJ families
            # and wide pairs the appearance contract cannot store; reject at
            # selection so Apply never hits the dataclass's ValueError.
            self.app.notify(
                "That icon uses a sequence this app cannot store; pick a single emoji.",
                severity="warning",
            )
            return
        self._selected_icon = char
        self._sync_emoji_highlight()
        self._sync_preview()
        self._focus_filter()

    def _sync_emoji_highlight(self) -> None:
        grid = self._grid()
        if grid is None:
            return
        for button in grid.query(EmojiButton):
            button.set_class(
                str(button.emoji_data.get("char", "")) == self._selected_icon,
                EMOJI_SELECTED_CLASS,
            )

    def _sync_swatch_highlight(self) -> None:
        for button in self.query(f".{SWATCH_CLASS}"):
            hex_color = getattr(button, "hex_color", None)
            is_selected = (
                self._selected_color is not None
                and hex_color == self._selected_color
            )
            button.set_class(is_selected, SWATCH_SELECTED_CLASS)

    def _sync_preview(self) -> None:
        try:
            preview = self.query_one(f"#{PREVIEW_ID}", Static)
        except (NoMatches, QueryError):
            return
        # PR #2480 review (#6): the preview Static is markup=False, so a
        # markup STRING would display literal tags. Build a styled Content
        # instead -- styles render, and the title stays escaped text.
        from textual.content import Content

        icon = self._selected_icon or "·"
        color = self._selected_color
        title = self._conversation_title or "this conversation"
        if color:
            preview.update(
                Content.assemble(
                    (f"{icon} ", "dim"),
                    (f"{title}  ", None),
                    (color, color),
                )
            )
        else:
            preview.update(Content.assemble((f"{icon} ", "dim"), (f"{title}  no color", None)))

    def _focus_filter(self) -> None:
        try:
            self.query_one(f"#{FILTER_INPUT_ID}", Input).focus()
        except (NoMatches, QueryError):
            return


def _swatch_label(hex_color: str) -> Any:
    """Return a colored square label for one swatch button."""
    from textual.content import Content

    return Content.from_markup(f"[{hex_color}]■[/]")
