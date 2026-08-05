"""Floating slash-command suggestion popup for the native Console composer.

Screen-owned overlay: the owning screen feeds it suggestions, routes
Up/Down/Enter/Tab/Escape to it while open, and it never takes focus. It
positions itself (``position: absolute`` + ``styles.offset``) so its bottom
edge sits just above the composer — the same anchored-overlay technique as
``Widgets/tooltip.py``.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from ...Chat.console_command_suggestions import CommandSuggestion

MAX_VISIBLE_ROWS = 8
MIN_WIDTH = 30


class SuggestionOption(Option):
    """OptionList row carrying its originating `CommandSuggestion`."""

    def __init__(self, suggestion: CommandSuggestion) -> None:
        prompt = Text(suggestion.label, style="bold")
        if suggestion.description:
            prompt.append("  ")
            prompt.append(suggestion.description, style="dim")
        super().__init__(prompt)
        self.suggestion = suggestion


class ConsoleCommandPopup(Widget):
    """Non-focusable overlay listing slash-command completions."""

    can_focus = False

    def __init__(self, **kwargs: Any) -> None:
        """Create the popup, hidden until `show_suggestions` is called.

        Args:
            **kwargs: Forwarded to ``Widget``. The popup forces its own
                ``id`` ("console-command-popup") when none is given.
        """
        kwargs.setdefault("id", "console-command-popup")
        super().__init__(**kwargs)
        self._suggestions: list[CommandSuggestion] = []
        self._desired_height = 0
        # Hidden by default in code (not just TCSS) so the widget is correct
        # even where the bundled stylesheet is not loaded (bare test apps).
        self.display = False

    def compose(self) -> ComposeResult:
        option_list = OptionList(id="console-command-popup-options")
        option_list.can_focus = False
        yield option_list

    @property
    def is_open(self) -> bool:
        """Return whether the popup is currently displayed."""
        return self.display

    def show_suggestions(self, suggestions: list[CommandSuggestion]) -> None:
        """Replace rows, reset the highlight, reposition, and show.

        Args:
            suggestions: Completions to list, in display order. An empty
                list still opens the popup at height 0 (caller filters).
        """
        self._suggestions = list(suggestions)
        option_list = self.query_one(OptionList)
        option_list.clear_options()
        option_list.add_options(
            [SuggestionOption(suggestion) for suggestion in self._suggestions]
        )
        self._desired_height = min(len(self._suggestions), MAX_VISIBLE_ROWS)
        self.styles.height = self._desired_height
        option_list.highlighted = 0
        self.reposition()
        self.display = True

    def hide(self) -> None:
        """Hide the popup and drop its rows."""
        self.display = False
        self._suggestions = []

    def move_highlight(self, delta: int) -> None:
        """Move the highlight by ``delta`` rows, wrapping at both ends.

        Args:
            delta: Rows to move the highlight by; negative moves up.
        """
        count = len(self._suggestions)
        if count == 0:
            return
        option_list = self.query_one(OptionList)
        current = option_list.highlighted or 0
        option_list.highlighted = (current + delta) % count

    def accept_selected(self) -> CommandSuggestion | None:
        """Return the highlighted suggestion, or ``None`` when unavailable."""
        highlighted = self.query_one(OptionList).highlighted
        if highlighted is None or not (0 <= highlighted < len(self._suggestions)):
            return None
        return self._suggestions[highlighted]

    def reposition(self) -> None:
        """Anchor the popup's bottom edge just above the composer.

        ``composer.region`` is Screen-relative and ``content_region`` is
        absolute, so the offset math works regardless of which container the
        popup is mounted in.
        """
        if self.parent is None:
            return
        try:
            composer = self.screen.query_one("#console-native-composer")
        except Exception:
            return
        anchor = composer.region
        origin = self.parent.content_region
        x = anchor.x - origin.x
        y = anchor.y - origin.y - self._desired_height
        self.styles.offset = (max(x, 0), max(y, 0))
        self.styles.width = max(anchor.width, MIN_WIDTH)
