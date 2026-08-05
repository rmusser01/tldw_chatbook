"""Floating slash-command suggestion popup for the native Console composer.

Screen-owned overlay: the owning screen feeds it suggestions, routes
Up/Down/Enter/Tab/Escape to it while open, and it never takes focus. It
positions itself (``position: absolute`` + ``styles.offset``) so its bottom
edge sits just above the composer — the same anchored-overlay technique as
``Widgets/tooltip.py``.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from rich.text import Text
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from ...Chat.console_command_suggestions import CommandSuggestion

MAX_VISIBLE_ROWS = 8
MIN_WIDTH = 30


class _SuggestionOption(Option):
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
        """Replace rows, reset the highlight, reposition, and show."""
        self._suggestions = list(suggestions)
        self._desired_height = min(len(self._suggestions), MAX_VISIBLE_ROWS)
        self.styles.height = self._desired_height
        # Set the final width BEFORE rebuilding the OptionList: option row
        # heights are computed (and cached) against the width at add time, so
        # adding options while the popup is still narrow wraps rows, inflates
        # the virtual size, and strands the list in a scrollbar-stuck state
        # that paints nothing.
        self.reposition()
        option_list = self.query_one(OptionList)
        option_list.clear_options()
        option_list.add_options(
            [_SuggestionOption(suggestion) for suggestion in self._suggestions]
        )
        option_list.highlighted = 0
        self.display = True
        # The shell's post-keystroke machinery (console-sync worker, guidance
        # dismissal) can reflow the composer a beat AFTER even the
        # post-refresh anchor runs (verified: both immediate and
        # call_after_refresh repositions can observe the pre-shift composer).
        # The trailing timer re-anchor covers that settle; every subsequent
        # keystroke re-anchors anyway. Idempotent and cheap.
        self.call_after_refresh(self.reposition)
        self.set_timer(0.1, self.reposition)

    def hide(self) -> None:
        """Hide the popup and drop its rows."""
        self.display = False
        self._suggestions = []

    def move_highlight(self, delta: int) -> None:
        """Move the highlight by ``delta`` rows, wrapping at both ends."""
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
        except NoMatches:
            # Composer not mounted (lifecycle transition) — nothing to anchor to.
            return
        except Exception as exception:
            # Unexpected failure: keep the popup from crashing the screen,
            # but leave a diagnostic trail instead of swallowing it silently.
            logger.warning(
                f"ConsoleCommandPopup.reposition failed: {exception!r}"
            )
            return
        anchor = composer.region
        origin = self.parent.content_region
        x = anchor.x - origin.x
        y = anchor.y - origin.y - self._desired_height
        self.styles.offset = (max(x, 0), max(y, 0))
        self.styles.width = max(anchor.width, MIN_WIDTH)
