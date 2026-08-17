"""Floating slash-command suggestion popup for the native Console composer.

Screen-owned overlay: the owning screen feeds it suggestions, routes
Up/Down/Enter/Tab/Escape to it while open, and it never takes focus. It
positions itself (``overlay: screen`` + ``styles.offset``) so its bottom
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
        # Last offset.y applied to the placement; region == flow_base + this,
        # so the two together always re-derive the flow base (see reposition).
        self._anchor_offset_y = 0
        # Hidden by default in code (not just TCSS) so the widget is correct
        # even where the bundled stylesheet is not loaded (bare test apps).
        self.display = False
        # Why not the stylesheet's `position: absolute`: Textual 8.x's
        # vertical layout resolves EVERY child's height against the
        # container's fr pool before the placement loop skips absolute
        # widgets, so an absolute popup still costs the workspace grid its
        # full popup-height in rows whenever autocomplete opens -- the
        # transcript visibly jumps up, a dead band opens under the status
        # row, and on short terminals the anchor clamps to the shell top.
        # `overlay: screen` is offset-positionable AND exempt from that
        # height deduction; the inline `position: relative` neutralizes the
        # stylesheet's `position: absolute` (which would keep the deduction
        # alive alongside the overlay rule).
        self.styles.position = "relative"
        self.styles.overlay = "screen"

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

        Idempotent when already open with the SAME suggestions: the rows and
        highlight are left alone (only the anchor is refreshed). Without
        this, the deferred `DraftChanged` sync arriving after a same-read
        `/`+Down pair would rebuild identical rows and yank the highlight
        the Down just moved back to 0 (task-3790). `CommandSuggestion` is a
        frozen dataclass, so `==` here is value equality.
        """
        if self.is_open and list(suggestions) == self._suggestions:
            self.reposition()
            return
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

        ``composer.region`` is Screen-relative and the flow base is derived
        from this widget's own placement (see the comment at the offset
        math), so the anchor holds regardless of which container the popup
        is mounted in.
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
        # DS-09 (TASK-2154.15): anchor the bottom edge above the topmost
        # visible strip in the composer's cluster (staged evidence, then the
        # prompt-queue shelf) rather than the composer itself, so the open
        # popup covers transcript rows (the conventional autocomplete
        # trade-off) instead of wiping mid-composition state. The status
        # chips now sit below the composer, out of the popup's reach -- the
        # anchor must never chase them down over the input row.
        bottom_y = anchor.y
        for strip_id in ("#console-staged-evidence-strip", "#console-prompt-queue"):
            try:
                strip = self.screen.query_one(strip_id)
            except NoMatches:
                continue
            if strip.display and strip.region.height > 0:
                bottom_y = min(bottom_y, strip.region.y)
        origin = self.parent.content_region
        x = anchor.x - origin.x
        y = max(bottom_y - self._desired_height, 0)
        # `overlay: screen` paints at (flow slot + offset), and the flow
        # slot sits wherever the parent's in-flow stack ends -- NOT the
        # parent's content origin the way `position: absolute` did. The
        # slot position is not directly readable (a fixed-height parent's
        # content_size reports its box, not its children's extent), so
        # re-derive it from the current placement: region.y is always
        # flow_base + the last offset.y this method applied. That also
        # tracks any reflow that moved the slot between keystrokes.
        if self.region.height > 0:
            flow_base_y = self.region.y - self._anchor_offset_y
            # May be negative: the popup paints ABOVE its flow slot at the
            # bottom of the shell. `y` already carries the screen-top clamp.
            offset_y = y - flow_base_y
        else:
            # Not laid out yet (first-ever show, display just flipping on):
            # anchor at the origin for this frame; the after-refresh
            # re-anchor in show_suggestions lands with a valid region.
            offset_y = max(y, 0)
        self._anchor_offset_y = offset_y
        self.styles.offset = (max(x, 0), offset_y)
        self.styles.width = max(anchor.width, MIN_WIDTH)
