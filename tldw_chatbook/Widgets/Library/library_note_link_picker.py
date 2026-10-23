"""The Library Notes ``[[`` link picker (TASK-34000.11).

A screen-owned overlay modelled on the Console's ``ConsoleCommandPopup``: the
controller feeds it rows, routes Up/Down/Enter/Tab/Escape to it while it is
open, and it never takes focus -- the body ``TextArea`` (insert mode) or the
Preview region (follow mode) keeps it, so typing keeps narrowing and Escape
leaves the typed text exactly as it was.

``overlay: screen`` + ``styles.offset`` place it under the body's caret row
(insert mode) or at the top of the Preview region (follow mode), clamped
inside the editor pane. It is yielded as its own root of the note work pane
rather than inside ``#library-note-editor-region``: that region is hidden
while Preview shows, and follow mode has to paint over Preview.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from rich.text import Text
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from tldw_chatbook.Utils.input_validation import escape_markup

MAX_VISIBLE_ROWS = 8
MIN_WIDTH = 30
MAX_WIDTH = 70


@dataclass(frozen=True)
class LinkCandidate:
    """One pickable note: its identity and the title the row shows."""

    note_id: str
    title: str


class _CandidateOption(Option):
    """OptionList row carrying its ``LinkCandidate``.

    The title goes through ``escape_markup`` and is rendered from a plain
    ``Text``: a title such as ``Bold [b] title`` must paint literally
    (lessons-textual, "Square brackets in widget text are console markup").
    """

    def __init__(self, candidate: LinkCandidate) -> None:
        # A plain ``Text`` is never parsed as markup. One row per title is
        # the SHEET's job (`.library-note-link-picker OptionList { text-wrap:
        # nowrap; text-overflow: ellipsis }`): ``OptionList`` re-wraps its
        # prompts itself, so a ``no_wrap`` Text did nothing -- live at
        # 160x45 and 80x24 the seed's very long title still took two rows
        # and pushed the eighth row out of view until the rule landed.
        super().__init__(Text(candidate.title), id=None)
        self.candidate = candidate


class LibraryNoteLinkPicker(Widget):
    """Non-focusable overlay listing note titles for a ``[[`` completion."""

    can_focus = False

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "library-note-link-picker")
        super().__init__(**kwargs)
        self._candidates: list[LinkCandidate] = []
        self._empty_query: str | None = None
        self._desired_height = 0
        self._desired_width = MIN_WIDTH
        # region == flow_base + the last offset applied; the two together
        # re-derive the flow base on every reposition (the Console rule).
        self._anchor_offset: tuple[int, int] = (0, 0)
        self._anchor_cell: tuple[int, int] = (0, 0)
        self._anchor_bounds: tuple[int, int, int, int] | None = None
        self.mode: Literal["insert", "follow"] = "insert"
        # Hidden in code, not only in TCSS, so a bare test app is right too.
        self.display = False
        # `position: relative` + `overlay: screen`: offset-positionable and
        # exempt from the vertical layout's fr-pool height deduction (see
        # ConsoleCommandPopup for the measured reason).
        self.styles.position = "relative"
        self.styles.overlay = "screen"

    def compose(self) -> ComposeResult:
        option_list = OptionList(id="library-note-link-picker-options")
        option_list.can_focus = False
        yield option_list

    @property
    def is_open(self) -> bool:
        """Return whether the picker is currently displayed."""
        return bool(self.display)

    @property
    def candidates(self) -> tuple[LinkCandidate, ...]:
        return tuple(self._candidates)

    def show_candidates(
        self,
        rows: list[LinkCandidate],
        *,
        mode: Literal["insert", "follow"] = "insert",
        empty_query: str | None = None,
    ) -> None:
        """Replace rows, reset the highlight, and show.

        Idempotent when already open in the same mode with the SAME rows:
        the highlight the user moved is left alone (task-3790's rule).

        Args:
            rows: The candidates, in the order to show them.
            mode: ``"insert"`` (the ``[[`` completion) or ``"follow"`` (the
                Preview's Enter picker).
            empty_query: When ``rows`` is empty, the query that found
                nothing -- rendered as one disabled row. ``None`` keeps the
                picker hidden on no rows (the loader has not answered yet).
        """
        rows = list(rows)
        if self.is_open and self.mode == mode and rows == self._candidates and (
            rows or self._empty_query == empty_query
        ):
            return
        if not rows and empty_query is None:
            self.hide()
            return
        self.mode = mode
        self._candidates = rows
        self._empty_query = empty_query if not rows else None
        self._desired_height = max(1, min(len(rows), MAX_VISIBLE_ROWS))
        longest = max((len(row.title) for row in rows), default=0)
        if not rows:
            longest = len(self._empty_row_text())
        self._desired_width = max(MIN_WIDTH, min(MAX_WIDTH, longest + 4))
        # The sheet's `round` border takes one row above and one below the
        # rows, so the box is rows + 2 (the Console popup has no border and
        # sets rows alone).
        # ds-runtime: the row count and the measured anchor decide the geometry.
        self.set_styles(height=self._desired_height + 2, width=self._desired_width)
        option_list = self.query_one(OptionList)
        option_list.clear_options()
        if rows:
            option_list.add_options([_CandidateOption(row) for row in rows])
            option_list.highlighted = 0
        else:
            option_list.add_option(Option(Text(self._empty_row_text()), disabled=True))
            option_list.highlighted = None
        self.display = True
        self._apply_anchor()
        self.call_after_refresh(self._apply_anchor)

    def _empty_row_text(self) -> str:
        return f"No note titled “{self._empty_query or ''}”"

    def hide(self) -> None:
        """Hide the picker and drop its rows."""
        self.display = False
        self._candidates = []
        self._empty_query = None

    def move_highlight(self, delta: int) -> None:
        """Move the highlight by ``delta`` rows, wrapping at both ends."""
        count = len(self._candidates)
        if count == 0:
            return
        option_list = self.query_one(OptionList)
        current = option_list.highlighted or 0
        option_list.highlighted = (current + delta) % count

    def accept_selected(self) -> LinkCandidate | None:
        """Return the highlighted candidate, or ``None`` when there is none."""
        if not self._candidates:
            return None
        highlighted = self.query_one(OptionList).highlighted
        if highlighted is None or not (0 <= highlighted < len(self._candidates)):
            return None
        return self._candidates[highlighted]

    def reposition(self, anchor: Widget, *, cell: tuple[int, int] | None = None) -> None:
        """Place the picker's top-left under ``cell``, clamped inside ``anchor``.

        Args:
            anchor: The widget whose screen region bounds the picker (the
                body's editor region in insert mode, the Preview region in
                follow mode).
            cell: The screen cell to sit under -- the body's caret in insert
                mode. ``None`` anchors at the region's own top-left.
        """
        region = anchor.region
        self._anchor_bounds = (region.x, region.y, region.right, region.bottom)
        if cell is None:
            self._anchor_cell = (region.x + 1, region.y)
        else:
            self._anchor_cell = cell
        self._apply_anchor()

    def _apply_anchor(self) -> None:
        if self.parent is None or not self.is_open or self._anchor_bounds is None:
            return
        try:
            left, top, right, bottom = self._anchor_bounds
            cell_x, cell_y = self._anchor_cell
            width = self._desired_width
            # Border rows + the rows themselves.
            height = self._desired_height + 2
            x = max(left, min(cell_x, right - width))
            x = max(x, 0)
            y = cell_y + 1
            if y + height > bottom:
                above = cell_y - height
                y = above if above >= top else max(top, bottom - height)
            y = max(y, 0)
            if self.region.height > 0:
                flow_x = self.region.x - self._anchor_offset[0]
                flow_y = self.region.y - self._anchor_offset[1]
                offset = (x - flow_x, y - flow_y)
            else:
                offset = (x, y)
            self._anchor_offset = offset
            # ds-runtime: the caret cell and the pane bounds decide the offset.
            self.styles.offset = offset
        except NoMatches:
            return
        except Exception as exception:  # noqa: BLE001 - never crash the screen
            logger.warning(f"LibraryNoteLinkPicker.reposition failed: {exception!r}")


__all__ = ["LibraryNoteLinkPicker", "LinkCandidate", "MAX_VISIBLE_ROWS", "escape_markup"]
