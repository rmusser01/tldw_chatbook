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
from typing import Any, Callable, Literal

from loguru import logger
from rich.text import Text
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import OptionList
from textual.widgets.option_list import Option

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

    The title is rendered from a plain rich ``Text``, which is never parsed
    as console markup: a title such as ``Bold [b] title`` paints literally
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
        self._empty_text: str | None = None
        #: Fix round 1 (I-2): the load generation the rows on show answered.
        #: An accept for any OTHER generation is stale -- the controller
        #: holds it (``accept_pending_for``) until the current answer lands.
        self.generation = 0
        self.accept_pending_for: int | None = None
        #: Fix round 2 (R2-1): a highlight move made while the rows on show
        #: were STALE. A held accept is cancelled by it -- the user never saw
        #: the row they would get highlighted in the current set.
        self.moved_on_stale = False
        #: The widget whose region bounds the box and a callable for the
        #: screen cell to sit under, re-read on every anchor pass (R2-5: a
        #: resize changes both).
        self._anchor_widget: Widget | None = None
        self._anchor_cell_provider: Callable[[], tuple[int, int] | None] | None = None
        #: One anchor pass per refresh. The offset math re-derives the flow
        #: slot from ``region - last offset``, which is only right once the
        #: region reflects that offset -- two passes in one frame (the
        #: same-rows re-show plus the controller's reposition) drifted the
        #: box off screen (fix round 2: offset (-72, -57) measured).
        self._anchor_pass_pending = False
        #: Fix round 4 (R4-3): what to call when a re-anchor pass finds the
        #: caret gone -- the controller's close path, so the generation
        #: bumps and the body's routing flag resets.
        self._on_caret_lost: Callable[[], None] | None = None
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

    @property
    def has_rows(self) -> bool:
        """Whether a real (acceptable) row is on show, not a hint row."""
        return bool(self._candidates)

    def show_candidates(
        self,
        rows: list[LinkCandidate],
        *,
        mode: Literal["insert", "follow"] = "insert",
        empty_text: str | None = None,
        generation: int = 0,
    ) -> None:
        """Replace rows, reset the highlight, and show.

        Idempotent when already open in the same mode with the SAME rows:
        the highlight the user moved is left alone (task-3790's rule); the
        generation stamp is refreshed either way.

        Args:
            rows: The candidates, in the order to show them.
            mode: ``"insert"`` (the ``[[`` completion) or ``"follow"`` (the
                Preview's Enter picker).
            empty_text: When ``rows`` is empty, the one disabled row to show
                ("No note titled …", "Keep typing…"). ``None`` keeps the
                picker hidden on no rows.
            generation: The load generation these rows answer (I-2).
        """
        rows = list(rows)
        self.generation = generation
        self.moved_on_stale = False
        if self.is_open and self.mode == mode and rows == self._candidates and (
            rows or self._empty_text == empty_text
        ):
            # Same rows: keep the highlight, but re-anchor -- the space beside
            # the caret may have changed (R2-5).
            self._apply_anchor()
            return
        if not rows and empty_text is None:
            self.hide()
            return
        self.mode = mode
        self._candidates = rows
        self._empty_text = empty_text if not rows else None
        self._desired_height = max(1, min(len(rows), MAX_VISIBLE_ROWS))
        longest = max((len(row.title) for row in rows), default=0)
        if not rows:
            longest = len(empty_text or "")
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
            option_list.add_option(Option(Text(self._empty_text or ""), disabled=True))
            option_list.highlighted = None
        self.display = True
        self._apply_anchor()
        self.call_after_refresh(self._apply_anchor)

    def hide(self) -> None:
        """Hide the picker and drop its rows."""
        self.display = False
        self._candidates = []
        self._empty_text = None
        self.accept_pending_for = None
        self.moved_on_stale = False

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

    def reposition(
        self,
        anchor: Widget,
        *,
        cell: tuple[int, int] | None = None,
        cell_provider: Callable[[], tuple[int, int] | None] | None = None,
        on_caret_lost: Callable[[], None] | None = None,
    ) -> None:
        """Place the picker's top-left under the caret, clamped inside ``anchor``.

        Args:
            anchor: The widget whose screen region bounds the picker (the
                body's editor region in insert mode, the Preview region in
                follow mode). Its region is re-read on every anchor pass.
            cell: The screen cell to sit under -- the body's caret in insert
                mode. ``None`` anchors at the region's own top-left.
            cell_provider: Re-reads that cell on later passes (a resize moves
                the caret on screen); wins over ``cell`` when it answers.
            on_caret_lost: Called when a later pass finds no caret cell (R4-3:
                the controller's close path). Without it the box hides itself.
        """
        self._anchor_widget = anchor
        self._anchor_cell_provider = cell_provider
        self._on_caret_lost = on_caret_lost
        region = anchor.region
        self._anchor_bounds = (region.x, region.y, region.right, region.bottom)
        if cell is None:
            self._anchor_cell = (region.x + 1, region.y)
        else:
            self._anchor_cell = cell
        self._apply_anchor()

    def reanchor(self) -> None:
        """Re-run the anchor pass against the anchor widget's CURRENT region
        and caret cell (fix round 2, R2-5: the note pane calls this on a
        resize, so a box cut to fit beside the caret grows back when the
        terminal does)."""
        if self._anchor_widget is None or not self.is_open:
            return
        try:
            region = self._anchor_widget.region
        except Exception:  # noqa: BLE001 - a detached anchor has no region
            return
        if region.width <= 0 or region.height <= 0:
            return
        self._anchor_bounds = (region.x, region.y, region.right, region.bottom)
        if self._anchor_cell_provider is not None:
            cell = self._anchor_cell_provider()
            if cell is None:
                # Fix round 3 (R3-2): the caret is not on screen -- a box
                # the user cannot see must never take Enter. Close -- through
                # the controller when it gave a route (R4-3: the generation
                # bumps and the body stops routing keys here), else alone.
                if self._on_caret_lost is not None:
                    self._on_caret_lost()
                else:
                    self.hide()
                return
            self._anchor_cell = cell
        else:
            # Fix round 3 (R3-4): no caret to follow (follow mode) -- the
            # anchor is the region's own top-left, recomputed every time
            # (it used to keep the cell from the previous size).
            self._anchor_cell = (region.x + 1, region.y)
        self._apply_anchor()

    def on_resize(self) -> None:
        self.reanchor()

    def _apply_anchor(self) -> None:
        """Request an anchor pass: synchronous while the box has no laid-out
        region yet (the first frame's absolute guess), else once after the
        next refresh, however many requests arrive in between."""
        if self.parent is None or not self.is_open or self._anchor_bounds is None:
            return
        if self.region.height <= 0:
            self._anchor_pass()
            return
        if self._anchor_pass_pending:
            return
        self._anchor_pass_pending = True
        self.call_after_refresh(self._anchor_pass)

    def _anchor_pass(self) -> None:
        self._anchor_pass_pending = False
        if self.parent is None or not self.is_open or self._anchor_bounds is None:
            return
        try:
            if self._anchor_widget is not None:
                region = self._anchor_widget.region
                if region.width > 0 and region.height > 0:
                    self._anchor_bounds = (region.x, region.y, region.right, region.bottom)
            left, top, right, bottom = self._anchor_bounds
            cell_x, cell_y = self._anchor_cell
            width = self._desired_width
            # Border rows + the rows themselves.
            height = self._desired_height + 2
            x = max(left, min(cell_x, right - width))
            x = max(x, 0)
            # Fix round 1 (M-5): the caret ROW stays visible. Below the caret
            # when the box fits there, else above when it fits there, else
            # the larger side with the rows cut to fit (never over the
            # caret: at 80x24 the box used to be clamped onto it).
            below = bottom - (cell_y + 1)
            above = cell_y - top
            if below >= height:
                y = cell_y + 1
            elif above >= height:
                y = cell_y - height
            elif below >= above:
                height = max(3, below)
                y = cell_y + 1
            else:
                height = max(3, above)
                y = cell_y - height
            y = max(y, 0)
            # Always, not only when cutting (R2-5): a box once cut beside the
            # caret has to grow back to its rows when the space returns.
            # ds-runtime: the rows are cut to the space beside the caret.
            self.set_styles(height=height)
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


__all__ = ["LibraryNoteLinkPicker", "LinkCandidate", "MAX_VISIBLE_ROWS"]
