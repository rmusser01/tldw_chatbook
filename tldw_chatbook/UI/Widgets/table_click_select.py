"""Make a single click on a `DataTable` row select it, not just move the cursor.

Textual fires `DataTable.RowSelected` on **activation** — Enter, or a second
click — while a single click produces `DataTable.RowHighlighted`. A pane that
handles only `RowSelected` therefore highlights on click and selects nothing:
the inspector stays empty and row actions stay disabled.

That defect made `Preview` / `Check now` / `Delete` unreachable by mouse in the
Watchlists Sources pane and ultimately hid a dead scrape path (TASK-1100,
TASK-1105). Every other table pane in the app still had it (TASK-1180).

**Why this gates on focus.** `RowHighlighted` is not a user-input event. It also
fires when a pane rebuilds its own table — `clear()` then `add_row()` moves the
cursor back to row 0 — and forwarding those turns a repopulation into a
selection. On the MCP workbench that is not merely noisy: a selection triggers
an awaited remove/mount in `MCPInspector`, which re-syncs the mode, which
repopulates the table, which highlights row 0 again. A first draft of this mixin
without the gate produced **157 selections from opening the Tools tab with no
user input at all**, and buried a genuine click under the repeats.

Focus separates the two cleanly, as measured on the real screen: a repopulating
table is not focused, while a click focuses the table before the cursor moves
and keyboard navigation requires focus by definition. So "the cursor moved on a
focused table" means a person moved it.
"""

from __future__ import annotations

from typing import Any

from textual.widgets import DataTable

__all__ = ["DataTableClickSelectMixin"]


class DataTableClickSelectMixin:
    """Forward user-driven `DataTable` cursor movement to the selection handler.

    List this **before** the widget base class so its handlers are found::

        class MyPane(DataTableClickSelectMixin, Vertical):
            def on_data_table_row_selected(self, event): ...

    The mixin re-dispatches to the pane's existing handler rather than defining
    its own selection hook: each pane resolves a row key back to its own domain
    object and posts its own message, and duplicating that in a second method is
    how the two drift apart.
    """

    #: Set False on a pane where cursor movement genuinely should not select --
    #: a multi-select table being marked up, for example. Prefer leaving the
    #: mixin off entirely; this exists so an opt-out is greppable.
    select_on_highlight: bool = True

    #: Last row key forwarded, so a table that re-highlights the same row does
    #: not re-post.
    _last_forwarded_row_key: Any = None

    #: Set while the pane is rebuilding its own rows. See `repopulating_table`.
    _suppress_row_selection: bool = False

    #: True only while the mixin is invoking the pane's own handler, so the
    #: dedup wrapper can tell its own call from a native activation.
    _forwarding_highlight: bool = False

    #: Row key a forwarded highlight just selected, consumed by the *next*
    #: native activation and then cleared. One-shot on purpose: it must
    #: suppress the Enter that completes an arrow-then-Enter gesture, and must
    #: NOT suppress a later, genuine re-selection of the same row -- which is
    #: what `goto_permission_row()` and the sub-view triggers do.
    _pending_activation_key: Any = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Make the pane's own `RowSelected` handler idempotent per row.

        Highlighting a row now selects it, so a user who arrows onto a row and
        *then* presses Enter would otherwise be processed twice: once by the
        forwarded highlight, once by Textual's native `RowSelected`. That is
        redundant work for an idempotent handler and a wrong count for anything
        that posts a message -- `test_row_selection_posts_entry_selected_with_
        synthetic_index` asserts exactly one event for exactly that gesture.

        The mixin cannot intercept by defining `on_data_table_row_selected`
        itself: a method on the subclass shadows one on a base class, and the
        subclass is where every pane defines its handler. Wrapping at class
        creation gets in front of that without any pane having to remember.
        """
        super().__init_subclass__(**kwargs)
        handler = cls.__dict__.get("on_data_table_row_selected")
        if handler is None or getattr(handler, "_tldw_dedup_wrapped", False):
            return

        def _deduped(self, event, _handler=handler):
            if self._forwarding_highlight:
                # This IS the mixin's forwarded call; let it through.
                return _handler(self, event)
            row_key = getattr(event, "row_key", None)
            value = getattr(row_key, "value", None)
            pending = self._pending_activation_key
            self._pending_activation_key = None
            if value is not None and value == pending:
                # The activation completing the gesture whose highlight already
                # selected this row; swallow it rather than re-posting.
                event.stop()
                return None
            return _handler(self, event)

        _deduped._tldw_dedup_wrapped = True  # type: ignore[attr-defined]
        _deduped.__name__ = handler.__name__
        _deduped.__doc__ = handler.__doc__
        cls.on_data_table_row_selected = _deduped  # type: ignore[assignment]

    def repopulating_table(self) -> None:
        """Declare that this pane is about to rebuild a table's rows.

        Call immediately before `clear()`/`add_row()`. Focus alone is not
        enough: a pane can rebuild a table the user is currently sitting in --
        switching the selected server repopulates the tools table while it
        still has focus -- and the resulting row-0 highlight then re-selects a
        tool, defeating the very clear that triggered the rebuild. An existing
        workbench test (`test_switching_selected_server_clears_tool_detail`)
        catches exactly that.

        Suppression is released after the next refresh rather than at the end of
        a `with` block, because Textual delivers the highlight messages a
        rebuild produces *after* the code that produced them has returned.
        """
        self._suppress_row_selection = True
        self._last_forwarded_row_key = None
        call_after_refresh = getattr(self, "call_after_refresh", None)
        if call_after_refresh is None:  # not mounted (unit-constructed pane)
            self._suppress_row_selection = False
            return
        call_after_refresh(self._resume_row_selection)

    def _resume_row_selection(self) -> None:
        self._suppress_row_selection = False

    def _should_forward(self, table: DataTable | None, row_key: Any) -> bool:
        if not self.select_on_highlight:
            return False
        if self._suppress_row_selection:
            return False
        if row_key is None or getattr(row_key, "value", None) is None:
            return False
        # A repopulating table is usually not focused; a clicked or arrowed one
        # always is. `repopulating_table()` covers the case where it is.
        if table is None or not table.has_focus:
            return False
        if row_key.value == self._last_forwarded_row_key:
            return False
        self._last_forwarded_row_key = row_key.value
        return True

    def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        """A click, or an arrow key, landing on a row."""
        handler = getattr(self, "on_data_table_row_selected", None)
        if handler is None:
            return
        table = getattr(event, "data_table", None)
        row_key = getattr(event, "row_key", None)
        if not self._should_forward(table, row_key):
            return
        event.stop()
        self._forwarding_highlight = True
        try:
            handler(
                DataTable.RowSelected(table, getattr(event, "cursor_row", 0), row_key)
            )
        finally:
            self._forwarding_highlight = False
            self._pending_activation_key = row_key.value

    def on_data_table_cell_highlighted(
        self, event: DataTable.CellHighlighted
    ) -> None:
        """The same, for a table whose cursor is cell-shaped rather than row-shaped."""
        table = getattr(event, "data_table", None)
        row_key = getattr(getattr(event, "cell_key", None), "row_key", None)
        if not self._should_forward(table, row_key):
            return

        cell_handler: Any = getattr(self, "on_data_table_cell_selected", None)
        if cell_handler is not None:
            event.stop()
            self._forwarding_highlight = True
            try:
                cell_handler(
                    DataTable.CellSelected(
                        table,
                        getattr(event, "value", None),
                        getattr(event, "coordinate", None),
                        event.cell_key,
                    )
                )
            finally:
                self._forwarding_highlight = False
            return

        row_handler = getattr(self, "on_data_table_row_selected", None)
        if row_handler is None:
            return
        event.stop()
        self._forwarding_highlight = True
        try:
            row_handler(
                DataTable.RowSelected(
                    table,
                    getattr(getattr(event, "coordinate", None), "row", 0) or 0,
                    row_key,
                )
            )
        finally:
            self._forwarding_highlight = False
