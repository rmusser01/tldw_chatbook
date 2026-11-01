"""Kanban board UI package (TASK-18934, ADR-237).

Board TUI ported and adapted from backlog-md-py's ``src/backlog_py/tui``
(``models.py`` / ``widgets.py`` / ``screens.py`` / ``app.py`` @ main
39e66a8): the view-model/filter/selection-preservation architecture and
the column/card/inspector widgets carry over; the markdown-project data
layer, orchestration queue, and run-history machinery are replaced by the
chatbook Kanban store (``Kanban_Interop.LocalKanbanService``). See
``backlog/decisions/237-kanban-board-port-and-data-ownership.md``.
"""

from __future__ import annotations

from tldw_chatbook.UI.Kanban.board_models import (
    BoardSnapshot,
    CardView,
    FilterState,
    SelectionState,
    filter_snapshot,
    select_after_refresh,
)
from tldw_chatbook.UI.Kanban.board_widgets import (
    BoardColumn,
    BoardHeader,
    CardInspector,
    CardWidget,
)
from tldw_chatbook.UI.Kanban.kanban_board_screen import KanbanBoardScreen

__all__ = [
    "BoardColumn",
    "BoardHeader",
    "BoardSnapshot",
    "CardInspector",
    "CardView",
    "CardWidget",
    "FilterState",
    "KanbanBoardScreen",
    "SelectionState",
    "filter_snapshot",
    "select_after_refresh",
]
