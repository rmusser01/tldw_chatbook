"""Legacy import shim for the embeddings window.

The active implementation lives in `SearchEmbeddingsWindow`; this module keeps
older imports and source-level DataTable selection checks working.
"""

from __future__ import annotations

from textual.widgets import DataTable

from .SearchEmbeddingsWindow import SearchEmbeddingsWindow


class EmbeddingsWindow(SearchEmbeddingsWindow):
    """Compatibility alias with legacy DataTable row-selection helpers."""

    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        table = event.data_table
        row_key = event.row_key
        table.get_row(row_key)
        table.update_cell(row_key, "✓", "✓")

    def select_all_rows(self, table: DataTable) -> None:
        for row_key in table.rows:
            table.update_cell(row_key, "✓", "✓")

    def clear_selected_rows(self, table: DataTable) -> None:
        for row_key in table.rows:
            table.update_cell(row_key, "✓", "")
