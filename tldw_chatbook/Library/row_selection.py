"""Per-source checked-row accumulator for the Library multi-select export.

Pure and Textual-free: the screen owns one instance per browsable source
(media/conversations/notes) and drives it from the row-press handlers, then
turns the checked ids into an ``ExportScope`` for the export canvas.
"""
from __future__ import annotations

from typing import Iterable

from tldw_chatbook.Library.library_export_scope import ExportScope


class RowSelection:
    def __init__(self, kind: str) -> None:
        # kind is one of "media" | "conversations" | "notes" — the ExportScope
        # single-source kind these ids belong to.
        self._kind = kind
        self._ids: set[str] = set()

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._ids)

    @property
    def count(self) -> int:
        return len(self._ids)

    def is_selected(self, row_id: str) -> bool:
        return row_id in self._ids

    def toggle(self, row_id: str) -> None:
        if not row_id:
            return
        self._ids.discard(row_id) if row_id in self._ids else self._ids.add(row_id)

    def select_all(self, rendered_ids: Iterable[str]) -> None:
        self._ids.update(rid for rid in rendered_ids if rid)

    def clear(self) -> None:
        self._ids.clear()

    def reconcile(self, rendered_ids: Iterable[str]) -> None:
        """Drop any selected id no longer present in the rendered rows."""
        self._ids &= set(rendered_ids)

    def export_scope(self) -> ExportScope:
        return ExportScope(kind=self._kind, ids=tuple(sorted(self._ids)))
