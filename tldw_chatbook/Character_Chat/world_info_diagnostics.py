"""Diagnostics data model for world-info (Lore) activation — the read model the
Try-it panel renders. Mirrors ``Chat_Dictionary_Lib``'s
``DictionaryProcessDiagnostics``/``DictionaryEntryDiagnostic`` shape, with
world-info-specific fields (source book, injection position, recursion depth).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WorldBookEntryDiagnostic:
    entry_id: Optional[int]
    source_book_id: Optional[int]
    source_book_name: str
    keys: List[str]
    activation_reason: str
    status: str  # "fired" | "skipped:disabled" | "skipped:secondary" | "skipped:budget"
    token_cost: int = 0
    injection_order: Optional[int] = None
    position: str = "before_char"
    content_preview: str = ""
    depth_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return the entry diagnostic as a plain dict for the Try-it renderer.

        Returns:
            A JSON-friendly dict with every field of this record.
        """
        return {
            "entry_id": self.entry_id,
            "source_book_id": self.source_book_id,
            "source_book_name": self.source_book_name,
            "keys": list(self.keys),
            "activation_reason": self.activation_reason,
            "status": self.status,
            "token_cost": self.token_cost,
            "injection_order": self.injection_order,
            "position": self.position,
            "content_preview": self.content_preview,
            "depth_level": self.depth_level,
        }


@dataclass
class WorldBookScanDiagnostics:
    entries: List[WorldBookEntryDiagnostic] = field(default_factory=list)
    matched: int = 0
    fired: int = 0
    skipped: int = 0
    tokens_used: int = 0
    token_budget: int = 0
    budget_exceeded: bool = False
    books_scanned: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return the scan diagnostics as a plain dict for the Try-it renderer.

        Returns:
            A JSON-friendly dict with the summary counters and a nested
            ``entries`` list of per-entry diagnostic dicts.
        """
        return {
            "entries": [record.to_dict() for record in self.entries],
            "matched": self.matched,
            "fired": self.fired,
            "skipped": self.skipped,
            "tokens_used": self.tokens_used,
            "token_budget": self.token_budget,
            "budget_exceeded": self.budget_exceeded,
            "books_scanned": self.books_scanned,
        }
