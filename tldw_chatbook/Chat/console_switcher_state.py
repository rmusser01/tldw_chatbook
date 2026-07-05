"""Pure result contracts for the Console session switcher (Ctrl+K)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
    ReverseKey,
)

CONSOLE_SWITCHER_RESULT_LIMIT = 20


@dataclass(frozen=True)
class ConsoleSwitcherEntry:
    """One selectable result in the Console session switcher."""

    row_key: str
    title: str
    subtitle: str
    native_session_id: str | None
    conversation_id: str | None
    scope_type: str
    workspace_id: str | None
    is_active: bool


def _matches(row: ConsoleConversationBrowserInputRow, tokens: list[str]) -> bool:
    haystack = " ".join((row.title, row.workspace_label, row.status)).lower()
    return all(token in haystack for token in tokens)


def build_console_switcher_entries(
    rows: Iterable[ConsoleConversationBrowserInputRow],
    *,
    query: str = "",
    limit: int = CONSOLE_SWITCHER_RESULT_LIMIT,
) -> tuple[ConsoleSwitcherEntry, ...]:
    """Build deduped, recent-first switcher results for a query.

    Args:
        rows: Browser input rows from the chat screen row builders.
        query: Whitespace-separated tokens; every token must match the row's
            title, workspace label, or status (case-insensitive substring).
        limit: Maximum number of entries returned.

    Returns:
        Up to ``limit`` entries, active row first, then most recent.
    """
    tokens = [token for token in query.lower().split() if token]
    seen: set[str] = set()
    deduped: list[ConsoleConversationBrowserInputRow] = []
    for row in rows:
        key = str(row.row_key or "")
        if not key or key in seen:
            continue
        seen.add(key)
        if tokens and not _matches(row, tokens):
            continue
        deduped.append(row)

    deduped.sort(
        key=lambda row: (
            not row.selected,
            ReverseKey(str(row.updated_sort or "")),
            row.title.casefold(),
            row.row_key,
        )
    )
    entries = []
    for row in deduped[: max(0, int(limit))]:
        subtitle = " - ".join(
            part
            for part in (row.workspace_label, row.status, row.updated_label)
            if str(part or "").strip()
        )
        entries.append(
            ConsoleSwitcherEntry(
                row_key=str(row.row_key),
                title=str(row.title or "Untitled conversation"),
                subtitle=subtitle,
                native_session_id=row.native_session_id,
                conversation_id=row.conversation_id,
                scope_type=str(row.scope_type or ""),
                workspace_id=row.workspace_id,
                is_active=bool(row.selected),
            )
        )
    return tuple(entries)
