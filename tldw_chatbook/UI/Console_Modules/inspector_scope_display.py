"""Original dictionary and world-book presentation from captured summaries."""

from ...Chat.console_display_state import ConsoleDisplayRow, ConsoleInspectorAction


def dictionary_rows(conversation_id, character_id, summary) -> tuple[ConsoleDisplayRow, ...]:
    if conversation_id is None and character_id is None:
        return (ConsoleDisplayRow("No active chat", ""),)
    dictionaries = (summary or {}).get("dictionaries") or []
    if not dictionaries:
        return (ConsoleDisplayRow("No dictionaries in play", ""),)
    rows = []
    for entry in dictionaries:
        if not isinstance(entry, dict):
            continue
        value = "from conversation" if entry.get("source") == "conversation" else "from character"
        if entry.get("shadowed"):
            value += " (shadowed)"
        if not entry.get("enabled", True):
            value += " (disabled)"
        rows.append(ConsoleDisplayRow(str(entry.get("name") or "Unnamed"), value))
    return tuple(rows)


def world_book_rows(conversation_id, summary) -> tuple[ConsoleDisplayRow, ...]:
    if conversation_id is None:
        return (ConsoleDisplayRow("No active chat", ""),)
    books = (summary or {}).get("world_books") or []
    if not books:
        return (ConsoleDisplayRow("No world books in play", ""),)
    rows = []
    for entry in books:
        if not isinstance(entry, dict):
            continue
        count = entry.get("entry_count")
        value = f"{count} entries" if isinstance(count, int) else "0 entries"
        if not entry.get("enabled", True):
            value += " (disabled)"
        rows.append(ConsoleDisplayRow(str(entry.get("name") or "Unnamed"), value))
    return tuple(rows)


def dictionary_actions(conversation_id, summary) -> tuple[ConsoleInspectorAction, ...]:
    dictionaries = (summary or {}).get("dictionaries") or []
    has_attached = any(
        isinstance(entry, dict) and entry.get("source") == "conversation"
        for entry in dictionaries
    )
    return (
        ConsoleInspectorAction(
            "console-inspector-dictionaries-attach", "Attach dictionary…",
            enabled=bool(conversation_id),
            disabled_reason="Start or load a conversation first",
        ),
        ConsoleInspectorAction(
            "console-inspector-dictionaries-detach", "Detach dictionary…",
            enabled=has_attached,
        ),
    )


def world_book_actions(conversation_id, summary) -> tuple[ConsoleInspectorAction, ...]:
    has_attached = bool((summary or {}).get("world_books"))
    return (
        ConsoleInspectorAction(
            "console-inspector-worldbooks-attach", "Attach world book…",
            enabled=bool(conversation_id),
            disabled_reason="Start or load a conversation first",
        ),
        ConsoleInspectorAction(
            "console-inspector-worldbooks-detach", "Detach world book…",
            enabled=has_attached,
        ),
    )
