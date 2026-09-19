"""Cell-safe layout shared by flat conversations and workspace rows."""

from rich.text import Text


def conversation_title_cells(title: str, available_cells: int) -> str:
    """Return one physical title line within the exact available cell budget."""
    if available_cells <= 0:
        return ""
    text = Text(" ".join(str(title).split()))
    text.truncate(max(0, available_cells), overflow="ellipsis")
    return text.plain


def conversation_action_width(*, ascii_mode: bool) -> int:
    """Reserve the same target through every state change in a rendering mode."""
    return 9 if ascii_mode else 4
