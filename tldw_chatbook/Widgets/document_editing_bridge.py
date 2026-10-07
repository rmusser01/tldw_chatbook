"""Apply qualified source edits through Textual's existing undo history."""

from textual.document._document import Selection
from textual.widgets import TextArea

from tldw_chatbook.Document_Editing.models import (
    DocumentRequest,
    FormatResult,
    RequestKey,
)


def _map_offset(source: str, replacement: str, offset: int) -> int:
    """Anchor to non-whitespace source characters in linear time, without a diff."""
    if offset == len(source):
        return len(replacement)
    rank = sum(not char.isspace() for char in source[:offset])
    for index, char in enumerate(replacement):
        if not char.isspace():
            if rank == 0:
                return index
            rank -= 1
    return len(replacement)


class TextAreaFormatBridge:
    """Hold only the latest request; the caller retains all save authority."""

    def __init__(self, editor: TextArea) -> None:
        self.editor = editor
        self._request: DocumentRequest | None = None

    def capture_request(self, request: DocumentRequest) -> None:
        """Register the immutable snapshot before dispatching a format request."""
        self._request = request

    def apply_format(
        self, result: FormatResult, current_key: RequestKey, editable: bool
    ) -> bool:
        """Apply one history transaction only while identity and source still match."""
        editor, request = self.editor, self._request
        if (
            request is None
            or result.key != current_key
            or request.key != current_key
            or request.is_excerpt
            or not editable
            or editor.read_only
            or editor.is_disabled
            or not editor.is_mounted
            or result.state != "formatted"
            or result.replacement is None
            or not result.check.format_allowed
            or editor.text != request.text
            or result.replacement == editor.text
        ):
            return False
        replacement = result.replacement
        offsets = [
            editor.document.get_index_from_location(point) for point in editor.selection
        ]
        mapped = [_map_offset(request.text, replacement, offset) for offset in offsets]
        scroll = editor.scroll_offset
        self._request = None
        editor.history.checkpoint()
        editor.replace(
            replacement, (0, 0), editor.document.end, maintain_selection_offset=True
        )
        editor.selection = Selection(
            *(editor.document.get_location_from_index(offset) for offset in mapped)
        )
        editor.history.checkpoint()
        editor.scroll_to(scroll.x, scroll.y, animate=False, force=True)
        return True
