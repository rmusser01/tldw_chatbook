"""Permanent Prompt work pane for Basic, Advanced, Info, and import."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widgets import Static

from .library_prompts_canvas import LibraryPromptsListCanvas


class LibraryPromptWorkPane(LibraryPromptsListCanvas):
    """Render non-list Prompt content while the concrete list stays mounted."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.remove_class("library-adaptive-reader-items")
        self.styles.min_width = 0

    def compose(self) -> ComposeResult:
        """Compose the active editor/task or a stable no-selection surface."""
        if self.import_open:
            yield Static(
                "Import prompts",
                id="library-prompt-import-heading",
                classes="destination-section",
                markup=False,
            )
            yield from self._compose_import_row()
            return
        if self.mode == "list":
            yield Static(
                "Select a prompt to edit it here.",
                id="library-prompt-work-empty",
                classes="destination-purpose",
                markup=False,
            )
            return
        yield from super().compose()

    def sync_state(self, **kwargs: Any) -> None:
        """Recompose only when the work projection actually changed."""
        if all(getattr(self, key, object()) == value for key, value in kwargs.items()):
            return
        super().sync_state(**kwargs)
