from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Input, TextArea, Button, Collapsible

#from ..Widgets.emoji_picker import


class NotesSidebarRight(VerticalScroll):
    """A sidebar for displaying and editing note details."""

    DEFAULT_CSS = """
    NotesSidebarRight {
        dock: right;
        width: 25%;
        min-width: 20;
        max-width: 80;
        background: $boost;
        padding: 1;
        border-left: thick $background-darken-1;
        overflow-y: auto;
        overflow-x: hidden;
    }
    NotesSidebarRight > .sidebar-title {
        text-style: bold underline;
        margin-bottom: 1;
        width: 100%;
        text-align: center;
    }
    NotesSidebarRight > Static.sidebar-label {
        margin-top: 1;
    }
    NotesSidebarRight > Input, NotesSidebarRight > TextArea {
        width: 100%;
        margin-bottom: 1;
    }
    NotesSidebarRight > Button, NotesSidebarRight > Collapsible > Button {
        width: 100%;
        margin-bottom: 1;
    }
    NotesSidebarRight > Collapsible {
        width: 100%;
        margin-bottom: 1;
    }
    .notes-keywords-textarea { /* Specific class from your app.py */
        height: 5; /* Example fixed height for keywords */
        /* width: 100%; (inherited) */
        /* margin-bottom: 1; (inherited) */
    }
    """

    def compose(self) -> ComposeResult:
        """Create child widgets for the notes details sidebar."""
        yield Static("Note Details", classes="sidebar-title", id="notes-details-sidebar-title")

        yield Static("Title:", classes="sidebar-label")
        yield Input(placeholder="Note title...", id="notes-title-input")

        yield Static("Keywords:", classes="sidebar-label")
        yield TextArea("", id="notes-keywords-area", classes="notes-keywords-textarea")
        yield Button("Save Note (from Editor)", id="notes-save-current-button",
                     variant="success")  # Or "Save Active Note"
        yield Button("Save Keywords", id="notes-save-keywords-button", variant="primary")

        # New Collapsible for Emojis
        with Collapsible(title="Emojis", collapsed=True):
            #yield EmojiPicker(id="notes-emoji-picker")
            pass  # Placeholder for EmojiPicker widget

        # Group export options
        with Collapsible(title="Export Options", collapsed=True):
            yield Button("Export as Markdown", id="notes-export-markdown-button")
            yield Button("Export as Text", id="notes-export-text-button")
        with Collapsible(title="Delete Note", collapsed=True):
            yield Button("Delete Selected Note", id="notes-delete-button", variant="error")