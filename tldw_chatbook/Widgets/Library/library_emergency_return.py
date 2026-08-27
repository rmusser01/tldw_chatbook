"""Narrow Library canvas return control."""

from __future__ import annotations

from textual.message import Message
from textual.widgets import Button

from ..glyph_fallback import ascii_glyph_mode


class LibraryEmergencyReturn(Button):
    """Presentation-only request control for the narrow canvas stage."""

    #: TASK-22858: BUNDLED_CSS, not DEFAULT_CSS — build_css.py lifts this
    #: into the widget-defaults tier of the app bundle. A class-level
    #: DEFAULT_CSS registers another stylesheet source against Textual's
    #: 64-entry parse cache (see Tests/UI/test_widget_css_consolidation.py).
    BUNDLED_CSS = """
    LibraryEmergencyReturn {
        width: 100%;
        height: 1;
        border: none;
    }
    """

    class ReturnRequested(Message):
        """Request a guarded return to the Library rail stage."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__("< Library" if ascii_glyph_mode() else "‹ Library", **kwargs)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Translate pointer and Enter activation into one typed request."""
        if event.button is not self:
            return
        event.stop()
        self.post_message(self.ReturnRequested())
