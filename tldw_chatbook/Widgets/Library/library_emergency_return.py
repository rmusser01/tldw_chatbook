"""Narrow Library canvas return control."""

from __future__ import annotations

from textual.message import Message
from textual.widgets import Button

from ..glyph_fallback import ascii_glyph_mode


class LibraryEmergencyReturn(Button):
    """Presentation-only request control for the narrow canvas stage."""

    DEFAULT_CSS = """
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
