"""A checkbox whose on/off state is a glyph, not a colour.

Stock ``ToggleButton`` paints the same ``X`` for both states and conveys
on/off through colour alone, so an unchecked box reads as checked in a
monochrome terminal, in a plain-text capture, and to anyone who cannot rely
on colour (WCAG 1.4.1). The first-run wizard solved this privately
(``SetupCheckbox``, TASK-21146) and the Library ingest canvas did the same
again (``StateGlyphCheckbox``, task-2043). This is the shared widget
(TASK-34100.4) for any surface outside those modules -- first the password
dialog's 'Show password' toggle, which is shared by the wizard, Settings and
the deprecated Tools window.
"""

from __future__ import annotations

from textual.widgets import Checkbox

from .glyph_fallback import ascii_glyph_mode

#: Inner glyph while checked; unchecked is a blank cell.
STATE_CHECKBOX_ON_GLYPH = "✓"
#: ASCII-mode substitute (one cell wide, like the button it replaces).
STATE_CHECKBOX_ON_ASCII = "x"


class StateCheckbox(Checkbox):
    """Checkbox showing ``✓`` when checked and a blank cell when not."""

    @property
    def _button(self):
        # BUTTON_INNER is ToggleButton's per-instance glyph seam; set it right
        # before the parent property renders so it shadows the class value.
        if self.value:
            self.BUTTON_INNER = (
                STATE_CHECKBOX_ON_ASCII if ascii_glyph_mode() else STATE_CHECKBOX_ON_GLYPH
            )
        else:
            self.BUTTON_INNER = " "
        return super()._button
