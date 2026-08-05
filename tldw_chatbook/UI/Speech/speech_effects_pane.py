"""Audio Effects: a placeholder that explains itself.

The rail entry previously shipped `disabled=True`. A greyed row called
"Audio Effects" tells the user nothing -- they cannot tell whether it is
broken, unavailable to them, or simply unbuilt, and there is no way to find
out from the screen.

One line, per the spec: what it will be, that it is not built, and the view
it belongs to. That is strictly more informative than a control that cannot
be pressed, and it costs a row rather than a screen.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

#: The whole surface. Kept as a constant so the copy is greppable and the
#: test asserts against the same string the user reads.
EFFECTS_PLACEHOLDER_TEXT = (
    "Sound effect generation is not built yet. It will live here as part of "
    "the Studio view, alongside voice cloning and multi-track assembly."
)


class SpeechEffectsPane(Vertical):
    """The Audio Effects view: a title and one line of explanation."""

    def __init__(self, **kwargs: Any) -> None:
        """Create the pane.

        Args:
            kwargs: Forwarded to ``Vertical``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"speech-effects-pane {classes}".strip(), **kwargs)

    def compose(self) -> ComposeResult:
        """Yield the title and the placeholder line.

        Returns:
            A ``ComposeResult`` with the pane title and one explanatory
            line.
        """
        yield Static("🎵 Audio Effects", classes="speech-pane-title")
        yield Static(
            EFFECTS_PLACEHOLDER_TEXT,
            id="speech-effects-placeholder",
            classes="speech-status-line",
            markup=False,
        )
