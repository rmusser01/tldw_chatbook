"""A TextArea that renders blank once detached (TASK-32049)."""

from __future__ import annotations

from textual.geometry import Region
from textual.strip import Strip
from textual.widgets import TextArea


class DetachSafeTextArea(TextArea):
    """TextArea whose queued repaint after removal renders blank instead of raising.

    Textual's ``Widget._message_loop_exit`` detaches the widget and then clears
    its component styles. A screen repaint already queued can still call
    ``render_lines``, where the theme step looks up ``text-area--gutter`` and
    raises ``KeyError``. ``_detach()`` runs first, so ``is_attached`` is already
    False whenever the styles are gone -- and a detached widget is never visible,
    so blank lines are the correct render.
    """

    def render_lines(self, crop: Region) -> list[Strip]:
        """Render normally while attached; blank lines once detached.

        Args:
            crop: Region of the widget to render.

        Returns:
            list[Strip]: One strip per row of ``crop``.
        """
        if not self.is_attached:
            return [Strip.blank(crop.width) for _ in range(crop.height)]
        return super().render_lines(crop)
