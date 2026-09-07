"""Whether the app's focus still belongs to a given screen.

Deliberately dependency-light: it imports only ``NoScreen`` so that screens
can use it without pulling in a module that defines widget classes (a
class-level ``DEFAULT_CSS`` costs a Textual stylesheet parse-cache slot).
"""

from __future__ import annotations

from typing import Any

from textual.dom import NoScreen

__all__ = ["focus_is_on_screen", "focused_id_on_screen"]


def focus_is_on_screen(focused: Any, screen: Any) -> bool:
    """Return whether ``focused`` is still attached to ``screen``.

    ``DOMNode.screen`` RAISES ``NoScreen`` for a detached node, so the
    obvious guard -- ``focused.screen is screen`` -- explodes before it can
    protect anything. ``App.focused`` legitimately outlives its screen: push
    the first-run wizard over Settings and pop it, and a worker-driven
    refresh can land with ``app.focused`` still pointing at a widget that has
    left the tree. In Settings' sync-row refresh that raised
    ``NoScreen('node has no screen')`` inside the worker and surfaced as a
    WorkerFailed (TASK-24652).

    Args:
        focused: The currently focused widget, or None.
        screen: The screen asking whether it still owns that focus.

    Returns:
        True only when ``focused`` is non-None and still attached to
        ``screen``; False for None, a detached node, or another screen's.
    """

    if focused is None:
        return False
    try:
        return focused.screen is screen
    except NoScreen:
        return False


def focused_id_on_screen(focused: Any, screen: Any) -> str | None:
    """Return ``focused``'s id, but only while it still lives on ``screen``.

    Args:
        focused: The currently focused widget, or None.
        screen: The screen asking whether it still owns that focus.

    Returns:
        The focused widget's id when it is attached to ``screen`` and has
        one, otherwise None.
    """

    if not focus_is_on_screen(focused, screen):
        return None
    return getattr(focused, "id", None) or None
