"""Let F6 leave a ``TextArea`` for the app's next-pane binding (TASK-34000.28 N-32).

Textual 8.2.8's ``TextArea`` binds ``f6`` to ``select_line`` (non-priority).
The app's F6 -- ``focus_next_workbench_pane`` -- is non-priority too, by
ADR-031, so Textual resolves a focused editor's own binding first: in the
Library's note body F6 SELECTED the current line, focus stayed put, and the
next keystroke replaced the line. LibraryScreen keeps F6 app-owned (no screen
binding; dev's pin ``test_library_keeps_f6_owned_by_the_app_global_binding``
and commit 6fab737230 "preserve global F6 ownership"), so the fix is at the
editor: release that one binding and the key travels up to the app.

``ChunkingLabScreen.on_mount`` applies the same release to its editors.
"""

from __future__ import annotations

from typing import TypeVar

from textual.widgets import TextArea

_EditorT = TypeVar("_EditorT", bound=TextArea)

#: The app-global next-pane key (``app.py``, ADR-031).
PANE_KEY = "f6"


def release_pane_key(editor: _EditorT) -> _EditorT:
    """Drop ``editor``'s own F6 binding so F6 reaches the app's pane key.

    Only this instance's map changes: ``DOMNode.__init__`` gives every node
    a copy of its class's merged bindings, so a stock ``TextArea`` elsewhere
    keeps Textual's F6 = select line. Every other editor key is untouched.

    Args:
        editor: A ``TextArea`` (or subclass) that owns a document body.

    Returns:
        The same editor, so a compose site can wrap the constructor call.
    """
    editor._bindings.key_to_bindings.pop(PANE_KEY, None)
    return editor
