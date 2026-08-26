"""App-level guard for Textual's text-selection MouseDown crash (task-14903).

The bug (present in Textual 8.2.8, the newest 8.x at the time of writing --
no fixed patch release exists inside this app's ``>=8.0.0,<9`` pin):

``Screen._forward_event``'s MouseDown branch begins a text selection by
resolving the clicked widget through ``Screen.get_widget_and_offset_at``,
which reads the COMPOSITOR's cached ``layers_visible`` map -- a snapshot
built at the last reflow. A widget pruned from the DOM mid-recompose
(``widget.parent`` already ``None``) stays resolvable in that stale map
until the next layout pass runs. For such a detached widget,
``Widget.region`` swallows ``NoScreen``/``NoWidget`` and returns
``NULL_REGION``, which drives ``get_widget_and_offset_at`` into its
``x < 0 or y < 0`` clamp branch and returns a NON-``None`` offset -- so
``_forward_event`` takes the content path::

    container = content_widget.parent        # None: widget is detached
    ...
    event.screen_offset - container.region.offset
    AttributeError: 'NoneType' object has no attribute 'region'

The exception propagates out of ``App.on_event`` (which called
``self.screen._forward_event(event)`` synchronously), reaches the message
pump's ``_handle_exception``, and terminates the whole application --
observed live in task-4023's verification as a click on the Library
Search/RAG canvas's ``#library-rag-query-quiet-line`` Static ~1s after a
50->24-row terminal resize. Reproduced deterministically in
``Tests/App/test_text_selection_crash_guard.py``.

Why THIS seam: the crash happens during event FORWARDING, before any
widget/screen handler dispatch, so no ``on_mouse_down`` handler anywhere
can intercept it; ``BaseAppScreen._forward_event`` would miss modal and
non-``BaseAppScreen`` screens; ``App._handle_exception`` is past the point
of recovery (the pump is already tearing down). Wrapping
``App.on_event`` -- the one dispatcher call every forwarded input event
passes through -- is the single choke point that covers every screen.

Why the filter cannot swallow real bugs: a caught ``AttributeError`` is
re-raised unless ALL of these hold -- the event is a ``MouseDown``, the
RAISING frame (innermost traceback frame) is Textual's own
``screen.py::_forward_event``, and that frame's ``container`` local is
``None``. That is exactly the selection-begin block dereferencing a
detached widget's parent, and nothing else. The check is deliberately
version-coupled to Textual internals and fails OPEN: if a future Textual
renames the function or the local, the predicate stops matching and the
exception propagates again (loudly), rather than the guard silently
eating unknown errors. The pinned reproduction test breaks alongside it.
"""

from __future__ import annotations

from types import FrameType
from typing import Optional

from loguru import logger
from textual import events

__all__ = [
    "TextSelectionCrashGuard",
    "match_selection_begin_container_crash",
]


def _raising_frame(error: BaseException) -> Optional[FrameType]:
    """The frame the exception was actually raised in, or None."""
    traceback = error.__traceback__
    if traceback is None:
        return None
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    return traceback.tb_frame


def match_selection_begin_container_crash(
    error: BaseException, event: object
) -> Optional[str]:
    """Match EXACTLY the task-14903 signature; anything else is not ours.

    Args:
        error: The exception that escaped event dispatch.
        event: The event whose dispatch raised ``error``.

    Returns:
        A short, log-safe description of the crash target (widget repr --
        Textual reprs carry id/classes, never content) when ``error`` is
        precisely Textual's selection-begin ``container = None``
        AttributeError for a ``MouseDown``; ``None`` for everything else,
        in which case the caller must re-raise.
    """
    if not isinstance(error, AttributeError):
        return None
    if not isinstance(event, events.MouseDown):
        return None
    frame = _raising_frame(error)
    if frame is None:
        return None
    code = frame.f_code
    if code.co_name != "_forward_event":
        return None
    filename = code.co_filename.replace("\\", "/")
    if not filename.endswith("textual/screen.py"):
        return None
    frame_locals = frame.f_locals
    if "container" not in frame_locals or frame_locals["container"] is not None:
        return None
    select_widget = frame_locals.get("select_widget")
    return repr(select_widget) if select_widget is not None else "<unknown widget>"


class TextSelectionCrashGuard:
    """App mixin: survive Textual's selection-begin crash on a stale click.

    Add BEFORE ``App`` in the base list (``class TldwCli(
    TextSelectionCrashGuard, ..., App[None])``). See the module docstring
    for the crash mechanics and why this is the narrowest viable seam.
    """

    async def on_event(self, event: events.Event) -> None:
        try:
            await super().on_event(event)  # type: ignore[misc]
        except AttributeError as error:
            target = match_selection_begin_container_crash(error, event)
            if target is None:
                raise
            # Complete the branch Textual aborted, exactly as its own
            # "target is not selectable" path would have
            # (screen.py: ``self._select_state = None``), so a stale
            # selection start from a previous interaction cannot linger.
            try:
                self.screen._select_state = None  # type: ignore[attr-defined]
            except Exception:
                pass
            logger.warning(
                "Dropped a MouseDown that hit Textual's text-selection begin "
                "path while its target widget was mid-recompose (detached "
                "parent). The click was not delivered; the app stays alive "
                "(task-14903)."
            )
