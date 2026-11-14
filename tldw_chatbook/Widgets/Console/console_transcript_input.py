"""Input ownership for the native Console transcript and its controls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial, wraps
from inspect import iscoroutinefunction
from typing import Any

from textual.dom import DOMNode
from textual.message import Message
from textual.widget import Widget


@dataclass(eq=False)
class TranscriptInputVisit:
    """One published view of a store activation."""

    published_at: float
    is_current: Callable[[], bool]


def _managed_transcript(node: object) -> Any:
    while isinstance(node, DOMNode):
        if getattr(node, "_transcript_surface", None) is not None:
            return node
        if (owner := getattr(node, "_transcript_input_owner", None)) is not None:
            node = owner
        else:
            node = node.parent
    return None


def _current_visit(owner: Any) -> TranscriptInputVisit | None:
    visit = getattr(owner, "_transcript_input_visit", None)
    return visit if visit is not None and visit.is_current() else None


def transcript_action_current(message: Message, receiver: Widget | None = None) -> bool:
    """Check the input's original visit, including after it bubbles."""
    origin = getattr(message, "_transcript_input_visit", None)
    if origin is not None:
        return origin.is_current()
    owner = (
        _managed_transcript(message._sender)
        or _managed_transcript(message.control)
        or _managed_transcript(receiver)
    )
    if owner is None:
        return True
    visit = _current_visit(owner)
    if visit is None or message.time < visit.published_at:
        return False
    message._transcript_input_visit = visit
    return True


def transcript_callback(
    node: Widget, callback: Callable[..., Any], *args: Any, **kwargs: Any
) -> Callable[[], Any]:
    """Preserve the visit when Textual schedules a transcript effect."""
    invoke = partial(callback, *args, **kwargs)
    owner = _managed_transcript(node)
    if owner is None:
        return invoke
    visit = _current_visit(owner)

    def run() -> Any:
        if visit is not None and visit.is_current():
            return invoke()
        return None

    return run


def guard_transcript_action(method):
    """Check a Console handler before it consumes a queued action."""

    def allowed(widget: Widget, message: Message) -> bool:
        if transcript_action_current(message, widget):
            return True
        message.stop().prevent_default()
        return False

    if iscoroutinefunction(method):

        @wraps(method)
        async def handle(widget, message):
            if allowed(widget, message):
                return await method(widget, message)
            return None
    else:

        @wraps(method)
        def handle(widget, message):
            if allowed(widget, message):
                return method(widget, message)
            return None

    return handle


def transcript_focus(
    node: Widget, target: Widget, *, scroll_visible: bool = True
) -> None:
    """Defer focus only while the requesting transcript visit is current."""
    if _managed_transcript(node) is None:
        target.focus(scroll_visible=scroll_visible)
        return
    screen = node.screen

    def set_focus() -> None:
        if target.is_attached and target.display:
            screen.set_focus(target, scroll_visible=scroll_visible)

    node.app.call_later(transcript_callback(node, set_focus))
