"""Mounted chat-message-widget index for TTS event delivery (task-14, F18d).

The TTS progress/completion handlers in ``tldw_chatbook/app_speech.py``
previously located the widget for an event with two full-DOM walks
(``app.query(ChatMessage)`` + ``app.query(ChatMessageEnhanced)``) plus a
linear ``message_id_internal`` scan per handler invocation. This module
replaces those walks with an O(1) ``dict`` lookup.

Ownership and import direction: the two widget classes
(``ChatMessage``, ``ChatMessageEnhanced``) register themselves from their
``on_mount``/``on_unmount`` hooks and re-key through
``watch_message_id_internal``; ``app_speech`` only reads. The registry
deliberately does NOT import the widget classes -- entries are opaque
objects -- so no import cycle is possible and the PIL/textual_image boot
cost of ``chat_message_enhanced`` stays off this module (TASK-21103).

Lifecycle: the mount/unmount hooks are the primary mechanism. Values are
weakrefs as a backstop so a widget whose ``on_unmount`` somehow never
fires cannot be kept alive (or resurrected) by the index; dead refs are
pruned on read. Entries are held per-instance with identity semantics --
two widgets may legitimately share one ``message_id_internal`` (variants,
edits), and each is removed individually.

All mutation happens on Textual's event loop thread (widget hooks and
event handlers alike), so plain dict/list operations are sufficient.
"""

from __future__ import annotations

import weakref
from typing import Any

#: ``message_id_internal`` -> weakrefs of currently mounted widgets.
_MESSAGE_WIDGETS: dict[str, list[weakref.ref[Any]]] = {}


def register_message_widget(message_id: str | None, widget: Any) -> None:
    """Index ``widget`` under ``message_id`` for TTS event delivery.

    Args:
        message_id: The widget's ``message_id_internal``. Falsy values are
            ignored: the TTS handlers only ever look up truthy event ids,
            so an id-less widget was never findable by the full-DOM scan
            this index replaces either.
        widget: The mounted widget (opaque; not type-checked to keep the
            import direction one-way).

    Returns:
        None. Re-registering an already-indexed instance is a no-op
        (identity comparison, not equality).
    """
    if not message_id:
        return
    refs = _MESSAGE_WIDGETS.setdefault(message_id, [])
    for existing in refs:
        if existing() is widget:
            return
    refs.append(weakref.ref(widget))


def unregister_message_widget(message_id: str | None, widget: Any) -> None:
    """Remove exactly ``widget`` (identity) from ``message_id``'s entries.

    Args:
        message_id: The key the widget was registered under.
        widget: The widget instance to remove. Other widgets sharing the
            key are untouched; dead weakrefs encountered along the way are
            pruned.

    Returns:
        None. Unknown keys or instances are a no-op.
    """
    if not message_id:
        return
    refs = _MESSAGE_WIDGETS.get(message_id)
    if refs is None:
        return
    remaining = [ref for ref in refs if ref() is not None and ref() is not widget]
    if not remaining:
        del _MESSAGE_WIDGETS[message_id]
    else:
        refs[:] = remaining


def get_message_widgets(message_id: str | None) -> tuple[Any, ...]:
    """Return the live widgets currently indexed under ``message_id``.

    Args:
        message_id: The event's message id (truthy in practice).

    Returns:
        Widgets registered under the id, in registration order. Dead
        weakrefs are dropped from the index as a side effect; a key whose
        entries have all died is removed entirely.
    """
    if not message_id:
        return ()
    refs = _MESSAGE_WIDGETS.get(message_id)
    if refs is None:
        return ()
    live: list[Any] = []
    kept: list[weakref.ref[Any]] = []
    for ref in refs:
        widget = ref()
        if widget is None:
            continue
        live.append(widget)
        kept.append(ref)
    if len(kept) != len(refs):
        if kept:
            refs[:] = kept
        else:
            del _MESSAGE_WIDGETS[message_id]
    return tuple(live)
