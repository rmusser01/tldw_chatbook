"""Trace sources for Console provider rows the ledger does not own (TASK-33621.2).

The durable Capture-On request gives every provider row a descriptor. A row
that is a saved message's exact value gets that message's saved revision.
Every other row is a provider artifact, and the request aggregate admits an
artifact only under the source its position allows: a leading system row
lands in the ``system`` category (``RENDERED_SYSTEM`` only) or, when marked as
conversation memory, the ``memory`` category; every later row lands in the
history, the active request or a tool loop. Before TASK-33621.2 (and
TASK-33940.3) every unsaved row was called ``ACTIVE_REQUEST``, so any chat
with a system row (a session prompt, a character template, or a seeded
greeting folded into it) failed the aggregate's ``system`` check and every
send was refused before the provider was contacted. A saved image result
failed later: provider history sends it as text only, but its saved revision
projects the image too.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_prepared_request import (
    MEMORY_OWNER_KEY,
    MEMORY_OWNER_VALUE,
)
from tldw_chatbook.Chat.console_trace_provenance import TraceProvenanceSource


def unsaved_trace_artifact_source(
    row: Mapping[str, Any], *, is_last: bool
) -> TraceProvenanceSource:
    """Label one provider row that has no saved revision, by its role.

    TASK-33940.3 introduced this mapping (the voice-capture builder's) for
    both durable builders; it lives here so one module owns every unsaved
    row's source.

    Args:
        row: The provider-visible row.
        is_last: Whether the row is the request's final (active) row.

    Returns:
        The artifact source its request category accepts.
    """
    role = row.get("role")
    if is_last:
        return TraceProvenanceSource.ACTIVE_REQUEST
    if role == ConsoleMessageRole.SYSTEM.value:
        return TraceProvenanceSource.RENDERED_SYSTEM
    if role == ConsoleMessageRole.TOOL.value:
        return TraceProvenanceSource.TOOL_RESULT
    if role == ConsoleMessageRole.ASSISTANT.value and row.get("tool_calls"):
        return TraceProvenanceSource.TOOL_CALL
    return TraceProvenanceSource.ACTIVE_REQUEST


def unsaved_row_sources(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[TraceProvenanceSource, ...]:
    """Return the artifact source each provider row takes when it is unsaved.

    Args:
        rows: The final provider rows, in request order.

    Returns:
        One source per row, matching the category the request aggregate puts
        the row in: leading system rows are rendered system context (or
        conversation memory, when marked as memory); a system row after the
        first non-system row is history; every other row is labelled by its
        role (:func:`unsaved_trace_artifact_source`).
    """
    system = ConsoleMessageRole.SYSTEM.value
    leading = 0
    while leading < len(rows) and rows[leading].get("role") == system:
        leading += 1
    last = len(rows) - 1
    return tuple(
        (
            TraceProvenanceSource.CONVERSATION_MEMORY
            if row.get(MEMORY_OWNER_KEY) == MEMORY_OWNER_VALUE
            else TraceProvenanceSource.RENDERED_SYSTEM
        )
        if index < leading
        else TraceProvenanceSource.ACTIVE_REQUEST
        if row.get("role") == system
        else unsaved_trace_artifact_source(row, is_last=index == last)
        for index, row in enumerate(rows)
    )


def saved_message_id(message: object, row: Mapping[str, Any]) -> str | None:
    """Return the saved id a provider row may be admitted under, if any.

    A saved revision projects every image its message holds. Provider history
    sends only the images its budget admits, and never an assistant's image
    result, so a row that leaves an image out is not the saved revision's
    value. Admitting it as one fails the dispatch-surface check
    (``surface_prefix_mismatch``) on every later send.

    Args:
        message: The live Console message the row was built from.
        row: The provider row as it will be sent.

    Returns:
        The message's persisted id, or None when the row omits saved media.
    """
    attachments = getattr(message, "attachments", None) or ()
    content = row.get("content")
    sent = (
        sum(
            1
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "image_url"
        )
        if isinstance(content, list)
        else 0
    )
    if sent < len(attachments):
        return None
    return getattr(message, "persisted_message_id", None)


__all__ = [
    "saved_message_id",
    "unsaved_row_sources",
    "unsaved_trace_artifact_source",
]
