"""Native branch cursor and context-summary persistence, owned by the Chat store."""

from __future__ import annotations

from loguru import logger


def session_context_summary(self, session_id: str) -> tuple[str | None, str | None]:
    """Return the session's in-memory ``(summary, boundary_native_id)`` pair.

    Console `/rewind` "summarize up to here" (SP2). ``(None, None)`` when
    no summary has been set (including a freshly created session).
    """
    self._session_or_raise(session_id)
    return self._context_summary_by_session.get(session_id, (None, None))


def set_session_context_summary(
    self,
    session_id: str,
    summary: str | None,
    boundary_native_id: str | None,
) -> None:
    """Set (or clear, with ``(None, None)``) a session's boundary summary.

    Updates the in-memory ``(summary, boundary_native_id)`` pair, then --
    when the session owns a persisted conversation and the persistence
    adapter exposes a raw ``db`` seam -- write-throughs the local-only
    ``conversations.context_summary`` / ``summary_boundary_message_id``
    columns (mapped to the boundary node's *persisted* id, or ``None``
    when the boundary is cleared or not yet persisted). A durable write
    failure is logged, never raised: the in-memory pair is authoritative
    and already updated, matching ``set_active_leaf``'s persist-through
    convention. Unlike ``set_active_leaf``, an unknown
    ``boundary_native_id`` does not raise -- it write-throughs a ``None``
    persisted id (treated as "not yet persisted"), matching the design's
    fail-open handling of a dangling boundary.

    Args:
        session_id: Native Console session ID.
        summary: The boundary summary text, or ``None`` to clear it.
        boundary_native_id: Native id of the message the summary covers
            up to, or ``None`` to clear it.

    Raises:
        KeyError: If the session is unknown.
    """
    self._session_or_raise(session_id)
    previous = self._context_summary_by_session.get(session_id, (None, None))
    updated = (summary, boundary_native_id)
    self._context_summary_by_session[session_id] = updated
    self._persist_context_summary(session_id, summary, boundary_native_id)
    self._bump_payload_revision(session_id)
    if updated != previous:
        self._bump_conversation_context_epoch(session_id)


def _persist_active_leaf(
    self,
    session_id: str,
    message_id: str | None,
    *,
    content_safe_diagnostic: bool = False,
) -> bool:
    """Write-through the local-only active-leaf pointer for a persisted conv.

    No-op unless the session owns a persisted conversation AND the
    persistence adapter exposes a raw ``db`` seam (mirrors the
    ``persistence_db = getattr(self.persistence, "db", None)`` pattern in
    ``persist_session_if_needed``). Maps the in-memory leaf to its persisted
    message id (``None`` when cleared or not yet persisted).

    Returns False only for an ownership refusal. Other I/O errors preserve
    the historical in-memory write-through behavior.
    """
    session = self._sessions.get(session_id)
    conversation_id = session.persisted_conversation_id if session is not None else None
    if conversation_id is None:
        return True
    persistence_db = getattr(self.persistence, "db", None)
    if persistence_db is None:
        return True
    leaf_persisted_id: str | None = None
    if message_id is not None:
        node = self._nodes_by_session.get(session_id, {}).get(message_id)
        leaf_persisted_id = node.persisted_message_id if node is not None else None
    return self._write_active_leaf(
        persistence_db,
        conversation_id,
        leaf_persisted_id,
        session_id=session_id,
        content_safe_diagnostic=content_safe_diagnostic,
    )


def _persist_context_summary(
    self,
    session_id: str,
    summary: str | None,
    boundary_native_id: str | None,
) -> None:
    """Write-through the local-only context-summary pair for a persisted conv.

    Console `/rewind` "summarize up to here" (SP2). No-op unless the
    session owns a persisted conversation AND the persistence adapter
    exposes a raw ``db`` seam (mirrors ``_persist_active_leaf``). Maps
    the in-memory boundary to its persisted message id (``None`` when
    cleared or not yet persisted).
    """
    session = self._sessions.get(session_id)
    conversation_id = session.persisted_conversation_id if session is not None else None
    if conversation_id is None:
        return
    persistence_db = getattr(self.persistence, "db", None)
    if persistence_db is None:
        return
    boundary_persisted_id: str | None = None
    if boundary_native_id is not None:
        node = self._nodes_by_session.get(session_id, {}).get(boundary_native_id)
        boundary_persisted_id = node.persisted_message_id if node is not None else None
    try:
        persistence_db.set_conversation_context_summary(
            conversation_id, summary, boundary_persisted_id
        )
    except Exception:  # noqa: BLE001 - preserve existing in-memory cursor fallback on storage errors.
        logger.bind(
            session_id=session_id,
            conversation_id=conversation_id,
        ).exception(
            "Failed to persist Console context-summary; the in-memory "
            "pair keeps the applied value."
        )


def _resolve_context_summary_on_resume(
    self, session_id: str, persisted_to_native: dict[str, str]
) -> None:
    """Map the persisted context-summary boundary back to a native id.

    Console `/rewind` resume mapping (SP2). Reads
    ``get_conversation_context_summary`` via the db seam (mirrors
    ``_persist_active_leaf``'s ``getattr(self.persistence, "db", None)``
    guard) -- no-op unless the session owns a persisted conversation and
    the persistence adapter exposes the seam. The stored boundary is a
    *persisted* message id; when it maps to a node on the just-loaded
    tree (``persisted_to_native``, built by ``_ingest_full_tree``), the
    in-memory summary state is set. Absent, unreadable, or dangling (no
    stored summary, or a boundary id not present on the loaded tree)
    leaves the in-memory state unset -- fail-open, matching the design's
    rule that an inert/dangling boundary falls back to full history. A
    DANGLING boundary additionally best-effort clears the stale
    persisted pair (``set_conversation_context_summary(conversation_id,
    None, None)``, guarded + exception-swallowed like the write-through
    above) so a permanently-orphaned boundary (e.g. its branch was
    hard-deleted, or a foreign client rewrote history) doesn't linger in
    the DB row indefinitely -- benign either way (this path already
    fails open, and the next summarize overwrites it), just misleading
    to anything that reads the column directly.
    """
    session = self._sessions.get(session_id)
    conversation_id = session.persisted_conversation_id if session is not None else None
    if conversation_id is None:
        return
    persistence_db = getattr(self.persistence, "db", None)
    if persistence_db is None:
        return
    try:
        summary, boundary_persisted_id = (
            persistence_db.get_conversation_context_summary(conversation_id)
        )
    except Exception:  # noqa: BLE001 - preserve existing in-memory cursor fallback on storage errors.
        logger.bind(
            session_id=session_id,
            conversation_id=conversation_id,
        ).exception("Failed to read Console context-summary on resume; leaving unset.")
        return
    if summary is None or boundary_persisted_id is None:
        return
    boundary_native_id = persisted_to_native.get(boundary_persisted_id)
    if boundary_native_id is None:
        # Dangling: the persisted boundary message isn't on the loaded
        # tree. Leave the in-memory state unset (fail-open) AND
        # best-effort clear the now-permanently-orphaned persisted pair
        # so it doesn't linger indefinitely -- non-fatal, mirrors
        # ``_persist_context_summary``'s write-through guard.
        try:
            persistence_db.set_conversation_context_summary(conversation_id, None, None)
        except Exception:  # noqa: BLE001 - preserve existing in-memory cursor fallback on storage errors.
            logger.bind(
                session_id=session_id,
                conversation_id=conversation_id,
            ).exception(
                "Failed to clear stale Console context-summary with a "
                "dangling boundary; the persisted pair may linger."
            )
        return
    self._context_summary_by_session[session_id] = (summary, boundary_native_id)
