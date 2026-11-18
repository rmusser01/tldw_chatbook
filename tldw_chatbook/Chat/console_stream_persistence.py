"""Stream folding and deferred writes under the native Chat store owner."""

from __future__ import annotations

import threading
from collections.abc import Iterator

from loguru import logger

from tldw_chatbook.Chat.console_chat_models import ConsoleChatMessage


def _fold_stream_buffer_without_persistence(
    self,
    message: ConsoleChatMessage,
) -> bool:
    """Fold buffered stream chunks into ``message.content`` without a write.

    TASK-259: after joining, the chunk list is collapsed to the single
    joined string (in place, preserving any outstanding list references),
    so the next materialize joins only the chunks that arrived since this
    one instead of re-walking the whole stream history on every 0.2s
    tick. The invariant ``"".join(buffer) == full streamed content`` is
    preserved for every reader.

    Args:
        message: Store-owned message whose visible content should
            reflect all chunks appended so far.

    Returns:
        Whether new chunks were folded into ``message.content``.
    """
    buffer = self._stream_chunks_by_message.get(message.id)
    if not buffer:
        return False
    if self._stream_materialized_counts.get(message.id) == len(buffer):
        return False
    message.content = "".join(buffer)
    buffer[:] = [message.content]
    self._stream_materialized_counts[message.id] = 1
    self._bump_message_speech_revision(message.id)
    return True


def visible_message_texts(store, session_id: str) -> Iterator[str]:
    """Yield a chat's visible message text lazily, copying no message.

    TASK-38688: the text ``messages_for_session`` would show, for a reader that
    needs only text: no fold, no deferred write, no per-message snapshot.
    Buffered stream text is the joined buffer (the TASK-259 invariant), and a
    quarantined message yields nothing, as its snapshot shows a placeholder.

    Raises:
        KeyError: If ``session_id`` does not identify a stored session.
    """
    store._session_or_raise(session_id)
    for message in store._messages_by_session[session_id]:
        if message.generation_projection_quarantined:
            continue
        buffer = store._stream_chunks_by_message.get(message.id)
        if buffer and store._stream_materialized_counts.get(message.id) != len(buffer):
            yield "".join(buffer)
        else:
            yield message.content


def _materialize_stream_buffer(self, message: ConsoleChatMessage) -> None:
    """Fold buffered chunks and persist a newly materialized pending row."""
    if self._fold_stream_buffer_without_persistence(message):
        if message.id in self._pending_terminal_receipts:
            # A receipt is only valid as part of its terminal row+mark
            # transaction. Leave the folded body live until that owner
            # persists it atomically rather than writing an orphan first.
            return
        self._persist_pending_message_if_ready(message)


def _materialize_stream_buffer_deferred(self, message: ConsoleChatMessage) -> None:
    """Fold buffered chunks inline and persist the pending row off-thread.

    task-33081: read paths running on the UI event loop (the 0.2s Console
    tick via ``messages_for_session``) must not write SQLite inline --
    under writer contention the inline pending-row INSERT blocks the
    whole loop for the connection's busy timeout. The fold itself is
    in-memory so readers still observe streamed text immediately; only
    the write moves to the store's single-slot persistence executor.
    """
    if self._fold_stream_buffer_without_persistence(message):
        if message.id in self._pending_terminal_receipts:
            return
        self._defer_pending_message_persistence(message)


def _defer_pending_message_persistence(self, message: ConsoleChatMessage) -> None:
    if self.persistence is None:
        return
    with self._stream_persistence_deferred_lock:
        if self._stream_persistence_executor_closed:
            return
        if message.id in self._stream_persistence_deferred_ids:
            return
        self._stream_persistence_deferred_ids.add(message.id)
    try:
        self._stream_persistence_executor.submit(
            self._persist_deferred_message, message
        )
    except RuntimeError:
        # Teardown raced the submit: the store is retiring, so this
        # pending write is moot and must never surface on the read path.
        with self._stream_persistence_deferred_lock:
            self._stream_persistence_deferred_ids.discard(message.id)


def _persist_deferred_message(self, message: ConsoleChatMessage) -> None:
    with self._stream_persistence_deferred_lock:
        self._stream_persistence_deferred_ids.discard(message.id)
    try:
        self._persist_pending_message_if_ready(message)
    except KeyError:
        # The message's session was retired between submit and run --
        # nothing durable remains to write for it.
        logger.debug(
            "Deferred stream persistence skipped a retired message ({})",
            message.id,
        )
    except Exception:  # noqa: BLE001 - deferred writer reports failure without crashing the UI reader.
        logger.exception(
            "Deferred stream persistence failed for message {}", message.id
        )


def flush_deferred_stream_persistence(self) -> None:
    """Block until every queued deferred stream write has completed.

    Test and teardown seam for the off-thread pending-row writes
    (task-33081): callers that must observe durable state immediately
    after a fold-driven read drain the executor here.
    """
    with self._stream_persistence_deferred_lock:
        executor = self._stream_persistence_executor
    if executor is None:  # pragma: no cover - always constructed today
        return
    barrier = threading.Event()
    executor.submit(barrier.set).add_done_callback(lambda _f: barrier.set())
    barrier.wait(timeout=30.0)
