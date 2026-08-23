"""Background ``messages_fts`` backfill for upgraded ChaChaNotes DBs (task-21100).

The v45->v46 migration originally rebuilt the whole ``messages_fts`` index
(``'delete-all'`` + reinsert of every non-deleted message) inside the boot
path's single version-bump transaction -- an O(total chat text) index rewrite
that froze first paint for the duration on large profiles. The migration now
only clears the index; :func:`backfill_chachanotes_messages_fts` is the pure,
testable loop that drives ``CharactersRAGDB.backfill_messages_fts`` to
completion, and ``app.py`` wires it into a ``run_worker(thread=True)`` call at
mount (next to the structurally identical subscriptions backfill,
``Subscriptions/fts_backfill.py``) so the reinsert never blocks boot.

Search semantics during the window, chosen deliberately (task-21100 notes):
the index is empty-but-consistent after the upgrade commits and fills oldest
rowid first; message-content search returns progressively more history until
the backfill completes, never errors, and never returns tombstoned rows. New
and edited messages are indexed immediately by the (v47-guarded) triggers
regardless of backfill progress. A database opened by something that never
runs this driver (a script, a test) simply keeps its resumable frontier until
the next driver run -- the "not yet indexed" state lives in the database
itself (``messages_fts_docsize`` membership), not in any caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .ChaChaNotes_DB import CharactersRAGDB


class ChaChaNotesFTSBackfillError(RuntimeError):
    """Raised when the backfill loop fails partway through a run.

    Carries ``rows_indexed`` -- how many rows this run had already indexed
    (and committed; each chunk is its own transaction) before the failing
    chunk -- so a caller's error log can report real progress instead of just
    "it failed".
    """

    def __init__(self, rows_indexed: int) -> None:
        super().__init__(
            f"ChaChaNotes messages FTS backfill failed after indexing "
            f"{rows_indexed} row(s) this run"
        )
        self.rows_indexed = rows_indexed


def backfill_chachanotes_messages_fts(
    db: "CharactersRAGDB", chunk_size: int = 500
) -> int:
    """Index every live ``messages`` row missing from ``messages_fts``.

    ``CharactersRAGDB.backfill_messages_fts`` indexes at most ``chunk_size``
    rows per call and reports ``0`` once nothing remains past its cursor.
    Looping it to completion here is what makes the wired path resumable and
    idempotent: an interrupted run (app killed mid-loop, worker cancelled)
    leaves some rows unindexed for the next call to pick up, and a call made
    after completion does no writes at all. The ascending in-run cursor only
    skips rows this run has already walked past; a restart begins at 0 and is
    always correct, because rows below the cursor can only be indexed (never
    un-indexed while live) by the triggers.

    Args:
        db: The ``CharactersRAGDB`` instance to backfill. Thread-local
            connections make sharing the app's instance safe from a worker
            thread.
        chunk_size: Rows to index per underlying call.

    Returns:
        Total number of rows indexed by this call (``0`` if there was
        nothing left to index).

    Raises:
        ChaChaNotesFTSBackfillError: If the underlying call raises partway
            through the run. Wraps the original exception (``raise ... from``)
            and records how many rows this run had already indexed.
    """
    total = 0
    cursor = 0
    while True:
        try:
            indexed, cursor = db.backfill_messages_fts(
                chunk_size=chunk_size, after_rowid=cursor
            )
        except Exception as exc:
            raise ChaChaNotesFTSBackfillError(total) from exc
        if indexed == 0:
            break
        total += indexed
        logger.debug(
            "ChaChaNotes messages FTS backfill: indexed {} row(s) this chunk "
            "({} total so far).",
            indexed,
            total,
        )

    if total:
        logger.info(
            "ChaChaNotes messages FTS backfill complete: indexed {} "
            "pre-existing message(s).",
            total,
        )
    else:
        logger.debug("ChaChaNotes messages FTS backfill: nothing to index.")

    return total
