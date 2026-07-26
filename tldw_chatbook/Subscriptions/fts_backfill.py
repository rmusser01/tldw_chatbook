"""Background FTS backfill for pre-existing subscription_items rows (task-688).

``SubscriptionsDB.backfill_items_fts`` (task-1a / Phase A) is chunked and
resumable, but nothing called it: the ``subscription_items_fts`` index is
created empty over a table that may already hold rows scraped before the
index existed, and only the insert/update triggers populate it going
forward. Without this wiring, every item scraped before upgrading stays
permanently unsearchable even though the search UI looks like it works.

:func:`backfill_subscription_items_fts` is the pure, testable core of the
fix -- it just drives ``backfill_items_fts`` to completion. ``app.py`` wires
it into a ``run_worker(thread=True)`` call at startup so a large backlog
never blocks app boot or screen mount; see
``TldwCli._backfill_subscription_items_fts``.
"""

from __future__ import annotations

from loguru import logger

from ..DB.Subscriptions_DB import SubscriptionsDB


class FTSBackfillError(RuntimeError):
    """Raised when the backfill loop fails partway through a run.

    Carries ``rows_indexed`` -- how many rows this run had already indexed
    (and committed; each chunk is its own transaction) before the failing
    chunk -- so a caller's error log can report real progress instead of
    just "it failed", without needing to duplicate the loop to track that
    count itself.
    """

    def __init__(self, rows_indexed: int) -> None:
        super().__init__(
            f"Subscription items FTS backfill failed after indexing "
            f"{rows_indexed} row(s) this run"
        )
        self.rows_indexed = rows_indexed


def backfill_subscription_items_fts(db: SubscriptionsDB, chunk_size: int = 500) -> int:
    """Index every pre-existing ``subscription_items`` row missing from FTS.

    ``SubscriptionsDB.backfill_items_fts`` indexes at most ``chunk_size`` rows
    per call and returns ``0`` once nothing remains. Looping it to completion
    here is what makes the *wired* path resumable and idempotent, not just
    the underlying method in isolation: the "not yet indexed" state lives in
    the database itself (the ``subscription_items_fts_docsize`` shadow
    table), not in any counter kept by this function or its caller, so an
    interrupted run (app killed mid-loop, worker cancelled, ...) simply
    leaves some rows unindexed for the next call to pick up, and a call made
    after completion does no writes at all rather than re-indexing or
    corrupting anything.

    Args:
        db: The ``SubscriptionsDB`` instance to backfill.
        chunk_size: Rows to index per underlying ``backfill_items_fts`` call.

    Returns:
        Total number of rows indexed by this call (``0`` if there was
        nothing left to index).

    Raises:
        FTSBackfillError: If the underlying ``backfill_items_fts`` call
            raises partway through the run. Wraps the original exception
            (via ``raise ... from``) and records how many rows this run had
            already indexed, so a caller logging the failure can report
            real progress rather than a bare "it failed".
    """
    total = 0
    while True:
        try:
            indexed = db.backfill_items_fts(chunk_size=chunk_size)
        except Exception as exc:
            raise FTSBackfillError(total) from exc
        if indexed == 0:
            break
        total += indexed
        logger.debug(
            "Subscription items FTS backfill: indexed {} row(s) this chunk "
            "({} total so far).",
            indexed,
            total,
        )

    if total:
        logger.info(
            "Subscription items FTS backfill complete: indexed {} "
            "pre-existing row(s).",
            total,
        )
    else:
        logger.debug("Subscription items FTS backfill: nothing to index.")

    return total
