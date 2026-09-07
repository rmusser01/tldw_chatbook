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

Pacing (TASK-22215, closing the half of TASK-22200's "backfills yield to
foreground work" that its ChaChaNotes sibling did not cover): this loop also
ran its chunks back to back, so on a profile with a large pre-index backlog it
held ``subscriptions.db``'s write lock essentially continuously while the app
was already serving screens -- and every watchlist ingest write, item status
change and briefing write is also ``BEGIN IMMEDIATE`` on that database. It now
sleeps :data:`~tldw_chatbook.DB.fts_backfill_pacing.INTER_CHUNK_PAUSE_SECONDS`
after every chunk that indexed rows, and the wait is abort-sliced so a
cancelled worker (app quit) stops within one slice instead of waiting out the
pause. Stopping early is safe for the same reason a crash is: the frontier is
``subscription_items_fts_docsize`` membership in the database itself.

Deliberately NOT copied from the ChaChaNotes driver: its lock-queue retry
schedule. That exists because chat writes race the messages backfill
constantly; here a chunk that loses the queue after the connection's own busy
timeout is rare enough that the existing fail-and-resume-next-boot contract is
still the honest behavior, and inventing a retry loop without a measurement to
size it would be the "clever, unstable" half of the owner's ruling.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from loguru import logger

from ..DB.fts_backfill_pacing import INTER_CHUNK_PAUSE_SECONDS, interruptible_sleep
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


def backfill_subscription_items_fts(
    db: SubscriptionsDB,
    chunk_size: int = 500,
    *,
    pause_seconds: float = INTER_CHUNK_PAUSE_SECONDS,
    should_abort: Optional[Callable[[], bool]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
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
        pause_seconds: Yield this long after every chunk that indexed rows, so
            foreground writers can take the write lock in the gap. ``0``
            restores the old back-to-back loop (tests, one-shot scripts).
        should_abort: Polled between chunks and inside every pause; when it
            returns True the run stops and returns what it indexed so far. The
            app passes the Textual worker's ``is_cancelled``, so quitting does
            not wait out a pause.
        sleep: Sleep implementation, injected by tests.

    Returns:
        Total number of rows indexed by this call (``0`` if there was
        nothing left to index, or if the run was aborted before its first
        chunk).

    Raises:
        FTSBackfillError: If the underlying ``backfill_items_fts`` call
            raises partway through the run. Wraps the original exception
            (via ``raise ... from``) and records how many rows this run had
            already indexed, so a caller logging the failure can report
            real progress rather than a bare "it failed".
    """
    total = 0
    while True:
        if should_abort is not None and should_abort():
            logger.debug(
                "Subscription items FTS backfill aborted after {} row(s); "
                "the remaining frontier resumes on the next run.",
                total,
            )
            return total
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
        # Only a chunk that did work pays the pause: the every-boot no-op
        # probe (one indexed scan finding nothing) must stay free.
        if interruptible_sleep(pause_seconds, should_abort, sleep):
            logger.debug(
                "Subscription items FTS backfill aborted mid-pause after {} "
                "row(s); the remaining frontier resumes on the next run.",
                total,
            )
            return total

    if total:
        logger.info(
            "Subscription items FTS backfill complete: indexed {} "
            "pre-existing row(s).",
            total,
        )
    else:
        logger.debug("Subscription items FTS backfill: nothing to index.")

    return total
