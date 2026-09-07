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

Pacing (task-22200): the loop originally ran its ``BEGIN IMMEDIATE`` chunks
back to back, so an upgrading user's whole first session contended with a
write-lock convoy -- every UI write (also ``BEGIN IMMEDIATE``, 15 s busy
timeout) had to race the next chunk's begin for the lock. The driver now
sleeps :data:`INTER_CHUNK_PAUSE_SECONDS` after every chunk that indexed rows,
bounding the backfill's write-lock duty cycle (measured ~3-5 ms of chunk work
per 100 ms pause on a default-size chunk) so a foreground writer acquires the
lock inside the gap instead. A chunk that itself dies on SQLite's plain
lock-queue timeout (``database is locked`` after the busy handler expires --
the RETRYABLE kind; the non-retryable snapshot-upgrade form only exists for
DEFERRED read-then-write transactions, which the IMMEDIATE chunk is not) is
retried through the bounded :data:`_LOCKED_RETRY_BACKOFF_SECONDS` schedule
instead of killing the run until the next boot. Both sleeps are cut at
:data:`_ABORT_POLL_SECONDS` granularity when the caller provides
``should_abort`` -- app shutdown must interrupt an in-flight pause, because a
cancel that only sets a flag cannot cut a plain ``time.sleep``.

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

import sqlite3
import time
from typing import Callable, Optional, TYPE_CHECKING

from loguru import logger

from .fts_backfill_pacing import (
    # Re-exported, not dead: this module is the documented home of the pacing
    # contract, and TASK-22200's tests import the slice size from here.
    ABORT_POLL_SECONDS as _ABORT_POLL_SECONDS,  # noqa: F401
    INTER_CHUNK_PAUSE_SECONDS,
    interruptible_sleep as _interruptible_sleep,
)

if TYPE_CHECKING:
    from .ChaChaNotes_DB import CharactersRAGDB

# TASK-22215 moved the two pacing primitives above into
# `DB/fts_backfill_pacing.py` so the subscription_items backfill (task-688)
# could adopt the same rules instead of a second copy of them. Names, values
# and semantics are unchanged; they are re-exported here because this module
# is where TASK-22200 built and documented them.

#: Escalating waits before retrying a chunk that lost the lock queue. Each
#: failed attempt already sat out the connection's 15 s busy handler, so
#: these are deliberately short -- the point is to survive one slow
#: foreground transaction, not to poll a wedged database forever.
_LOCKED_RETRY_BACKOFF_SECONDS = (0.5, 1.0, 2.0)


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


def _is_lock_queue_timeout(exc: BaseException) -> bool:
    """True when ``exc`` (or its cause chain) is SQLite's plain busy/locked
    timeout -- the contention signal the backoff schedule exists for. Walks
    the chain because the DB layer sometimes wraps ``sqlite3`` errors in
    ``CharactersRAGDBError`` (``raise ... from``)."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, sqlite3.OperationalError):
            message = str(current).lower()
            if "database is locked" in message or "database is busy" in message:
                return True
        current = current.__cause__ or current.__context__
    return False


def backfill_chachanotes_messages_fts(
    db: "CharactersRAGDB",
    chunk_size: int = 500,
    *,
    pause_seconds: float = INTER_CHUNK_PAUSE_SECONDS,
    should_abort: Optional[Callable[[], bool]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Index every live ``messages`` row missing from ``messages_fts``.

    ``CharactersRAGDB.backfill_messages_fts`` indexes at most ``chunk_size``
    rows per call and reports ``0`` once nothing remains past its cursor.
    Looping it to completion here is what makes the wired path resumable and
    idempotent: an interrupted run (app killed mid-loop, worker cancelled,
    ``should_abort`` fired) leaves some rows unindexed for the next call to
    pick up, and a call made after completion does no writes at all. The
    ascending in-run cursor only skips rows this run has already walked past;
    a restart begins at 0 and is always correct, because rows below the
    cursor can only be indexed (never un-indexed while live) by the triggers.

    Pacing (task-22200): after every chunk that indexed rows the loop sleeps
    ``pause_seconds`` so foreground ``BEGIN IMMEDIATE`` writers acquire the
    lock between chunks; a chunk lost to the busy-handler timeout is retried
    through ``_LOCKED_RETRY_BACKOFF_SECONDS`` before the run gives up. The
    no-work path (first chunk finds nothing -- every boot after completion)
    performs no sleep at all, so the boot probe stays one indexed scan.

    Args:
        db: The ``CharactersRAGDB`` instance to backfill. Thread-local
            connections make sharing the app's instance safe from a worker
            thread.
        chunk_size: Rows to index per underlying call.
        pause_seconds: Sleep between chunks. ``0`` disables pacing (tests,
            offline tools with exclusive access).
        should_abort: Polled between chunks and inside every sleep; when it
            returns True the run stops cleanly, returning the rows indexed so
            far. Stopping is not failing -- the frontier lives in the DB and
            the next run resumes it.
        sleep: Injection seam for the pacing/backoff waits (tests).

    Returns:
        Total number of rows indexed by this call (``0`` if there was
        nothing left to index; may be partial if ``should_abort`` fired).

    Raises:
        ChaChaNotesFTSBackfillError: If the underlying call raises partway
            through the run (after exhausting the lock-timeout retries, for
            contention errors). Wraps the original exception
            (``raise ... from``) and records how many rows this run had
            already indexed.
    """
    total = 0
    cursor = 0
    locked_retries = 0
    while True:
        if should_abort is not None and should_abort():
            logger.info(
                "ChaChaNotes messages FTS backfill stopping on abort signal "
                "after {} row(s); the next run resumes from the database's "
                "own frontier.",
                total,
            )
            return total
        try:
            indexed, cursor = db.backfill_messages_fts(
                chunk_size=chunk_size, after_rowid=cursor
            )
        except Exception as exc:
            if (
                _is_lock_queue_timeout(exc)
                and locked_retries < len(_LOCKED_RETRY_BACKOFF_SECONDS)
            ):
                backoff = _LOCKED_RETRY_BACKOFF_SECONDS[locked_retries]
                locked_retries += 1
                logger.warning(
                    "ChaChaNotes messages FTS backfill chunk lost the lock "
                    "queue (attempt {}/{}); backing off {}s before retrying.",
                    locked_retries,
                    len(_LOCKED_RETRY_BACKOFF_SECONDS),
                    backoff,
                )
                if _interruptible_sleep(backoff, should_abort, sleep):
                    logger.info(
                        "ChaChaNotes messages FTS backfill stopping on abort "
                        "signal during contention backoff after {} row(s).",
                        total,
                    )
                    return total
                continue
            raise ChaChaNotesFTSBackfillError(total) from exc
        locked_retries = 0
        if indexed == 0:
            break
        total += indexed
        logger.debug(
            "ChaChaNotes messages FTS backfill: indexed {} row(s) this chunk "
            "({} total so far).",
            indexed,
            total,
        )
        if _interruptible_sleep(pause_seconds, should_abort, sleep):
            logger.info(
                "ChaChaNotes messages FTS backfill stopping on abort signal "
                "during the inter-chunk pause after {} row(s).",
                total,
            )
            return total

    if total:
        logger.info(
            "ChaChaNotes messages FTS backfill complete: indexed {} "
            "pre-existing message(s).",
            total,
        )
    else:
        logger.debug("ChaChaNotes messages FTS backfill: nothing to index.")

    return total
