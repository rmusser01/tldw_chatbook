"""One startup sweep that un-wedges work an earlier process never finished.

Task-19561. Three tables in ``SubscriptionsDB`` carry an in-progress status
that only the process doing the work can ever move off: ``local_watchlist_
runs`` (``queued``/``running``), ``briefings`` (``generating``), and the
``briefing_scripts``/``briefing_audio`` rows hanging off them. If that
process stops existing -- ``SIGTERM``, a crash, a laptop losing power -- the
row keeps its in-progress status forever, and because several of those
statuses also act as one-at-a-time guards, the *feature* stays wedged, not
just the row.

**Why a startup sweep and not a shutdown reconcile.** A shutdown reconcile
only covers the terminations the process survives long enough to notice. Power
loss, ``SIGKILL`` and a hard crash all skip it by construction, and those are
precisely the cases that strand a row. Reconciling on the way *in* covers
every one of them, including the ones a graceful shutdown handles too --
which is why the graceful path does not need its own duplicate of this logic.
``AgentRunsDB.reconcile_orphaned_runs`` already establishes the pattern in
this codebase; this is the same contract for the subscriptions side.

**Why the sweep is bounded by a row-id boundary.** The first version of this
module argued it was safe to sweep unscoped, because "it runs once, during app
startup, before any claim can have been taken in this process". Independent
review (Qodo, PR #1972) showed that reasoning does not survive contact with
``app.py``: the scheduler worker starts inside ``on_mount``, and this sweep is
created *later*, as a deferred startup task after post-mount setup. The
scheduler's ``run()`` ticks immediately after loading its queue, so a due
watchlist check can launch a real ``queued``/``running`` row seconds before
the sweep runs -- and the sweep then failed it as "interrupted". That is not a
two-process corner case: it is single-process, every launch, and it was
reproduced end to end against a real scheduler (see
``Tests/Watchlists/test_startup_reconcile_scheduler_race.py``).

The fix is a boundary rather than an ordering rule. ``capture_prior_process_
boundary`` records the highest row id present in each table at the moment the
``SubscriptionsDB`` is opened -- which happens in ``TldwCli.__init__``, before
an event loop exists at all, let alone a scheduler -- and the sweep only
touches rows at or below it. All four tables are ``INTEGER PRIMARY KEY
AUTOINCREMENT``, which SQLite guarantees is strictly increasing and never
reused (the ``sqlite_sequence`` counter is not rolled back by a delete), so
*every* row this process creates is provably above the boundary and provably
out of the sweep's reach.

Ordering was the other candidate fix -- move the sweep ahead of the scheduler
and pin it with a test. It was rejected because it stays correct only while
nobody edits ``on_mount``, and ``on_mount`` is edited constantly. A boundary
captured before the loop exists cannot be undone by reordering anything that
happens after it, and the parameter is required, so the scoped call cannot
silently decay into the unscoped one.

**The remaining exposure, stated plainly.** A second tldw_chatbook started
while the first is mid-generation can still fail the first one's live rows,
because those rows already existed when the second process opened the database
and are therefore below its boundary. The boundary narrows that window rather
than closing it: rows the first instance creates *after* the second one
launches are now spared, where before they were fair game for the whole
startup. Closing it completely needs a process/owner marker on the row, which
is a schema change this task does not carry. It is the same exposure
``AgentRunsDB.reconcile_orphaned_runs`` shipped with, and the same one the
app's own second-instance warning exists to flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

__all__ = [
    "INTERRUPTED_RUN_ERROR",
    "PriorProcessBoundary",
    "capture_prior_process_boundary",
    "fail_interrupted_watchlist_runs",
    "reconcile_interrupted_subscription_work",
]

#: What a swept ``local_watchlist_runs`` row says happened to it. Distinct,
#: searchable text: "failed" alone would be indistinguishable from a check
#: that ran and genuinely failed.
INTERRUPTED_RUN_ERROR = "Interrupted: the application stopped before this run finished."

#: The two statuses a stopped process can strand. ``queued`` matters as much
#: as ``running``: a queued run is dispatched by an in-process worker that no
#: longer exists, so nothing will ever pick it up.
_UNFINISHED_RUN_STATUSES = ("queued", "running")

#: The four tables the sweep touches, in ``(result key, table)`` form. Every
#: one of them declares ``id INTEGER PRIMARY KEY AUTOINCREMENT``, which is
#: what makes a ``MAX(id)`` boundary sound rather than merely plausible:
#: AUTOINCREMENT keeps its counter in ``sqlite_sequence`` and never hands the
#: same id out twice, so a delete cannot let a later insert land at or below a
#: boundary captured earlier. A plain ``INTEGER PRIMARY KEY`` would reuse the
#: highest freed rowid and quietly break that guarantee.
_BOUNDED_TABLES = (
    ("runs", "local_watchlist_runs"),
    ("briefings", "briefings"),
    ("scripts", "briefing_scripts"),
    ("audio", "briefing_audio"),
)


@dataclass(frozen=True)
class PriorProcessBoundary:
    """The highest row id each table held before this process could write.

    Every field is the table's ``MAX(id)`` at capture time, or ``None``. The
    two ways to get ``None`` mean the same thing to the sweep and are both
    handled by not sweeping that table at all:

    * the table was empty, so there is nothing a previous process could have
      stranded in it; or
    * reading it raised, in which case declining to sweep leaves a row wedged
      (recoverable next launch) instead of failing a live one (not).
    """

    runs: int | None = None
    briefings: int | None = None
    scripts: int | None = None
    audio: int | None = None


def capture_prior_process_boundary(db: "SubscriptionsDB") -> PriorProcessBoundary:
    """Record where the previous process's rows end and this one's begin.

    Must be called before anything in this process can insert into these
    tables. In the app that is ``TldwCli.__init__``'s wiring, at the moment
    the ``SubscriptionsDB`` is constructed -- there is no event loop yet, so
    no scheduler, no handler and no UI action can have run.

    Synchronous and blocking (SQLite), and deliberately so: four ``MAX(id)``
    lookups against a primary-key index, on a connection the constructor has
    just opened anyway.

    Args:
        db: An open ``SubscriptionsDB``.

    Returns:
        The boundary. A table that cannot be read contributes ``None``, which
        excludes it from the sweep entirely rather than sweeping it unscoped.
    """
    boundaries: dict[str, int | None] = {}
    for key, table in _BOUNDED_TABLES:
        try:
            with db.transaction() as conn:
                row = conn.execute(f"SELECT MAX(id) AS max_id FROM {table}").fetchone()
            boundaries[key] = None if row is None else row["max_id"]
        except Exception as exc:  # noqa: BLE001 - a launch never dies on this
            logger.warning(
                f"Could not read the startup reconcile boundary for {table} "
                f"type={type(exc).__name__}; that table will not be swept."
            )
            boundaries[key] = None
    return PriorProcessBoundary(**boundaries)


def fail_interrupted_watchlist_runs(
    db: "SubscriptionsDB", max_row_id: int | None
) -> int:
    """Fail unfinished ``local_watchlist_runs`` rows up to ``max_row_id``.

    Only ``queued``/``running`` rows are touched -- finished history keeps
    its status, its stats and its own error text.

    Args:
        db: An open ``SubscriptionsDB``.
        max_row_id: The highest id this sweep may touch, from
            ``capture_prior_process_boundary``. Required, not defaulted: an
            unbounded sweep fails the live runs this process's own scheduler
            launched (Qodo, PR #1972), so the bound must be impossible to
            forget. ``None`` sweeps nothing at all.

    Returns:
        How many rows were failed.

    Raises:
        Exception: Re-raised from ``transaction()`` on any database error;
            the caller decides whether an unreconciled sweep is fatal (it is
            not -- see ``reconcile_interrupted_subscription_work``).
    """
    if max_row_id is None:
        return 0
    # Same timestamp shape LocalWatchlistsService writes (`_utc_now`, an
    # aware ISO-8601 string), NOT SQLite's CURRENT_TIMESTAMP -- the Runs pane
    # sorts and formats these as one column, so a swept row must not be the
    # only one in a different format.
    now = datetime.now(timezone.utc).isoformat()
    placeholders = ",".join("?" for _ in _UNFINISHED_RUN_STATUSES)
    with db.transaction() as conn:
        count = conn.execute(
            "UPDATE local_watchlist_runs "
            "SET status = 'failed', "
            "    error_msg = COALESCE(error_msg, ?), "
            "    finished_at = COALESCE(finished_at, ?), "
            "    updated_at = ? "
            f"WHERE status IN ({placeholders}) AND id <= ?",
            (
                INTERRUPTED_RUN_ERROR,
                now,
                now,
                *_UNFINISHED_RUN_STATUSES,
                int(max_row_id),
            ),
        ).rowcount
    if count:
        logger.info(f"failed {count} interrupted watchlist run(s)")
    return count


def reconcile_interrupted_subscription_work(
    db: "SubscriptionsDB", boundary: PriorProcessBoundary
) -> dict[str, int]:
    """Sweep the in-progress subscriptions rows a dead process left behind.

    Synchronous and blocking (SQLite); callers on the event loop must hop it
    onto a thread. Each sweep is contained separately: one failing table must
    not stop the other three from being reconciled, because a wedged guard in
    any one of them is a feature the user cannot use.

    Args:
        db: An open ``SubscriptionsDB``.
        boundary: From ``capture_prior_process_boundary``, taken before this
            process could insert anything. Required rather than optional so
            that no caller -- and no future edit to ``app.py``'s startup
            ordering -- can reach the unscoped sweep that failed live rows.

    Returns:
        Rows reconciled per table, keyed ``runs``/``briefings``/``scripts``/
        ``audio``. A key is absent if that sweep raised.
    """
    from tldw_chatbook.Subscriptions.briefing_audio import fail_interrupted_audio
    from tldw_chatbook.Subscriptions.briefing_cast import fail_interrupted_scripts
    from tldw_chatbook.Subscriptions.briefing_service import (
        fail_interrupted_briefings,
    )

    sweeps = (
        ("runs", lambda: fail_interrupted_watchlist_runs(db, boundary.runs)),
        (
            "briefings",
            lambda: (
                0
                if boundary.briefings is None
                else fail_interrupted_briefings(db, max_row_id=boundary.briefings)
            ),
        ),
        (
            "scripts",
            lambda: (
                0
                if boundary.scripts is None
                else fail_interrupted_scripts(db, max_row_id=boundary.scripts)
            ),
        ),
        (
            "audio",
            lambda: (
                0
                if boundary.audio is None
                else fail_interrupted_audio(db, max_row_id=boundary.audio)
            ),
        ),
    )
    reconciled: dict[str, int] = {}
    for name, sweep in sweeps:
        try:
            reconciled[name] = int(sweep() or 0)
        except Exception as exc:  # noqa: BLE001 - one table must not veto the rest
            logger.warning(
                f"Startup reconcile of interrupted {name} failed "
                f"type={type(exc).__name__}"
            )
    return reconciled
