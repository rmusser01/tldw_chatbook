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

**Why it is safe to sweep unscoped.** The existing UI-gated sweeps take great
care to spare rows belonging to a *live, in-process* generation, via
``exclude``/``exclude_watchlists`` snapshots of the claim registry. This one
needs none of that: it runs once, during app startup, before any claim can
have been taken in this process, so every in-progress row it can see belongs
to a process that no longer exists.

**The one exposure, stated plainly.** A second tldw_chatbook started while the
first is mid-generation will fail the first one's live rows. That is the same
exposure ``AgentRunsDB.reconcile_orphaned_runs`` shipped with, and the same
one the app's own second-instance warning exists to flag. It fails a row that
is genuinely running rather than leaving a row that is genuinely dead; the
work is re-runnable either way, and the alternative (never sweeping) leaves
the guard shut forever.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

__all__ = [
    "INTERRUPTED_RUN_ERROR",
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


def fail_interrupted_watchlist_runs(db: "SubscriptionsDB") -> int:
    """Fail every unfinished ``local_watchlist_runs`` row; return the count.

    Only ``queued``/``running`` rows are touched -- finished history keeps
    its status, its stats and its own error text.

    Args:
        db: An open ``SubscriptionsDB``.

    Returns:
        How many rows were failed.

    Raises:
        Exception: Re-raised from ``transaction()`` on any database error;
            the caller decides whether an unreconciled sweep is fatal (it is
            not -- see ``reconcile_interrupted_subscription_work``).
    """
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
            f"WHERE status IN ({placeholders})",
            (INTERRUPTED_RUN_ERROR, now, now, *_UNFINISHED_RUN_STATUSES),
        ).rowcount
    if count:
        logger.info(f"failed {count} interrupted watchlist run(s)")
    return count


def reconcile_interrupted_subscription_work(db: "SubscriptionsDB") -> dict[str, int]:
    """Sweep every in-progress subscriptions row left by a dead process.

    Synchronous and blocking (SQLite); callers on the event loop must hop it
    onto a thread. Each sweep is contained separately: one failing table must
    not stop the other three from being reconciled, because a wedged guard in
    any one of them is a feature the user cannot use.

    Args:
        db: An open ``SubscriptionsDB``.

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
        ("runs", lambda: fail_interrupted_watchlist_runs(db)),
        ("briefings", lambda: fail_interrupted_briefings(db)),
        ("scripts", lambda: fail_interrupted_scripts(db)),
        ("audio", lambda: fail_interrupted_audio(db)),
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
