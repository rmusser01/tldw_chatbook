"""Handler for scheduled watchlist/subscription check tasks."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
from tldw_chatbook.Subscriptions.local_watchlists_service import (
    EXECUTABLE_SOURCE_TYPES,
    LocalWatchlistsService,
)
from tldw_chatbook.Subscriptions.monitoring_engine import FeedMonitor, URLMonitor

_WATCHLIST_TASK_PREFIX = "watchlist"

# Shadow mode only -- see `_check_in_shadow`. The real path asks
# `EXECUTABLE_SOURCE_TYPES` instead.
_SHADOW_FEED_TYPES = ("rss", "atom", "json_feed", "podcast")
_SHADOW_URL_TYPES = ("url", "url_list")

_STATUS_SUCCESS = "success"
_STATUS_ERROR = "error"
_STATUS_SKIPPED = "skipped"
_STATUS_MISSING = "missing"
_STATUS_UNKNOWN_TYPE = "unknown_type"


class WatchlistCheckHandler:
    """Execute a scheduled watchlist check as a real, visible watchlist run.

    The handler is stateless: all persistent subscription state lives in
    ``SubscriptionsDB``.

    TASK-1383. This used to be a parallel reimplementation of
    ``LocalWatchlistsService``'s execution path, and it sank its results
    exclusively into ``SubscriptionsDB.record_check_result`` -- which writes
    ``subscription_stats``, a daily aggregate whose only reader
    (``get_subscription_health``) has no callers at all. The Watchlists Runs
    pane reads ``local_watchlist_runs``, and only ``launch_run`` ever wrote to
    that table, so a source checked *only* by the scheduler -- the normal
    unattended case this feature exists for -- produced no record on the one
    screen built to show what a check did.

    Being a second implementation, it had also drifted into two outright bugs
    that are fixed here by deletion rather than by patching:

    * its URL type tuple omitted ``sitemap``, so scheduled sitemap sources hit
      the "unknown subscription type" branch and were never checked at all;
    * ``url_list`` was passed whole to a single ``check_url`` call, so a
      scheduled 50-URL source checked exactly one URL.

    Routing through ``launch_run`` + ``execute_run`` yields the run row, its
    ``stats_json`` dispositions (TASK-1362), per-URL baselines (TASK-1361),
    filters and alerts for free, and leaves this class with the job that is
    genuinely its own: parsing the task id, resolving and gating the
    subscription, and reporting scheduler metrics.

    ``shadow_mode`` keeps a deliberate, separate fork; see ``_check_in_shadow``.
    """

    def __init__(
        self,
        subscriptions_db: SubscriptionsDB,
        feed_monitor: FeedMonitor | None = None,
        url_monitor: URLMonitor | None = None,
        shadow_mode: bool = False,
        watchlists_service: LocalWatchlistsService | None = None,
    ) -> None:
        """Initialize the handler.

        Args:
            subscriptions_db: Persistent subscription store used to read
                subscriptions and record run results/errors.
            feed_monitor: Monitor for RSS/Atom/JSON feed checks, used by the
                shadow path only. A default ``FeedMonitor`` is created when
                ``None``.
            url_monitor: Monitor for URL change checks, used by the shadow path
                only. A default ``URLMonitor`` bound to ``subscriptions_db`` is
                created when ``None``.
            shadow_mode: When ``True``, execute checks without mutating
                ``subscriptions_db`` and emit metrics with a ``shadow`` label.
            watchlists_service: The service that owns run execution. A default
                bound to ``subscriptions_db`` is created when ``None``.
        """
        self.subscriptions_db = subscriptions_db
        self.feed_monitor = feed_monitor if feed_monitor is not None else FeedMonitor()
        self.url_monitor = (
            url_monitor
            if url_monitor is not None
            else URLMonitor(db=subscriptions_db, persist_snapshots=not shadow_mode)
        )
        self.shadow_mode = shadow_mode
        # A constructor parameter rather than a lookup inside `handle`, for the
        # same reason `feed_monitor`/`url_monitor` are: a test can hand in a
        # service wired to a stub `run_executor` without monkeypatching a
        # module-level name, and production wiring (`app.py`) stays zero-config
        # because the default binds to the db it was already given.
        self.watchlists_service = (
            watchlists_service
            if watchlists_service is not None
            else LocalWatchlistsService(db_factory=lambda: self.subscriptions_db)
        )

    async def handle(self, task: dict[str, Any]) -> None:
        """Process a single watchlist check task.

        Args:
            task: Projected scheduled task dict from ``WatchlistProjection``.
        """
        start_time = time.time()
        subscription_id: int | None = None
        subscription_type = "unknown"
        status = _STATUS_MISSING
        run_id: Any = None

        try:
            subscription_id = self._parse_subscription_id(task.get("id"))
            if subscription_id is None:
                return

            subscription = self.subscriptions_db.get_subscription(subscription_id)
            if subscription is None:
                logger.warning(f"Subscription {subscription_id} not found")
                return

            subscription_type = subscription.get("type", "unknown")

            if subscription.get("is_paused") or not subscription.get("is_active"):
                logger.info(f"Skipping paused/inactive subscription {subscription_id}")
                status = _STATUS_SKIPPED
                return

            logger.info(
                f"Checking subscription '{subscription.get('name')}' "
                f"(ID: {subscription_id})"
            )

            if self.shadow_mode:
                if not await self._check_in_shadow(subscription, subscription_type):
                    status = _STATUS_UNKNOWN_TYPE
                    return
                status = _STATUS_SUCCESS
                return

            if subscription_type not in EXECUTABLE_SOURCE_TYPES:
                # Kept as an early return rather than letting `execute_run`
                # raise, so the scheduler metric still distinguishes "this type
                # has no executor" from "the check failed" -- the TASK-1212
                # distinction. `sitemap` is inside this set now; it never was
                # in the tuple this replaced.
                logger.warning(f"Unknown subscription type: {subscription_type}")
                status = _STATUS_UNKNOWN_TYPE
                return

            launched = await self.watchlists_service.launch_run(
                source_id=subscription_id
            )
            run_id = launched["run_id"]
            # `execute_run` records its own result -- including
            # `record_check_result` -- and does not re-raise a fetch failure, so
            # this handler must neither record a second time nor expect an
            # exception for the ordinary failure case.
            executed = await self.watchlists_service.execute_run(run_id)
            run_status = str(executed.get("status") or "")
            status = _STATUS_ERROR if run_status == "failed" else _STATUS_SUCCESS
            stats = executed.get("stats") or {}
            logger.info(
                f"Subscription check complete: '{subscription.get('name')}' - "
                f"run {run_id} {run_status or 'completed'}, "
                f"{stats.get('new_items_found', 0)} new items"
            )

        except Exception as exc:
            status = _STATUS_ERROR
            logger.error(f"Error checking subscription {subscription_id}: {exc}")
            await self._record_failure(subscription_id, run_id, exc)

        finally:
            duration = time.time() - start_time
            labels: dict[str, Any] = {
                "status": status,
                "subscription_type": subscription_type,
            }
            if self.shadow_mode:
                labels["shadow"] = "true"
            log_counter("watchlist_checks", labels=labels)
            log_histogram("watchlist_check_duration", duration, labels=labels)

    async def _record_failure(
        self, subscription_id: int | None, run_id: Any, exc: BaseException
    ) -> None:
        """Persist a failure that escaped ``execute_run``'s own handling.

        Only failures *around* execution reach here -- ``launch_run`` raising,
        or ``execute_run`` failing before its internal ``try`` (a subscription
        deleted between the two calls). A fetch failure is already recorded by
        ``execute_run``.

        Auto-pause parity: the run-failure path ends in
        ``SubscriptionsDB.record_check_error`` (``local_watchlists_service.py``
        ``:492``), which is the exact call this handler used to make itself, so
        ``consecutive_failures`` is bumped identically either way. Calling it
        again here would double-count it, so the ``run_id`` branch does not.
        """
        if self.shadow_mode or subscription_id is None:
            return
        if run_id is not None:
            # A row already exists at `queued`/`running`; leaving it there is
            # the TASK-1090 failure. `record_run_failure` marks it failed *and*
            # calls `record_check_error`.
            try:
                await self.watchlists_service.record_run_failure(
                    run_id, source_id=subscription_id, error=exc
                )
                return
            except Exception:
                logger.opt(exception=True).warning(
                    f"Watchlists: could not mark scheduled run {run_id} failed; "
                    f"falling back to recording the error on the subscription."
                )
        self.subscriptions_db.record_check_error(subscription_id, str(exc))

    async def _check_in_shadow(
        self, subscription: dict[str, Any], subscription_type: str
    ) -> bool:
        """Probe a subscription without writing anything. The deliberate fork.

        Shadow mode (``[scheduling] watchlist_checks_shadow``) exists to prove
        the scheduler wiring end to end -- projection, queue, dispatch, fetch --
        on an installation where a real check must not touch the database. It
        therefore cannot go through ``launch_run``/``execute_run`` at all: those
        exist precisely to write a run row, items and snapshots.

        So this stays a direct-monitor call, and stays deliberately coarser than
        the real path: it probes one URL for a ``url_list`` and does not
        enumerate a ``sitemap``. That is a diagnostic's fidelity, not a check's,
        and it is safe here only because nothing it returns is persisted.

        Args:
            subscription: The subscription row to probe.
            subscription_type: Its ``type``.

        Returns:
            ``True`` when the type was probed, ``False`` when it has no shadow
            arm (reported as ``unknown_type``).
        """
        if subscription_type in _SHADOW_FEED_TYPES:
            items = await self.feed_monitor.check_feed(subscription)
        elif subscription_type in _SHADOW_URL_TYPES:
            result, _disposition = await self.url_monitor.check_url(subscription)
            items = [result] if result is not None else []
        else:
            logger.warning(f"Unknown subscription type: {subscription_type}")
            return False
        logger.info(
            f"Shadow check complete: '{subscription.get('name')}' - "
            f"{len(items)} item(s) observed, nothing written"
        )
        return True

    def _parse_subscription_id(self, task_id: Any) -> int | None:
        """Extract the numeric subscription id from a ``watchlist:<id>`` task id."""
        if not isinstance(task_id, str) or ":" not in task_id:
            logger.warning(f"Invalid watchlist task id: {task_id!r}")
            return None

        prefix, raw_id = task_id.split(":", 1)
        if prefix != _WATCHLIST_TASK_PREFIX:
            logger.warning(f"Invalid watchlist task id prefix: {task_id!r}")
            return None

        try:
            return int(raw_id)
        except ValueError:
            logger.warning(f"Invalid watchlist subscription id: {task_id!r}")
            return None

    async def __call__(self, task: dict[str, Any]) -> None:
        """Allow the handler to be invoked directly by the scheduler loop."""
        await self.handle(task)
