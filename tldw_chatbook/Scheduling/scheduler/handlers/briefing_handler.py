"""Handler for scheduled per-watchlist briefing generation tasks."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
from tldw_chatbook.Scheduling.services.briefing_projection import parse_briefing_task_id
from tldw_chatbook.Subscriptions.briefing_keep import KeepRefused, keep_briefing
from tldw_chatbook.Subscriptions.briefing_service import (
    STATUS_COMPLETE,
    STATUS_EMPTY,
    STATUS_FAILED,
    GenerationInFlightError,
    active_briefing_claims,
    generate_briefing,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

#: What `_run_generation` measures: this label fires once, in the `finally`,
#: whenever the spawned coroutine ran to completion without either of the
#: two exception branches below catching something -- i.e. `generate_briefing`
#: returned (whatever status it wrote to the row: complete, empty, or its own
#: internally-handled `failed`). It does NOT mean "was dispatched": nothing is
#: recorded at the moment `handle` spawns the task, only when it finishes, so
#: a generation still in flight -- or one whose process is killed before this
#: `finally` runs -- is never counted here at all. Review round 1 caught the
#: previous name (`"dispatched"`) claiming the opposite.
_STATUS_COMPLETED = "completed"
_STATUS_SKIPPED_CLAIMED = "skipped_claimed"
_STATUS_RACED = "raced"
_STATUS_ERROR = "error"


class BriefingJobHandler:
    """Fire-and-forget scheduled briefing generation.

    Locked Decision 3 of the briefings phase 4 plan: `SchedulerLoop.tick`
    awaits every handler serially, inline (`loop.py`), and a briefing
    generation is a multi-minute LLM call where a watchlist check is a
    quick HTTP fetch. Awaiting `generate_briefing` here would stall every
    other due task -- reminders, watchlist checks, and every other briefing
    -- behind whichever provider is slowest. `handle` therefore does only
    synchronous, in-memory work (the claim snapshot check) and spawns the
    generation as an independent `asyncio.Task`, returning before it has had
    a chance to run at all. The one DB read the generation itself needs
    (`default_briefing_preset_id`) is deliberately NOT done in `handle` --
    see `_default_preset_id`'s own docstring for why it moved inside the
    spawned task instead (review round 1).

    The stateless-handler shape follows `WatchlistCheckHandler`: all
    persistent state lives in `SubscriptionsDB` and in `briefing_service`'s
    own in-process claim registry, not on this object -- except for the
    strong references to in-flight generation tasks below, which exist
    purely so a spawned task cannot be garbage-collected mid-flight (a bare
    `asyncio.create_task` result with no other reference is only weakly
    held by the event loop).

    Task-19561 gave that same set a second job. Because these tasks are not
    Textual workers, they were invisible to app shutdown, which only ever
    cancelled `App.workers` -- so quitting destroyed a live generation
    mid-flight instead of cancelling it. `shutdown()` (below) is the seam
    `app.py`'s `on_unmount` now calls to reach them while the loop is still
    running.

    Task 3 (kept-briefings, task-1780): once a spawned generation resolves
    `complete`, `_run_generation` auto-mirrors it into ChaChaNotes via
    `briefing_keep.keep_briefing(..., origin="scheduled")` -- see
    `_auto_keep`'s own docstring for the full containment story. This is
    strictly best-effort: an absent ChaChaNotes handle or any failure from
    the keep service never touches the generation outcome or the
    `briefings` row.

    Review round 1 (Task 3): the ChaChaNotes handle is taken as a
    zero-arg **getter**, resolved fresh inside `_auto_keep` every time a
    generation completes -- never a plain instance captured once at
    construction time. `app.py` constructs this handler (in
    `_wire_watchlists_and_notifications_services`) BEFORE its own
    `self.chachanotes_db` attribute is assigned later in `__init__`;
    capturing the instance directly at that point would freeze `None`
    into this handler for the rest of the process's life, making
    auto-keep permanently inert in production even once
    `self.chachanotes_db` exists -- "wired but never live", the exact
    recurring bug class this stream keeps catching. A getter
    (`lambda: getattr(self, "chachanotes_db", None)`) sidesteps
    construction order entirely: every keep attempt re-reads whatever the
    attribute currently holds.
    """

    def __init__(
        self,
        subscriptions_db: Any,
        generate: Callable[..., Awaitable[dict[str, Any]]] = generate_briefing,
        chachanotes_db_getter: Callable[[], CharactersRAGDB | None] | None = None,
        dispatch_service: Any | None = None,
        notification_app_getter: Callable[[], Any | None] | None = None,
        incident_recorder: Any | None = None,
    ) -> None:
        """Initialize the handler.

        Args:
            subscriptions_db: The `SubscriptionsDB` briefings and watchlist
                settings are read from and written to. Shared with the
                manual "Generate" button's own instance in production
                (`app.py`), since the in-process claim registry is only
                effective when every caller shares one process -- sharing
                the db instance is not itself what makes the claim work,
                but keeping a second, divergent connection around invites
                exactly the kind of drift this stream keeps finding.
            generate: The generation seam. Defaults to
                `briefing_service.generate_briefing`; tests inject a fake
                to control timing and outcome without touching the real
                claim registry's generation path.
            chachanotes_db_getter: A zero-arg callable returning an open
                `CharactersRAGDB`, or `None`, or `None` itself (the
                default). Called fresh every time a generation resolves
                `complete` (`_auto_keep`), never once at construction time
                -- see the class docstring's "Review round 1" paragraph
                for why that distinction is load-bearing, not stylistic.
                A getter that itself returns `None` (genuinely no
                ChaChaNotes handle available yet, or ever), and the bare
                `None` default (no getter configured at all -- every
                existing test that does not care about auto-keep), both
                simply disable auto-keep for that attempt; nothing about
                generation itself depends on this parameter.
            dispatch_service: `dispatch_service`/`notification_app_getter`
                follow the same optional-collaborator discipline as
                `chachanotes_db_getter`: absent means headless/tests and
                every notification path is a no-op. When given (production
                wires `NotificationDispatchService` here, the same seam
                `ReminderHandler` takes), one `category="briefing"`
                notification is dispatched per generation completion or
                failure -- never for a claim-race skip. Notification
                failures are contained exactly like `_auto_keep`'s.
            notification_app_getter: The app is a *getter* for the same
                late-binding reason `chachanotes_db` is: passed as `app=`
                to `dispatch` at call time (the dispatch service only
                uses it for transient toast delivery, never persistence),
                resolved fresh on every notification.
        """
        self.subscriptions_db = subscriptions_db
        self._generate = generate
        self._chachanotes_db_getter = chachanotes_db_getter
        self.dispatch_service = dispatch_service
        self._notification_app_getter = notification_app_getter
        # TASK-26027: optional failure-incident recorder. None keeps
        # today's behavior (every failure notifies). When present it
        # groups repeat failures of one watchlist's brief by error
        # signature and suppresses re-notification for a grouped/acked
        # incident; a success closes the incident.
        self._incident_recorder = incident_recorder
        #: Strong references to spawned generation tasks, keyed by nothing
        #: in particular -- a plain set, discarded from on completion. See
        #: the class docstring for why this exists at all.
        self._pending_generations: set[asyncio.Task[Any]] = set()

    async def handle(self, task: dict[str, Any]) -> None:
        """Process one scheduled briefing task.

        Parses the task id, refuses a watchlist that already has a live
        in-process claim (a re-emitted job for a generation still running
        from an earlier tick, or from a concurrent manual "Generate"
        press), and otherwise spawns the generation and returns
        immediately -- never awaiting it.

        Args:
            task: Projected scheduled task dict from `BriefingProjection`.
        """
        watchlist_id = parse_briefing_task_id(task.get("id"))
        if watchlist_id is None:
            logger.warning(f"Invalid briefing task id: {task.get('id')!r}")
            return

        if watchlist_id in active_briefing_claims():
            # Not an error: the projection re-emits this job every reload
            # until the watchlist's `last_completed_at` moves, and a
            # multi-minute generation can easily span several polls. The
            # claim -- held by this same run, or by a concurrent manual
            # Generate press -- is the refusal; queuing a second attempt
            # would only race `generate_briefing`'s own guard for nothing.
            logger.info(
                f"Skipping scheduled briefing for watchlist {watchlist_id}: "
                f"a generation is already in flight for it."
            )
            log_counter(
                "briefing_schedule_runs", labels={"status": _STATUS_SKIPPED_CLAIMED}
            )
            return

        spawned = asyncio.create_task(
            self._run_generation(watchlist_id),
            name=f"briefing_generation_watchlist_{watchlist_id}",
        )
        self._pending_generations.add(spawned)
        spawned.add_done_callback(self._pending_generations.discard)

    async def shutdown(self, *, timeout: float = 5.0) -> int:
        """Cancel and settle every generation this handler still owns.

        Task-19561. `handle` deliberately spawns generations as bare
        `asyncio.Task`s rather than Textual workers (Locked Decision 3 --
        a multi-minute LLM call must not stall the scheduler tick), which
        also means they are absent from `App.workers`, the only collection
        app shutdown was cancelling. They therefore survived teardown as
        detached tasks whose event loop was about to close underneath
        them: asyncio's "Task was destroyed but it is pending", a `generating`
        row nobody would move again, and any write in flight abandoned
        wherever it happened to be. Shutdown now reaches them through this
        method; `app.py`'s `on_unmount` calls it while the loop is still
        alive, so the cancellation is actually delivered and awaited.

        The row a cancelled generation leaves behind stays `generating`, on
        purpose: writing a terminal status from inside a cancellation, on a
        loop that is closing, is exactly the racing background write
        `local_watchlists_service` documents as trading one stale row for a
        stale row plus a destroyed task. The startup sweep
        (`Subscriptions/startup_reconcile.py`) reconciles it instead, which
        also covers the terminations no shutdown hook can ever run for.

        Idempotent, and safe to call with nothing in flight.

        Args:
            timeout: Seconds to wait for the cancelled tasks to settle
                before giving up on them. Exceeding it is logged, never
                raised -- a shutdown must not fail on this.

        Returns:
            How many in-flight generations were cancelled.
        """
        pending = [task for task in self._pending_generations if not task.done()]
        if not pending:
            return 0
        for task in pending:
            task.cancel()
        # `asyncio.wait`, NOT `wait_for(gather(...))`: on expiry `wait_for`
        # cancels what it is waiting on and then awaits that cancellation,
        # so a task that swallows `CancelledError` hangs the very call whose
        # timeout was supposed to bound it. `wait` just returns and reports.
        _, unsettled = await asyncio.wait(pending, timeout=timeout)
        for task in pending:
            if task.done() and not task.cancelled():
                # Retrieve any exception so a cancelled-at-shutdown task
                # cannot surface as "exception was never retrieved".
                task.exception()
        if unsettled:
            logger.warning(
                f"{len(unsettled)} scheduled briefing generation(s) did not "
                f"settle within {timeout}s of cancellation."
            )
        return len(pending)

    def _default_preset_id(self, watchlist_id: int) -> int | None:
        """The watchlist's stored `default_briefing_preset_id`, or `None`.

        Matches `watchlists_collections_screen._read_watchlist_briefing_
        settings`'s own read of the same column -- there is no
        service-layer getter for it either. That method's own docstring is
        explicit: "Always called through `asyncio.to_thread`; never call
        this directly from the UI thread" -- and `_run_generation` (the
        only caller, review round 1) follows the same rule, NOT
        `WatchlistCheckHandler.handle`, which calls the service method
        `get_subscription()`, not raw `.conn` SQL, and reads a table this
        handler's own spawned generations never write to concurrently.
        Both distinctions matter here: `SubscriptionsDB` waits up to
        `Subscriptions_DB.BUSY_TIMEOUT_MS` (5 s) for a contended write --
        pinned explicitly by task-19562, previously the inherited sqlite3
        default, and measured rather than assumed -- and THIS handler's
        own `generate_briefing` calls write to `watchlists`'/`briefings`'
        shared connection from `asyncio.to_thread` workers -- so a direct,
        synchronous call here could block on a lock its own spawned work
        is holding, self-inflicting exactly the tick stall Locked Decision
        3 exists to prevent. Being inside the spawned task (never inside
        `handle`) means a wait here only delays this one generation's own
        start, never the scheduler tick.

        The read runs inside `with subscriptions_db.transaction() as
        conn:`, not a bare `subscriptions_db.conn.execute` (Qodo rule
        1011851: every accessor this stream has shipped goes through
        `transaction()`, reads included, so rollback-on-exception is
        consistently wired even for read paths).
        """
        with self.subscriptions_db.transaction() as conn:
            row = conn.execute(
                "SELECT default_briefing_preset_id FROM watchlists WHERE id = ?",
                (watchlist_id,),
            ).fetchone()
        if row is None:
            return None
        return row["default_briefing_preset_id"]

    async def _run_generation(self, watchlist_id: int) -> None:
        """Run one generation to completion, containing every failure.

        This coroutine is the whole body of the spawned task, so nothing
        it does may raise: an exception escaping a task nobody awaits
        becomes asyncio's "Task exception was never retrieved" -- an
        unhandled-exception event this handler must never produce.
        `generate_briefing` already turns provider failures into `failed`
        rows and returns normally (Task 1's contract), so the two branches
        below are for the cases that are NOT already a row: losing the
        in-process claim race, and any other exception escaping the
        service (documented as possible for a database error -- see
        `generate_briefing`'s own docstring). Neither is logged with
        anything beyond the exception's type name: briefing content must
        never reach a log line, and a database error's own message could
        embed a query fragment carrying it.

        The `default_briefing_preset_id` read lives here, off the event
        loop (`asyncio.to_thread`), rather than in `handle` -- see
        `_default_preset_id`'s own docstring. A DB error surfacing from
        that read is contained by the same `except Exception` branch below
        as a DB error from `generate_briefing` itself; either way, nothing
        escapes this task.

        Auto-keep (Task 3) runs from the `else` clause below, i.e. only
        once generation itself has already resolved without raising --
        deliberately outside the `try`, so a failure inside `_auto_keep`
        (already fully contained by its own docstring's terms) can never
        be mistaken by the `except` branches here for a generation
        failure, and can never flip `status` away from `_STATUS_COMPLETED`.
        """
        start = time.time()
        status = _STATUS_COMPLETED
        try:
            preset_id = await asyncio.to_thread(self._default_preset_id, watchlist_id)
            result = await self._generate(
                self.subscriptions_db, watchlist_id, preset_id=preset_id
            )
        except GenerationInFlightError:
            # Lost the race between this handler's claim snapshot and
            # `generate_briefing`'s own atomic check-then-add: harmless,
            # since the other caller's generation is the one of record.
            status = _STATUS_RACED
            logger.debug(
                f"Scheduled briefing for watchlist {watchlist_id} lost the "
                f"generation claim race to another in-process caller."
            )
        except Exception as exc:  # noqa: BLE001 - must never escape uncaught
            status = _STATUS_ERROR
            logger.warning(
                f"Scheduled briefing generation for watchlist {watchlist_id} "
                f"failed outside the service's own handling: "
                f"{type(exc).__name__}"
            )
            await self._notify_error(
                watchlist_id, signature=f"{type(exc).__name__}: {exc}"
            )
        else:
            await self._auto_keep(result)
            await self._notify_result(watchlist_id, result)
        finally:
            duration = time.time() - start
            log_counter("briefing_schedule_runs", labels={"status": status})
            log_histogram(
                "briefing_schedule_duration", duration, labels={"status": status}
            )

    async def _auto_keep(self, briefing_row: dict[str, Any]) -> None:
        """Mirror a just-finished scheduled generation into ChaChaNotes.

        Spec's Keep-service "Auto path": a scheduled generation that
        resolved `complete` is mirrored with `origin="scheduled"` so it
        survives the watchlist being deleted later, even if the user never
        presses the (future) manual Keep button. Every other case is a
        deliberate, silent no-op:

        - `briefing_row["status"] != STATUS_COMPLETE` (`empty` or
          `failed`): returned before any thread hop, getter call, or DB
          call -- reading `generate_briefing`'s own returned row directly,
          never re-querying it (that row already knows its own status).
        - `self._chachanotes_db_getter` is `None`, or calling it returns
          `None`: no ChaChaNotes handle available for this attempt (see
          the class/`__init__` docstrings for why this is a getter,
          resolved HERE and not once at construction time); logged at
          DEBUG.
        - `keep_briefing` itself raises `KeepRefused`: the belt-and-braces
          case -- this method's own status check above already keeps
          `empty`/`failed` rows from ever reaching `keep_briefing`, but the
          service refuses independently too (e.g. a `complete` row whose
          body reads back blank), so this is treated exactly as
          expected-and-benign, not an error.
        - Any other exception from `keep_briefing` (or from calling the
          getter itself): logged with `type(exc).__name__` only -- never
          a message, which could embed briefing content or a query
          fragment -- and swallowed.

        This coroutine is called from `_run_generation`'s `else` clause
        (i.e. after generation already resolved without raising), and must
        itself never raise: nothing it does may alter `_run_generation`'s
        own `status` var, retroactively look like a generation failure, or
        touch the `briefings` row (`keep_briefing` only reads
        `subscriptions_db`; every write it makes lands in whatever the
        getter returned). A lost mirror costs nothing permanent --
        `keep_briefing` is additive-idempotent, so the next scheduled run
        for this watchlist re-keeps whatever this attempt missed.

        Args:
            briefing_row: Whatever `self._generate` returned for this run
                -- the finished `briefings` row as a dict, per
                `generate_briefing`'s own contract (Task 1).
        """
        try:
            if briefing_row.get("status") != STATUS_COMPLETE:
                return
            chachanotes_db = (
                self._chachanotes_db_getter()
                if self._chachanotes_db_getter is not None
                else None
            )
            if chachanotes_db is None:
                logger.debug(
                    f"Skipping auto-keep for briefing "
                    f"{briefing_row.get('id')!r}: no ChaChaNotes handle "
                    f"available for this handler right now."
                )
                return
            briefing_id = briefing_row["id"]
            try:
                await asyncio.to_thread(
                    keep_briefing,
                    self.subscriptions_db,
                    chachanotes_db,
                    briefing_id,
                    origin="scheduled",
                )
            except KeepRefused as exc:
                logger.debug(
                    f"Auto-keep refused for briefing {briefing_id}: "
                    f"{type(exc).__name__}"
                )
        except Exception as exc:  # noqa: BLE001 - must never escape uncaught
            logger.warning(
                f"Auto-keep for a scheduled briefing failed outside the "
                f"keep service's own handling: {type(exc).__name__}"
            )

    async def _notify_result(self, watchlist_id: int, result: dict[str, Any]) -> None:
        """Dispatch one completion notification for a finished generation.

        No-op without a dispatch service; never raises (same containment rule
        as `_auto_keep` -- a notification failure must never surface as a
        scheduling failure).
        """
        if self.dispatch_service is None:
            return
        try:
            status = str(result.get("status") or "")
            if status not in (STATUS_COMPLETE, STATUS_EMPTY, STATUS_FAILED):
                return
            # TASK-26027 AC#4: a completed brief resolves any open incident
            # for this watchlist, so a later recurrence alerts afresh.
            if status == STATUS_COMPLETE:
                self._close_incident(watchlist_id)
            name = await asyncio.to_thread(self._watchlist_name, watchlist_id)
            briefing_id = result.get("id")
            if status == STATUS_COMPLETE:
                title = "Daily brief ready"
                message = f"{name} finished its scheduled brief."
                severity = "information"
            else:
                title = "Daily brief needs attention"
                error = str(result.get("error") or "").strip()
                message = (
                    f"{name} finished its scheduled brief with status "
                    f"'{status}'" + (f": {error}" if error else "") + "."
                )
                severity = "warning"
            app = (
                self._notification_app_getter()
                if self._notification_app_getter is not None
                else None
            )
            self.dispatch_service.dispatch(
                app=app,
                category="briefing",
                title=title,
                message=message,
                severity=severity,
                source_entity_kind="briefing",
                source_entity_id=(
                    str(briefing_id) if briefing_id is not None else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Briefing completion notification for watchlist "
                f"{watchlist_id} failed: {type(exc).__name__}"
            )

    def _should_notify_failure(
        self, watchlist_id: int, signature: str | None
    ) -> bool:
        """Record the failure incident and decide whether to notify (26027)."""
        recorder = self._incident_recorder
        if recorder is None or signature is None:
            return True
        try:
            from datetime import datetime, timezone

            from tldw_chatbook.Scheduling.task_incidents import (
                normalize_error_signature,
            )

            _incident_id, should_notify = recorder.record_task_failure(
                f"briefing:{watchlist_id}",
                "briefing_job",
                normalize_error_signature(signature),
                datetime.now(timezone.utc),
            )
            return bool(should_notify)
        except Exception:  # noqa: BLE001 -- incident failure never blocks alerting
            logger.opt(exception=True).debug("incident record_failure failed")
            return True

    def _close_incident(self, watchlist_id: int) -> None:
        """Resolve any open incident for this watchlist on success (26027)."""
        recorder = self._incident_recorder
        if recorder is None:
            return
        try:
            from datetime import datetime, timezone

            recorder.record_task_success(
                f"briefing:{watchlist_id}", datetime.now(timezone.utc)
            )
        except Exception:  # noqa: BLE001 -- never blocks the success path
            logger.opt(exception=True).debug("incident record_success failed")

    async def _notify_error(
        self, watchlist_id: int, signature: str | None = None
    ) -> None:
        """Dispatch one attention notification for a crashed generation.

        TASK-26027: when an incident recorder is wired and a signature is
        given, a repeat of the same failure (grouped or acknowledged) is
        recorded but NOT re-notified -- only the first of a signature, or a
        recurrence after a resolving success, alerts.
        """
        if self.dispatch_service is None:
            return
        if not self._should_notify_failure(watchlist_id, signature):
            return
        try:
            name = await asyncio.to_thread(self._watchlist_name, watchlist_id)
            app = (
                self._notification_app_getter()
                if self._notification_app_getter is not None
                else None
            )
            self.dispatch_service.dispatch(
                app=app,
                category="briefing",
                title="Daily brief failed",
                message=(
                    f"{name}'s scheduled brief failed outside the briefing "
                    "service's own handling. See the Watchlists artifacts "
                    "pane for the failed row."
                ),
                severity="error",
                source_entity_kind="watchlist",
                source_entity_id=str(watchlist_id),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Briefing error notification for watchlist {watchlist_id} "
                f"failed: {type(exc).__name__}"
            )

    def _watchlist_name(self, watchlist_id: int) -> str:
        """The watchlist's name, or a stable fallback (same read pattern as
        `_default_preset_id`)."""
        with self.subscriptions_db.transaction() as conn:
            row = conn.execute(
                "SELECT name FROM watchlists WHERE id = ?", (watchlist_id,)
            ).fetchone()
        if row is None:
            return f"Watchlist {watchlist_id}"
        return str(row["name"] or f"Watchlist {watchlist_id}")

    async def __call__(self, task: dict[str, Any]) -> None:
        """Allow the handler to be invoked directly by the scheduler loop."""
        await self.handle(task)
