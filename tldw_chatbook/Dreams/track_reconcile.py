"""Startup reconcile of Dream-owned watches with the persisted master switch.

``[dreams] enabled`` can change without the Settings toggle ever running:
the user edits ``config.toml`` by hand, or a switch was interrupted (or
made by a build that had none) and left an old state behind. Dreams'
boot wiring therefore asks, once, whether its watches are on the side of
the switch the config says NOW, and brings them there (review of #2921,
item 3c).

This module is deliberately light -- it is imported, and its one worker
started, on every boot, whatever the switch says: "off" includes a
``[dreams]`` section the user deleted, and that is exactly a state that
must be reconciled. So the cost is kept down inside the worker instead,
and this is all of it (one thread hop):

* Dreams off, no database file: one ``stat``; nothing is created.
* Dreams off, a database file: an ordinary open -- which initialises the
  schema in a write transaction and migrates a file from an older build
  -- then two small reads, then every connection is closed again. (A
  read-only look first would be cheaper, but the private-SQLite policy
  of this store's owner, ``db.base``, does not admit read-only targets;
  see ``DB/private_sqlite.py``.)
* Dreams on: the app's own handle (which the boot catch-up opens anyway,
  creating the database on first use) and the same two reads.

The Dreams service chain (``track_service`` and what it pulls in) is
imported only when there is work.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from tldw_chatbook.config import get_dreams_db_path, get_subscriptions_db_path
from tldw_chatbook.Dreams.settings import dreams_setting

#: ``settled`` values of a wrapper's suspension record (the record itself
#: is documented in ``track_service``). They live HERE because this
#: module answers the boot question from them and must stay light.
SETTLED_OFF = "off"  # every stop the record names has been applied
SETTLED_ON = "on"    # a resume restored all it may; the rest is held


def settled_state(raw: Any) -> str | None:
    """How a stored suspension record says its wrapper is settled.

    Args:
        raw: The ``dream_tracked_items.suspension`` text, or None.

    Returns:
        ``SETTLED_OFF``, ``SETTLED_ON``, or ``None`` for a wrapper with
        no record, an unfinished one, or one that cannot be read.
    """
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except (TypeError, ValueError):
        return None
    state = record.get("settled") if isinstance(record, dict) else None
    return state if state in (SETTLED_OFF, SETTLED_ON) else None


def _records_need_switch(records: list, *, enabled: bool) -> bool:
    """The rule behind :func:`needs_switch`, on records already read."""
    for raw in records:
        if enabled:
            if raw and settled_state(raw) != SETTLED_ON:
                return True
        elif settled_state(raw) != SETTLED_OFF:
            return True
    return False


def needs_switch(dreams_db: Any, *, enabled: bool) -> bool:
    """Whether any live watch is not settled where the switch says (sync).

    One read (``DreamsDB.live_tracked_suspensions``). With Dreams OFF a
    live wrapper needs work unless its record is settled "off" -- no
    record at all (a hand-edited config), or a stop that was recorded but
    never finished. With Dreams ON it needs work when it still carries a
    record that is not settled "on": a resume that never ran or could not
    finish. A wrapper held because its page is not a public address IS
    settled "on" and does not count -- it is retried by the next
    switch-on or re-track, not by every start (review of #2921, round 3,
    items T2/T3/T19).

    Args:
        dreams_db: A ``DreamsDB``.
        enabled: The master switch as the config has it now.

    Returns:
        True when a switch pass has something to do.
    """
    return _records_need_switch(
        dreams_db.live_tracked_suspensions(), enabled=enabled)


def _boot_db(app: Any) -> tuple[Any, bool, bool, bool]:
    """Find the Dreams DB and what the start owes it, if anything (sync).

    Runs on a worker thread: it may stat the profile and open SQLite.

    Returns:
        ``(dreams_db, owned, switch, backfill)``. ``dreams_db`` is None
        when there is nothing to do -- Dreams has no database, every live
        watch is settled on the side of the switch, and no subscription
        awaits its egress mark. ``owned`` is True when the handle was
        opened here and must be closed by the caller: with Dreams OFF the
        app-owned builder builds nothing (an off feature gets no storage
        handle), so an EXISTING file is opened directly. ``switch`` and
        ``backfill`` say which of the two jobs there is.
    """
    enabled = bool(dreams_setting("enabled"))
    owned = False
    if enabled:
        dreams_db = app.get_dreams_db()
    else:
        path = get_dreams_db_path()
        if not path.exists():
            return None, False, False, False
        from tldw_chatbook.DB.Dreams_DB import DreamsDB

        dreams_db = DreamsDB(path)
        owned = True
    if dreams_db is None:
        return None, False, False, False
    records, backfill = dreams_db.tracking_boot_state()
    switch = _records_need_switch(records, enabled=enabled)
    if switch or backfill:
        return dreams_db, owned, switch, backfill
    if owned:
        dreams_db.close_all()
    return None, False, False, False


class DreamsDispatchGate(asyncio.Event):
    """Bounded startup waiting and durable-work permission are separate facts."""

    def __init__(self, on_ready=None) -> None:
        super().__init__()
        self.dispatch_ready = False
        self.source_ids: frozenset[int] | None = None
        self._on_ready = on_ready

    def authorize(self) -> None:
        if not self.dispatch_ready:
            self.dispatch_ready = True
            if callable(self._on_ready):
                self._on_ready()

    def permits(self, task: dict, task_type: str) -> bool:
        if self.dispatch_ready:
            return True
        if task_type in {"dreams_cycle", "dream_track_check"}:
            return False
        if task.get("link_type") == "dream_tracked_item":
            return False
        if task_type != "watchlist_job":
            return True
        # A failed ownership read cannot classify a watchlist as unrelated.
        if self.source_ids is None:
            return False
        try:
            prefix, raw_id = str(task.get("id", "")).split(":", 1)
            return prefix == "watchlist" and int(raw_id) not in self.source_ids
        except (TypeError, ValueError):
            return False


def first_dispatch_gate(scheduler_loop: Any) -> asyncio.Event:
    """Make the scheduler's first dispatch wait for the boot reconcile.

    Review of #2921, round 3, T14a. The scheduler worker starts just
    before the reconcile. With Dreams switched off by hand in
    ``config.toml`` its watches are still enabled when the app starts,
    and an overdue Dreams source or reminder could be dispatched by the
    scheduler's first tick before the suspend reached it. The scheduler
    therefore waits -- bounded, see ``SchedulerLoop.first_dispatch_gates``
    -- for the event returned here, which :func:`reconcile_at_boot` sets
    as soon as its cheap check finds nothing that has to land first (the
    ordinary start: one thread hop), and otherwise when its work is done.
    A separate per-task guard remains after timeout/error until persisted
    obligations are verified; opening the event alone does not authorize work.

    Called synchronously from the app's Dreams wiring, in the slice that
    also starts the scheduler worker, so the gate is registered before
    either coroutine has run.

    Args:
        scheduler_loop: The app's ``SchedulerLoop`` (anything without a
            ``first_dispatch_gates`` list is left alone).

    Returns:
        The event to pass to :func:`reconcile_at_boot`.
    """
    gate = DreamsDispatchGate(getattr(scheduler_loop, "request_reload", None))
    guards = getattr(scheduler_loop, "dispatch_guards", None)
    if guards is None:
        guards = []
        scheduler_loop.dispatch_guards = guards
    guards.append(gate.permits)
    gates = getattr(scheduler_loop, "first_dispatch_gates", None)
    if isinstance(gates, list):
        gates.append(gate.wait)
    return gate


def _stores_may_run(app: Any, dreams_db: Any) -> bool:
    """Whether every store the reconcile writes is cleared for it (sync).

    The reconcile disables and re-arms reminders (the scheduled-tasks
    store), switches, stamps and re-rules subscriptions (the subscriptions
    store) and rewrites its own watches' records (the Dreams store), so it
    holds exactly where their own background work holds (review of #2921,
    round 3, T25; round 4, B5):

    * ``db.scheduled_tasks`` -- the question ``app.py`` asks before it
      starts the scheduler worker and ``SchedulerLoop.run`` asks before
      every tick;
    * ``db.subscriptions`` -- the question the subscriptions store's own
      boot work asks (``app_feature_glue._backfill_subscription_items_fts``).
      Round 3 asked only the first, although the egress backfill writes
      nothing but subscriptions;
    * ``db.dreams`` -- the store of the suspension and provenance records
      every pass writes.

    Each is false for a restored generation still awaiting the user's
    review (ADR-126), where a false answer "permits local inspection
    only". It is a check made once, before the work, as ``app.py`` makes
    its own -- not an ``activation.execution_scope`` retained through it.
    This pass is many awaits long: thread hops, watchlists service calls,
    name lookups. A storage maintenance pause (a backup capture) drains
    live storage leases before it proceeds and gives up at its deadline
    (``storage_admission._LocalPause.drain``), so holding a scope's lease
    through those awaits would make a capture wait on name lookups. The
    cost of checking once: a restore that begins after this check is not
    held off by it. File I/O; the caller thread-offloads it.

    Returns:
        True when every store there is to ask about is cleared for
        execution (a runtime without a scheduler store, or without the
        watchlists service, is not asked about that one).
    """
    from tldw_chatbook.Backup_Recovery import activation

    # Preserve the existing scheduler/Subscriptions checks and their order;
    # the captured Dreams store also owns durable suspension/provenance writes.
    stores: list[tuple[str, Any]] = []
    scheduler_db_path = getattr(
        getattr(getattr(app, "scheduler_loop", None), "db", None),
        "db_path", None)
    if scheduler_db_path is not None:
        stores.append(("db.scheduled_tasks", scheduler_db_path))
    if getattr(app, "local_watchlists_service", None) is not None:
        stores.append(("db.subscriptions", get_subscriptions_db_path()))
    stores.append(("db.dreams", getattr(dreams_db, "db_path", None)))
    return all(
        activation.execution_allowed((owner,), path) for owner, path in stores)


#: Boot passes in flight, owned HERE (review of #2921, round 4): a strong
#: reference, because a bare task is only weakly held by the event loop.
_BOOT_PASSES: set[asyncio.Task] = set()


async def reconcile_at_boot(
    app: Any, first_dispatch: asyncio.Event | None = None
) -> None:
    """Run the boot reconcile as a pass the calling worker cannot cut short.

    The switch pass inside it already runs as a task its caller cannot
    cancel (``track_service.reconcile_tracking``). The worker around it
    used to own the two things that must outlive a cancelled worker: the
    temporary handle of a Dreams that is off -- closed in the worker's
    ``finally`` while the pass was still using it, so the pass's later
    hops reopened connections nothing ever closed -- and the scheduler's
    first-dispatch gate, opened by a worker that was giving up while the
    suspend had not landed. Both now belong to the pass
    (:func:`_boot_pass`): the handle is closed, and the gate opened, when
    the pass is over.
    """
    task = asyncio.get_running_loop().create_task(
        _boot_pass(app, first_dispatch), name="dreams_boot_reconcile")
    _BOOT_PASSES.add(task)
    task.add_done_callback(_BOOT_PASSES.discard)
    await asyncio.shield(task)


async def _boot_pass(
    app: Any, first_dispatch: asyncio.Event | None = None
) -> None:
    """Bring Dream-owned watches in line with ``[dreams] enabled`` at boot.

    The body of :func:`reconcile_at_boot`, the app worker started from
    Dreams' scheduler wiring: off the UI thread, and never raising (an
    escaped error would take the app down). With Dreams off and a watch
    still running -- a hand-edited
    config -- the watch is suspended; with Dreams on and a watch still
    carrying an unsettled suspension record -- an old "off" state, a
    switch that was cut short -- it is resumed. The common boot finds
    nothing to do and stops after one read.

    The same read says whether any subscription Dreams created still
    awaits its egress mark (``track_service.backfill_egress_marks``, T1);
    that one-time backfill runs from here too, whatever the switch says.

    When there IS work, it is held by the gates the scheduler honours
    (T25, :func:`_stores_may_run`): it waits out storage maintenance,
    and does nothing while a store it is about to write is not cleared
    for execution.

    Args:
        app: The running app; read for ``get_dreams_db``,
            ``local_watchlists_service``, ``scheduling_service.db`` and
            ``scheduler_loop.request_reload``, each tolerated missing.
        first_dispatch: The gate from :func:`first_dispatch_gate`, if the
            scheduler is waiting on one. It is set as soon as nothing
            that must precede the first dispatch is left -- at once when
            there is nothing to suspend and no source to mark -- and in
            every case when this returns. That wake-up never grants per-task
            permission after an error; the separate dispatch guard persists.
    """
    try:
        dreams_db, owned, switch, backfill = await asyncio.to_thread(
            _boot_db, app)
        # What has to land BEFORE anything is dispatched: stopping the
        # watches of a Dreams that is off, and the egress-mark backfill
        # (T1 -- written when an unmarked source was a trusted egress
        # origin; since ADR-207 it is fetched under the public-only rule
        # anyway). A resume has no such claim.
        must_precede_dispatch = backfill or (
            switch and not bool(dreams_setting("enabled")))
        if first_dispatch is not None and not must_precede_dispatch:
            if isinstance(first_dispatch, DreamsDispatchGate):
                first_dispatch.authorize()
            first_dispatch.set()
        if dreams_db is None:
            return
        try:
            if isinstance(first_dispatch, DreamsDispatchGate):
                first_dispatch.source_ids = frozenset(
                    await asyncio.to_thread(dreams_db.dream_owned_subscription_ids))
            # Only now: there is something to do -- if the scheduler may.
            # While storage maintenance (a backup capture) has admission
            # closed the scheduler waits, and so does this; where the
            # scheduler would not run at all, this changes nothing and
            # the next start finds the same work (T25).
            wait_for_admission = getattr(
                getattr(app, "scheduler_loop", None),
                "wait_for_maintenance_admission", None)
            if wait_for_admission is not None and not (
                    await wait_for_admission()):
                # The scheduler was stopped while the capture was still
                # open: nothing may be written now.
                logger.info(
                    "Dreams startup reconcile held: storage maintenance "
                    "was still under way when the scheduler stopped")
                return
            if not await asyncio.to_thread(_stores_may_run, app, dreams_db):
                logger.info(
                    "Dreams startup reconcile held: a store it writes is "
                    "not cleared for execution")
                return
            from tldw_chatbook.Dreams.track_service import (
                backfill_egress_marks,
                reconcile_tracking,
            )

            subs_service = getattr(app, "local_watchlists_service", None)
            # There is work, so the subscriptions are about to be read
            # anyway: the mark is checked against every Dreams source's
            # REAL options, not only against the marker in this store
            # (round 4, B4 -- the two live in different stores, and only
            # the subscriptions store is backed up).
            await backfill_egress_marks(subs_service, dreams_db, verify=True)
            if switch:
                await reconcile_tracking(
                    subs_service,
                    dreams_db,
                    scheduling_db_getter=lambda: getattr(
                        getattr(app, "scheduling_service", None), "db", None),
                    on_settled=getattr(
                        getattr(app, "scheduler_loop", None),
                        "request_reload", None),
                )
            # Reconcile deliberately contains per-write failures. Returning
            # from it does not prove that the persisted obligations settled.
            records, pending_marks = await asyncio.to_thread(dreams_db.tracking_boot_state)
            if (isinstance(first_dispatch, DreamsDispatchGate)
                    and not pending_marks
                    and not _records_need_switch(records, enabled=bool(dreams_setting("enabled")))):
                first_dispatch.authorize()
        finally:
            if owned:
                # Every connection, not just one thread's (T7): the hops
                # above each opened their own on an executor thread.
                await asyncio.to_thread(dreams_db.close_all)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a worker body must not raise
        logger.warning(
            "Dreams startup reconcile failed: {}", type(exc).__name__)
    finally:
        if first_dispatch is not None:
            first_dispatch.set()
