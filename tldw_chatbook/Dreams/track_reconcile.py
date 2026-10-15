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
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

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


#: ``link_type`` of the reminders Dreams' watches own (the track service's
#: ``_REMINDER_LINK_TYPE``; written out here so this module stays light).
_REMINDER_LINK_TYPE = "dream_tracked_item"


def _parsed(raw: Any) -> dict | None:
    """A stored suspension record as a dict; None when there is none."""
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _url_key(url: Any) -> tuple[str, str, str, str]:
    """A URL compared the way Dreams wrote it: scheme and host case-folded,
    no fragment, no trailing slash on the path."""
    parts = urlsplit(str(url or "").strip())
    return (parts.scheme.lower(), parts.netloc.lower(),
            parts.path.rstrip("/"), parts.query)


def is_dreams_source(
    processing_options: Any, source_url: Any, story_urls: Iterable[Any],
) -> bool:
    """Whether a subscription is still the source a Dreams watch created.

    Review of #3123, M1. ``subscriptions.id`` is AUTOINCREMENT and a
    restore puts the table's id counter back too: restore only the
    subscriptions from a backup older than a watch, and the user's next
    new source takes the id Dreams' source had, while the Dreams store
    still names that id as Dreams' own. So a Dreams path that changes a
    source by id asks this first, from durable data on both sides, and a
    "no" means the watch's source is gone -- handled like a deleted one,
    and the source with that id is never touched:

    * its stored ``url_provenance`` is not ``user_entered`` (Dreams marks
      what it creates as nobody's typing; a source the user typed, or
      whose address the user changed, is theirs);
    * its URL is the URL of a story Dreams made it for (Dreams writes the
      story's URL into the source it creates; stories are never deleted).
      A watch with no story leaves only the first test.

    Args:
        processing_options: The source's stored options (text or dict).
        source_url: The source's URL.
        story_urls: The URLs of the stories of its Dreams watches.
    """
    options = processing_options
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except ValueError:
            options = None
    # ``Utils.egress.URL_PROVENANCE_OPTION`` / ``UrlProvenance.USER_ENTERED``,
    # written out so this module stays light.
    if isinstance(options, dict) and options.get("url_provenance") == "user_entered":
        return False
    urls = [url for url in story_urls if url]
    return not urls or any(_url_key(source_url) == _url_key(url) for url in urls)


def _changed_before(changed: Any, at: Any) -> bool:
    """Whether a row last changed no later than a record was written.

    Review of #3123, M2. Users can switch a reminder or a change rule back
    on themselves (Schedules, Watchlists) -- unlike a source -- and the
    start check must not undo that. Both rows carry ``updated_at``, and a
    record the switch-off or a retirement wrote carries ``at``: a row
    changed after the record is the user's; one changed no later is as
    Dreams left it, or as a restore put it back. Either time missing or
    unreadable (a record an earlier build wrote) is "cannot tell" -- not
    taken for a restore.
    """
    changed_at, written_at = _moment(changed), _moment(at)
    return changed_at is not None and written_at is not None and (
        changed_at <= written_at)


def changed_after(changed: Any, at: Any) -> bool:
    """Whether a row demonstrably changed after a record was written.

    Review of #3123, N4: the user's change, which a re-track must not undo.
    Either time missing or unreadable is "cannot tell" -- False.
    """
    changed_at, written_at = _moment(changed), _moment(at)
    return changed_at is not None and written_at is not None and (
        changed_at > written_at)


def _moment(value: Any) -> datetime | None:
    """An ISO 8601 time as an aware UTC-assumed datetime, or None."""
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


class _StoreReadRefused(RuntimeError):
    """A store refused the start check's read (review of #3123, N3).

    For instance ``RecoveryRequired`` from a restored profile still
    awaiting review. Carries the store and the type of the refusal only.
    """

    def __init__(self, store: str, cause: BaseException) -> None:
        super().__init__(store)
        self.store = store
        self.reason = type(cause).__name__


@dataclass
class _Drift:
    """What a restore put out of step, as the work that brings it back.

    Attributes:
        records: ``(tracked_item_id, record_text)`` writes on live watches,
            for the switch pass to take up.
        sources: Dreams-created sources of retired watches to switch off.
        reminders: Reminders of retired watches to disarm.
    """

    records: list[tuple[int, str]] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)
    reminders: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.records or self.sources or self.reminders)


def _restore_drift(
    rows: list[dict],
    gone: list[dict],
    run_states: dict[int, dict],
    rule_states: dict[int, dict],
    reminders: list[dict],
    *,
    enabled: bool,
) -> _Drift:
    """Where the stores disagree with Dreams' records (pure).

    TASK-34924 (owner decision 2026-10-14, a check at every start). A
    backup restores the subscriptions and scheduling stores apart from
    the Dreams store, so a restore can bring back sources, rules and
    reminders that run while Dreams' records say they are stopped -- or
    stopped while the records owe them nothing. What it finds:

    * Dreams off, a live watch settled "off" whose Dreams-created source
      runs, whose Dreams rule is enabled, or whose reminder is armed: its
      record is written back unsettled, and the switch-off pass stops
      what runs and settles it again. A source runs when ``is_active`` is
      set -- an auto-paused one too, which Watchlists' Resume restarts;
      Resume clears ``is_paused`` only, so a source the switch disabled
      while it was paused (``is_active`` 0, ``is_paused`` 1) does not.
    * Dreams off, a retired watch whose retirement record says Dreams
      switched its source off (and no live watch rides it) while that
      source runs, or lists a reminder Dreams disarmed that is armed
      again: that source is switched off and that reminder disarmed.
    * Dreams on, a live watch that Dreams' own source rides (``created_by_
      dreams``), on a source that exists and is off, while the watch's
      record owes nothing for it: the record gets a restart debt, and the
      switch-on pass restarts it after its public-address check.

    Everything but that restart follows a record. The restart reads "off"
    as a restore's doing only for a source Dreams created under a live
    watch: nothing in Watchlists or the agent tools switches an existing
    source off today. A future control that does must leave a mark this
    check can read, or a start would turn the source back on. A rule or
    reminder that is off under a live watch with Dreams on is left alone:
    the user may have switched it off, and no record says otherwise.

    Pure; the reads are the caller's.
    """
    armed: dict[str, list[dict]] = {}
    for reminder in reminders:
        if reminder.get("enabled") and not reminder.get("last_run_at"):
            armed.setdefault(str(reminder.get("link_id")), []).append(reminder)
    drift = _Drift()
    def dreams_own(row: dict, state: dict | None) -> bool:
        # A source id Dreams no longer owns -- a restore handed it to the
        # user's new source (review of #3123, M1) -- is not looked at.
        return (bool(int(row.get("created_by_dreams") or 0))
                and state is not None
                and is_dreams_source(state.get("processing_options"),
                                     state.get("source"),
                                     [row.get("story_url")]))

    for row in rows:
        record = _parsed(row.get("suspension"))
        state = run_states.get(int(row.get("subscription_id") or 0))
        created = dreams_own(row, state)
        if not enabled:
            if settled_state(row.get("suspension")) != SETTLED_OFF:
                continue  # not settled: the pass takes it up anyway
            # A rule or reminder counts only when it last changed before
            # the switch-off wrote this record (M2): a later change is the
            # user's.
            at = (record or {}).get("at")
            rule = rule_states.get(int(row.get("alert_rule_id") or 0)) or {}
            if ((created and state["is_active"])
                    or (rule.get("enabled")
                        and _changed_before(rule.get("updated_at"), at))
                    or any(_changed_before(reminder.get("updated_at"), at)
                           for reminder in armed.get(str(row["id"]), ()))):
                unsettled = dict(record or {})
                unsettled.pop("settled", None)
                drift.records.append((int(row["id"]), json.dumps(unsettled)))
        elif (created and not state["is_active"]
                and not (record or {}).get("subscription")):
            owed = {"reminders": [], "rules": [], **(record or {}),
                    "subscription": True}
            owed.pop("settled", None)
            drift.records.append((int(row["id"]), json.dumps(owed)))
    if enabled:
        return drift
    ridden = {int(row.get("subscription_id") or 0) for row in rows}
    for row in gone:
        record = _parsed(row.get("suspension")) or {}
        source_id = int(row.get("subscription_id") or 0)
        state = run_states.get(source_id)
        if (record.get("subscription") and dreams_own(row, state)
                and state["is_active"] and source_id not in ridden
                and source_id not in drift.sources):
            drift.sources.append(source_id)
        disarmed = {str(value) for value in record.get("reminders") or []}
        drift.reminders.extend(
            str(reminder["id"]) for reminder in armed.get(str(row["id"]), ())
            if str(reminder["id"]) in disarmed
            and _changed_before(reminder.get("updated_at"), record.get("at")))
    return drift


def _read_drift(
    app: Any, rows: list[dict], gone: list[dict], *, enabled: bool,
) -> _Drift:
    """The start check's reads of the other two stores, then the compare.

    Both stores are built and opened by the app's wiring on every start
    (``app.subscriptions_db``, the scheduling service's database); this
    reads them by key and never builds one. With Dreams on only the live
    watches' sources are read; with Dreams off also their rules, the
    sources a retired watch's record says Dreams switched off, and the
    reminders -- these only when a live watch has an event date (the
    only kind that owns one) or a retired watch's record lists one. A
    store the runtime does not have is not read (sync, on the boot
    worker).

    Measured (TASK-34924, 2026-10-15, 20 live watches, real stores in the
    test profile, machine load ~10): the queries take 0.4 ms on an open
    connection; what a start pays is opening one on the boot worker
    through the private-SQLite admission (a plain ``sqlite3`` open is
    1 ms) -- about 80 ms with Dreams on (one store), 150 ms with Dreams
    off (two) -- and nothing when there is no watch to look at. It runs
    before the first-dispatch gate opens, so the scheduler's first
    dispatch waits for it too.

    A store that refuses the read -- ``RecoveryRequired`` on a restored
    profile still awaiting review, say -- raises :class:`_StoreReadRefused`
    (review of #3123, N3). The start does not fail and unrelated work runs,
    but Dreams' work stays held for the whole session, as after any error
    in the boot reconcile: a restored source or reminder the check could
    not look at is not stopped either, though nothing dispatches it (a
    manual "Check now" in Watchlists would still run it). The next start
    asks again.
    """
    subscriptions_db = getattr(app, "subscriptions_db", None)
    scheduling_db = getattr(getattr(app, "scheduling_service", None), "db", None)
    run_states: dict[int, dict] = {}
    rule_states: dict[int, dict] = {}
    reminders: list[dict] = []
    gone_owing = [row for row in gone
                  if (_parsed(row.get("suspension")) or {}).get("subscription")]
    if subscriptions_db is not None:
        try:
            run_states, rule_states = subscriptions_db.run_states(
                [row["subscription_id"] for row in rows + gone_owing
                 if row.get("subscription_id")],
                () if enabled else [row["alert_rule_id"] for row in rows
                                    if row.get("alert_rule_id")])
        except Exception as exc:  # re-raised, named
            raise _StoreReadRefused("subscriptions", exc) from exc
        finally:
            # Its connections are per thread: this worker's goes now, as
            # ``Subscriptions.db_offload`` does for every off-loop call.
            subscriptions_db.close()
    gone_listing = [row for row in gone
                    if (_parsed(row.get("suspension")) or {}).get("reminders")]
    if (scheduling_db is not None and not enabled
            and (any(row.get("event_date") for row in rows) or gone_listing)):
        try:
            reminders = scheduling_db.list_reminder_tasks_by_link(
                _REMINDER_LINK_TYPE, [row["id"] for row in rows + gone_listing])
        except Exception as exc:  # re-raised, named
            raise _StoreReadRefused("scheduled tasks", exc) from exc
    return _restore_drift(
        rows, gone, run_states, rule_states, reminders, enabled=enabled)


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


def _boot_db(app: Any) -> tuple[Any, bool, bool, bool, _Drift]:
    """Find the Dreams DB and what the start owes it, if anything (sync).

    Runs on a worker thread: it may stat the profile and open SQLite.

    Returns:
        ``(dreams_db, owned, switch, backfill, drift)``. ``dreams_db`` is
        None when there is nothing to do -- Dreams has no database, every
        live watch is settled on the side of the switch and in step with
        the other stores (:func:`_restore_drift`), and no subscription
        awaits its egress mark. ``owned`` is True when the handle was
        opened here and must be closed by the caller: with Dreams OFF the
        app-owned builder builds nothing (an off feature gets no storage
        handle), so an EXISTING file is opened directly. ``switch`` and
        ``backfill`` say which of the two jobs there is; ``drift`` is what
        brings a restore's disagreement back in step (:class:`_Drift`).
    """
    enabled = bool(dreams_setting("enabled"))
    owned = False
    if enabled:
        dreams_db = app.get_dreams_db()
    else:
        path = get_dreams_db_path()
        if not path.exists():
            return None, False, False, False, _Drift()
        from tldw_chatbook.DB.Dreams_DB import DreamsDB

        dreams_db = DreamsDB(path)
        owned = True
    if dreams_db is None:
        return None, False, False, False, _Drift()
    rows, gone, backfill = dreams_db.tracking_boot_rows(retired=not enabled)
    switch = _records_need_switch(
        [row.get("suspension") for row in rows], enabled=enabled)
    drift = _read_drift(app, rows, gone, enabled=enabled) if (
        rows or gone) else _Drift()
    if switch or backfill or drift:
        return (dreams_db, owned, switch or bool(drift.records), backfill,
                drift)
    if owned:
        dreams_db.close_all()
    return None, False, False, False, _Drift()


def _write_drift(dreams_db: Any, drift: list[tuple[int, str]]) -> None:
    """Write the records :func:`_restore_drift` decided (sync)."""
    for tracked_item_id, record in drift:
        dreams_db.set_tracked_suspension(tracked_item_id, record, only_live=True)


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
        dreams_db, owned, switch, backfill, drift = await asyncio.to_thread(
            _boot_db, app)
        # What has to land BEFORE anything is dispatched: stopping the
        # watches of a Dreams that is off, and the egress-mark backfill
        # (T1 -- written when an unmarked source was a trusted egress
        # origin; since ADR-207 it is fetched under the public-only rule
        # anyway). A resume has no such claim.
        must_precede_dispatch = backfill or bool(
            drift.sources or drift.reminders) or (
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
            # What a restore put out of step becomes records the pass
            # below takes up (TASK-34924), written only now that the
            # stores are cleared for it.
            if drift.records:
                await asyncio.to_thread(_write_drift, dreams_db, drift.records)
            from tldw_chatbook.Dreams.track_service import (
                backfill_egress_marks,
                reconcile_tracking,
                stop_restored_leftovers,
            )

            subs_service = getattr(app, "local_watchlists_service", None)
            # There is work, so the subscriptions are about to be read
            # anyway: the mark is checked against every Dreams source's
            # REAL options, not only against the marker in this store
            # (round 4, B4 -- the two live in different stores, which a
            # backup restores as parts of different groups).
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
            if drift.sources or drift.reminders:
                # A failure here raises past the authorization below: the
                # dispatch guard keeps holding Dreams' work.
                await stop_restored_leftovers(
                    subs_service,
                    getattr(getattr(app, "scheduling_service", None), "db", None),
                    drift.sources, drift.reminders, dreams_db)
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
    except _StoreReadRefused as refused:
        # Review of #3123, N3: Dreams' work stays held until the next start
        # (an error never authorizes); said once, store and reason type.
        logger.warning(
            "Dreams startup check: the {} store refused its read ({}); "
            "Dreams work stays held until the next start",
            refused.store, refused.reason)
    except Exception as exc:  # noqa: BLE001 - a worker body must not raise
        logger.warning(
            "Dreams startup reconcile failed: {}", type(exc).__name__)
    finally:
        if first_dispatch is not None:
            first_dispatch.set()
