# Tests/Dreams/test_track_reconcile.py
"""Review of #2921, item 3c: Dreams' startup path reconciles its watches
with the persisted master switch.

The Settings toggle is not the only way ``[dreams] enabled`` changes: the
user can edit config.toml by hand, and a suspension that was interrupted
(or never ran, in a build without one) leaves an old state behind. At
boot the watches are brought in line with the switch as it is NOW --
off the UI thread, and without creating storage or importing the Dreams
service chain when there is nothing to do.
"""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.Dreams import track_reconcile


class _Sources:
    """The two watchlists-service calls the switch makes, recorded."""

    def __init__(self):
        self.active: dict[int, bool] = {}
        self.updates: list[tuple[int, bool]] = []
        self.marked: list[tuple[int, dict]] = []
        self.options: dict[int, dict] = {}
        self.reads = 0

    async def get_source(self, source_id):
        self.reads += 1
        return {"id": source_id, "url": "https://example.com/page",
                "active": self.active.get(int(source_id), True),
                "paused": False,
                "settings": {"processing_options": dict(
                    self.options.get(int(source_id), {}))}}

    async def update_source(self, source_id, payload):
        if "processing_options" in payload:
            self.options[int(source_id)] = dict(payload["processing_options"])
            self.marked.append(
                (int(source_id), dict(payload["processing_options"])))
        if "active" in payload:
            self.active[int(source_id)] = bool(payload["active"])
            self.updates.append((int(source_id), bool(payload["active"])))
        return {"id": source_id}

    async def list_alert_rules(self, *, job_id=None, source_id=None):
        return []


@pytest.fixture(autouse=True)
def _subscriptions_store_path(tmp_path, monkeypatch):
    """Where the reconcile looks for the subscriptions store when it asks
    whether that store is cleared for execution (round 4, B5). The real
    lookup reads the config, which fails closed under the per-test
    config redirect."""
    monkeypatch.setattr(
        track_reconcile, "get_subscriptions_db_path",
        lambda: tmp_path / "subscriptions.db")


@pytest.fixture()
def settings(monkeypatch):
    overrides: dict = {"enabled": True}

    def get(section, key, default=None):
        if section == "dreams" and key in overrides:
            return overrides[key]
        return default

    monkeypatch.setattr("tldw_chatbook.Dreams.settings.get_cli_setting", get)
    return overrides


@pytest.fixture()
def egress_seams(monkeypatch):
    from tldw_chatbook.Utils import egress

    monkeypatch.setattr(
        egress, "get_cli_setting", lambda section, key, default=None: default)

    async def _public(_host):
        return ["93.184.216.34"]

    monkeypatch.setattr(egress, "_resolve_async", _public)


def _seed_watch(path, *, suspension=None, egress_marked=1) -> int:
    db = DreamsDB(path, "boot-test")
    try:
        item_id = db.create_tracked_item(
            mechanism="page", intent="topic", cadence_seconds=43200,
            subscription_id=7, created_by_dreams=1,
            egress_marked=egress_marked)
        if suspension is not None:
            db.set_tracked_suspension(item_id, json.dumps(suspension))
        return item_id
    finally:
        db.close()


def _app(path, sources, *, dreams_db=None, reloads=None):
    """The slice of the app the boot reconcile reads."""
    return SimpleNamespace(
        get_dreams_db=lambda: dreams_db,
        local_watchlists_service=sources,
        scheduling_service=None,
        scheduler_loop=SimpleNamespace(
            request_reload=lambda: (reloads if reloads is not None
                                    else []).append(1)),
    )


def _suspension(path, item_id):
    db = DreamsDB(path, "boot-test")
    try:
        return db.get_tracked_item(item_id)["suspension"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_boot_suspends_watches_left_running_by_a_hand_edited_off(
        tmp_path, settings, monkeypatch):
    """config.toml says Dreams is off, but nothing ever suspended its
    watches: the Dreams database exists, so it is opened (and closed
    again -- the app gains no handle), and the running watch is stopped."""
    path = tmp_path / "dreams.sqlite"
    item_id = _seed_watch(path)
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = False
    sources, reloads = _Sources(), []
    app = _app(path, sources, reloads=reloads)

    await track_reconcile.reconcile_at_boot(app)

    assert sources.updates == [(7, False)]
    assert json.loads(_suspension(path, item_id))["subscription"] is True
    assert reloads == [1], "the scheduler queue reloads once the pass settles"
    assert not hasattr(app, "dreams_db"), (
        "a disabled Dreams gets no storage handle from the reconcile"
    )


@pytest.mark.asyncio
async def test_boot_creates_no_storage_for_a_dreams_that_was_never_enabled(
        tmp_path, settings, monkeypatch):
    """The path's directory EXISTS (round 3, T8): it is the ``exists()``
    guard that keeps the file from appearing. With a missing parent
    directory the open failed and was swallowed, and this test passed with
    the guard deleted."""
    path = tmp_path / "dreams.sqlite"
    assert path.parent.is_dir()
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = False
    sources = _Sources()

    await track_reconcile.reconcile_at_boot(_app(path, sources))

    assert not path.exists()
    assert list(tmp_path.glob("dreams.sqlite*")) == [], "nor its -wal/-shm"
    assert sources.updates == []


@pytest.mark.asyncio
async def test_boot_resumes_watches_an_old_off_state_left_suspended(
        tmp_path, settings, monkeypatch, egress_seams):
    """Dreams is on again (re-enabled by hand, or a resume was cut short),
    and a watch still carries its suspension record."""
    path = tmp_path / "dreams.sqlite"
    item_id = _seed_watch(
        path, suspension={"subscription": True, "reminders": [], "rules": []})
    sources = _Sources()
    sources.active[7] = False
    dreams_db = DreamsDB(path, "boot-app")
    try:
        await track_reconcile.reconcile_at_boot(
            _app(path, sources, dreams_db=dreams_db))
    finally:
        dreams_db.close()

    assert sources.updates == [(7, True)]
    assert _suspension(path, item_id) is None


def _forget_track_service(monkeypatch) -> None:
    """Make "was it imported?" answerable: drop the module from
    ``sys.modules`` AND from its package (``from pkg import mod`` is
    satisfied by the attribute alone). monkeypatch puts both back."""
    import sys

    import tldw_chatbook.Dreams as dreams_package

    monkeypatch.delitem(
        sys.modules, "tldw_chatbook.Dreams.track_service", raising=False)
    monkeypatch.delattr(dreams_package, "track_service", raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("switch", [True, False], ids=["dreams-on", "dreams-off"])
async def test_boot_with_nothing_to_switch_never_loads_the_track_service(
        tmp_path, settings, monkeypatch, switch):
    """The common boot: every watch is settled where the switch says. One
    cheap read, and the Dreams service chain is not IMPORTED for it.

    Proved on ``sys.modules`` (round 3, T21): the module is taken out
    first, so finding it there afterwards means the reconcile imported
    it. (Patching ``track_service.reconcile_tracking`` by its dotted name,
    as this test used to, imports the module itself and only shows the
    function was not called.)"""
    import sys

    path = tmp_path / "dreams.sqlite"
    _seed_watch(path, suspension=None if switch else {"settled": "off"})
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = switch
    sources = _Sources()
    _forget_track_service(monkeypatch)
    dreams_db = DreamsDB(path, "boot-app") if switch else None
    try:
        await track_reconcile.reconcile_at_boot(
            _app(path, sources, dreams_db=dreams_db))
    finally:
        if dreams_db is not None:
            dreams_db.close_all()

    assert "tldw_chatbook.Dreams.track_service" not in sys.modules
    assert sources.updates == [] and sources.reads == 0


@pytest.mark.asyncio
async def test_boot_with_work_does_load_the_track_service(
        tmp_path, settings, monkeypatch):
    """The control for the test above: the same probe sees the import
    when there IS a watch to switch."""
    import sys

    path = tmp_path / "dreams.sqlite"
    _seed_watch(path)
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = False
    _forget_track_service(monkeypatch)

    await track_reconcile.reconcile_at_boot(_app(path, _Sources()))

    assert "tldw_chatbook.Dreams.track_service" in sys.modules


@pytest.mark.asyncio
async def test_boot_with_nothing_to_do_leaves_no_handle_on_a_disabled_database(
        tmp_path, settings, monkeypatch):
    """Round 3, T15: with Dreams off the file is opened just to ask; when
    the answer is "nothing to do" that handle is closed again."""
    path = tmp_path / "dreams.sqlite"
    _seed_watch(path, suspension={"settled": "off"})
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = False

    await track_reconcile.reconcile_at_boot(_app(path, _Sources()))

    assert _open_handles_on(path) == 0


@pytest.mark.asyncio
async def test_boot_reconcile_never_raises_into_its_worker(
        tmp_path, settings, monkeypatch):
    """It runs as an app worker; an escaped error would take the app down."""
    def _broken():
        raise RuntimeError("profile path unavailable")

    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", _broken)
    settings["enabled"] = False

    await track_reconcile.reconcile_at_boot(_app(tmp_path / "x", _Sources()))


def _wire(settings_value, monkeypatch, *, with_gates: bool = False
          ) -> tuple[list, SimpleNamespace]:
    """Run the app's Dreams wiring on a stand-in app; return worker starts."""
    from tldw_chatbook.app_service_wiring import ServiceWiringMixin

    def get(section, key, default=None):
        if (section, key) == ("dreams", "enabled") and settings_value is not None:
            return settings_value
        return default

    monkeypatch.setattr("tldw_chatbook.Dreams.settings.get_cli_setting", get)
    started: list = []

    def run_worker(work, **kwargs):
        started.append((getattr(work, "__name__", ""), kwargs))
        work.close()  # never awaited here: the worker would run it

    scheduler_loop = SimpleNamespace(queue=SimpleNamespace(), handlers={})
    if with_gates:
        scheduler_loop.first_dispatch_gates = []
    app = SimpleNamespace(
        scheduler_loop=scheduler_loop,
        get_dreams_db=lambda: None,
        _start_dreams_boot_catchup=lambda: None,
        _dreams_cycle_deps=lambda: None,
        run_worker=run_worker,
    )
    ServiceWiringMixin._wire_dreams_scheduler_integration(app)
    return started, app


@pytest.mark.parametrize(
    "switch", [True, False, None], ids=["on", "off", "never-set"])
def test_dreams_wiring_starts_the_boot_reconcile_whatever_the_switch_says(
        monkeypatch, switch):
    """The app's Dreams wiring -- the one place Dreams comes up at boot --
    hands the reconcile to a worker. Off is exactly the hand-edited case
    it exists for, and so is a ``[dreams]`` section the user deleted
    (no key at all, so the default -- off -- applies) while watches Dreams
    made are still running."""
    started, app = _wire(switch, monkeypatch)

    assert started == [
        ("reconcile_at_boot", {"exclusive": False, "group": "dreams"})]
    assert "dream_track_check" in app.scheduler_loop.handlers


@pytest.mark.asyncio
async def test_boot_suspends_watches_when_the_dreams_section_was_deleted(
        tmp_path, monkeypatch):
    """No ``[dreams] enabled`` key at all: the shipped default (off) is
    the current value, and running watches are brought in line with it."""
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default=None: default)
    path = tmp_path / "dreams.sqlite"
    item_id = _seed_watch(path)
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    sources = _Sources()

    await track_reconcile.reconcile_at_boot(_app(path, sources))

    assert sources.updates == [(7, False)]
    assert _suspension(path, item_id) is not None


# --- T1: the one-time egress-mark backfill rides the boot reconcile -------------


@pytest.mark.asyncio
@pytest.mark.parametrize("switch", [True, False], ids=["dreams-on", "dreams-off"])
async def test_boot_marks_unmarked_dreams_sources_once(
        tmp_path, settings, monkeypatch, switch):
    """A source Dreams created before the egress mark existed is stamped at
    the next start -- with Dreams OFF too: the source still exists and the
    user can run it from Watchlists. Once done it never runs again."""
    path = tmp_path / "dreams.sqlite"
    _seed_watch(
        path, egress_marked=0,
        suspension=None if switch else {"settled": "off"})
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = switch
    sources = _Sources()
    dreams_db = DreamsDB(path, "boot-app") if switch else None
    try:
        app = _app(path, sources, dreams_db=dreams_db)
        await track_reconcile.reconcile_at_boot(app)

        assert sources.marked == [(7, {"url_provenance": "unknown"})]
        assert sources.updates == [], "marking switches nothing on or off"

        reads = sources.reads
        await track_reconcile.reconcile_at_boot(app)

        assert sources.marked == [(7, {"url_provenance": "unknown"})]
        assert sources.reads == reads, "the second start asks nothing of it"
    finally:
        if dreams_db is not None:
            dreams_db.close()


# --- Round 4, B4: the marker is a hint; the source's own options are the fact ----
#
# "Already marked" is remembered in dreams.sqlite, the mark itself lives in
# the subscriptions store, and only that store is backed up. Restore it from
# before the mark and the sources come back unmarked while the marker says
# done.


@pytest.mark.asyncio
async def test_a_boot_that_has_work_checks_every_dreams_source_for_its_mark(
        tmp_path, settings, monkeypatch):
    """The reconcile is about to read the subscriptions anyway (a watch is
    left running by a hand-edited "off"), so it looks at the real options
    of EVERY source Dreams owns instead of trusting the marker -- here the
    source of a watch retired long ago, which no switch pass touches."""
    path = tmp_path / "dreams.sqlite"
    _seed_watch(path, egress_marked=1)  # live, running: the boot's work
    db = DreamsDB(path, "boot-test")
    try:
        retired = db.create_tracked_item(
            mechanism="page", intent="topic", cadence_seconds=43200,
            subscription_id=9, created_by_dreams=1, egress_marked=1)
        db.set_tracked_status(retired, "retired", retired_reason="manual")
    finally:
        db.close()
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = False
    sources = _Sources()  # ...whose (restored) sources carry no mark

    await track_reconcile.reconcile_at_boot(_app(path, sources))

    assert (9, {"url_provenance": "unknown"}) in sources.marked


@pytest.mark.asyncio
async def test_an_ordinary_boot_still_asks_nothing_of_the_subscriptions(
        tmp_path, settings, monkeypatch):
    """The limit that is documented, pinned: with no other work a start
    does not read the subscriptions store to re-verify the marker."""
    path = tmp_path / "dreams.sqlite"
    _seed_watch(path, egress_marked=1, suspension={"settled": "off"})
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = False
    sources = _Sources()

    await track_reconcile.reconcile_at_boot(_app(path, sources))

    assert sources.reads == 0 and sources.marked == []


# --- T7: the temporary handle of a disabled Dreams leaves nothing open ----------


def _open_handles_on(path) -> int:
    """How many file descriptors this process holds on the database file
    (and its -wal/-shm companions). macOS/Linux; psutil ships with dev."""
    import psutil

    names = {str(path), f"{path}-wal", f"{path}-shm"}
    resolved = {str(Path(name).resolve()) for name in names}
    return sum(1 for handle in psutil.Process().open_files()
               if handle.path in names or handle.path in resolved)


def test_close_all_closes_connections_opened_on_other_threads(tmp_path):
    """``close()`` only closes the calling thread's connection; every
    ``asyncio.to_thread`` hop may have opened another."""
    import threading

    path = tmp_path / "dreams.sqlite"
    db = DreamsDB(path, "close-all")

    def touch():
        db.live_tracked_suspensions()

    threads = [threading.Thread(target=touch) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert _open_handles_on(path) >= 5, "fixture: one connection per thread"

    db.close()
    assert _open_handles_on(path) >= 4, "close() is this thread's only"
    db.close_all()

    assert _open_handles_on(path) == 0
    # The handle is still usable afterwards: it opens a fresh connection.
    assert db.live_tracked_suspensions() == []
    db.close_all()
    assert _open_handles_on(path) == 0


class _RotatingPool(ThreadPoolExecutor):
    """Hands each job to the next of four long-lived threads -- what the
    app's executor does over a session, made deterministic: every thread
    that touches the database keeps its own connection open."""

    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self._pools = [ThreadPoolExecutor(max_workers=1) for _ in range(4)]
        self._next = 0

    def submit(self, fn, /, *args, **kwargs):
        pool = self._pools[self._next % len(self._pools)]
        self._next += 1
        return pool.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True, *, cancel_futures=False):
        for pool in self._pools:
            pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        super().shutdown(wait=wait, cancel_futures=cancel_futures)


@pytest.mark.asyncio
async def test_boot_reconcile_leaves_no_handle_on_a_disabled_dreams_database(
        tmp_path, settings, monkeypatch):
    """With Dreams off the reconcile opens the file itself (the app keeps no
    handle for an off feature). When there was work, its thread hops left
    a dozen connections open on dreams.sqlite for the rest of the session:
    ``close()`` reaches only the connection of the thread that calls it."""
    path = tmp_path / "dreams.sqlite"
    _seed_watch(path)
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = False
    sources = _Sources()
    pool = _RotatingPool()
    asyncio.get_running_loop().set_default_executor(pool)
    try:
        await track_reconcile.reconcile_at_boot(_app(path, sources))

        assert sources.updates == [(7, False)], "fixture: there was work"
        assert _open_handles_on(path) == 0
    finally:
        pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_a_cancelled_boot_worker_does_not_close_the_handle_under_the_pass(
        tmp_path, settings, monkeypatch):
    """Review round 4. The switch pass runs as a task its caller cannot
    cancel. The boot worker closed the temporary handle in its own
    ``finally`` -- so a worker cancelled mid-pass closed every connection
    while the pass was still using the handle, the pass's later hops
    reopened them, and nothing ever closed those (9 handles left). The
    handle now lives exactly as long as the pass; and the scheduler's gate
    opens when the pass is over, not when the worker gives up."""
    path = tmp_path / "dreams.sqlite"
    item_id = _seed_watch(path)
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = False
    in_the_stop, release = asyncio.Event(), asyncio.Event()

    class _Parked(_Sources):
        async def update_source(self, source_id, payload):
            if "active" in payload:
                in_the_stop.set()
                await release.wait()
            return await super().update_source(source_id, payload)

    class _NoReminders:
        def list_reminder_tasks_by_link(self, link_type, link_ids):
            return []

    sources = _Parked()
    app = _app(path, sources)
    # With a scheduling DB the pass settles the record AFTER the stop: a
    # database hop that lands after the cancelled worker has gone.
    app.scheduling_service = SimpleNamespace(db=_NoReminders())
    gate = track_reconcile.first_dispatch_gate(
        SimpleNamespace(first_dispatch_gates=[]))
    pool = _RotatingPool()
    asyncio.get_running_loop().set_default_executor(pool)
    try:
        worker = asyncio.create_task(
            track_reconcile.reconcile_at_boot(app, gate))
        await asyncio.wait_for(in_the_stop.wait(), timeout=10)

        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker
        assert not gate.is_set(), (
            "the suspend has not landed: the scheduler stays held"
        )

        release.set()
        await asyncio.wait_for(gate.wait(), timeout=10)

        assert sources.updates == [(7, False)], "the pass ran to its end"
        assert gate.is_set()
        assert _open_handles_on(path) == 0
        assert json.loads(_suspension(path, item_id))["settled"] == "off"
    finally:
        release.set()
        pool.shutdown(wait=True)


# --- T25: the boot reconcile holds where the scheduler would hold ---------------
#
# The scheduler does not start, and does not tick, while the scheduled-tasks
# store is not cleared for execution (a restored generation awaiting review,
# ADR-126), and it waits while storage maintenance (a backup capture) has its
# admission closed. The boot reconcile writes to the same stores.


class _WatchedGate(asyncio.Event):
    """An event that says when somebody started waiting on it."""

    def __init__(self) -> None:
        super().__init__()
        self.waited = asyncio.Event()

    async def wait(self):
        self.waited.set()
        return await super().wait()


def _gated_app(path, sources, *, maintenance_open: bool = True,
               admitted: bool = True):
    app = _app(path, sources)
    gate = _WatchedGate()
    if maintenance_open:
        gate.set()
    app.scheduler_loop.db = SimpleNamespace(db_path=path.parent / "scheduled.db")

    async def wait_for_maintenance_admission() -> bool:
        await gate.wait()
        return admitted

    # The scheduler's PUBLIC seam (round 4); a fake that only had the
    # private event would not be waited on at all.
    app.scheduler_loop.wait_for_maintenance_admission = (
        wait_for_maintenance_admission)
    return app, gate


@pytest.mark.asyncio
async def test_boot_reconcile_holds_while_execution_is_not_allowed(
        tmp_path, settings, monkeypatch):
    """A restored profile awaiting review: the scheduler stays off, and the
    reconcile changes nothing either. The work is still there for the start
    after the review."""
    from tldw_chatbook.Backup_Recovery import activation

    path = tmp_path / "dreams.sqlite"
    item_id = _seed_watch(path, egress_marked=0)
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = False
    asked: list[tuple] = []

    def not_allowed(owners, db_path=None):
        asked.append((owners, db_path))
        return False

    monkeypatch.setattr(activation, "execution_allowed", not_allowed)
    sources = _Sources()
    app, _gate = _gated_app(path, sources)

    await track_reconcile.reconcile_at_boot(app)

    assert asked[0] == (("db.scheduled_tasks",), tmp_path / "scheduled.db"), (
        "the very question the scheduler asks before it starts"
    )
    assert sources.updates == [] and sources.marked == [] and sources.reads == 0
    assert _suspension(path, item_id) is None
    assert _open_handles_on(path) == 0

    # Review done: the next start does the work.
    monkeypatch.setattr(activation, "execution_allowed", lambda *a, **k: True)
    await track_reconcile.reconcile_at_boot(app)
    assert sources.updates == [(7, False)]
    assert sources.marked == [(7, {"url_provenance": "unknown"})]


@pytest.mark.asyncio
@pytest.mark.parametrize("refused", ["db.scheduled_tasks", "db.subscriptions", "db.dreams"])
async def test_boot_reconcile_asks_about_every_store_it_is_about_to_write(
        tmp_path, settings, monkeypatch, refused):
    """Review round 4, B5. The reconcile asked only about the scheduled-
    tasks store, but the backfill and the switch write the subscriptions
    store too -- whose own background work asks about ``db.subscriptions``
    (``app_feature_glue._backfill_subscription_items_fts``). Either store
    still awaiting review holds the whole reconcile."""
    from tldw_chatbook.Backup_Recovery import activation

    path = tmp_path / "dreams.sqlite"
    item_id = _seed_watch(path, egress_marked=0)
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = False
    asked: list[tuple] = []

    def allowed(owners, db_path=None):
        asked.append((owners, db_path))
        return owners != (refused,)

    monkeypatch.setattr(activation, "execution_allowed", allowed)
    sources = _Sources()
    app, _gate = _gated_app(path, sources)

    await track_reconcile.reconcile_at_boot(app)

    assert (("db.subscriptions",), tmp_path / "subscriptions.db") in asked or (
        refused == "db.scheduled_tasks"), "the subscriptions store is asked about"
    assert (("db.scheduled_tasks",), tmp_path / "scheduled.db") in asked
    if refused == "db.dreams":
        assert (("db.dreams",), path) in asked
    assert sources.updates == [] and sources.marked == [] and sources.reads == 0
    assert _suspension(path, item_id) is None

    monkeypatch.setattr(activation, "execution_allowed", lambda *a, **k: True)
    await track_reconcile.reconcile_at_boot(app)
    assert sources.updates == [(7, False)]


@pytest.mark.asyncio
async def test_boot_reconcile_waits_for_storage_maintenance_to_finish(
        tmp_path, settings, monkeypatch):
    from tldw_chatbook.Backup_Recovery import activation

    path = tmp_path / "dreams.sqlite"
    _seed_watch(path)
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    monkeypatch.setattr(activation, "execution_allowed", lambda *a, **k: True)
    settings["enabled"] = False
    sources = _Sources()
    app, gate = _gated_app(path, sources, maintenance_open=False)

    reconcile = asyncio.create_task(track_reconcile.reconcile_at_boot(app))
    await asyncio.wait_for(gate.waited.wait(), timeout=10)
    await asyncio.sleep(0.05)
    assert sources.updates == [] and not reconcile.done(), (
        "held while a backup capture has admission closed"
    )

    gate.set()
    await asyncio.wait_for(reconcile, timeout=10)

    assert sources.updates == [(7, False)]


@pytest.mark.asyncio
async def test_boot_reconcile_holds_when_the_scheduler_stopped_under_a_capture(
        tmp_path, settings, monkeypatch):
    """Review round 4. ``SchedulerLoop.stop()`` wakes everything that waits
    for maintenance to end -- while the capture may still be open. The
    reconcile used to read the private event and write regardless; it now
    asks the scheduler, and holds on "no"."""
    from tldw_chatbook.Backup_Recovery import activation

    path = tmp_path / "dreams.sqlite"
    item_id = _seed_watch(path)
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    monkeypatch.setattr(activation, "execution_allowed", lambda *a, **k: True)
    settings["enabled"] = False
    sources = _Sources()
    app, _gate = _gated_app(path, sources, admitted=False)

    await track_reconcile.reconcile_at_boot(app)

    assert sources.updates == [] and sources.marked == [] and sources.reads == 0
    assert _suspension(path, item_id) is None
    assert _open_handles_on(path) == 0


# --- T14a: nothing is dispatched before the boot suspend has landed -------------
#
# The scheduler worker starts just before the reconcile. After a hand-edited
# "off", an overdue Dreams source or reminder could be dispatched by the
# scheduler's first tick before the suspend reached it. The scheduler's first
# dispatch now waits for a gate the reconcile opens -- at once when its cheap
# check finds nothing to stop, after the pass when it does.


def test_dreams_wiring_gates_the_schedulers_first_dispatch(monkeypatch):
    started, app = _wire(False, monkeypatch, with_gates=True)

    (gate_wait,) = app.scheduler_loop.first_dispatch_gates
    assert started == [
        ("reconcile_at_boot", {"exclusive": False, "group": "dreams"})]
    assert callable(gate_wait)


@pytest.mark.asyncio
async def test_first_dispatch_gate_opens_at_once_when_nothing_is_to_be_stopped(
        tmp_path, settings, monkeypatch):
    """The ordinary start: the scheduler waits one cheap check, no more."""
    path = tmp_path / "dreams.sqlite"
    _seed_watch(path)  # Dreams on, the watch simply running
    loop = SimpleNamespace(first_dispatch_gates=[])
    gate = track_reconcile.first_dispatch_gate(loop)
    dreams_db = DreamsDB(path, "boot-app")
    try:
        await track_reconcile.reconcile_at_boot(
            _app(path, _Sources(), dreams_db=dreams_db), gate)
    finally:
        dreams_db.close_all()

    assert gate.is_set()
    assert loop.first_dispatch_gates == [gate.wait]


@pytest.mark.asyncio
async def test_first_dispatch_gate_stays_shut_until_the_boot_suspend_landed(
        tmp_path, settings, monkeypatch):
    path = tmp_path / "dreams.sqlite"
    _seed_watch(path)
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    settings["enabled"] = False  # hand-edited off, the watch still running
    gate = track_reconcile.first_dispatch_gate(
        SimpleNamespace(first_dispatch_gates=[]))
    states_seen_by_the_stop: list[bool] = []

    class _Watching(_Sources):
        async def update_source(self, source_id, payload):
            states_seen_by_the_stop.append(gate.is_set())
            return await super().update_source(source_id, payload)

    sources = _Watching()

    await track_reconcile.reconcile_at_boot(_app(path, sources), gate)

    assert sources.updates == [(7, False)]
    # Two writes since round 4, B4: the verifying backfill stamps the
    # (unmarked) source, then the suspend stops it. Both precede dispatch.
    assert states_seen_by_the_stop and not any(states_seen_by_the_stop), (
        "the scheduler was still held while the source was marked and stopped"
    )
    assert gate.is_set(), "and is released once the pass is over"


@pytest.mark.asyncio
async def test_first_dispatch_gate_opens_even_when_the_reconcile_fails(
        tmp_path, settings, monkeypatch):
    def _broken():
        raise RuntimeError("profile path unavailable")

    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", _broken)
    settings["enabled"] = False
    gate = track_reconcile.first_dispatch_gate(
        SimpleNamespace(first_dispatch_gates=[]))

    await track_reconcile.reconcile_at_boot(
        _app(tmp_path / "x", _Sources()), gate)

    assert gate.is_set(), "a failed reconcile must never hold the scheduler"


# --- Review round 4: the boot suspend against the real stores --------------------
#
# Everything above drives the reconcile with a fake watchlists service and no
# scheduling DB, where a suspend can never settle (settling needs both). This
# one is the real thing end to end: a real subscriptions store and watchlists
# service, a real scheduled-tasks store, a hand-edited "off".


@pytest.mark.asyncio
async def test_boot_suspend_settles_against_the_real_stores(
        tmp_path, settings, egress_seams, monkeypatch):
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Dreams.track_service import track_page
    from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
    from tldw_chatbook.Subscriptions import LocalWatchlistsService

    # ``create_source`` freezes a config default through the guarded
    # loader, which fails closed under the per-test config redirect.
    monkeypatch.setattr(
        "tldw_chatbook.DB.Subscriptions_DB.get_cli_setting",
        lambda section, key, default=None: default)
    path = tmp_path / "dreams.sqlite"
    monkeypatch.setattr(track_reconcile, "get_dreams_db_path", lambda: path)
    subs_db = SubscriptionsDB(tmp_path / "subscriptions.db", "boot-real")
    scheduling_db = ScheduledTasksDB(tmp_path / "scheduled.db", "boot-real")
    service = LocalWatchlistsService(db_factory=lambda: subs_db)
    reloads: list[int] = []
    app = SimpleNamespace(
        get_dreams_db=lambda: None,  # Dreams is off: the app builds no handle
        local_watchlists_service=service,
        scheduling_service=SimpleNamespace(db=scheduling_db),
        scheduler_loop=SimpleNamespace(
            request_reload=lambda: reloads.append(1)),
    )
    try:
        seeded = DreamsDB(path, "boot-real")
        try:
            tracked = await track_page(
                service, seeded, url="https://example.com/festival",
                title="Festival", intent="event", event_date="2026-10-01",
                origin_story_id=1,
                scheduling_db_getter=lambda: scheduling_db)
        finally:
            seeded.close_all()
        item_id, source_id = tracked["tracked_item_id"], tracked["subscription_id"]
        (reminder,) = scheduling_db.list_reminder_tasks(owner_id="local")
        (rule,) = await service.list_alert_rules(job_id=source_id)
        assert reminder["enabled"] and rule["enabled"], "fixture"
        settings["enabled"] = False  # edited by hand; nothing was suspended

        await track_reconcile.reconcile_at_boot(app)

        assert int(subs_db.get_subscription(source_id)["is_active"]) == 0
        assert not scheduling_db.get_reminder_task(reminder["id"])["enabled"]
        (rule,) = await service.list_alert_rules(job_id=source_id)
        assert not rule["enabled"]
        record = json.loads(_suspension(path, item_id))
        assert record.pop("at"), "when the switch-off settled it (#3123, M2)"
        assert record == {
            "subscription": True, "reminders": [reminder["id"]],
            "rules": [rule["rule_id"]], "settled": "off"}
        assert reloads == [1], "the scheduler queue was asked to reload"
        assert _open_handles_on(path) == 0

        # Settled: the next start finds nothing to do and touches nothing.
        updated_at = subs_db.get_subscription(source_id)["updated_at"]
        await track_reconcile.reconcile_at_boot(app)
        assert reloads == [1]
        assert subs_db.get_subscription(source_id)["updated_at"] == updated_at
        assert _open_handles_on(path) == 0
    finally:
        subs_db.close()
        scheduling_db.close()


# --- TASK-34924: a restore that undoes the switch is caught at the next start --
#
# Switching Dreams off stops the sources, rules and reminders its watches
# own; they live in the subscriptions and scheduling stores, which a backup
# restores on their own, while Dreams' records (another store) do not
# change. A restore is modelled as what it is to these stores: their files
# replaced by an earlier copy while the app is down.


class _RealStores:
    """The three real stores and the slice of the app the boot reads."""

    def __init__(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "tldw_chatbook.DB.Subscriptions_DB.get_cli_setting",
            lambda section, key, default=None: default)
        self.tmp = tmp_path
        self.dreams_path = tmp_path / "dreams.sqlite"
        monkeypatch.setattr(
            track_reconcile, "get_dreams_db_path", lambda: self.dreams_path)
        self.paths = {"subs": tmp_path / "subscriptions.db",
                      "sched": tmp_path / "scheduled.db"}
        self._open()

    def _open(self):
        from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
        from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
        from tldw_chatbook.Subscriptions import LocalWatchlistsService

        self.subs_db = SubscriptionsDB(self.paths["subs"], "restore-test")
        self.scheduling_db = ScheduledTasksDB(self.paths["sched"], "restore-test")
        self.service = LocalWatchlistsService(db_factory=lambda: self.subs_db)

    def close(self):
        self.subs_db.close()
        self.scheduling_db.close()

    def backup(self, name, keys=("subs", "sched")):
        import sqlite3

        for key, path in self.paths.items():
            if key not in keys:
                continue
            source = sqlite3.connect(path)
            target = sqlite3.connect(self.tmp / f"{name}-{key}.db")
            with target:
                source.backup(target)
            source.close()
            target.close()

    def restore(self, name, keys=("subs", "sched")):
        """The app is down; the store files are put back as they were."""
        import shutil

        self.close()
        for key, path in self.paths.items():
            if key not in keys:
                continue
            for suffix in ("-wal", "-shm"):
                Path(str(path) + suffix).unlink(missing_ok=True)
            shutil.copyfile(self.tmp / f"{name}-{key}.db", path)
        self._open()

    def app(self, dreams_db=None):
        return SimpleNamespace(
            get_dreams_db=lambda: dreams_db,
            local_watchlists_service=self.service,
            subscriptions_db=self.subs_db,
            scheduling_service=SimpleNamespace(db=self.scheduling_db),
            scheduler_loop=SimpleNamespace(request_reload=lambda: None),
        )

    async def switch(self, settings, *, on: bool):
        from tldw_chatbook.Dreams.track_service import set_tracking_suspended

        settings["enabled"] = on
        db = DreamsDB(self.dreams_path, "restore-test")
        try:
            await set_tracking_suspended(
                self.service, db, suspended=not on,
                scheduling_db_getter=lambda: self.scheduling_db)
        finally:
            db.close_all()

    async def track(self):
        from tldw_chatbook.Dreams.track_service import track_page

        db = DreamsDB(self.dreams_path, "restore-test")
        try:
            return await track_page(
                self.service, db, url="https://example.com/festival",
                # Far ahead: the switch-on and the boot run the lifecycle
                # sweep on the real clock, which retires a passed event.
                title="Festival", intent="event", event_date="2099-10-01",
                origin_story_id=1,
                scheduling_db_getter=lambda: self.scheduling_db)
        finally:
            db.close_all()

    async def running(self, source_id):
        (reminder,) = self.scheduling_db.list_reminder_tasks(owner_id="local")
        (rule,) = await self.service.list_alert_rules(job_id=source_id)
        return (bool(int(self.subs_db.get_subscription(source_id)["is_active"])),
                bool(reminder["enabled"]), bool(rule["enabled"]))


@pytest.mark.asyncio
async def test_a_restore_with_dreams_off_runs_nothing_dreams_created(
        tmp_path, settings, egress_seams, monkeypatch):
    """AC 1: a backup taken while Dreams was on is restored into a profile
    where Dreams is off. The restored source, rule and reminder ran again
    -- the records said everything was stopped, so the start read none of
    them. The start now finds them out of step and stops them."""
    stores = _RealStores(tmp_path, monkeypatch)
    try:
        tracked = await stores.track()
        source_id = tracked["subscription_id"]
        stores.backup("while-on")
        await stores.switch(settings, on=False)
        assert await stores.running(source_id) == (False, False, False)

        stores.restore("while-on")
        assert await stores.running(source_id) == (True, True, True), "fixture"
        await track_reconcile.reconcile_at_boot(stores.app())

        assert await stores.running(source_id) == (False, False, False)
        record = json.loads(_suspension(stores.dreams_path,
                                        tracked["tracked_item_id"]))
        assert record["settled"] == "off" and record["subscription"] is True
    finally:
        stores.close()


@pytest.mark.asyncio
async def test_a_restore_with_dreams_on_runs_every_live_watchs_source(
        tmp_path, settings, egress_seams, monkeypatch):
    """AC 2: a backup taken while Dreams was off is restored into a profile
    where Dreams is on (only the subscriptions and schedules came back).
    Dreams' records owe nothing -- switching on cleared them -- so the
    source stayed off under a live watch. The start now gives the watch a
    restart debt and the switch-on pass restarts it."""
    stores = _RealStores(tmp_path, monkeypatch)
    try:
        tracked = await stores.track()
        source_id = tracked["subscription_id"]
        await stores.switch(settings, on=False)
        stores.backup("while-off")
        await stores.switch(settings, on=True)
        assert (await stores.running(source_id))[0] is True

        stores.restore("while-off")
        assert (await stores.running(source_id))[0] is False, "fixture"
        dreams_db = DreamsDB(stores.dreams_path, "boot-app")
        try:
            await track_reconcile.reconcile_at_boot(stores.app(dreams_db))
        finally:
            dreams_db.close_all()

        assert (await stores.running(source_id))[0] is True
        assert _suspension(stores.dreams_path, tracked["tracked_item_id"]) is None
    finally:
        stores.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("on", [True, False], ids=["dreams-on", "dreams-off"])
async def test_a_start_that_finds_the_stores_in_step_writes_and_loads_nothing(
        tmp_path, settings, egress_seams, monkeypatch, on):
    """The two reads are all an ordinary start pays: nothing is written
    and the Dreams service chain is not imported."""
    import sys

    stores = _RealStores(tmp_path, monkeypatch)
    try:
        tracked = await stores.track()
        source_id = tracked["subscription_id"]
        if not on:
            await stores.switch(settings, on=False)
        before = (stores.subs_db.get_subscription(source_id)["updated_at"],
                  _suspension(stores.dreams_path, tracked["tracked_item_id"]))
        _forget_track_service(monkeypatch)
        dreams_db = DreamsDB(stores.dreams_path, "boot-app") if on else None
        try:
            await track_reconcile.reconcile_at_boot(stores.app(dreams_db))
        finally:
            if dreams_db is not None:
                dreams_db.close_all()

        assert "tldw_chatbook.Dreams.track_service" not in sys.modules
        assert (stores.subs_db.get_subscription(source_id)["updated_at"],
                _suspension(stores.dreams_path,
                            tracked["tracked_item_id"])) == before
    finally:
        stores.close()


def test_a_paused_source_the_switch_disabled_is_not_seen_as_running():
    """Watchlists' Resume clears ``is_paused`` only: a source with
    ``is_active`` 0 does not run, paused or not, and the start must not
    take it for one a restore switched back on (it would every start)."""
    rows = [{"id": 1, "subscription_id": 7, "created_by_dreams": 1,
             "alert_rule_id": 0, "event_date": None,
             "suspension": json.dumps({"subscription": True, "reminders": [],
                                       "rules": [], "settled": "off"})}]

    def state(active):
        return {"is_active": active, "is_paused": True,
                "source": "https://example.com/page", "processing_options": None}

    assert not track_reconcile._restore_drift(
        rows, [], {7: state(False)}, {}, [], enabled=False)
    assert track_reconcile._restore_drift(
        rows, [], {7: state(True)}, {}, [], enabled=False)


@pytest.mark.asyncio
async def test_a_restore_with_dreams_off_stops_what_a_retired_watch_left(
        tmp_path, settings, egress_seams, monkeypatch):
    """AC 1, a retired watch: untracking it switched its source off and
    disarmed its reminder, and its retirement record says so. A backup
    from before the untrack brings both back running into a profile where
    Dreams is now off; no switch pass reads a retired watch, so the start
    stops them itself."""
    from tldw_chatbook.Dreams.track_service import untrack

    stores = _RealStores(tmp_path, monkeypatch)
    try:
        tracked = await stores.track()
        source_id = tracked["subscription_id"]
        stores.backup("before-untrack")
        db = DreamsDB(stores.dreams_path, "restore-test")
        try:
            await untrack(stores.service, db, tracked["tracked_item_id"],
                          scheduling_db_getter=lambda: stores.scheduling_db)
        finally:
            db.close_all()
        await stores.switch(settings, on=False)
        assert (await stores.running(source_id))[:2] == (False, False), "fixture"

        stores.restore("before-untrack")
        assert (await stores.running(source_id))[:2] == (True, True), "fixture"
        await track_reconcile.reconcile_at_boot(stores.app())

        assert (await stores.running(source_id))[:2] == (False, False)
    finally:
        stores.close()


# --- Review of #3123, M1: a reused source id is never Dreams' ------------------
#
# ``subscriptions.id`` is AUTOINCREMENT, and a restore puts the table's id
# counter back too. Restore only the subscriptions from a backup older than
# a watch, and the user's next new source gets the id Dreams' source had,
# while the Dreams store still names that id as Dreams' own.


@pytest.mark.asyncio
@pytest.mark.parametrize("on", [False, True], ids=["dreams-off", "dreams-on"])
async def test_the_start_never_acts_on_a_users_source_on_a_reused_id(
        tmp_path, settings, egress_seams, monkeypatch, on):
    """Dreams off: the user's active source used to be switched OFF at the
    next start. Dreams on: the user's source, created inactive on purpose,
    used to be switched ON (and a user-entered address skips the public
    check)."""
    stores = _RealStores(tmp_path, monkeypatch)
    try:
        settings["enabled"] = on
        if not on:
            settings["enabled"] = True  # track with Dreams on, then off
        tracked = None
        stores.backup("before-the-watch", keys=("subs",))
        tracked = await stores.track()
        if not on:
            await stores.switch(settings, on=False)
        stores.restore("before-the-watch", keys=("subs",))
        from tldw_chatbook.Utils.egress import UrlProvenance

        users = await stores.service.create_source({
            "name": "My NAS", "url": "https://nas.example/status",
            "source_type": "url", "active": not on,
            "url_provenance": UrlProvenance.USER_ENTERED})
        source_id = users["source_id"]
        assert source_id == tracked["subscription_id"], "fixture: id reused"
        dreams_db = DreamsDB(stores.dreams_path, "boot-app") if on else None
        try:
            await track_reconcile.reconcile_at_boot(stores.app(dreams_db))
        finally:
            if dreams_db is not None:
                dreams_db.close_all()

        assert bool(int(stores.subs_db.get_subscription(source_id)[
            "is_active"])) is (not on), "the user's source is left as it was"
    finally:
        stores.close()


@pytest.mark.asyncio
async def test_the_switch_off_never_stops_a_users_source_on_a_reused_id(
        tmp_path, settings, egress_seams, monkeypatch):
    """The suspend pass took the reused id for Dreams' source and stopped
    the user's own; the watch's source is gone and is handled as such."""
    from tldw_chatbook.Utils.egress import UrlProvenance

    stores = _RealStores(tmp_path, monkeypatch)
    try:
        stores.backup("before-the-watch", keys=("subs",))
        tracked = await stores.track()
        stores.restore("before-the-watch", keys=("subs",))
        users = await stores.service.create_source({
            "name": "My NAS", "url": "https://nas.example/status",
            "source_type": "url", "active": True,
            "url_provenance": UrlProvenance.USER_ENTERED})
        source_id = users["source_id"]
        assert source_id == tracked["subscription_id"], "fixture: id reused"

        await stores.switch(settings, on=False)

        assert int(stores.subs_db.get_subscription(source_id)["is_active"]) == 1
        record = json.loads(_suspension(stores.dreams_path,
                                        tracked["tracked_item_id"]))
        assert record["settled"] == "off" and record["subscription"] is False
    finally:
        stores.close()


# --- Review of #3123, M2: a reminder or rule the user re-arms is theirs --------
#
# With Dreams off, a reminder (or Dreams' change rule) the user switches back
# on in Schedules or Watchlists was disarmed again at every start -- no
# restore needed. Rows carry ``updated_at``; the records now carry when the
# switch-off (or the retirement) wrote them, and only a row last changed
# before that is taken for one a restore brought back.


def test_record_and_row_times_compare_to_the_millisecond():
    """A record's ``at`` is canonical (milliseconds, ADR-173); a rule's
    ``updated_at`` carries microseconds. A row Dreams changed in the same
    millisecond it wrote the record is not the user's."""
    from tldw_chatbook.Dreams.track_reconcile import (
        _changed_before,
        changed_after,
    )

    at = "2026-10-15T12:00:00.123Z"
    same_ms = "2026-10-15T12:00:00.123456+00:00"
    assert _changed_before(same_ms, at)
    assert not changed_after(same_ms, at)
    assert changed_after("2026-10-15T12:00:00.124000+00:00", at)
    assert not _changed_before(None, at) and not changed_after(None, at)


@pytest.mark.asyncio
async def test_a_reminder_and_rule_the_user_re_arms_while_off_stay_armed(
        tmp_path, settings, egress_seams, monkeypatch):
    stores = _RealStores(tmp_path, monkeypatch)
    try:
        tracked = await stores.track()
        source_id = tracked["subscription_id"]
        await stores.switch(settings, on=False)
        (reminder,) = stores.scheduling_db.list_reminder_tasks(owner_id="local")
        (rule,) = await stores.service.list_alert_rules(job_id=source_id)
        stores.scheduling_db.update_reminder_task(reminder["id"], enabled=True)
        await stores.service.update_alert_rule(rule["rule_id"], enabled=True)

        await track_reconcile.reconcile_at_boot(stores.app())

        assert await stores.running(source_id) == (False, True, True), (
            "the user's reminder and rule stand; the source stays off")
    finally:
        stores.close()


@pytest.mark.asyncio
async def test_a_retired_watchs_reminder_the_user_re_arms_while_off_stays_armed(
        tmp_path, settings, egress_seams, monkeypatch):
    from tldw_chatbook.Dreams.track_service import untrack

    stores = _RealStores(tmp_path, monkeypatch)
    try:
        tracked = await stores.track()
        db = DreamsDB(stores.dreams_path, "restore-test")
        try:
            await untrack(stores.service, db, tracked["tracked_item_id"],
                          scheduling_db_getter=lambda: stores.scheduling_db)
        finally:
            db.close_all()
        await stores.switch(settings, on=False)
        (reminder,) = stores.scheduling_db.list_reminder_tasks(owner_id="local")
        assert not reminder["enabled"], "fixture: untrack disarmed it"
        stores.scheduling_db.update_reminder_task(reminder["id"], enabled=True)

        await track_reconcile.reconcile_at_boot(stores.app())

        assert stores.scheduling_db.get_reminder_task(reminder["id"])["enabled"]
    finally:
        stores.close()


# --- Review of #3123, N3: a store that refuses the start check's read ----------


@pytest.mark.asyncio
async def test_a_store_that_refuses_the_start_read_holds_dreams_work_and_says_so(
        tmp_path, settings, egress_seams, monkeypatch):
    """The start does not fail and unrelated work runs, but Dreams' work
    stays held until the next start (an error never authorizes), and one
    warning names the store and the kind of refusal -- nothing else."""
    from loguru import logger

    stores = _RealStores(tmp_path, monkeypatch)
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)),
                      level="WARNING", format="{message}")
    try:
        await stores.track()

        class Refused(RuntimeError):
            pass

        def refuse(*args, **kwargs):
            raise Refused("restored profile awaiting review")

        monkeypatch.setattr(stores.subs_db, "run_states", refuse)
        gate = track_reconcile.DreamsDispatchGate()
        dreams_db = DreamsDB(stores.dreams_path, "boot-app")
        try:
            await track_reconcile.reconcile_at_boot(stores.app(dreams_db), gate)
        finally:
            dreams_db.close_all()

        assert gate.is_set() and not gate.dispatch_ready
        refusals = [m for m in messages if "refused" in m]
        assert len(refusals) == 1, messages
        assert "subscriptions" in refusals[0] and "Refused" in refusals[0]
        assert "awaiting review" not in refusals[0], "the reason type only"
    finally:
        logger.remove(sink)
        stores.close()
