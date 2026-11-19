"""Time-based ``sync_log`` retention (task-34389, ADR-229).

The reachability retention from task-19564 bounds ``sync_log`` to the rows
its readers can reach, but the frontier rows of live entities still live
forever, and the media DB journal has no reachability bound at all -- the
prune APIs shipped with zero production callers. task-34389 adds the missing
half: a time-based window (default 90 days, ``[sync]
sync_log_retention_days``, ``0`` disables) swept by one parameterized
``DELETE`` per store, wired into the app's post-startup maintenance path
(next to change-review retention).

What this pins:

* Both stores' ``delete_sync_log_entries_older_than`` delete ONLY rows
  strictly older than the cutoff, including rows written by the real
  trigger/``_log_sync_event`` paths (so the ISO ``YYYY-MM-DDTHH:MM:SS.mmmZ``
  timestamp shapes are what actually get compared).
* A row at exactly the cutoff survives (strictly-older semantics).
* The app-level pass reads the config knob, prunes BOTH stores in one pass,
  and ``0`` -- the documented off-switch -- touches neither.
* The shipped config carries the knob and a fresh profile resolves its
  default through the real loader.
"""

from __future__ import annotations

import asyncio
import tomllib
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

OLD_TS = "2020-01-01T00:00:00.000Z"


def _cutoff(days: int = 90) -> str:
    """The cutoff string the app pass computes (mirrors app.py)."""
    return (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@pytest.fixture()
def chacha_db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "sync_retention.db", "retention_client")
    try:
        yield database
    finally:
        database.close_connection()


@pytest.fixture()
def media_db(tmp_path: Path) -> MediaDatabase:
    # File-backed on purpose: the app pass runs the DELETE on a worker
    # thread, and MediaDatabase keeps thread-local connections, so an
    # in-memory DB would be a different (empty) database over there --
    # production DBs are file-backed, and so is this fixture.
    database = MediaDatabase(db_path=str(tmp_path / "media_sync_retention.db"), client_id="retention_client")
    try:
        yield database
    finally:
        database.close_connection()


@pytest.fixture()
def dreams_db(tmp_path: Path):
    from tldw_chatbook.DB.Dreams_DB import DreamsDB

    database = DreamsDB(tmp_path / "dreams.sqlite", "retention_client")
    try:
        yield database
    finally:
        database.close_all()


@pytest.fixture()
def guardian_db(tmp_path: Path):
    from tldw_chatbook.DB.Guardian_DB import GuardianDB

    database = GuardianDB(tmp_path / "guardian.sqlite", "retention_client")
    try:
        yield database
    finally:
        database.close()


def _media_log_event(db: MediaDatabase, entity_uuid: str, version: int = 1) -> None:
    """Write one sync event through the real production logging path."""
    with db.transaction() as conn:
        db._log_sync_event(
            conn,
            entity="Media",
            entity_uuid=entity_uuid,
            operation="create",
            version=version,
            payload={"uuid": entity_uuid, "content": "payload body"},
        )


# ---------------------------------------------------------------------------
# ChaChaNotes_DB.delete_sync_log_entries_older_than
# ---------------------------------------------------------------------------
def test_chachanotes_prune_deletes_only_rows_older_than_cutoff(
    chacha_db: CharactersRAGDB,
):
    """Rows backdated past the window leave; fresh trigger-written rows stay."""
    cutoff = _cutoff()
    conversation_id = chacha_db.add_conversation(
        {"title": "retention probe", "character_id": 1}
    )
    fresh_message_id = chacha_db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "fresh body",
        }
    )

    total_rows = chacha_db.execute_query(
        "SELECT COUNT(*) AS n FROM sync_log"
    ).fetchone()["n"]
    assert total_rows > 0

    # Backdate everything EXCEPT the fresh message's rows, via direct SQL.
    chacha_db.execute_query(
        "UPDATE sync_log SET timestamp = ? WHERE NOT (entity = 'messages' "
        "AND entity_id = ?)",
        (OLD_TS, fresh_message_id),
        commit=True,
    )
    backdated = chacha_db.execute_query(
        "SELECT COUNT(*) AS n FROM sync_log WHERE timestamp < ?",
        (cutoff,),
    ).fetchone()["n"]
    assert backdated > 0

    deleted = chacha_db.delete_sync_log_entries_older_than(cutoff)

    assert deleted == backdated
    # Compare in SQL: the DATETIME column comes back as a Python datetime.
    stale_left = chacha_db.execute_query(
        "SELECT COUNT(*) AS n FROM sync_log WHERE timestamp < ?",
        (cutoff,),
    ).fetchone()["n"]
    assert stale_left == 0
    remaining = chacha_db.execute_query(
        "SELECT COUNT(*) AS n FROM sync_log"
    ).fetchone()["n"]
    assert remaining > 0, "the fresh message's sync rows must survive"


def test_chachanotes_prune_keeps_row_at_exact_cutoff(chacha_db: CharactersRAGDB):
    """Strictly-older semantics: a row stamped exactly AT the cutoff survives."""
    cutoff = _cutoff()
    with chacha_db.transaction() as conn:
        conn.execute(
            "INSERT INTO sync_log(entity, entity_id, operation, timestamp, "
            "client_id, version, payload) VALUES('messages', 'm-boundary', "
            "'create', ?, 'retention_client', 1, '{}')",
            (cutoff,),
        )
        conn.execute(
            "INSERT INTO sync_log(entity, entity_id, operation, timestamp, "
            "client_id, version, payload) VALUES('messages', 'm-ancient', "
            "'create', ?, 'retention_client', 1, '{}')",
            (OLD_TS,),
        )

    deleted = chacha_db.delete_sync_log_entries_older_than(cutoff)

    assert deleted == 1
    survivors = {
        row["entity_id"]
        for row in chacha_db.execute_query(
            "SELECT entity_id FROM sync_log"
        ).fetchall()
    }
    # Membership, not an exact set: a fresh profile seeds its own (current-
    # timestamp) rows, which legitimately survive.
    assert "m-boundary" in survivors
    assert "m-ancient" not in survivors


def test_chachanotes_prune_writes_still_land_after_prune(
    chacha_db: CharactersRAGDB,
):
    """Pruning is a pure DELETE: triggers keep emitting afterwards."""
    chacha_db.delete_sync_log_entries_older_than(OLD_TS)
    conversation_id = chacha_db.add_conversation(
        {"title": "post-prune", "character_id": 1}
    )
    hits = chacha_db.execute_query(
        "SELECT COUNT(*) AS n FROM sync_log WHERE entity = 'conversations' "
        "AND entity_id = ?",
        (conversation_id,),
    ).fetchone()["n"]
    assert hits >= 1


@pytest.mark.parametrize("bad_cutoff", ["", None, 123])
def test_chachanotes_prune_rejects_invalid_cutoff(
    chacha_db: CharactersRAGDB, bad_cutoff
):
    with pytest.raises(ValueError):
        chacha_db.delete_sync_log_entries_older_than(bad_cutoff)


# ---------------------------------------------------------------------------
# Client_Media_DB_v2.delete_sync_log_entries_older_than
# ---------------------------------------------------------------------------
def test_media_prune_deletes_only_rows_older_than_cutoff(media_db: MediaDatabase):
    """Events backdated past the window leave; fresh ones stay."""
    for i in range(3):
        _media_log_event(media_db, f"uuid-{i}")

    media_db.execute_query(
        "UPDATE sync_log SET timestamp = ? WHERE entity_uuid IN ('uuid-0', 'uuid-1')",
        (OLD_TS,),
        commit=True,
    )

    deleted = media_db.delete_sync_log_entries_older_than(_cutoff())

    assert deleted == 2
    survivors = {
        row["entity_uuid"]
        for row in media_db.execute_query(
            "SELECT entity_uuid FROM sync_log"
        ).fetchall()
    }
    assert survivors == {"uuid-2"}


def test_media_prune_keeps_row_at_exact_cutoff(media_db: MediaDatabase):
    cutoff = _cutoff()
    media_db.execute_query(
        "INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, "
        "client_id, version, payload) VALUES('Media', 'uuid-boundary', "
        "'create', ?, 'retention_client', 1, '{}')",
        (cutoff,),
        commit=True,
    )

    assert media_db.delete_sync_log_entries_older_than(cutoff) == 0
    assert (
        media_db.execute_query(
            "SELECT COUNT(*) AS n FROM sync_log"
        ).fetchone()["n"]
        == 1
    )


@pytest.mark.parametrize("bad_cutoff", ["", None, 123])
def test_media_prune_rejects_invalid_cutoff(media_db: MediaDatabase, bad_cutoff):
    with pytest.raises(ValueError):
        media_db.delete_sync_log_entries_older_than(bad_cutoff)


# ---------------------------------------------------------------------------
# The app-level pass: TldwCli._perform_sync_log_retention
# ---------------------------------------------------------------------------
def _fake_app(**dbs) -> SimpleNamespace:
    return SimpleNamespace(loguru_logger=loguru_logger, **dbs)


def _run_pass(monkeypatch: pytest.MonkeyPatch, fake_app: SimpleNamespace, days) -> None:
    """Drive LifecycleMixin._perform_sync_log_retention without booting the app."""
    from tldw_chatbook import app_lifecycle as lifecycle_module

    # Patch the name the code actually reads: the pass lives on LifecycleMixin
    # and resolves get_cli_setting in app_lifecycle's namespace (the perf
    # snapshot carried a seam pointing at tldw_chatbook.app, the pre-port
    # home -- patching there silently missed, so the "0 disables" case ran a
    # real 90-day prune and failed).
    monkeypatch.setattr(
        lifecycle_module,
        "get_cli_setting",
        lambda *args, **kwargs: days,
    )
    asyncio.run(
        lifecycle_module.LifecycleMixin._perform_sync_log_retention(fake_app)
    )


def test_app_pass_prunes_both_stores(
    chacha_db: CharactersRAGDB,
    media_db: MediaDatabase,
    monkeypatch: pytest.MonkeyPatch,
):
    """One pass prunes both journals; nothing under the window is touched."""
    chacha_db.add_conversation({"title": "app pass", "character_id": 1})
    chacha_db.execute_query(
        "UPDATE sync_log SET timestamp = ?", (OLD_TS,), commit=True
    )
    _media_log_event(media_db, f"uuid-{uuid.uuid4().hex[:8]}")
    media_db.execute_query(
        "UPDATE sync_log SET timestamp = ?", (OLD_TS,), commit=True
    )

    _run_pass(monkeypatch, _fake_app(chachanotes_db=chacha_db, media_db=media_db), 90)

    # The contract: NOTHING older than the window survives in either store.
    # (A fresh profile's per-thread connection may legitimately seed new
    # current-timestamp rows during the pass; those are not the target.)
    for db in (chacha_db, media_db):
        stale_left = db.execute_query(
            "SELECT COUNT(*) AS n FROM sync_log WHERE timestamp < ?",
            (_cutoff(),),
        ).fetchone()["n"]
        assert stale_left == 0


def test_app_pass_retires_the_handles_it_opens_on_its_worker_thread(
    chacha_db: CharactersRAGDB,
    media_db: MediaDatabase,
    monkeypatch: pytest.MonkeyPatch,
):
    """TASK-38955: the pass opens a handle per store on a default-executor
    thread. Left open there, its storage lease keeps a backup pause from
    draining, and no close on the UI thread can reach it."""
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook import app_lifecycle as lifecycle_module
    from tldw_chatbook.Backup_Recovery import storage_admission

    def live_leases() -> set:
        with storage_admission._lock:
            return set(storage_admission._live_leases)

    chacha_db.add_conversation({"title": "app pass", "character_id": 1})
    _media_log_event(media_db, f"uuid-{uuid.uuid4().hex[:8]}")
    connections_before = chacha_db.registered_connection_count()
    leases_before = live_leases()
    monkeypatch.setattr(lifecycle_module, "get_cli_setting", lambda *a, **k: 90)
    # Kept alive past the pass, as the app's default pool threads are.
    pool = ThreadPoolExecutor(max_workers=1)

    async def one_pass():
        asyncio.get_running_loop().set_default_executor(pool)
        await lifecycle_module.LifecycleMixin._perform_sync_log_retention(
            _fake_app(chachanotes_db=chacha_db, media_db=media_db)
        )
        return chacha_db.registered_connection_count(), live_leases() - leases_before

    try:
        connections_after, new_leases = asyncio.run(one_pass())
    finally:
        pool.shutdown(wait=True)

    # (registered before, registered after, new live leases)
    assert (connections_before, connections_after, len(new_leases)) == (1, 1, 0)


def _unmount_owner(
    monkeypatch: pytest.MonkeyPatch, *, on_workers_settled=lambda: None, **stores
):
    """A ``LifecycleMixin`` stub around real stores, for a real ``on_unmount``."""
    from unittest.mock import AsyncMock, Mock

    from tldw_chatbook import app_lifecycle as lifecycle_module

    owner = object.__new__(type("_Owner", (lifecycle_module.LifecycleMixin,), {}))
    owner.__dict__.update(
        **stores,
        loguru_logger=Mock(),
        ui_responsiveness_monitor=None,
        _rich_log_handler=None,
        db_status_manager=SimpleNamespace(stop_periodic_updates=Mock()),
        _shutdown_actor_pack_recovery=AsyncMock(return_value=None),
        _shutdown_workflow_session=AsyncMock(),
        _settle_speech_initialization=AsyncMock(return_value=None),
        _stop_served_canvas_control=AsyncMock(),
        _shutdown_app_owned_lifecycles=AsyncMock(),
        _cancel_and_settle_workers=AsyncMock(
            side_effect=lambda *_: on_workers_settled()
        ),
        _close_owned_tts_resources=AsyncMock(),
        _stop_ui_responsiveness_monitor=Mock(),
        _shutdown_ingest_parse_pool=Mock(),
        _stop_footer_status_timers=Mock(),
    )
    monkeypatch.setattr(lifecycle_module, "arm_exit_watchdog", Mock())
    monkeypatch.setattr(lifecycle_module, "persist_event", Mock())
    monkeypatch.setattr(lifecycle_module, "unregister_running_app", Mock())
    monkeypatch.setattr(lifecycle_module.logging, "shutdown", Mock())
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle."
        "stop_all_server_processes",
        lambda app: [],
    )
    return owner


async def test_app_unmount_retires_both_stores_on_every_thread(
    chacha_db: CharactersRAGDB,
    media_db: MediaDatabase,
    monkeypatch: pytest.MonkeyPatch,
):
    """TASK-38955: quit closed neither store, so their handles and storage
    leases outlived the app, on the UI thread and on worker threads."""
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.Utils.private_paths import lexical_path

    paths = {lexical_path(chacha_db.db_path), lexical_path(media_db.db_path)}

    def store_leases() -> list:
        with storage_admission._lock:
            return [
                lease
                for lease in storage_admission._live_leases
                if lease.resource_path in paths
            ]

    chacha_db.add_conversation({"title": "unmount", "character_id": 1})
    _media_log_event(media_db, f"uuid-{uuid.uuid4().hex[:8]}")
    pool = ThreadPoolExecutor(max_workers=1)
    # A worker-thread handle that nothing on that thread closes.
    await asyncio.get_running_loop().run_in_executor(pool, chacha_db.get_connection)
    assert chacha_db.registered_connection_count() == 2

    # ChaChaNotes handles still registered when the workers settled.
    open_when_workers_settled: list[int] = []
    owner = _unmount_owner(
        monkeypatch,
        on_workers_settled=lambda: open_when_workers_settled.append(
            chacha_db.registered_connection_count()
        ),
        chachanotes_db=chacha_db,
        media_db=media_db,
    )
    try:
        await owner.on_unmount()
        left = (chacha_db.registered_connection_count(), len(store_leases()))
    finally:
        pool.shutdown(wait=True)

    # (registered ChaChaNotes handles, live leases on either store); the
    # stores close after the workers settle, not before.
    assert (left, open_when_workers_settled) == ((0, 0), [2])


def _lease_threads(db) -> list[str]:
    """Threads holding a live storage lease on ``db``'s file."""
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.Utils.private_paths import lexical_path

    path = lexical_path(db.db_path)
    with storage_admission._lock:
        return sorted(
            lease.resource_thread.name
            for lease in storage_admission._live_leases
            if lease.resource_path == path
        )


@pytest.mark.parametrize("store", ["media", "dreams", "guardian"])
def test_app_unmount_retires_a_worker_thread_handle_of_each_store(
    store: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
):
    """TASK-39047: a handle media, Dreams or Guardian opened on a pool thread
    (the scope picker's tag lister, an ``asyncio.to_thread`` read) outlived
    the quit with its storage lease, and kept a backup pause from draining."""
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook.Chat.scope_picker_listers import build_keyword_tag_lister

    db = request.getfixturevalue(f"{store}_db")
    opens_on_a_pool_thread = {
        "media": lambda: build_keyword_tag_lister(SimpleNamespace(media_db=db))(""),
        "dreams": lambda: asyncio.to_thread(db.list_profile),
        "guardian": lambda: asyncio.to_thread(db.get_meta, "quit"),
    }[store]
    kept = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kept")

    async def quit_after_a_worker_read() -> tuple[list[str], list[str]]:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(kept)
        await opens_on_a_pool_thread()
        opened = _lease_threads(db)
        # The worker stays alive across the quit; the quit hops elsewhere.
        loop.set_default_executor(ThreadPoolExecutor(1, thread_name_prefix="quit"))
        await _unmount_owner(monkeypatch, **{f"{store}_db": db}).on_unmount()
        return opened, _lease_threads(db)

    try:
        opened, left = asyncio.run(quit_after_a_worker_read())
    finally:
        kept.shutdown(wait=True)

    assert "kept_0" in opened
    assert left == []


def _loop_gaps() -> tuple[list[float], asyncio.Task]:
    """Record the event loop's gaps between 20 ms ticks."""
    import time

    gaps: list[float] = []

    async def ticker() -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    return gaps, asyncio.create_task(ticker())


async def test_app_unmount_store_close_never_aborts_a_straggler_write(
    chacha_db: CharactersRAGDB,
):
    """TASK-38955 review: a worker can outlive the unmount's cancel grace.
    A waiting quiescence barrier made its next statement raise and rolled its
    transaction back; the quit close must never abort a write."""
    import threading
    from unittest.mock import Mock

    from tldw_chatbook import app_lifecycle as lifecycle_module

    chacha_db.get_connection().execute(
        "CREATE TABLE IF NOT EXISTS quit_probe (n INTEGER NOT NULL)"
    )
    stop, running = threading.Event(), threading.Event()
    errors: list[str] = []
    committed = [0]

    def straggler() -> None:
        try:
            while not stop.is_set():
                with chacha_db.transaction() as cursor:
                    cursor.execute("INSERT INTO quit_probe (n) VALUES (?)", (committed[0],))
                    cursor.execute("INSERT INTO quit_probe (n) VALUES (?)", (committed[0],))
                committed[0] += 1
                running.set()
        except Exception as exc:  # noqa: BLE001 - the assertion reports it
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            chacha_db.close_connection()

    writer = threading.Thread(target=straggler)
    writer.start()
    assert running.wait(5)
    owner = SimpleNamespace(chachanotes_db=chacha_db, loguru_logger=Mock())
    try:
        await lifecycle_module.LifecycleMixin._close_main_databases(owner)
        await asyncio.sleep(0.2)  # the straggler keeps writing past the close
    finally:
        stop.set()
        writer.join(10)

    rows = chacha_db.get_connection().execute(
        "SELECT COUNT(*) FROM quit_probe"
    ).fetchone()[0]
    assert (errors, rows) == ([], 2 * committed[0])


_STRAGGLER_WRITES = {
    "media": ("quit_probe", "INSERT INTO quit_probe (n) VALUES (?)"),
    "dreams": (
        "dream_seen_items",
        (
            "INSERT INTO dream_seen_items (url, title_digest, first_seen, last_seen)"
            " VALUES (?, 'digest', 'now', 'now')"
        ),
    ),
    "guardian": (
        "guardian_meta",
        "INSERT INTO guardian_meta (key, value) VALUES (?, 'v')",
    ),
}


def _rows(db, table: str) -> int:
    with db.transaction() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.mark.parametrize("store", ["media", "dreams", "guardian"])
async def test_app_unmount_store_close_never_aborts_a_straggler_write_in_each_store(
    store: str, request: pytest.FixtureRequest
):
    """TASK-39047: the quit's idle close of media, Dreams and Guardian, as for
    ChaChaNotes above. A worker writing across the close keeps its open
    transaction and commits it, and the close returns at once."""
    import threading
    import time
    from unittest.mock import Mock

    from tldw_chatbook import app_lifecycle as lifecycle_module

    db = request.getfixturevalue(f"{store}_db")
    table, insert = _STRAGGLER_WRITES[store]
    if store == "media":  # Dreams and Guardian refuse a table their schema lacks
        with db.transaction() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS quit_probe (n INTEGER NOT NULL)")
    rows_before = _rows(db, table)
    entered, release = threading.Event(), threading.Event()
    outcome: list[str] = []

    def straggler() -> None:
        try:
            with db.transaction() as conn:
                conn.execute(insert, ("a",))
                entered.set()
                release.wait(20)
                conn.execute(insert, ("b",))
            outcome.append("committed")
        except Exception as exc:  # noqa: BLE001 - the assertion reports it
            outcome.append(f"{type(exc).__name__}: {exc}")
            entered.set()
        finally:
            db.close()

    writer = threading.Thread(target=straggler)
    writer.start()
    assert entered.wait(5)
    owner = SimpleNamespace(**{f"{store}_db": db}, loguru_logger=Mock())
    started = time.monotonic()
    try:
        await lifecycle_module.LifecycleMixin._close_main_databases(owner)
        took = time.monotonic() - started
    finally:
        release.set()
        writer.join(10)

    assert (outcome, _rows(db, table) - rows_before, took < 1.0) == (
        ["committed"],
        2,
        True,
    ), took


@pytest.mark.parametrize("store", ["media", "dreams", "guardian"])
def test_idle_close_refuses_while_a_getter_is_opening_a_handle(
    store: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
):
    """TASK-39047: a getter holds a registry acquisition while it opens or
    reuses a handle, so the idle close cannot close one it is returning. The
    getter is called raw here, outside any core operation."""
    import threading

    from tldw_chatbook.DB import Client_Media_DB_v2 as media_module

    db = request.getfixturevalue(f"{store}_db")
    opening, release = threading.Event(), threading.Event()
    if store == "media":
        real, owner, name, get = (
            media_module.connect_private_sqlite,
            media_module,
            "connect_private_sqlite",
            db.get_connection,
        )
    else:
        real, owner, name, get = (
            db._open_connection,
            db,
            "_open_connection",
            db._held_connection,
        )

    def slow_open(*args, **kwargs):
        opening.set()
        release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, slow_open)

    def open_on_a_worker() -> None:
        try:
            get()
        finally:
            db.close()

    worker = threading.Thread(target=open_on_a_worker)
    worker.start()
    try:
        assert opening.wait(5)
        while_opening = db.close_idle_connections()
    finally:
        release.set()
        worker.join(10)

    assert (while_opening, db.close_idle_connections()) == (False, True)


def test_kernel_idle_close_is_a_no_op_for_a_store_without_a_registry(tmp_path):
    """TASK-39047 review: only Dreams and Guardian keep a registry; the idle
    close of any other kernel store raised AttributeError."""
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    db = WorkspaceDB(tmp_path / "workspace.sqlite")
    try:
        assert db.close_idle_connections() is True
    finally:
        db.close()


@contextmanager
def _raw_handle_in_a_core_operation(db):
    """A raw handle held inside a counted core operation, outside a transaction."""
    from tldw_chatbook.Backup_Recovery.participants import core_operation

    if isinstance(db, (CharactersRAGDB, MediaDatabase)):
        with core_operation(db):
            yield db.get_connection()
    else:
        with db.connection() as conn:  # the kernel's own counted scope
            yield conn


@pytest.mark.parametrize("store", ["chachanotes", "media", "dreams", "guardian"])
async def test_app_unmount_store_close_skips_a_store_with_a_core_operation_running(
    store: str, request: pytest.FixtureRequest
):
    """TASK-39047, TASK-39048: a worker inside a core operation holds a raw
    handle between two statements, with no transaction open. The quit close
    must leave the store to close with the process; closing the handle under
    it raised ``ProgrammingError: Cannot operate on a closed database``."""
    import threading
    from unittest.mock import Mock

    from tldw_chatbook import app_lifecycle as lifecycle_module

    db = request.getfixturevalue("chacha_db" if store == "chachanotes" else f"{store}_db")
    held, release = threading.Event(), threading.Event()
    outcome: list[str] = []

    def straggler() -> None:
        try:
            with _raw_handle_in_a_core_operation(db) as conn:
                conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchall()
                held.set()
                release.wait(20)
                conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchall()
            outcome.append("read twice")
        except Exception as exc:  # noqa: BLE001 - the assertion reports it
            outcome.append(f"{type(exc).__name__}: {exc}")
            held.set()
        finally:
            db.close()

    writer = threading.Thread(target=straggler)
    writer.start()
    assert held.wait(5)
    logger = Mock()
    owner = SimpleNamespace(**{f"{store}_db": db}, loguru_logger=logger)
    try:
        await lifecycle_module.LifecycleMixin._close_main_databases(owner)
    finally:
        release.set()
        writer.join(10)

    label = {
        "chachanotes": "ChaChaNotes",
        "media": "Media",
        "dreams": "Dreams",
        "guardian": "Guardian",
    }[store]
    assert (outcome, [call.args[0] for call in logger.info.call_args_list]) == (
        ["read twice"],
        [f"{label} still in use at quit; its handles close with the process"],
    )


async def test_app_unmount_store_close_skips_a_busy_store_without_blocking(
    chacha_db: CharactersRAGDB,
):
    """TASK-38955 review: a straggler mid-transaction, holding the WAL write
    lock. The close must not wait for it, freeze the loop (the UI-thread
    close's checkpoint used to wait out busy_timeout, 15 s) or abort it; the
    busy store is left to close with the process."""
    import threading
    import time
    from unittest.mock import Mock

    from tldw_chatbook import app_lifecycle as lifecycle_module

    chacha_db.add_conversation({"title": "ui handle", "character_id": 1})
    chacha_db.get_connection().execute(
        "CREATE TABLE IF NOT EXISTS quit_probe (n INTEGER NOT NULL)"
    )
    entered, release = threading.Event(), threading.Event()
    outcome: list[str] = []

    def stuck_writer() -> None:
        try:
            with chacha_db.transaction() as cursor:
                cursor.execute("INSERT INTO quit_probe (n) VALUES (1)")
                entered.set()
                release.wait(20)
            outcome.append("committed")
        except Exception as exc:  # noqa: BLE001 - the assertion reports it
            outcome.append(f"{type(exc).__name__}: {exc}")
            entered.set()
        finally:
            chacha_db.close_connection()

    writer = threading.Thread(target=stuck_writer)
    writer.start()
    assert entered.wait(5)
    gaps, beat = _loop_gaps()
    await asyncio.sleep(0.1)
    owner = SimpleNamespace(chachanotes_db=chacha_db, loguru_logger=Mock())
    started = time.monotonic()
    try:
        await lifecycle_module.LifecycleMixin._close_main_databases(owner)
        took = time.monotonic() - started
        await asyncio.sleep(0.05)
    finally:
        beat.cancel()
        release.set()
        writer.join(10)

    assert (took < 1.0, max(gaps) < 1.0, outcome) == (True, True, ["committed"]), (
        took,
        max(gaps),
    )
    assert chacha_db.registered_connection_count() >= 1  # left for process exit


async def test_app_unmount_ui_close_bounds_its_checkpoint_wait(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    """TASK-38955 review: a UI-thread handle opened while the idle close ran
    on its worker is closed on the loop, and that close checkpoints. With
    another connection holding the write lock it waited out busy_timeout
    (15 s); the quit close bounds the wait."""
    import sqlite3
    import time
    from unittest.mock import Mock

    from tldw_chatbook import app_lifecycle as lifecycle_module

    chacha_db.add_conversation({"title": "ui handle", "character_id": 1})
    chacha_db.get_connection().execute(
        "CREATE TABLE IF NOT EXISTS quit_probe (n INTEGER NOT NULL)"
    )
    # Stands in for an idle close that finished before the UI handle opened.
    monkeypatch.setattr(chacha_db, "close_idle_connections", lambda: True)
    holder = sqlite3.connect(chacha_db.db_path_str, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("INSERT INTO quit_probe (n) VALUES (1)")
    gaps, beat = _loop_gaps()
    await asyncio.sleep(0.1)
    owner = SimpleNamespace(chachanotes_db=chacha_db, loguru_logger=Mock())
    started = time.monotonic()
    try:
        await lifecycle_module.LifecycleMixin._close_main_databases(owner)
        took = time.monotonic() - started
        await asyncio.sleep(0.05)
    finally:
        beat.cancel()
        holder.rollback()
        holder.close()

    bound = lifecycle_module._QUIT_CLOSE_BUSY_SECONDS
    assert (took < bound + 1.0, max(gaps) < bound + 1.0) == (True, True), (
        took,
        max(gaps),
    )

def test_app_pass_zero_disables(
    chacha_db: CharactersRAGDB,
    media_db: MediaDatabase,
    monkeypatch: pytest.MonkeyPatch,
):
    """``[sync] sync_log_retention_days = 0`` is the documented off-switch."""
    chacha_db.add_conversation({"title": "keep everything", "character_id": 1})
    chacha_db.execute_query(
        "UPDATE sync_log SET timestamp = ?", (OLD_TS,), commit=True
    )
    _media_log_event(media_db, f"uuid-{uuid.uuid4().hex[:8]}")
    media_db.execute_query(
        "UPDATE sync_log SET timestamp = ?", (OLD_TS,), commit=True
    )

    _run_pass(monkeypatch, _fake_app(chachanotes_db=chacha_db, media_db=media_db), 0)

    # Ancient rows are all still there: the pass never ran a DELETE.
    for db in (chacha_db, media_db):
        stale_left = db.execute_query(
            "SELECT COUNT(*) AS n FROM sync_log WHERE timestamp < ?",
            (_cutoff(),),
        ).fetchone()["n"]
        assert stale_left > 0


def test_app_pass_skips_missing_dbs(monkeypatch: pytest.MonkeyPatch):
    """A store that is not open (media_db is per-host, may be None) is skipped."""

    class RecordingDB:
        def __init__(self):
            self.cutoffs: list[str] = []

        def delete_sync_log_entries_older_than(self, cutoff: str) -> int:
            self.cutoffs.append(cutoff)
            return 5

    recording = RecordingDB()
    _run_pass(monkeypatch, _fake_app(chachanotes_db=recording, media_db=None), 90)
    assert len(recording.cutoffs) == 1
    # The cutoff is the ISO-Z shape both journals store.
    assert recording.cutoffs[0].endswith("Z")
    assert "T" in recording.cutoffs[0]


def test_app_pass_invalid_config_value_is_contained(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    """A garbage knob value logs and skips; it never raises out of the pass."""
    chacha_db.add_conversation({"title": "invalid knob", "character_id": 1})
    before = chacha_db.execute_query(
        "SELECT COUNT(*) AS n FROM sync_log"
    ).fetchone()["n"]

    _run_pass(
        monkeypatch,
        _fake_app(chachanotes_db=chacha_db, media_db=None),
        "not-a-number",
    )

    assert (
        chacha_db.execute_query("SELECT COUNT(*) AS n FROM sync_log").fetchone()["n"]
        == before
    )


# ---------------------------------------------------------------------------
# The shipped config carries the knob (Tests/test_config_chunking_defaults shape)
# ---------------------------------------------------------------------------
def test_config_template_ships_sync_section():
    from tldw_chatbook import config as config_module

    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)

    assert template["sync"]["sync_log_retention_days"] == 90


def test_default_config_tree_carries_sync_section():
    from tldw_chatbook import config as config_module

    assert config_module.DEFAULT_CONFIG_FROM_TOML["sync"]["sync_log_retention_days"] == 90


def test_real_loader_resolves_sync_retention_default(tmp_path, monkeypatch):
    """A fresh profile (missing config file) still resolves the knob."""
    from Tests.Backup_Recovery.config_test_support import select_config_source

    # Select the fresh profile through the ADR-126-compliant seam: a raw
    # TLDW_CONFIG_PATH redirect fails the config-participant admission closed
    # (RecoveryRequired) because the imported source stays bound to the
    # bootstrap selection. select_config_source installs an independent
    # source module, exactly as Tests/test_config_chunking_defaults.py does.
    source = select_config_source(monkeypatch, tmp_path / "missing-config.toml")

    settings = source.load_settings(force_reload=True)

    assert settings["sync"]["sync_log_retention_days"] == 90
    assert source.get_cli_setting("sync", "sync_log_retention_days", None) == 90
