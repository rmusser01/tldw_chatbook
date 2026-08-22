"""task-19562 part B: `LocalWatchlistsService` sqlite calls must not run on the loop.

22 `async def` methods on `LocalWatchlistsService` used to call `SubscriptionsDB`
synchronously inline, each blocking the event loop -- the whole TUI -- for its
own query duration. They now hop through `db_offload.run_db_off_loop`
(`asyncio.to_thread` under the hood), mirroring the pattern task-15463 already
established for `launch_run`/`execute_run`/`record_run_result` in this same
service.

The work order this task started from enumerated 19 of the 22 by an AST scan
that matched calls to *named* `SubscriptionsDB` methods (`db.get_subscription`,
`db.delete_subscription`, ...). `get_alert_rule`, `list_runs` and
`list_alert_rules` read through a bare `db.conn.cursor()` instead, so that
scan missed them -- but all three are exactly the same class of loop-blocking
read, and `get_alert_rule` is directly embedded in two of the 19
(`create_alert_rule`/`update_alert_rule` both end by awaiting it), so leaving
it inline would have left those two still blocking on their last statement.
19 + 3 = 22, which is exactly the count task-19562's own description names.

Threading note, same reasoning as `test_watchlists_db_instance_and_off_loop.py`:
every database here MUST be file-backed (`tmp_path`), never `:memory:`.
`run_db_off_loop` deliberately runs its callable INLINE, on the calling
thread, when `db.is_memory_db is True` -- `SubscriptionsDB` keeps thread-local
connections and builds its schema on the constructing thread, so an
in-memory database handed to a worker thread would be a private, empty
database nobody else can see. That carve-out means an in-memory DB here would
make every assertion below vacuously pass whether or not the offload actually
happened -- proving nothing. A file-backed database is what actually exercises
the `asyncio.to_thread` branch.

Two probes are used, matching the two shapes the service's edited call sites
take:

* Most methods call a *named* `SubscriptionsDB` method directly
  (`db.get_subscription`, `db.delete_subscription`, ...). `_spy` patches that
  bound method on the instance and records `threading.get_ident()` per call.
* The four methods that used to hold a `db.transaction()` block open
  (`cancel_run`, `create_alert_rule`, `update_alert_rule`,
  `delete_alert_rule`) were rewritten so the WHOLE transaction hops as one
  synchronous helper (per `run_db_off_loop`'s own contract: it must not be
  handed a callable that holds a transaction open across the await boundary).
  `_spy_transaction` patches `db.transaction` itself so entering it is what
  gets timestamped, regardless of how many statements run inside.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.Subscriptions.local_watchlists_service import (
    LocalWatchlistsService,
)

pytestmark = pytest.mark.unit


def _build_service_and_seed(tmp_path):
    """A file-backed `SubscriptionsDB` with one subscription and one item.

    Returns:
        ``(service, db, source_id, item_id)``.
    """
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    source_id = db.add_subscription(
        name="Feed", type="rss", source="https://example.com/feed.xml"
    )
    with db.transaction() as conn:
        item_id = persist_subscription_item(
            conn,
            source_id,
            {
                "url": "https://example.com/one/",
                "title": "One",
                "content_hash": "hash-off-loop",
            },
            run_id=None,
            now="2026-08-21T09:00:00+00:00",
        )
    return service, db, source_id, item_id


def _spy(db: SubscriptionsDB, name: str) -> list[int]:
    """Patch `db.<name>` to record the thread id of every call.

    Instance-attribute assignment shadows the class method, same technique
    `test_watchlists_db_instance_and_off_loop.py` and
    `test_local_watchlists_service.py` already use to spy on `SubscriptionsDB`.
    """
    threads: list[int] = []
    real = getattr(db, name)

    def wrapper(*args, **kwargs):
        threads.append(threading.get_ident())
        return real(*args, **kwargs)

    setattr(db, name, wrapper)
    return threads


def _spy_transaction(db: SubscriptionsDB) -> list[int]:
    """Patch `db.transaction` to record the thread id it is ENTERED on.

    The four rewritten methods hop the entire `with db.transaction():` block
    as one synchronous helper, so the meaningful timestamp is "what thread
    opened the transaction", not any one statement inside it.
    """
    threads: list[int] = []
    real_transaction = db.transaction

    @contextmanager
    def wrapper():
        threads.append(threading.get_ident())
        with real_transaction() as conn:
            yield conn

    db.transaction = wrapper
    return threads


# --- simple reads/writes: one db-method call per service call ---------------

_SIMPLE_CASES = [
    pytest.param(
        "get_all_subscriptions",
        lambda service, ctx: service.list_sources(),
        id="list_sources",
    ),
    pytest.param(
        "get_subscription",
        lambda service, ctx: service.get_source(ctx["source_id"]),
        id="get_source",
    ),
    pytest.param(
        "get_new_items",
        lambda service, ctx: service.list_items(),
        id="list_items",
    ),
    pytest.param(
        "get_item_status",
        lambda service, ctx: service.get_item_status(ctx["item_id"]),
        id="get_item_status",
    ),
    pytest.param(
        "get_item_content",
        lambda service, ctx: service.get_item_content(ctx["item_id"]),
        id="get_item_content",
    ),
    pytest.param(
        "get_url_snapshots",
        lambda service, ctx: service.get_url_snapshots(
            ctx["source_id"], "https://example.com/one/"
        ),
        id="get_url_snapshots",
    ),
    pytest.param(
        "mark_item_status",
        lambda service, ctx: service.update_item(
            item_id=ctx["item_id"], status="ingested"
        ),
        id="update_item",
    ),
    pytest.param(
        "mark_all_read",
        lambda service, ctx: service.mark_all_read(),
        id="mark_all_read",
    ),
    pytest.param(
        "restore_items_new",
        lambda service, ctx: service.restore_items_new(item_ids=[ctx["item_id"]]),
        id="restore_items_new",
    ),
    pytest.param(
        "set_item_flagged",
        lambda service, ctx: service.set_item_flagged(
            item_id=ctx["item_id"], flagged=True
        ),
        id="set_item_flagged",
    ),
    pytest.param(
        "get_subscription_id_by_source",
        lambda service, ctx: service.find_source_id_by_url(
            "https://example.com/feed.xml"
        ),
        id="find_source_id_by_url",
    ),
    pytest.param(
        "delete_subscription",
        lambda service, ctx: service.delete_source(ctx["source_id"]),
        id="delete_source",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("db_method_name", "call"), _SIMPLE_CASES)
async def test_simple_db_calls_run_off_the_event_loop_thread(
    tmp_path, db_method_name, call
):
    """Each of these service methods' single db call must hop to a worker thread.

    Mutation: revert any one of these methods' `run_db_off_loop` wrapping back
    to a bare inline call and its own case reddens, naming the db method that
    ran on the loop thread.
    """
    service, db, source_id, item_id = _build_service_and_seed(tmp_path)
    ctx = {"source_id": source_id, "item_id": item_id}
    threads = _spy(db, db_method_name)
    loop_thread = threading.get_ident()

    await call(service, ctx)

    assert threads, f"SubscriptionsDB.{db_method_name} must have been called"
    assert all(thread_id != loop_thread for thread_id in threads), (
        f"SubscriptionsDB.{db_method_name} ran on the event-loop thread: "
        f"{threads}"
    )


# --- multi-call methods -------------------------------------------------


@pytest.mark.asyncio
async def test_create_source_both_db_calls_run_off_the_event_loop_thread(tmp_path):
    """`create_source` does an INSERT then a re-read; both must hop."""
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    service = LocalWatchlistsService(db_factory=lambda: db)
    add_threads = _spy(db, "add_subscription")
    get_threads = _spy(db, "get_subscription")
    loop_thread = threading.get_ident()

    created = await service.create_source(
        {"name": "New Feed", "url": "https://example.com/new.xml"}
    )

    assert created["title"] == "New Feed"
    assert add_threads and all(t != loop_thread for t in add_threads)
    assert get_threads and all(t != loop_thread for t in get_threads)


@pytest.mark.asyncio
async def test_update_source_both_db_calls_run_off_the_event_loop_thread(tmp_path):
    """`update_source` does an UPDATE then a re-read; both must hop."""
    service, db, source_id, _ = _build_service_and_seed(tmp_path)
    update_threads = _spy(db, "update_subscription")
    get_threads = _spy(db, "get_subscription")
    loop_thread = threading.get_ident()

    updated = await service.update_source(source_id, {"name": "Renamed"})

    assert updated["title"] == "Renamed"
    assert update_threads and all(t != loop_thread for t in update_threads)
    assert get_threads and all(t != loop_thread for t in get_threads)


@pytest.mark.asyncio
async def test_resume_source_both_db_calls_run_off_the_event_loop_thread(tmp_path):
    """`resume_source`'s reset-then-read pair must both hop, in order."""
    service, db, source_id, _ = _build_service_and_seed(tmp_path)
    reset_threads = _spy(db, "reset_subscription_errors")
    get_threads = _spy(db, "get_subscription")
    loop_thread = threading.get_ident()

    await service.resume_source(source_id)

    assert reset_threads and all(t != loop_thread for t in reset_threads)
    assert get_threads and all(t != loop_thread for t in get_threads)


# --- the four rewritten `db.transaction()` blocks ---------------------------


@pytest.mark.asyncio
async def test_cancel_run_transaction_block_runs_off_the_event_loop_thread(tmp_path):
    """`cancel_run`'s whole transaction must hop as one unit."""
    service, db, source_id, _ = _build_service_and_seed(tmp_path)
    launched = await service.launch_run(source_id=source_id)
    run_id = launched["run_id"]
    txn_threads = _spy_transaction(db)
    loop_thread = threading.get_ident()

    cancelled = await service.cancel_run(run_id)

    assert cancelled["status"] == "cancelled"
    assert txn_threads, "db.transaction() must have been entered"
    assert all(t != loop_thread for t in txn_threads), (
        f"cancel_run's transaction opened on the event-loop thread: {txn_threads}"
    )


@pytest.mark.asyncio
async def test_create_alert_rule_transaction_block_runs_off_the_event_loop_thread(
    tmp_path,
):
    """`create_alert_rule`'s whole transaction must hop as one unit."""
    service, db, source_id, _ = _build_service_and_seed(tmp_path)
    txn_threads = _spy_transaction(db)
    loop_thread = threading.get_ident()

    rule = await service.create_alert_rule(
        name="No items", condition_type="no_items", source_id=source_id
    )

    assert rule["name"] == "No items"
    assert txn_threads, "db.transaction() must have been entered"
    assert all(t != loop_thread for t in txn_threads), (
        f"create_alert_rule's transaction opened on the event-loop thread: "
        f"{txn_threads}"
    )


@pytest.mark.asyncio
async def test_update_alert_rule_transaction_block_runs_off_the_event_loop_thread(
    tmp_path,
):
    """`update_alert_rule`'s whole transaction must hop as one unit."""
    service, db, source_id, _ = _build_service_and_seed(tmp_path)
    rule = await service.create_alert_rule(
        name="No items", condition_type="no_items", source_id=source_id
    )
    txn_threads = _spy_transaction(db)
    loop_thread = threading.get_ident()

    updated = await service.update_alert_rule(rule["rule_id"], name="Renamed rule")

    assert updated["name"] == "Renamed rule"
    assert txn_threads, "db.transaction() must have been entered"
    assert all(t != loop_thread for t in txn_threads), (
        f"update_alert_rule's transaction opened on the event-loop thread: "
        f"{txn_threads}"
    )


@pytest.mark.asyncio
async def test_delete_alert_rule_transaction_block_runs_off_the_event_loop_thread(
    tmp_path,
):
    """`delete_alert_rule`'s whole transaction must hop as one unit."""
    service, db, source_id, _ = _build_service_and_seed(tmp_path)
    rule = await service.create_alert_rule(
        name="No items", condition_type="no_items", source_id=source_id
    )
    txn_threads = _spy_transaction(db)
    loop_thread = threading.get_ident()

    result = await service.delete_alert_rule(rule["rule_id"])

    assert result["deleted"] is True
    assert txn_threads, "db.transaction() must have been entered"
    assert all(t != loop_thread for t in txn_threads), (
        f"delete_alert_rule's transaction opened on the event-loop thread: "
        f"{txn_threads}"
    )


@pytest.mark.asyncio
async def test_get_alert_rule_reads_off_the_event_loop_thread(tmp_path):
    """`get_alert_rule` was added to the offload sweep (not on the original
    enumerated list -- see the module docstring in `local_watchlists_service.py`
    at `get_alert_rule`) because `create_alert_rule`/`update_alert_rule` both
    end by awaiting it; leaving it inline would have left those two methods
    still blocking the loop on their last statement.
    """
    service, db, source_id, _ = _build_service_and_seed(tmp_path)
    rule = await service.create_alert_rule(
        name="No items", condition_type="no_items", source_id=source_id
    )
    # `get_alert_rule` reads via a bare `db.conn.cursor()`, not a named
    # SubscriptionsDB method, so it is not spy-able the way the simple cases
    # above are. Use the thread-local `set_trace_callback` probe instead
    # (same technique as `test_a_scheduled_check_runs_no_sqlite_on_the_
    # event_loop` in `test_watchlists_db_instance_and_off_loop.py`): any SQL
    # seen on the loop thread's OWN connection is, by construction, SQL that
    # ran inline -- `SubscriptionsDB` connections are thread-local, so a
    # worker thread's statements are invisible to this callback.
    loop_statements: list[str] = []
    db.conn.set_trace_callback(loop_statements.append)
    try:
        rule_again = await service.get_alert_rule(rule["rule_id"])
    finally:
        db.conn.set_trace_callback(None)

    assert rule_again["rule_id"] == rule["rule_id"]
    assert not loop_statements, (
        "get_alert_rule ran SQL on the event-loop thread: "
        f"{loop_statements[:3]}"
    )


@pytest.mark.asyncio
async def test_list_runs_reads_off_the_event_loop_thread(tmp_path):
    """`list_runs` also reads via a bare `db.conn.cursor()` -- the trace probe."""
    service, db, source_id, _ = _build_service_and_seed(tmp_path)
    await service.launch_run(source_id=source_id)

    loop_statements: list[str] = []
    db.conn.set_trace_callback(loop_statements.append)
    try:
        runs = await service.list_runs()
    finally:
        db.conn.set_trace_callback(None)

    assert len(runs) == 1, "the seeded run must have been read, or this proves nothing"
    assert not loop_statements, (
        f"list_runs ran SQL on the event-loop thread: {loop_statements[:3]}"
    )


@pytest.mark.asyncio
async def test_list_alert_rules_reads_off_the_event_loop_thread(tmp_path):
    """`list_alert_rules` also reads via a bare `db.conn.cursor()` -- the trace probe."""
    service, db, source_id, _ = _build_service_and_seed(tmp_path)
    await service.create_alert_rule(
        name="No items", condition_type="no_items", source_id=source_id
    )

    loop_statements: list[str] = []
    db.conn.set_trace_callback(loop_statements.append)
    try:
        rules = await service.list_alert_rules(source_id=source_id)
    finally:
        db.conn.set_trace_callback(None)

    assert len(rules) == 1, (
        "the seeded alert rule must have been read, or this proves nothing"
    )
    assert not loop_statements, (
        f"list_alert_rules ran SQL on the event-loop thread: {loop_statements[:3]}"
    )
