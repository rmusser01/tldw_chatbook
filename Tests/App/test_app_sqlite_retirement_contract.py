"""App retirement preserves actual borrowers, not merely cached references."""

import asyncio
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.app_lifecycle import LifecycleMixin
from tldw_chatbook.Backup_Recovery.participants import _core_operation
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _handles(app):
    return (
        (app.chachanotes_db, app.chachanotes_db._local, "conn"),
        (
            app.evaluation_orchestrator.db,
            app.evaluation_orchestrator.db._local,
            "connection",
        ),
        (
            app.local_library_collections_db,
            app.local_library_collections_db._thread_local,
            "conn",
        ),
        (app.subscriptions_db, app.subscriptions_db._local, "conn"),
        (app.local_workspace_db, app.local_workspace_db._thread_local, "conn"),
    )


def _assert_closed(connection):
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


@pytest.fixture
def app(tmp_path):
    application = SimpleNamespace(
        chachanotes_db=CharactersRAGDB(tmp_path / "chat.sqlite", "app-retirement"),
        evaluation_orchestrator=EvaluationOrchestrator(tmp_path / "evals.sqlite"),
        local_library_collections_db=LibraryCollectionsDB(
            tmp_path / "collections.sqlite"
        ),
        subscriptions_db=SubscriptionsDB(tmp_path / "subscriptions.sqlite"),
        local_workspace_db=WorkspaceDB(tmp_path / "workspace.sqlite"),
        _chat_sqlite_borrowed=(None, None),
    )
    yield application
    # Cleanup is deliberately after native assertions, never their proof.
    for owner, cache, attribute in _handles(application):
        connection = getattr(cache, attribute, None)
        if connection is not None:
            connection.rollback()
        owner.close()


async def test_cold_handles_really_close_and_retirement_is_repeat_safe(app):
    connections = [getattr(cache, attribute) for _, cache, attribute in _handles(app)]
    await LifecycleMixin._retire_app_sqlite_handles(app)
    for connection in connections:
        _assert_closed(connection)
    await LifecycleMixin._retire_app_sqlite_handles(app)


@pytest.mark.parametrize("replace", [False, True], ids=["warm-borrower", "replacement"])
async def test_shared_chat_borrower_is_preserved_by_native_identity(app, replace):
    db = app.chachanotes_db
    previous = db.get_connection()
    app._chat_sqlite_borrowed = (db, previous)
    if replace:
        db.close_connection()
    current = db.get_connection()
    await LifecycleMixin._retire_app_sqlite_handles(app)
    if replace:
        assert current is not previous
        _assert_closed(current)
    else:
        assert current is previous
        assert current.execute("SELECT 1").fetchone()[0] == 1


@pytest.mark.parametrize(
    "index",
    range(5),
    ids=["chat", "evals", "collections", "subscriptions", "workspace"],
)
@pytest.mark.parametrize(
    "managed", [False, True], ids=["raw-transaction", "managed-scope"]
)
async def test_busy_native_work_is_not_rolled_back_or_revoked(app, index, managed):
    owner, cache, attribute = _handles(app)[index]
    connection = getattr(cache, attribute)
    if managed:
        with (
            _core_operation(owner),
            pytest.raises(RuntimeError, match="sqlite_retirement_incomplete"),
        ):
            await LifecycleMixin._retire_app_sqlite_handles(app)
            # This scope remains admitted through the refused retirement.
    else:
        connection.execute("BEGIN")
        with pytest.raises(RuntimeError, match="sqlite_retirement_incomplete"):
            await LifecycleMixin._retire_app_sqlite_handles(app)
        assert connection.in_transaction
        connection.rollback()
    assert getattr(cache, attribute) is connection
    assert connection.execute("SELECT 1").fetchone()[0] == 1
    await LifecycleMixin._retire_app_sqlite_handles(app)
    _assert_closed(connection)


async def test_live_tracked_chat_cursor_survives_refused_retirement(app):
    connection = app.chachanotes_db.get_connection()
    cursor = connection.execute("SELECT 1 UNION ALL SELECT 2")
    try:
        with pytest.raises(RuntimeError, match="sqlite_retirement_incomplete"):
            await LifecycleMixin._retire_app_sqlite_handles(app)
        assert [row[0] for row in cursor.fetchall()] == [1, 2]
    finally:
        cursor.close()
    await LifecycleMixin._retire_app_sqlite_handles(app)
    _assert_closed(connection)


@pytest.mark.parametrize(
    "index",
    range(5),
    ids=["chat", "evals", "collections", "subscriptions", "workspace"],
)
async def test_native_close_failure_keeps_cache_and_handle_visible(
    app, index, monkeypatch
):
    _, cache, attribute = _handles(app)[index]
    connection = getattr(cache, attribute)

    def fail_close(_connection):
        raise sqlite3.OperationalError("controlled_native_close_failure")

    with monkeypatch.context() as patch:
        patch.setattr(type(connection), "close", fail_close)
        with pytest.raises(RuntimeError, match="sqlite_retirement_incomplete"):
            await LifecycleMixin._retire_app_sqlite_handles(app)
        assert getattr(cache, attribute) is connection
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    await LifecycleMixin._retire_app_sqlite_handles(app)
    _assert_closed(connection)


@pytest.mark.parametrize(
    "shared", [False, True], ids=["owned-workspace", "shared-chat"]
)
async def test_foreign_live_native_cache_is_not_closed_by_app_thread(app, shared):
    db = app.chachanotes_db if shared else app.local_workspace_db
    current = db.get_connection() if shared else db._thread_local.conn
    entered, release = threading.Event(), threading.Event()
    failures = []

    def foreign_borrower():
        try:
            connection = db.get_connection() if shared else db._held_connection()
            entered.set()
            with _core_operation(db):
                assert release.wait(5)
                assert connection.execute("SELECT 1").fetchone()[0] == 1
            db.close()
        except BaseException as error:  # noqa: BLE001 - propagate the foreign thread's assertion/error after join.
            failures.append(error)

    thread = threading.Thread(target=foreign_borrower)
    thread.start()
    try:
        assert entered.wait(5)
        if shared:
            await LifecycleMixin._retire_app_sqlite_handles(app)
            _assert_closed(current)
        else:
            with pytest.raises(RuntimeError, match="sqlite_retirement_incomplete"):
                await LifecycleMixin._retire_app_sqlite_handles(app)
            assert current.execute("SELECT 1").fetchone()[0] == 1
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not failures
    await LifecycleMixin._retire_app_sqlite_handles(app)
    _assert_closed(current)


async def test_evaluation_tasks_settle_before_their_real_native_close(app):
    connection = app.evaluation_orchestrator.db.get_connection()
    entered, settled = asyncio.Event(), asyncio.Event()

    async def evaluation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            assert connection.execute("SELECT 1").fetchone()[0] == 1
            settled.set()

    task = asyncio.create_task(evaluation())
    app.evaluation_orchestrator._active_tasks["actual-task"] = task
    try:
        await entered.wait()
        await LifecycleMixin._retire_app_sqlite_handles(app)
        assert task.done() and settled.is_set()
        _assert_closed(connection)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_evaluation_finalizer_transaction_is_not_implicitly_rolled_back(app):
    connection = app.evaluation_orchestrator.db.get_connection()
    entered = asyncio.Event()

    async def evaluation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            connection.execute("BEGIN")

    task = asyncio.create_task(evaluation())
    app.evaluation_orchestrator._active_tasks["transaction-tail"] = task
    try:
        await entered.wait()
        with pytest.raises(RuntimeError, match="sqlite_retirement_incomplete"):
            await LifecycleMixin._retire_app_sqlite_handles(app)
        assert task.done()
        assert connection.in_transaction
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        connection.rollback()
        await LifecycleMixin._retire_app_sqlite_handles(app)
        _assert_closed(connection)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_evaluation_preparation_owner_settles_before_native_close(app):
    connection = app.evaluation_orchestrator.db.get_connection()
    entered, settled = asyncio.Event(), asyncio.Event()

    async def preparation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            assert connection.execute("SELECT 1").fetchone()[0] == 1
            app.evaluation_orchestrator._maintenance_runs.pop(asyncio.current_task())
            settled.set()

    task = asyncio.create_task(preparation())
    app.evaluation_orchestrator._maintenance_runs[task] = 1
    try:
        await entered.wait()
        await LifecycleMixin._retire_app_sqlite_handles(app)
        assert task.done() and settled.is_set()
        _assert_closed(connection)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("failure", ["producer", "textual", "textual-cancel"])
async def test_root_preserves_primary_shutdown_failure_and_native_handles(app, failure):
    primary = (
        asyncio.CancelledError()
        if failure == "textual-cancel"
        else RuntimeError(f"controlled_{failure}_shutdown_failure")
    )

    class TextualBoundary:
        async def _shutdown(self):
            if failure.startswith("textual"):
                raise primary

    class Application(LifecycleMixin, TextualBoundary):
        async def _shutdown_app_owned_lifecycles(self):
            if failure == "producer":
                raise primary

    application = Application()
    application.__dict__.update(vars(app))
    connections = [getattr(cache, attribute) for _, cache, attribute in _handles(app)]
    with pytest.raises(type(primary)) as raised:
        await application._shutdown()
    assert raised.value is primary
    for connection in connections:
        assert connection.execute("SELECT 1").fetchone()[0] == 1


async def test_root_retires_only_after_native_shutdown_callbacks_finish(app):
    connections = [getattr(cache, attribute) for _, cache, attribute in _handles(app)]

    class TextualBoundary:
        async def _shutdown(self):
            for connection in connections:
                assert connection.execute("SELECT 1").fetchone()[0] == 1

    class Application(LifecycleMixin, TextualBoundary):
        async def _shutdown_app_owned_lifecycles(self):
            for connection in connections:
                assert connection.execute("SELECT 1").fetchone()[0] == 1

    application = Application()
    application.__dict__.update(vars(app))
    await application._shutdown()
    for connection in connections:
        _assert_closed(connection)
