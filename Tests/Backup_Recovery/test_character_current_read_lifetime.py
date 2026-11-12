"""Current Character callback lifetime, with original native scopes and reads."""

import asyncio
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from Tests.UI.test_console_character_context import _controller
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationNavigationService,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.character_context import (
    ConsoleCharacterContextController,
)

pytestmark = [
    pytest.mark.bootstrap_profile,
    pytest.mark.usefixtures("local_root"),
    pytest.mark.asyncio,
]


def _closed(connection):
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        return True
    return False


@contextmanager
def _original_metadata(database, hold=False):
    original = ConsoleCharacterContextController._read_database_scope_metadata
    code = original.__code__
    loop_thread = threading.current_thread()
    before_main, before_thread = sys.getprofile(), threading.getprofile()
    entered, release = threading.Event(), threading.Event()
    facts = {
        "calls": 0,
        "connections": [],
        "entered": entered,
        "release": release,
        "lifecycle": [],
    }

    def observed(frame, event, value):
        if frame.f_code is not code or frame.f_locals.get("database") is not database:
            return
        assert threading.current_thread() is not loop_thread
        if event == "call":
            facts["calls"] += 1
        elif event == "return":
            connection = database._local.conn
            assert connection is not None and not _closed(connection)
            if not any(item is connection for item in facts["connections"]):
                facts["connections"].append(connection)
            if hold and (hold == "all" or not entered.is_set()):
                entered.set()
                assert release.wait(10), "original Character callback was not released"

    threading.setprofile_all_threads(observed)
    try:
        yield facts
    finally:
        release.set()
        threading.setprofile_all_threads(before_thread)
        sys.setprofile(before_main)
        assert (
            ConsoleCharacterContextController._read_database_scope_metadata is original
        )
        assert original.__code__ is code


@pytest.mark.parametrize("cancellation", ["single", "repeated", "all_created"])
async def test_current_character_capture_keeps_physical_owner_until_callback_returns(
    tmp_path, cancellation
):
    database = CharactersRAGDB(tmp_path / "capture.sqlite", "current-character")
    controller = _controller(database_accessor=lambda: database)
    pending = None
    try:
        with _original_metadata(database, hold=True) as facts:
            prior = asyncio.all_tasks()
            pending = asyncio.create_task(controller._capture_scope())
            try:
                assert await asyncio.to_thread(facts["entered"].wait, 10)
                assert facts["connections"] and worker_leases(database)
                if cancellation == "all_created":
                    selected = asyncio.all_tasks() - prior
                    assert pending in selected
                    for task in selected:
                        task.cancel()
                else:
                    pending.cancel()
                await asyncio.sleep(0)
                if cancellation == "repeated":
                    pending.cancel()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                returned_early = pending.done()
                assert worker_leases(database)
            finally:
                facts["release"].set()
                if pending is not None:
                    await asyncio.gather(pending, return_exceptions=True)
                async with asyncio.timeout(10):
                    while worker_leases(database):
                        await asyncio.sleep(0.01)
            assert all(_closed(item) for item in facts["connections"])
            assert not worker_leases(database)
            assert pending.cancelled()
            assert not returned_early, (
                "Character caller returned while original native callback was live"
            )
    finally:
        database.close()


async def test_current_character_display_reuses_only_same_owner_observation(tmp_path):
    database = CharactersRAGDB(tmp_path / "cadence.sqlite", "current-character")
    controller = _controller(
        database_accessor=lambda: database,
        current_character_accessor=lambda: None,
        service_factory=CharacterConversationNavigationService,
    )
    store = ConsoleChatStore()
    store.create_session(
        session_id="current",
        ephemeral=True,
        settings=ConsoleSessionSettings(provider="synthetic", model="synthetic"),
    )
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(chachanotes_db=database, app_config={}),
        _console_chat_store=store,
        _console_chat_tearing_down=False,
    )
    try:
        await controller.refresh_if_scope_changed(force=True)
        with _original_metadata(database) as direct:
            for _ in range(6):
                await controller.refresh_if_scope_changed()
        assert direct["calls"] == 12
        assert direct["connections"] and all(
            _closed(item) for item in direct["connections"]
        )
        assert not worker_leases(database)
        facade = getattr(controller, "refresh_presentation_if_scope_changed", None)
        with _original_metadata(database) as displayed:
            for _ in range(6):
                if facade is None:
                    await controller.refresh_if_scope_changed()
                else:
                    await facade(screen)
        assert displayed["connections"] and all(
            _closed(item) for item in displayed["connections"]
        )
        assert not worker_leases(database)
        assert displayed["calls"] == 2, (
            "same-owner display repeats original metadata observation"
        )
    finally:
        database.close()


@pytest.mark.parametrize(
    "route", ["file", "borrowed_transaction", "memory", "subclass", "custom"]
)
async def test_character_finite_call_preserves_borrowers_arguments_errors_and_custom_routes(
    tmp_path, route
):
    class DeclaredDatabase(CharactersRAGDB):
        pass

    loop = asyncio.get_running_loop()
    prior_executor = loop._default_executor
    executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="character-lifetime-control"
    )
    loop.set_default_executor(executor)
    database = connection = transaction = None
    marker, expected = object(), object()
    error = ValueError("declared Character callback error")
    seen = []
    try:
        if route == "custom":
            database = SimpleNamespace()
        else:
            constructor = DeclaredDatabase if route == "subclass" else CharactersRAGDB
            path = ":memory:" if route == "memory" else tmp_path / "owner.sqlite"
            database = await asyncio.to_thread(constructor, path, "declared-character")
            if route in {"memory", "subclass", "borrowed_transaction"}:
                connection = await asyncio.to_thread(database.get_connection)
                assert not _closed(connection)
            else:
                # Retire the fixture constructor's worker handle before the
                # first finite callback, so that callback really creates one.
                await asyncio.to_thread(database.close_connection)
            if route == "borrowed_transaction":
                transaction = database.transaction()
                await asyncio.to_thread(transaction.__enter__)
                assert connection.in_transaction
        controller = _controller(database_accessor=lambda: database)

        def callback(argument, *, named, fail=False):
            assert argument is marker and named is expected
            assert threading.current_thread() is not threading.main_thread()
            if route != "custom":
                actual = database.get_connection()
                actual.execute("SELECT 1").fetchone()
                seen.append(actual)
                if route == "borrowed_transaction":
                    assert actual is connection and actual.in_transaction
            if fail:
                raise error
            return expected

        assert (
            await controller._await_database_work(
                database, callback, marker, named=expected
            )
            is expected
        )
        with pytest.raises(ValueError) as caught:
            await controller._await_database_work(
                database, callback, marker, named=expected, fail=True
            )
        assert caught.value is error
        assert not controller._preparation_reads
        if route == "file":
            assert len(seen) == 2 and all(_closed(item) for item in seen)
            assert not worker_leases(database)
        elif route != "custom":
            assert all(item is connection for item in seen)
            assert not _closed(connection)
            if route == "borrowed_transaction":
                assert connection.in_transaction and worker_leases(database)
    finally:
        if transaction is not None:
            await asyncio.to_thread(transaction.__exit__, None, None, None)
        if database is not None and route != "custom":
            await asyncio.to_thread(database.close)
            if connection is not None:
                assert _closed(connection)
        executor.shutdown(wait=True)
        if prior_executor is None:
            loop._default_executor = None
        else:
            loop.set_default_executor(prior_executor)


@pytest.mark.parametrize("terminal", ["remove", "host_drain", "app_shutdown"])
@pytest.mark.parametrize(
    "cancellation", ["none", "repeated", "all_created", "settle_entry"]
)
async def test_current_character_terminal_retains_original_native_callback(
    tmp_path, terminal, cancellation
):
    from Tests.UI.test_console_character_context import _CharacterApp
    from tldw_chatbook.UI.Console_Modules.view_workers import (
        capture_console_view_workers,
        drain_console_view_workers,
    )
    from tldw_chatbook.Widgets.Console.console_character_context import (
        ConsoleCharacterContext,
    )

    database = CharactersRAGDB(tmp_path / "terminal.sqlite", "current-character")
    controller = _controller(
        database_accessor=lambda: database,
        current_character_accessor=lambda: None,
        service_factory=CharacterConversationNavigationService,
    )
    app = _CharacterApp(controller, controller.state)
    mounted, close = asyncio.Event(), asyncio.Event()
    widgets = []

    async def app_session():
        async with app.run_test():
            widgets.append(app.screen.query_one(ConsoleCharacterContext))
            mounted.set()
            await close.wait()

    pending = original_task = None
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()
    entry_cancelled = []
    with _original_metadata(database, hold=True) as facts:
        session = asyncio.create_task(app_session())
        try:
            await asyncio.wait_for(mounted.wait(), 5)
            assert await asyncio.to_thread(facts["entered"].wait, 5)
            widget = widgets[0]
            original_task = getattr(widget, "_controller_task", None) or widget._task
            assert original_task is not None and not original_task.done()
            assert facts["connections"] and worker_leases(database)
            captured = capture_console_view_workers(app)
            captured_original = any(
                node is widget and task is original_task
                for worker, node, task, work in captured[-1]
            )
            prior = asyncio.all_tasks()
            if cancellation == "settle_entry":
                from types import CodeType

                settle_code = next(
                    code
                    for code in drain_console_view_workers.__code__.co_consts
                    if type(code) is CodeType and code.co_name == "settle"
                )
                assert original_factory is None

                def entry_factory(loop, coroutine, **kwargs):
                    task = asyncio.Task(coroutine, loop=loop, **kwargs)
                    if getattr(coroutine, "cr_code", None) is settle_code:
                        entry_cancelled.append(task)
                        task.cancel()
                    return task

                loop.set_task_factory(entry_factory)
            if terminal == "remove":
                pending = asyncio.ensure_future(widget.remove())
            elif terminal == "host_drain":
                pending = asyncio.create_task(drain_console_view_workers(captured))
            else:
                close.set()
                pending = session
            # Let the real teardown reach its existing boundary while the
            # original native callback independently remains held.
            await asyncio.wait({pending}, timeout=0.2)
            if cancellation not in {"none", "settle_entry"}:
                if cancellation == "all_created":
                    selected = asyncio.all_tasks() - prior
                    facts["selected"] = [
                        (
                            task.get_name(),
                            repr(task.get_coro()),
                            [
                                (frame.f_code.co_name, frame.f_lineno)
                                for frame in task.get_stack()
                            ],
                        )
                        for task in selected
                    ]
                    for task in selected:
                        task.cancel()
                pending.cancel()
                await asyncio.sleep(0)
                pending.cancel()
                await asyncio.wait({pending}, timeout=0.2)
            loop.set_task_factory(original_factory)
            if cancellation == "settle_entry":
                assert entry_cancelled, (
                    "actual drain never created its settle coordinator"
                )
            returned_early = pending.done()
            assert worker_leases(database)
            facts["release"].set()
            await asyncio.wait_for(asyncio.gather(pending, return_exceptions=True), 5)
            await asyncio.gather(original_task, return_exceptions=True)
            async with asyncio.timeout(5):
                while worker_leases(database):
                    await asyncio.sleep(0.01)
            assert all(_closed(item) for item in facts["connections"])
            assert not returned_early, (
                f"{terminal} returned while original native callback was live"
            )
            if terminal == "host_drain":
                assert captured_original, (
                    "actual host manager omitted issued Character work"
                )
        finally:
            loop.set_task_factory(original_factory)
            facts["release"].set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            if original_task is not None:
                await asyncio.gather(original_task, return_exceptions=True)
            close.set()
            await asyncio.gather(session, return_exceptions=True)
            database.close()


@pytest.mark.parametrize(
    "outcome",
    [
        "pending",
        "worker_error",
        "coordinator_source_error",
        "factory_raise",
        "factory_enqueue_raise",
        "factory_cancel",
        "factory_enqueue_cancel",
    ],
)
async def test_character_host_drain_preserves_native_custody_and_outcomes(
    tmp_path, outcome
):
    from types import CodeType

    from textual.app import App
    from textual.worker import Worker, WorkerState

    from tldw_chatbook.UI.Console_Modules.view_workers import (
        capture_console_view_workers,
        drain_console_view_workers,
    )

    database = CharactersRAGDB(tmp_path / "outcome.sqlite", "current-character")
    controller = _controller(database_accessor=lambda: database)
    host = App()
    error = (
        asyncio.CancelledError("declared factory cancellation")
        if outcome in {"factory_cancel", "factory_enqueue_cancel"}
        else ValueError("declared worker error")
    )
    loop = asyncio.get_running_loop()
    factory, original_wait = loop.get_task_factory(), Worker.wait
    changed = []
    created = []
    drain = None

    async def work():
        try:
            await controller._capture_scope()
        except asyncio.CancelledError:
            if outcome == "worker_error":
                raise error from None
            raise
        if outcome == "worker_error":
            raise error

    async def replacement_wait(self):
        raise AssertionError("replacement Worker.wait must never be called")

    with host._context(), _original_metadata(database, hold=True) as facts:
        worker = host.run_worker(
            work(), group="console-character-context-refresh", exit_on_error=False
        )
        try:
            assert await asyncio.to_thread(facts["entered"].wait, 5)
            assert worker_leases(database)
            captured = capture_console_view_workers(host)
            assert any(row[0] is worker for row in captured[-1])
            if outcome == "pending":
                # The real wait's existing PENDING refusal must still join the
                # exact issued Task in the captured final retirement phase.
                worker.state = WorkerState.PENDING
            elif outcome in {
                "coordinator_source_error",
                "factory_raise",
                "factory_enqueue_raise",
                "factory_cancel",
                "factory_enqueue_cancel",
            }:
                assert factory is None
                settle_code = next(
                    code
                    for code in drain_console_view_workers.__code__.co_consts
                    if type(code) is CodeType and code.co_name == "settle"
                )

                def source_factory(loop, coroutine, **kwargs):
                    if getattr(coroutine, "cr_code", None) is settle_code:
                        changed.append(coroutine)
                        if outcome in {"factory_raise", "factory_cancel"}:
                            raise error
                        task = asyncio.Task(coroutine, loop=loop, **kwargs)
                        created.append(task)
                        if outcome in {
                            "factory_enqueue_raise",
                            "factory_enqueue_cancel",
                        }:
                            raise error
                        Worker.wait = replacement_wait
                        return task
                    return asyncio.Task(coroutine, loop=loop, **kwargs)

                loop.set_task_factory(source_factory)

            async def capture_outcome():
                try:
                    await drain_console_view_workers(captured)
                except BaseException as failure:  # noqa: BLE001 - assert exact cancellation/error object.
                    return failure
                return None

            drain = asyncio.create_task(capture_outcome())
            await asyncio.wait({drain}, timeout=0.2)
            returned_early = drain.done()
            if outcome in {
                "coordinator_source_error",
                "factory_raise",
                "factory_enqueue_raise",
                "factory_cancel",
                "factory_enqueue_cancel",
            }:
                assert changed
                drain.cancel()
                await asyncio.sleep(0)
                drain.cancel()
                await asyncio.wait({drain}, timeout=0.05)
            assert worker_leases(database)
            loop.set_task_factory(factory)
            facts["release"].set()
            (result,) = await asyncio.wait_for(
                asyncio.gather(drain, return_exceptions=True), 5
            )
            await asyncio.gather(worker._task, return_exceptions=True)
            assert not worker_leases(database)
            assert all(_closed(item) for item in facts["connections"])
            assert not returned_early, (
                "coordinator outcome released a live original Character callback"
            )
            if outcome == "coordinator_source_error":
                assert type(result) is RuntimeError
                assert str(result) == "console_view_worker_owner_changed"
            elif outcome in {
                "factory_raise",
                "factory_enqueue_raise",
                "factory_cancel",
                "factory_enqueue_cancel",
            }:
                import inspect

                assert result is error
                assert all(
                    inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
                    for coro in changed
                )
                assert all(task.done() for task in created)
            else:
                assert result is None
                if outcome == "worker_error":
                    assert worker.error is error
        finally:
            facts["release"].set()
            loop.set_task_factory(factory)
            Worker.wait = original_wait
            for task in created:
                await asyncio.gather(task, return_exceptions=True)
            for coroutine in changed:
                coroutine.close()
            if drain is not None:
                await asyncio.gather(drain, return_exceptions=True)
            await asyncio.gather(worker._task, return_exceptions=True)
            database.close()


async def test_character_widget_preserves_overlapping_workers_and_message_pump(
    tmp_path,
):
    from Tests.UI.test_console_character_context import _CharacterApp
    from tldw_chatbook.UI.Console_Modules.view_workers import (
        capture_console_view_workers,
    )
    from tldw_chatbook.Widgets.Console.console_character_context import (
        ConsoleCharacterContext,
    )

    database = CharactersRAGDB(tmp_path / "overlap.sqlite", "current-character")
    controller = _controller(
        database_accessor=lambda: database,
        current_character_accessor=lambda: None,
        service_factory=CharacterConversationNavigationService,
    )
    app = _CharacterApp(controller, controller.state)
    removal = None
    with _original_metadata(database, hold="all") as facts:
        try:
            async with app.run_test():
                widget = app.screen.query_one(ConsoleCharacterContext)
                assert await asyncio.to_thread(facts["entered"].wait, 5)
                first = widget._controller_task
                widget._start(controller._capture_scope())
                second = widget._controller_task
                async with asyncio.timeout(5):
                    while len(facts["connections"]) != 2:
                        await asyncio.sleep(0.01)
                assert first is not second and not first.done() and not second.done()
                message_pump = widget._task
                assert message_pump not in {first, second} and not message_pump.done()
                captured = capture_console_view_workers(app, widget)
                assert {row[2] for row in captured[-1]} == {first, second}
                removal = asyncio.ensure_future(widget.remove())
                await asyncio.wait({removal}, timeout=0.2)
                returned_early = removal.done()
                assert worker_leases(database)
                facts["release"].set()
                await asyncio.wait_for(removal, 5)
                assert first.done() and second.done() and message_pump.done()
                assert all(_closed(item) for item in facts["connections"])
                assert not worker_leases(database)
                assert not returned_early
        finally:
            facts["release"].set()
            if removal is not None:
                await asyncio.gather(removal, return_exceptions=True)
            database.close()


async def test_character_scope_pair_keeps_original_reads_on_one_finite_handle(tmp_path):
    database = CharactersRAGDB(tmp_path / "pair.sqlite", "current-character")
    controller = _controller(database_accessor=lambda: database)
    try:
        with _original_metadata(database) as facts:
            snapshot = await controller._capture_scope()
        assert snapshot.database is database
        assert (
            snapshot.fingerprint.data_authority_id == database.get_local_authority_id()
        )
        assert facts["calls"] == 2
        assert facts["connections"] and all(
            _closed(item) for item in facts["connections"]
        )
        assert not worker_leases(database)
        assert len(facts["connections"]) == 1, (
            "paired scope reads reopen their finite native handle"
        )
    finally:
        database.close()


@pytest.mark.parametrize(
    "mutation", ["database", "character", "conversation", "revision"]
)
async def test_character_file_pair_rechecks_original_ambient_on_issuing_thread(
    tmp_path, mutation
):
    first = CharactersRAGDB(tmp_path / "first.sqlite", "first-character")
    second = CharactersRAGDB(tmp_path / "second.sqlite", "second-character")
    active, current, conversation = [first], [None], [None]
    issuing = threading.current_thread()
    accessor_threads = []

    def read(slot):
        accessor_threads.append(threading.current_thread())
        assert threading.current_thread() is issuing
        return slot[0]

    controller = _controller(
        database_accessor=lambda: read(active),
        current_character_accessor=lambda: read(current),
        open_conversation_accessor=lambda: read(conversation),
    )
    pending = None
    try:
        with _original_metadata(first, hold=True) as facts:
            pending = asyncio.create_task(controller._capture_scope())
            try:
                assert await asyncio.to_thread(facts["entered"].wait, 5)
                if mutation == "database":
                    active[0] = second
                elif mutation == "character":
                    current[0] = (9, "New character")
                elif mutation == "conversation":
                    conversation[0] = "new-conversation"
                else:
                    first.add_character_card({"name": "Real revision change"})
                facts["release"].set()
                snapshot = await asyncio.wait_for(pending, 5)
                assert snapshot.database is active[0]
                assert snapshot.fingerprint.current_character_id == (
                    None if current[0] is None else 9
                )
                assert snapshot.fingerprint.open_conversation_id == (
                    conversation[0] or ""
                )
                assert (
                    snapshot.fingerprint.data_revision
                    == active[0].get_character_conversation_search_revision()
                )
                assert accessor_threads and all(
                    thread is issuing for thread in accessor_threads
                )
                assert facts["connections"] and all(
                    _closed(item) for item in facts["connections"]
                )
                assert not worker_leases(first) and not worker_leases(second)
            finally:
                facts["release"].set()
                if pending is not None:
                    await asyncio.gather(pending, return_exceptions=True)
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("custom", ["metadata", "ambient"])
async def test_character_custom_pair_callbacks_keep_preceding_finite_route(
    tmp_path, custom
):
    from types import MethodType

    database = CharactersRAGDB(tmp_path / "custom-pair.sqlite", "current-character")
    controller = _controller(database_accessor=lambda: database)
    calls = []
    if custom == "metadata":
        original = controller._read_database_scope_metadata

        def callback(database):
            calls.append(threading.current_thread())
            return original(database)

        controller._read_database_scope_metadata = callback
    else:
        original = controller._ambient_scope_matches

        def callback(self, *args):
            calls.append(threading.current_thread())
            return original(*args)

        controller._ambient_scope_matches = MethodType(callback, controller)
    try:
        with _original_metadata(database) as facts:
            snapshot = await controller._capture_scope()
        assert snapshot.database is database and facts["calls"] == 2
        assert facts["connections"] and all(
            _closed(item) for item in facts["connections"]
        )
        assert not worker_leases(database)
        assert len(facts["connections"]) == 2, (
            "custom callback was moved into the new paired route"
        )
        if custom == "metadata":
            assert len(calls) == 2 and all(
                thread is not threading.current_thread() for thread in calls
            )
        else:
            assert calls and all(
                thread is threading.current_thread() for thread in calls
            )
    finally:
        database.close()


@pytest.mark.parametrize("database_route", ["file", "custom"])
async def test_character_middle_ambient_error_retains_exact_original_error(
    tmp_path, database_route
):
    database = (
        CharactersRAGDB(tmp_path / "ambient-error.sqlite", "current-character")
        if database_route == "file"
        else SimpleNamespace(
            get_local_authority_id=lambda: "authority",
            get_character_conversation_search_revision=lambda: 1,
        )
    )
    error = LookupError("declared transient ambient failure")
    cause = ValueError("declared accessor cause")
    error.__cause__ = cause
    calls = []
    issuing = threading.current_thread()

    def current_character():
        calls.append(threading.current_thread())
        assert threading.current_thread() is issuing
        if len(calls) == 2:
            raise error

    controller = _controller(
        database_accessor=lambda: database, current_character_accessor=current_character
    )
    caught = None
    try:
        with pytest.raises(LookupError) as captured:
            await controller._capture_scope()
        caught = captured.value
        if database_route == "file":
            assert not worker_leases(database)
        assert caught is error, (
            f"middle ambient error changed to {type(caught).__name__}"
        )
        assert caught.__cause__ is cause
        assert len(calls) == 2
    finally:
        if database_route == "file":
            database.close()


@pytest.mark.parametrize("phase", ["before", "held"])
@pytest.mark.parametrize("mutation", ["body", "defaults"])
async def test_character_pair_stock_source_change_uses_current_direct_route(
    tmp_path, phase, mutation
):
    from types import FunctionType

    database = CharactersRAGDB(tmp_path / "source.sqlite", "current-character")
    controller = _controller(database_accessor=lambda: database)
    function = ConsoleCharacterContextController._read_database_scope_metadata
    code, defaults = function.__code__, function.__defaults__
    namespace = function.__globals__
    original = FunctionType(
        code, namespace, function.__name__, defaults, function.__closure__
    )
    original.__kwdefaults__ = function.__kwdefaults__
    calls = []
    names = ("_pair_review_original", "_pair_review_calls")
    assert all(name not in namespace for name in names)
    namespace[names[0]], namespace[names[1]] = original, calls
    exec(  # noqa: S102 - fixed test body in the original defining namespace.
        "def _pair_review_body(database):\n"
        "    _pair_review_calls.append(database)\n"
        "    return _pair_review_original(database)\n",
        namespace,
    )
    replacement = namespace.pop("_pair_review_body")
    pending = None

    def change():
        if mutation == "body":
            function.__code__ = replacement.__code__
        else:
            function.__defaults__ = ()

    try:
        with _original_metadata(database, hold=phase == "held") as facts:
            try:
                if phase == "before":
                    change()
                pending = asyncio.create_task(controller._capture_scope())
                if phase == "held":
                    assert await asyncio.to_thread(facts["entered"].wait, 5)
                    change()
                    facts["release"].set()
                snapshot = await asyncio.wait_for(pending, 5)
                expected = 2 if phase == "before" else 3
                assert snapshot.database is database
                assert facts["calls"] == expected
                assert len(facts["connections"]) == expected
                assert all(_closed(item) for item in facts["connections"])
                assert not worker_leases(database)
                if mutation == "body":
                    assert calls == [database, database]
            finally:
                facts["release"].set()
                if pending is not None:
                    await asyncio.gather(pending, return_exceptions=True)
                function.__code__, function.__defaults__ = code, defaults
    finally:
        for name in names:
            namespace.pop(name)
        database.close()


async def test_character_file_pair_preserves_existing_worker_transaction(tmp_path):
    loop = asyncio.get_running_loop()
    prior = loop._default_executor
    executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="character-pair-borrower"
    )
    loop.set_default_executor(executor)
    database = connection = transaction = None
    try:
        database = await asyncio.to_thread(
            CharactersRAGDB, tmp_path / "borrowed-pair.sqlite", "character-borrower"
        )
        connection = await asyncio.to_thread(database.get_connection)
        transaction = database.transaction()
        await asyncio.to_thread(transaction.__enter__)
        assert connection.in_transaction and worker_leases(database)
        controller = _controller(database_accessor=lambda: database)
        with _original_metadata(database) as facts:
            snapshot = await controller._capture_scope()
        assert snapshot.database is database and facts["calls"] == 2
        assert facts["connections"] == [connection]
        assert connection.in_transaction and not _closed(connection)
        assert worker_leases(database)
    finally:
        if transaction is not None:
            await asyncio.to_thread(transaction.__exit__, None, None, None)
        if database is not None:
            await asyncio.to_thread(database.close)
        if connection is not None:
            assert _closed(connection)
        loop._default_executor = prior
        executor.shutdown(wait=True)
