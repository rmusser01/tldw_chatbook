"""One real main-DB display lifetime; each section keeps its own publisher."""

import asyncio
import inspect
import sqlite3
import sys
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_character_presentation_cadence import _character
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root  # noqa: PLC0414
from Tests.Chat.test_console_first_send_atomicity import _controller
from Tests.UI.test_console_workspace_controller import _workspace_controller
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules import config_sync
from tldw_chatbook.UI.Console_Modules.character_context import character_presentation_work
from tldw_chatbook.UI.Console_Modules.console_spend_projection import ConsoleContextReadSnapshot

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]


@contextmanager
def _worker_connections(database):
    connections = []
    main = threading.current_thread()
    code = inspect.unwrap(CharactersRAGDB._get_thread_connection).__code__
    previous, previous_main = threading.getprofile(), sys.getprofile()

    def observe(frame, event, result):
        if (event == "return" and frame.f_code is code
            and frame.f_locals.get("self") is database
            and threading.current_thread() is not main
            and isinstance(result, sqlite3.Connection)
            and not any(result is item for item in connections)):
            connections.append(result)

    threading.setprofile_all_threads(observe)
    try:
        yield connections
    finally:
        threading.setprofile_all_threads(previous)
        sys.setprofile(previous_main)


def _display(tmp_path):
    database, store, controller, _gateway = _controller(tmp_path)
    store.append_message("session-1", role=ConsoleMessageRole.USER, content="seed", persist=True)
    character, screen, *_ = _character(database)
    screen._console_chat_store = store
    screen._console_chat_controller = controller
    screen._character_context = character
    screen._console_context_read_snapshot = ConsoleContextReadSnapshot()
    service = ChatConversationService(database)
    screen.app_instance.local_chat_conversation_service = service
    workspace = _workspace_controller(app_instance=screen.app_instance)
    workspace._publish_display_reads = lambda: None
    screen._workspace = workspace
    workspace._console_appearance_map_cache = {}
    workspace._console_appearance_map_service = service
    workspace._console_appearance_map_queried_ids = {store.sessions()[0].persisted_conversation_id}
    return database, screen


def _configured_display(tmp_path):
    from Tests.UI.test_console_checked_display_scope import _screen

    database, store, controller, screen, tasks = _screen(tmp_path)
    screen._character_context = _character(database)[0]
    screen._console_chat_controller = controller
    screen._console_chat_tearing_down = False
    screen._console_sync_in_progress = True
    service = ChatConversationService(database)
    screen.app_instance.local_chat_conversation_service = service
    workspace = screen._workspace = _workspace_controller(app_instance=screen.app_instance)
    workspace._publish_display_reads = lambda: None
    workspace._console_appearance_map_cache = {}
    workspace._console_appearance_map_service = service
    workspace._console_appearance_map_queried_ids = {store.sessions()[0].persisted_conversation_id}
    snapshot = ConsoleContextReadSnapshot.for_screen(screen, max_age=1.0)
    assert snapshot.config_projection is screen._console_readiness_config_projection
    return database, screen, tasks


@contextmanager
def _context_preparation_calls(screen, *, hold_config=None):
    from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read

    controller = screen._console_chat_controller
    projection = screen._console_context_read_snapshot.config_projection
    codes = {
        type(projection).warm.__code__: "config_warm",
        projection.read_current.__code__: "config_read",
        controller._context_repository.has_active_memory_selection.__func__.__code__: "probe",
        controller._durable_context_snapshots.__func__.__code__: "lineage",
    }
    native = run_preparation_read.__code__
    calls = dict.fromkeys((*codes.values(), "context_native"), 0)
    frames = []
    main = threading.current_thread()
    previous, previous_main = threading.getprofile(), sys.getprofile()

    def observe(frame, event, result):
        prior = previous_main if threading.current_thread() is main else previous
        if prior is not None:
            prior(frame, event, result)
        if event != "call" or any(frame is seen for seen in frames):
            return
        name = codes.get(frame.f_code)
        if frame.f_code is native and frame.f_locals.get("creator") is controller:
            name = "context_native"
        if name is None:
            return
        frames.append(frame)
        calls[name] += 1
        if name == "config_read" and hold_config is not None:
            entered, release = hold_config
            entered.set()
            assert release.wait(30), "test did not release the original config reader"

    threading.setprofile_all_threads(observe)
    try:
        yield calls
    finally:
        threading.setprofile_all_threads(previous)
        sys.setprofile(previous_main)


@pytest.mark.asyncio
async def test_ready_stock_config_joins_display_without_warming_again(tmp_path):
    from Tests.UI.test_console_checked_display_scope import _warm

    database, screen, tasks = _configured_display(tmp_path)
    snapshot = screen._console_context_read_snapshot
    try:
        projection = await _warm(screen, tasks)
        proof, at = projection._display_proof, projection.at
        assert proof is not None
        with _worker_connections(database) as connections, _context_preparation_calls(screen) as calls:
            await config_sync.prepare_main_db_display(screen)
        assert calls["config_warm"] == calls["config_read"] == calls["context_native"] == 0
        assert calls["probe"] == 1
        assert len(connections) == 1
        assert snapshot.value is not None
        assert projection._display_proof is proof and projection.at == at
        assert projection.max_age == snapshot.max_age == 1.0
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["cold", "expired", "pending", "busy"])
async def test_unready_context_keeps_full_free_of_config_and_independent_reads(tmp_path, state):
    from Tests.UI.test_console_checked_display_scope import _warm

    database, screen, tasks = _configured_display(tmp_path)
    snapshot = screen._console_context_read_snapshot
    projection = snapshot.config_projection
    entered, release = threading.Event(), threading.Event()
    task = None
    locked = False
    try:
        if state != "cold":
            await _warm(screen, tasks)
        if state in {"expired", "pending"}:
            projection.at -= projection.max_age + 1
        if state == "busy":
            await snapshot.lock.acquire()
            locked = True
        with _context_preparation_calls(
            screen, hold_config=(entered, release) if state == "pending" else None,
        ) as calls:
            if state == "pending":
                projection.run(lambda: None)
                assert await asyncio.to_thread(entered.wait, 10)
                assert projection.pending and projection._reads
            prior_reads = calls["config_read"]
            task = asyncio.create_task(config_sync.prepare_main_db_display(screen))
            done, _ = await asyncio.wait({task}, timeout=10)
            assert task in done, "FULL awaited context's existing config/lock work"
            await task
            assert calls["config_warm"] == 0
            assert calls["config_read"] == prior_reads
            assert calls["probe"] == calls["context_native"] == 0
        assert snapshot.value is None
        assert screen._character_context.state.scope_fingerprint is not None
    finally:
        release.set()
        if locked:
            snapshot.lock.release()
        if task is not None and not task.done():
            task.cancel()
        await asyncio.gather(*([task] if task is not None else []), *tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_positive_context_probe_leaves_lineage_to_original_scheduled_reader(tmp_path):
    from Tests.Chat.test_console_memory_presence import _reset
    from Tests.UI.test_console_checked_display_scope import _warm

    database, screen, tasks = _configured_display(tmp_path)
    controller = screen._console_chat_controller
    store = controller.store
    session = store.sessions()[0]
    row = store.messages_for_session(session.id)[0]
    controller._context_repository.insert_memory_selection(
        _reset(session.persisted_conversation_id, row.persisted_message_id)
    )
    snapshot = screen._console_context_read_snapshot
    try:
        await _warm(screen, tasks)
        with _context_preparation_calls(screen) as calls:
            await config_sync.prepare_main_db_display(screen)
        assert calls["probe"] == 1
        assert calls["lineage"] == calls["context_native"] == 0
        assert snapshot.value is None
        assert await snapshot.warm(controller, session.id)
        assert snapshot.value is not None
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_sealed_display_cannot_start_cold_context_work(tmp_path):
    database, screen, tasks = _configured_display(tmp_path)
    snapshot = screen._console_context_read_snapshot
    batch = config_sync._MainDbDisplayRead(screen, database)
    try:
        await batch.run([])
        with _context_preparation_calls(screen) as calls:
            assert not await snapshot.warm(screen._console_chat_controller, "session-1", _display_read=batch)
        assert not any(calls.values())
        assert snapshot.value is None
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_shared_context_refuses_changed_config_method_before_publication(tmp_path, monkeypatch):
    from Tests.UI.test_console_checked_display_scope import _warm

    database, screen, tasks = _configured_display(tmp_path)
    controller = screen._console_chat_controller
    snapshot = screen._console_context_read_snapshot
    loop = asyncio.get_running_loop()
    installed = threading.Event()
    called = []
    probe = controller._context_repository.has_active_memory_selection.__func__.__code__
    previous, previous_main = threading.getprofile(), sys.getprofile()

    def changed_key():
        called.append(True)
        raise RuntimeError("changed config method belongs to the original route")

    def replace_key():
        monkeypatch.setattr(snapshot.config_projection, "_key", changed_key)
        installed.set()

    def observe(frame, event, _result):
        if event == "return" and frame.f_code is probe and not installed.is_set():
            loop.call_soon_threadsafe(replace_key)
            assert installed.wait(10), "test did not replace the config method on its loop"

    try:
        await _warm(screen, tasks)
        threading.setprofile_all_threads(observe)
        await config_sync.prepare_main_db_display(screen)
        assert installed.is_set()
        assert not called
        assert snapshot.value is None
        assert screen._character_context.state.scope_fingerprint is not None
    finally:
        threading.setprofile_all_threads(previous)
        sys.setprofile(previous_main)
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


async def _prepare(screen):
    prepare = getattr(config_sync, "prepare_main_db_display", None)
    if prepare is not None:
        return await prepare(screen)
    # Actual pre-change operations, so RED measures separate physical lifetimes.
    await character_presentation_work(screen._character_context, screen)
    await screen._console_context_read_snapshot.warm(screen._console_chat_controller, "session-1")
    workspace = screen._workspace
    service = screen.app_instance.local_chat_conversation_service
    await workspace._refresh_appearances(
        service.db, service.get_conversation_appearances,
        tuple(workspace._console_appearance_map_queried_ids), workspace._console_appearance_map_cache,
    )


@pytest.mark.asyncio
async def test_stock_sections_share_one_retired_main_database_connection(tmp_path):
    database, screen = _display(tmp_path)
    try:
        with _worker_connections(database) as connections:
            await _prepare(screen)
        assert screen._console_context_read_snapshot.value is not None
        assert screen._character_context.state.scope_fingerprint is not None
        assert screen._workspace._console_appearance_map_cached_at > 0
        assert len(connections) == 1, "context, Character and appearances opened separate physical connections"
        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError):
                _ = connection.in_transaction
        assert not worker_leases(database)
        assert not screen._character_context._preparation_reads
        assert not screen._console_chat_controller._preparation_reads
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("callback", ["context_control_presentation_inputs", "context_control_inputs", "_global_context_policy_overrides"])
async def test_full_batch_keeps_custom_context_on_original_scheduled_route(tmp_path, monkeypatch, callback):
    database, screen = _display(tmp_path)
    called = []

    def custom(*args, **kwargs):
        called.append(True)
        raise RuntimeError("custom context work belongs to the original scheduled refresh")

    if callback == "context_control_presentation_inputs":
        async def custom_read(*args, **kwargs):
            custom(*args, **kwargs)
        replacement = custom_read
    else:
        replacement = custom
    monkeypatch.setattr(screen._console_chat_controller, callback, replacement)
    try:
        await config_sync.prepare_main_db_display(screen)
        assert not called
        assert screen._console_context_read_snapshot.value is None
        assert screen._character_context.state.scope_fingerprint is not None
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["repository", "persistence"])
@pytest.mark.parametrize("kind", ["custom", "memory"])
async def test_full_batch_keeps_non_native_context_database_on_original_route(tmp_path, owner, kind):
    database, screen = _display(tmp_path)
    controller = screen._console_chat_controller
    source = controller._context_repository if owner == "repository" else controller.store.persistence
    replacement = (
        CharactersRAGDB(":memory:", "display-memory") if kind == "memory"
        else SimpleNamespace(transaction=database.transaction, is_memory_db=False)
    )
    original = source.db
    source.db = replacement
    try:
        await config_sync.prepare_main_db_display(screen)
        assert screen._console_context_read_snapshot.value is None
        assert screen._character_context.state.scope_fingerprint is not None
    finally:
        source.db = original
        if kind == "memory":
            replacement.close()
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("callback", ["context_control_presentation_inputs", "_global_context_policy_overrides"])
async def test_full_batch_refuses_context_callback_changed_after_enlistment(tmp_path, monkeypatch, callback):
    database, screen = _display(tmp_path)
    called = []

    def custom(*args, **kwargs):
        called.append(True)
        raise RuntimeError("late custom context belongs to the scheduled refresh")

    async def custom_read(*args, **kwargs):
        custom(*args, **kwargs)

    snapshot = screen._console_context_read_snapshot
    code = ConsoleContextReadSnapshot._warm.__code__
    previous = sys.getprofile()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event == "call" and frame.f_code is code and frame.f_locals.get("self") is snapshot:
            monkeypatch.setattr(
                screen._console_chat_controller, callback,
                custom_read if callback == "context_control_presentation_inputs" else custom,
            )

    sys.setprofile(observe)
    try:
        await config_sync.prepare_main_db_display(screen)
        assert not called
        assert screen._console_context_read_snapshot.value is None
        assert screen._character_context.state.scope_fingerprint is not None
    finally:
        sys.setprofile(previous)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("definition", ["__defaults__", "__kwdefaults__"])
async def test_full_batch_keeps_changed_public_context_defaults_on_original_route(tmp_path, monkeypatch, definition):
    database, screen = _display(tmp_path)
    reader = screen._console_chat_controller.context_control_presentation_inputs.__func__
    # Deliberate source customization, not an observation wrapper around stock work.
    monkeypatch.setattr(
        reader, definition,
        ("custom-default-session",) if definition == "__defaults__"
        else {"_presentation_global_overrides": None},
    )
    try:
        await config_sync.prepare_main_db_display(screen)
        assert screen._console_context_read_snapshot.value is None
        assert screen._character_context.state.scope_fingerprint is not None
    finally:
        database.close()


@pytest.mark.asyncio
async def test_shared_appearance_refresh_removes_deleted_conversation(tmp_path):
    database, screen = _display(tmp_path)
    workspace = screen._workspace
    conversation_id, = workspace._console_appearance_map_queried_ids
    workspace._console_appearance_map_cache = {conversation_id: ("old-icon", "blue")}
    row = database.get_conversation_by_id(conversation_id)
    assert database.soft_delete_conversation(conversation_id, row["version"])
    try:
        await config_sync.prepare_main_db_display(screen)
        assert conversation_id not in workspace._console_appearance_map_cache
        assert workspace._console_appearance_map_cached_at > 0
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_at", ["before_capture", "before_read", "after_read"])
async def test_shared_appearance_keeps_custom_db_reader_on_original_route(
    tmp_path, monkeypatch, replace_at
):
    database, screen = _display(tmp_path)
    workspace = screen._workspace
    (conversation_id,) = workspace._console_appearance_map_queried_ids
    original = database.get_conversations_metadata_by_ids
    calls = []

    def custom(ids):
        calls.append(tuple(ids))
        return original(ids)

    def replace_reader():
        monkeypatch.setattr(database, "get_conversations_metadata_by_ids", custom)

    if replace_at == "before_capture":
        replace_reader()
    previous_main, previous = sys.getprofile(), threading.getprofile()
    capture_code = config_sync._MainDbDisplayRead.capture.__code__
    read_code = original.__func__.__code__

    def observe(frame, event, _result):
        if event != "return":
            return
        if (
            replace_at == "before_read"
            and frame.f_code is capture_code
            and frame.f_locals["section"] == "appearance"
        ) or (
            replace_at == "after_read"
            and frame.f_code is read_code
            and frame.f_locals["self"] is database
        ):
            replace_reader()

    try:
        threading.setprofile_all_threads(observe)
        try:
            await config_sync.prepare_main_db_display(screen)
        finally:
            threading.setprofile_all_threads(previous)
            sys.setprofile(previous_main)
        assert not calls
        assert workspace._console_appearance_map_cached_at == 0
        assert screen._character_context.state.scope_fingerprint is not None

        # No coverage from FULL: the original synchronous display reader
        # remains responsible for the customized metadata method.
        assert conversation_id in workspace._console_conversation_appearance_map(
            [conversation_id]
        )
        assert calls == [(conversation_id,)]
    finally:
        database.close()


@pytest.mark.asyncio
async def test_shared_read_failure_does_not_publish_or_suppress_other_sections(tmp_path):
    database = CharactersRAGDB(tmp_path / "independent.sqlite", "display")
    screen = SimpleNamespace(app_instance=SimpleNamespace(chachanotes_db=database))
    batch = config_sync._MainDbDisplayRead(screen, database)
    observers = [set(), set(), set()]
    published, refused = [], []

    async def section(name, reads):
        checks = 0

        def current():
            nonlocal checks
            checks += 1
            if name == "context" and checks == 2:
                raise RuntimeError("source replaced during read")

        read = batch.capture(
            name, database, database.get_character_conversation_search_revision,
            reads=reads, require_current=current,
        )
        try:
            value = await read
        except RuntimeError:
            refused.append(name)
        else:
            assert not worker_leases(database)
            published.append((name, value))

    try:
        with _worker_connections(database) as connections:
            await batch.run([section(name, reads) for name, reads in zip(
                ("character", "context", "appearance"), observers, strict=True,
            )])
        assert [name for name, _value in published] == ["character", "appearance"]
        assert refused == ["context"]
        assert len(connections) == 1
        assert all(not reads for reads in observers)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_shared_cancellation_waits_for_physical_retirement(tmp_path):
    database = CharactersRAGDB(tmp_path / "cancel.sqlite", "display")
    screen = SimpleNamespace(app_instance=SimpleNamespace(chachanotes_db=database))
    batch = config_sync._MainDbDisplayRead(screen, database)
    entered, release = threading.Event(), threading.Event()
    observers, published = [set(), set()], []

    def blocked():
        value = database.get_character_conversation_search_revision()
        entered.set()
        assert release.wait(10), "test-owned native read not released"
        return value

    async def section(name, reads):
        future = batch.capture(name, database, blocked, reads=reads)
        published.append(await future)

    task = None
    try:
        with _worker_connections(database) as connections:
            task = asyncio.create_task(batch.run([
                section("character", observers[0]), section("context", observers[1]),
            ]))
            assert await asyncio.to_thread(entered.wait, 10)
            assert all(reads for reads in observers)
            assert next(iter(observers[0])) is next(iter(observers[1]))
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert worker_leases(database)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not published
        assert len(connections) == 1
        assert all(not reads for reads in observers)
        assert not worker_leases(database)
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_shared_capture_rejects_other_instances_and_late_followups(tmp_path):
    database = CharactersRAGDB(tmp_path / "exact.sqlite", "display")
    other = CharactersRAGDB(tmp_path / "other.sqlite", "display-other")
    screen = SimpleNamespace(app_instance=SimpleNamespace(chachanotes_db=database))
    batch = config_sync._MainDbDisplayRead(screen, database)
    reads = set()
    try:
        assert batch.capture("context", other, lambda: None, reads=reads) is None
        await batch.run([])
        assert batch.capture("context", database, lambda: None, reads=reads) is None
        assert not reads
    finally:
        other.close()
        database.close()
