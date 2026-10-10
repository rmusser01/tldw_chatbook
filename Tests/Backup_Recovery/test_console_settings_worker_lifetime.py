"""Actual settings CAS workers retire exact finite Notes ownership."""

import asyncio
import sqlite3
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery.test_finite_db_counted_interval import live_operations
from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import local_root as local_root  # noqa: PLC0414
from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleSettingsComponent
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_generation_settings_metadata import (
    parse_console_generation_settings,
    snapshot_from_session_settings,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsAction,
    ConsoleSettingsDraftState,
    ConsoleSettingsSubmission,
    ConsoleSettingsSurface,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.usefixtures("local_root")]


@pytest.fixture
def settings_owner(tmp_path):
    database, store, _controller_owner, _gateway = _controller(tmp_path)
    store.append_message(
        "session-1", role=ConsoleMessageRole.USER, content="seed", persist=True
    )
    database.close()
    try:
        yield database, store
    finally:
        # Exact fixture-owned maintenance cleanup also retires the legacy RED
        # handle; it never clears unrelated global admission state.
        registry = database._connection_quiescence
        token = registry.begin_quiescence(timeout_seconds=5)
        try:
            registry.close_registered(token)
        finally:
            registry.end_quiescence(token)
        store.end_app_runtime()
        database.close()


def settings_commit(store):
    submission = ConsoleSettingsSubmission(
        submission_id="native-settings",
        action=ConsoleSettingsAction.APPLY_TO_CHAT,
        surface=ConsoleSettingsSurface.FULL_SETTINGS,
        origin=store.capture_console_settings_origin("session-1"),
        draft=ConsoleSettingsDraftState(
            settings=ConsoleSessionSettings(
                provider="openai",
                model="persisted-settings-model",
                temperature=0.61,
                streaming=False,
            ),
            context_policy_overrides=ConsoleContextPolicyOverrides(
                compaction_mode=ContextCompactionMode.OFF
            ),
            field_drafts=(),
            model_drafts=(),
            endpoint_draft=None,
        ),
        user_display_name_override=None,
        default_field_mask=frozenset(),
    )
    return store.commit_console_settings_live(submission)


@contextmanager
def observe_settings(
    database, persistence, *, held=None, event="call", entered=None, release=None
):
    """Observe installed original code objects without replacing custody seams."""
    writer_codes = {
        ChatPersistenceService.update_conversation_generation_settings.__code__: "generation",
        ChatPersistenceService.update_conversation_context_policy.__code__: "context",
    }
    getter_code = CharactersRAGDB.get_connection.__code__
    observations, connections, statements = [], [], []
    previous_thread, previous_main = threading.getprofile(), sys.getprofile()
    blocked = False

    def observe(frame, kind, value):
        nonlocal blocked
        if threading.current_thread() is threading.main_thread():
            return
        if frame.f_code is getter_code and kind == "return" and value is not None:
            if frame.f_locals.get("self") is database and all(
                value is not old for old in connections
            ):
                connections.append(value)
                value.set_trace_callback(statements.append)
        member = writer_codes.get(frame.f_code)
        if (
            member is not None
            and frame.f_locals.get("self") is persistence
            and kind == event
        ):
            observations.append((member, tuple(live_operations(database))))
            if member == held and not blocked:
                blocked = True
                entered.set()
                assert release.wait(10)

    threading.setprofile_all_threads(observe)
    try:
        yield observations, connections, statements
    finally:
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous_main)


@pytest.mark.asyncio
async def test_real_settings_writers_commit_sql_and_retire_each_native_callback(
    settings_owner,
):
    database, store = settings_owner
    persistence = store.persistence
    commit = settings_commit(store)
    with observe_settings(database, persistence) as (
        observations,
        connections,
        statements,
    ):
        outcome = await store.persist_console_settings_commit_serialized(commit)
    assert outcome.written_components == frozenset(ConsoleSettingsComponent)
    assert not worker_leases(
        database
    ), "completed Settings CAS left an ordinary worker handle live"
    assert [(member, len(operations)) for member, operations in observations] == [
        ("generation", 1),
        ("context", 1),
    ]
    assert len(connections) == 2, "each finite writer owns its freshly opened handle"
    assert any("UPDATE conversations" in sql for sql in statements)
    assert any(
        "INSERT INTO console_conversation_context_policy" in sql for sql in statements
    )
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    row = database.get_conversation_by_id(commit.persisted_conversation_id)
    assert (
        parse_console_generation_settings(row["metadata"]).snapshot.model
        == "persisted-settings-model"
    )
    assert (
        persistence.context_repository.load_policy(
            commit.persisted_conversation_id
        ).overrides.compaction_mode
        is ContextCompactionMode.OFF
    )
    database.close()
    assert not live_operations(database)
    pause = storage._begin_local_pause()
    try:
        assert pause.drain(time.monotonic() + 1)
    finally:
        pause.resume()


@pytest.mark.asyncio
@pytest.mark.parametrize("member", ["generation", "context"])
async def test_actual_writer_counts_before_sql_and_survives_cancelled_waiter(
    settings_owner, member
):
    database, store = settings_owner
    commit = settings_commit(store)
    database.close()
    entered, release = threading.Event(), threading.Event()
    pause = None
    pending = None
    with observe_settings(
        database, store.persistence, held=member, entered=entered, release=release
    ):
        try:
            pending = asyncio.create_task(
                store.persist_console_settings_commit_serialized(commit)
            )
            assert await asyncio.to_thread(entered.wait, 10)
            assert live_operations(
                database
            ), "Settings callback must count before its first SQL"
            lifecycle = store._settings_persistence_lifecycles["session-1"]
            drain = lifecycle.drain.task
            pause = storage._begin_local_pause()
            assert not pause.drain(time.monotonic() + 0.02)
            with pytest.raises(
                bootstrap.RecoveryRequired, match="storage_locally_paused"
            ):
                await store._run_console_settings_writer(
                    store.persistence,
                    store.persistence.update_conversation_generation_settings,
                    conversation_id=commit.persisted_conversation_id,
                    snapshot=snapshot_from_session_settings(commit.settings),
                    expected_snapshot=None,
                )
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert drain in lifecycle.tasks and not drain.done()
            release.set()
            outcome = await drain
            if member == "generation":
                assert (
                    ConsoleSettingsComponent.GENERATION_SETTINGS
                    in outcome.written_components
                )
                assert (
                    ConsoleSettingsComponent.CONTEXT_POLICY in outcome.failed_components
                )
            else:
                assert outcome.written_components == frozenset(ConsoleSettingsComponent)
            assert not live_operations(database) and not worker_leases(database)
            assert pause.drain(time.monotonic() + 1)
        finally:
            release.set()
            if pause is not None:
                pause.resume()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            lifecycle = store._settings_persistence_lifecycles.get("session-1")
            if lifecycle is not None and lifecycle.tasks:
                await asyncio.gather(*lifecycle.tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("member", ["generation", "context"])
async def test_repeated_drain_cancellation_keeps_actual_native_writer_registered(
    settings_owner, member
):
    database, store = settings_owner
    commit = settings_commit(store)
    database.close()
    entered, release = threading.Event(), threading.Event()
    pending, pause = None, None
    with observe_settings(
        database,
        store.persistence,
        held=member,
        event="return",
        entered=entered,
        release=release,
    ):
        try:
            pending = asyncio.create_task(
                store.persist_console_settings_commit_serialized(commit)
            )
            assert await asyncio.to_thread(entered.wait, 10)
            lifecycle = store._settings_persistence_lifecycles["session-1"]
            drain = lifecycle.drain.task
            assert live_operations(database) and worker_leases(database)
            pause = storage._begin_local_pause()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            for _ in range(2):
                drain.cancel()
                await asyncio.sleep(0.01)
                assert not drain.done() and drain in lifecycle.tasks
                assert live_operations(database) and worker_leases(database)
                assert not pause.drain(time.monotonic() + 0.02)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await drain
            await asyncio.sleep(0)
            assert drain not in lifecycle.tasks
            assert not worker_leases(database) and not live_operations(database)
            assert pause.drain(time.monotonic() + 1)
        finally:
            release.set()
            if pause is not None:
                pause.resume()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            lifecycle = store._settings_persistence_lifecycles.get("session-1")
            if lifecycle is not None and lifecycle.tasks:
                await asyncio.gather(*lifecycle.tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["adapter", "database", "repository", "repository-database", "writer"]
)
async def test_real_context_writer_rejects_changed_captured_source_after_sql(
    settings_owner, tmp_path, mutation
):
    from tldw_chatbook.Chat.console_context_repository import ConsoleContextRepository

    database, store = settings_owner
    persistence = store.persistence
    repository = persistence.context_repository
    original_writer = persistence.update_conversation_context_policy
    other = CharactersRAGDB(tmp_path / "other.sqlite", "other-settings")
    other.close()
    commit = settings_commit(store)
    database.close()
    entered, release = threading.Event(), threading.Event()
    pending = None
    prior = store.sessions()[0].context_policy_durable_revision
    with observe_settings(
        database,
        persistence,
        held="context",
        event="return",
        entered=entered,
        release=release,
    ):
        try:
            pending = asyncio.create_task(
                store.persist_console_settings_commit_serialized(commit)
            )
            assert await asyncio.to_thread(entered.wait, 10)
            if mutation == "adapter":
                store.persistence = ChatPersistenceService(other)
            elif mutation == "database":
                persistence.db = other
            elif mutation == "repository":
                persistence.context_repository = ConsoleContextRepository(other)
            elif mutation == "repository-database":
                repository.db = other
            else:
                persistence.update_conversation_context_policy = lambda **_kwargs: None
            release.set()
            outcome = await pending
            assert ConsoleSettingsComponent.CONTEXT_POLICY in outcome.failed_components
            assert store.sessions()[0].context_policy_durable_revision == prior
            assert not worker_leases(database) and not worker_leases(other)
            assert not live_operations(database) and not live_operations(other)
        finally:
            release.set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            store.persistence = persistence
            persistence.db = database
            persistence.context_repository = repository
            repository.db = database
            persistence.update_conversation_context_policy = original_writer
            other.close()


@pytest.mark.asyncio
async def test_queued_actual_writer_rejects_replaced_database_before_body(
    settings_owner, tmp_path, monkeypatch
):
    database, store = settings_owner
    persistence = store.persistence
    other = CharactersRAGDB(tmp_path / "replacement.sqlite", "replacement")
    other.close()
    commit = settings_commit(store)
    database.close()
    release = threading.Event()
    queued = asyncio.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(release.wait, 10)

        async def pinned(call, *args, **kwargs):
            queued.set()
            return await asyncio.get_running_loop().run_in_executor(
                executor, partial(call, *args, **kwargs)
            )

        monkeypatch.setattr(asyncio, "to_thread", pinned)
        pending = asyncio.create_task(
            store.persist_console_settings_commit_serialized(commit)
        )
        with observe_settings(database, persistence) as (
            observations,
            _connections,
            _statements,
        ):
            try:
                await asyncio.wait_for(queued.wait(), 5)
                persistence.db = other
                persistence.context_repository.db = other
                release.set()
                outcome = await pending
                assert (
                    ConsoleSettingsComponent.GENERATION_SETTINGS
                    in outcome.failed_components
                )
                assert not any(
                    member == "generation" for member, _ in observations
                ), "replaced source reached original writer body"
                assert not worker_leases(database) and not worker_leases(other)
            finally:
                release.set()
                await asyncio.gather(pending, return_exceptions=True)
                blocker.result(10)
                persistence.db = database
                persistence.context_repository.db = database
                other.close()


@pytest.mark.asyncio
async def test_standard_writer_preserves_borrowed_native_transaction(
    settings_owner, monkeypatch
):
    database, store = settings_owner
    persistence = store.persistence
    commit = settings_commit(store)
    database.close()
    snapshot = snapshot_from_session_settings(commit.settings)
    with ThreadPoolExecutor(max_workers=1) as executor:

        async def pinned(call, *args, **kwargs):
            return await asyncio.get_running_loop().run_in_executor(
                executor, partial(call, *args, **kwargs)
            )

        monkeypatch.setattr(asyncio, "to_thread", pinned)

        def borrow():
            connection = database.get_connection()
            connection.execute("BEGIN")
            return connection

        connection = await pinned(borrow)
        try:
            await store._run_console_settings_writer(
                persistence,
                persistence.update_conversation_generation_settings,
                conversation_id=commit.persisted_conversation_id,
                snapshot=snapshot,
                expected_snapshot=None,
            )

            def verify():
                assert database.get_connection() is connection
                assert connection.in_transaction
                return connection.execute("SELECT 17").fetchone()[0]

            assert await pinned(verify) == 17
            assert worker_leases(
                database
            ), "borrowed native owner was retired by the callback"
        finally:
            await pinned(lambda: (connection.rollback(), database.close()))
        assert not worker_leases(database)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape", ["custom", "subclass", "memory", "overridden-writer", "foreign-repository"]
)
async def test_unqualified_settings_writer_preserves_original_callback_contract(
    settings_owner, shape
):
    database, store = settings_owner
    persistence = store.persistence
    observations = []

    def original_route(*args, **kwargs):
        observations.append(
            (kwargs, tuple(live_operations(database)), threading.current_thread().name)
        )
        return 42

    if shape == "custom":
        adapter = SimpleNamespace(db=database)
        writer = original_route
    elif shape == "subclass":
        adapter = type("CustomPersistence", (ChatPersistenceService,), {})(database)
        writer = original_route
    elif shape == "memory":
        adapter = ChatPersistenceService(CharactersRAGDB(":memory:", "memory-settings"))
        writer = original_route
    elif shape == "overridden-writer":
        adapter = persistence
        adapter.update_conversation_generation_settings = original_route
        writer = adapter.update_conversation_generation_settings
    else:
        adapter = persistence
        adapter.context_repository = SimpleNamespace(
            db=database, save_policy_if_revision=original_route
        )
        writer = adapter.update_conversation_context_policy

    try:
        kwargs = (
            {"marker": 13}
            if shape != "foreign-repository"
            else {
                "conversation_id": "declared-id",
                "overrides": ConsoleContextPolicyOverrides(),
                "expected_revision": None,
            }
        )
        assert await store._run_console_settings_writer(adapter, writer, **kwargs) == 42
        assert observations and not observations[0][1]
        assert observations[0][2] != "MainThread"
    finally:
        if shape == "memory":
            adapter.db.close()


@pytest.mark.asyncio
async def test_actual_settings_write_error_retires_native_handle_and_keeps_other_component(
    settings_owner,
):
    database, store = settings_owner
    commit = settings_commit(store)
    with database.transaction() as cursor:
        cursor.execute(
            "CREATE TRIGGER settings_write_error BEFORE UPDATE OF metadata ON conversations BEGIN SELECT RAISE(ABORT, 'settings-control-error'); END"
        )
    database.close()
    with observe_settings(database, store.persistence) as (
        observations,
        connections,
        _statements,
    ):
        outcome = await store.persist_console_settings_commit_serialized(commit)
    assert ConsoleSettingsComponent.GENERATION_SETTINGS in outcome.failed_components
    assert ConsoleSettingsComponent.CONTEXT_POLICY in outcome.written_components
    assert [(member, len(operations)) for member, operations in observations] == [
        ("generation", 1),
        ("context", 1),
    ]
    assert not worker_leases(database) and not live_operations(database)
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    policy = store.persistence.context_repository.load_policy(
        commit.persisted_conversation_id
    )
    assert policy.overrides.compaction_mode is ContextCompactionMode.OFF


@pytest.mark.asyncio
async def test_native_writer_error_during_cancel_drain_keeps_cancellation_precedence(
    settings_owner,
):
    database, store = settings_owner
    commit = settings_commit(store)
    with database.transaction() as cursor:
        cursor.execute(
            "CREATE TRIGGER settings_write_error BEFORE UPDATE OF metadata ON conversations BEGIN SELECT RAISE(ABORT, 'settings-control-error'); END"
        )
    database.close()
    entered, release = threading.Event(), threading.Event()
    pending = None
    with observe_settings(
        database,
        store.persistence,
        held="generation",
        event="return",
        entered=entered,
        release=release,
    ):
        try:
            pending = asyncio.create_task(
                store.persist_console_settings_commit_serialized(commit)
            )
            assert await asyncio.to_thread(entered.wait, 10)
            lifecycle = store._settings_persistence_lifecycles["session-1"]
            drain = lifecycle.drain.task
            assert live_operations(database) and worker_leases(database)
            drain.cancel()
            await asyncio.sleep(0.01)
            assert not drain.done() and drain in lifecycle.tasks
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert not worker_leases(database) and not live_operations(database)
        finally:
            release.set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            lifecycle = store._settings_persistence_lifecycles.get("session-1")
            if lifecycle is not None and lifecycle.tasks:
                await asyncio.gather(*lifecycle.tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("member", ["generation", "context", "repository"])
async def test_actual_generation_writer_cannot_redirect_during_database_aba(
    settings_owner, tmp_path, member
):
    from tldw_chatbook.Chat.console_context_repository import ConsoleContextRepository

    database, store = settings_owner
    persistence = store.persistence
    repository = persistence.context_repository
    commit = settings_commit(store)
    database.close()
    other_path = tmp_path / "same-initial-conversation.sqlite"
    shutil.copyfile(database.db_path, other_path)
    other = CharactersRAGDB(other_path, "other-settings-aba")
    other_repository = ConsoleContextRepository(other)
    other.close()
    if member == "generation":
        writer_code = (
            ChatPersistenceService.update_conversation_generation_settings.__code__
        )
        receiver = persistence
    elif member == "context":
        writer_code = ChatPersistenceService.update_conversation_context_policy.__code__
        receiver = persistence
    else:
        writer_code = ConsoleContextRepository.save_policy_if_revision.__code__
        receiver = repository
    previous_thread, previous_main = threading.getprofile(), sys.getprofile()
    entered = False

    def observe(frame, event, _result):
        nonlocal entered
        if frame.f_code is writer_code and frame.f_locals.get("self") is receiver:
            if event == "call":
                entered = True
                if member == "generation":
                    persistence.db = other
                elif member == "context":
                    persistence.context_repository = other_repository
                else:
                    repository.db = other
            elif event == "return":
                persistence.db = database
                persistence.context_repository = repository
                repository.db = database

    threading.setprofile_all_threads(observe)
    try:
        outcome = await store.persist_console_settings_commit_serialized(commit)
    finally:
        threading.setprofile_all_threads(previous_thread)
        sys.setprofile(previous_main)
        persistence.db = database
        persistence.context_repository = repository
        repository.db = database
    try:
        assert entered
        assert not worker_leases(
            other
        ), "transient replacement acquired a native worker handle"
        original_row = database.get_conversation_by_id(commit.persisted_conversation_id)
        other_row = other.get_conversation_by_id(commit.persisted_conversation_id)
        original_snapshot = parse_console_generation_settings(
            original_row["metadata"]
        ).snapshot
        other_snapshot = parse_console_generation_settings(
            other_row["metadata"]
        ).snapshot
        failed = (
            ConsoleSettingsComponent.GENERATION_SETTINGS
            if member == "generation"
            else ConsoleSettingsComponent.CONTEXT_POLICY
        )
        assert failed in outcome.failed_components
        if member == "generation":
            assert original_snapshot is None
        else:
            assert original_snapshot.model == "persisted-settings-model"
            assert (
                persistence.context_repository.load_policy(
                    commit.persisted_conversation_id
                ).revision
                is None
            )
        assert other_snapshot is None
        assert (
            other_repository.load_policy(commit.persisted_conversation_id).revision
            is None
        )
    finally:
        registry = other._connection_quiescence
        token = registry.begin_quiescence(timeout_seconds=5)
        try:
            registry.close_registered(token)
        finally:
            registry.end_quiescence(token)
        other.close()
