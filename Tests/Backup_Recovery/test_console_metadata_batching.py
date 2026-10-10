"""Finite Console snapshots avoid repeated native preparation and stale reads."""

import asyncio
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest

from Tests.Backup_Recovery.test_finite_db_retirement import worker_leases
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from Tests.UI.test_console_character_context import _controller
from tldw_chatbook.DB import private_sqlite
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.Workspaces.models import (
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceRuntimeBinding,
)
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


pytestmark = pytest.mark.usefixtures("local_root")


def observe_worker_connections(monkeypatch, database, accessor):
    """Call through to SQLite and retain each real worker handle and its SQL."""
    origin = threading.current_thread()
    original = getattr(database, accessor)
    connections = []
    statements = []

    def observed():
        connection = original()
        if threading.current_thread() is not origin and not any(
            item is connection for item in connections
        ):
            connections.append(connection)
            connection.set_trace_callback(
                lambda sql: statements.append((len(connections), sql))
            )
        return connection

    monkeypatch.setattr(database, accessor, observed)
    return connections, statements


@pytest.mark.asyncio
async def test_character_scope_pairs_share_one_retired_native_connection(
    tmp_path, monkeypatch
):
    """Splitting paired metadata into offloads doubles real native setup."""
    database = CharactersRAGDB(tmp_path / "notes.db", "batching")
    authority = database.get_local_authority_id()
    revision = database.get_character_conversation_search_revision()
    controller = _controller(database_accessor=lambda: database)
    connections, statements = observe_worker_connections(
        monkeypatch, database, "get_connection"
    )
    preparations = []
    prepare = private_sqlite.prepare_in_helper

    def observed_prepare(*args, **kwargs):
        preparations.append(threading.current_thread())
        return prepare(*args, **kwargs)

    monkeypatch.setattr(private_sqlite, "prepare_in_helper", observed_prepare)
    try:
        snapshot = await controller._capture_scope()
        assert snapshot.fingerprint.data_authority_id == authority
        assert snapshot.fingerprint.data_revision == revision
        assert sum("SELECT local_authority_id" in sql for _, sql in statements) == 2
        assert sum("SELECT data_revision" in sql for _, sql in statements) == 2
        assert len(connections) == 1, "both metadata pairs need only one native handle"
        if preparations:  # Windows prepares in-process; POSIX uses the helper.
            assert len(preparations) == 1
        assert not worker_leases(database)
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("member", ["authority", "revision"])
async def test_paired_scope_retries_real_database_metadata_changes(
    tmp_path, monkeypatch, member
):
    """Dropping either paired read can publish pre-mutation authority metadata."""
    database = CharactersRAGDB(tmp_path / "notes.db", "batching")
    controller = _controller(database_accessor=lambda: database)
    entered, release = threading.Event(), threading.Event()
    getter = (
        "get_local_authority_id"
        if member == "authority"
        else "get_character_conversation_search_revision"
    )
    read = getattr(database, getter)
    blocked = False

    def delayed_read():
        nonlocal blocked
        value = read()
        if threading.current_thread() is not threading.main_thread() and not blocked:
            blocked = True
            entered.set()
            assert release.wait(10)
        return value

    monkeypatch.setattr(database, getter, delayed_read)
    pending = asyncio.create_task(controller._capture_scope())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        if member == "authority":
            authority = str(uuid4())
            with database.transaction() as cursor:
                cursor.execute(
                    "UPDATE rag_identity_context SET local_authority_id = ? "
                    "WHERE context_name = 'default'",
                    (authority,),
                )
            revision = database.get_character_conversation_search_revision()
        else:
            authority = database.get_local_authority_id()
            revision = database.increment_character_conversation_search_revision()
        release.set()
        snapshot = await pending
        assert snapshot.fingerprint.data_authority_id == authority
        assert snapshot.fingerprint.data_revision == revision
        assert not worker_leases(database)
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("member", ["database", "character", "conversation"])
async def test_midpoint_scope_switch_skips_remaining_stale_metadata(
    tmp_path, monkeypatch, member
):
    """A midpoint scope change must stop stale reads on the UI thread."""
    first = CharactersRAGDB(tmp_path / "first.db", "first")
    second = CharactersRAGDB(tmp_path / "second.db", "second")
    active = [first]
    character = [(1, "Original")]
    conversation = ["original"]
    origin = threading.current_thread()
    accessor_threads = []

    def on_loop(value):
        accessor_threads.append(threading.current_thread())
        return value

    controller = _controller(
        database_accessor=lambda: on_loop(active[0]),
        current_character_accessor=lambda: on_loop(character[0]),
        open_conversation_accessor=lambda: on_loop(conversation[0]),
    )
    connections, statements = observe_worker_connections(
        monkeypatch, first, "get_connection"
    )
    entered, release = threading.Event(), threading.Event()
    read_revision = first.get_character_conversation_search_revision
    blocked = False

    def delayed_revision():
        nonlocal blocked
        revision = read_revision()
        if threading.current_thread() is not origin and not blocked:
            blocked = True
            entered.set()
            assert release.wait(10)
        return revision

    monkeypatch.setattr(
        first, "get_character_conversation_search_revision", delayed_revision
    )
    pending = asyncio.create_task(controller._capture_scope())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        if member == "database":
            active[0] = second
        elif member == "character":
            character[0] = (2, "Replacement")
        else:
            conversation[0] = "replacement"
        release.set()
        snapshot = await pending
        assert snapshot.database is active[0]
        assert snapshot.fingerprint.current_character_id == character[0][0]
        assert snapshot.fingerprint.current_character_label == character[0][1]
        assert snapshot.fingerprint.open_conversation_id == conversation[0]
        assert set(accessor_threads) == {origin}
        # The rejected first capture stops immediately after its first pair.
        first_pair_reads = [sql for number, sql in statements if number == 1]
        assert sum("SELECT local_authority_id" in sql for sql in first_pair_reads) == 1
        assert sum("SELECT data_revision" in sql for sql in first_pair_reads) == 1
        assert connections
        assert not worker_leases(first)
        assert not worker_leases(second)
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        first.close()
        second.close()


@pytest.mark.asyncio
async def test_cancelled_character_capture_keeps_running_handle_owned_until_retired(
    tmp_path, monkeypatch
):
    """Awaiter cancellation cannot retire the paired callback's active handle."""
    database = CharactersRAGDB(tmp_path / "notes.db", "batching")
    controller = _controller(database_accessor=lambda: database)
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    read_revision = database.get_character_conversation_search_revision
    blocked = False

    def delayed_revision():
        nonlocal blocked
        revision = read_revision()
        if not blocked:
            blocked = True
            entered.set()
            try:
                assert release.wait(10)
            finally:
                exited.set()
        return revision

    monkeypatch.setattr(
        database, "get_character_conversation_search_revision", delayed_revision
    )
    pending = asyncio.create_task(controller._capture_scope())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        pending.cancel()
        for _ in range(20):
            await asyncio.sleep(0)
        assert (
            not pending.done()
        ), "native Character cancellation must await its held callback"
        assert worker_leases(database)
        pending.cancel()
        for _ in range(20):
            await asyncio.sleep(0)
        assert (
            not pending.done()
        ), "repeated cancellation abandoned native Character custody"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert (
            exited.is_set()
        ), "cancellation returned before the original native reader"
        assert await asyncio.to_thread(exited.wait, 10)
        for _ in range(200):
            if not worker_leases(database):
                break
            await asyncio.sleep(0.005)
        assert not worker_leases(database)
        assert (await controller._capture_scope()).database is database
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        assert await asyncio.to_thread(exited.wait, 10)
        database.close()


@pytest.mark.asyncio
async def test_workspace_snapshot_reads_bindings_once_and_rechecks_real_folder_status(
    tmp_path, monkeypatch
):
    """Duplicate listers repeat SQL; stored readiness must still be rechecked."""
    database = WorkspaceDB(tmp_path / "workspaces.db")
    registry = LocalWorkspaceRegistryService(database)
    default = registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="named", name="Named")
    folder = tmp_path / "project"
    folder.mkdir()
    local = registry.add_folder_binding("named", folder)
    other = registry.save_runtime_binding(
        WorkspaceRuntimeBinding(
            workspace_id="named",
            binding_id="git",
            binding_kind=RuntimeBindingKind.GIT_WORKTREE,
            label="Repository",
            locator=str(folder),
            status=RuntimeBindingStatus.INSPECT_ONLY,
            metadata={"branch": "dev"},
        )
    )
    controller = object.__new__(ConsoleWorkspaceController)
    controller.app_instance = SimpleNamespace(workspace_registry_service=registry)
    connections, statements = observe_worker_connections(
        monkeypatch, database, "_held_connection"
    )
    try:
        for ready in (True, False):
            statements.clear()
            if not ready:
                folder.rmdir()
            availability, by_id = await asyncio.to_thread(
                controller._capture_workspace_files_availability,
                (default.workspace_id, "named"),
            )
            assert availability == {default.workspace_id: False, "named": ready}
            assert by_id[default.workspace_id] == ()
            assert [binding.binding_id for binding in by_id["named"]] == [
                local.binding_id,
                other.binding_id,
            ]
            assert by_id["named"][0].status is (
                RuntimeBindingStatus.READY if ready else RuntimeBindingStatus.MISSING
            )
            assert by_id["named"][1] == other
            # Each public runtime read executes one SELECT, including Default's
            # stale-binding check. Both workspaces need exactly two SQL reads.
            binding_reads = [
                sql
                for _, sql in statements
                if "SELECT" in sql and "FROM workspace_runtime_bindings" in sql
            ]
            assert len(binding_reads) == 2
            assert not worker_leases(database)
        assert len(connections) == 2, "each finite snapshot retires its own handle"
    finally:
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [True, False])
async def test_character_midpoint_rendezvous_retires_after_cancellation_or_expiry(
    tmp_path, monkeypatch, cancel
):
    """A cancelled awaiter or a lost UI callback cannot strand a native handle."""
    from tldw_chatbook.UI.Console_Modules import character_context

    database = CharactersRAGDB(tmp_path / "notes.db", "rendezvous")
    accessor_calls = []

    def current_character():
        accessor_calls.append(threading.current_thread())
        return None

    controller = _controller(
        database_accessor=lambda: database,
        current_character_accessor=current_character,
    )
    loop = asyncio.get_running_loop()
    schedule = loop.call_soon_threadsafe
    queued = []
    entered = threading.Event()
    delivered = False

    def hold_midpoint(callback, *args, **kwargs):
        if callback.__name__ == "validate_ambient_on_loop":
            queued.append(callback)
            entered.set()
            return None
        return schedule(callback, *args, **kwargs)

    monkeypatch.setattr(loop, "call_soon_threadsafe", hold_midpoint)
    deadline = character_context._SCOPE_AMBIENT_CHECK_TIMEOUT_SECONDS
    if not cancel:
        monkeypatch.setattr(
            character_context, "_SCOPE_AMBIENT_CHECK_TIMEOUT_SECONDS", 0.02
        )
    pending = asyncio.create_task(controller._capture_scope())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        if cancel:
            pending.cancel()
            for _ in range(20):
                await asyncio.sleep(0)
            assert not pending.done(), "native cancellation outran the queued midpoint"
            assert worker_leases(database), "a queued midpoint still owns its handle"
            queued[0]()
            delivered = True
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert not worker_leases(
                database
            ), "cancellation returned before native retirement"
        else:
            with pytest.raises(character_context._ConsoleCharacterScopeReadError):
                await pending
            assert not worker_leases(database)
            call_count = len(accessor_calls)
            queued[0]()
            delivered = True
            assert (
                len(accessor_calls) == call_count
            ), "expired checks cannot inspect UI state"
        for _ in range(200):
            if not worker_leases(database):
                break
            await asyncio.sleep(0.005)
        assert not worker_leases(database)
        assert set(accessor_calls) == {threading.current_thread()}
        monkeypatch.setattr(loop, "call_soon_threadsafe", schedule)
        monkeypatch.setattr(
            character_context, "_SCOPE_AMBIENT_CHECK_TIMEOUT_SECONDS", deadline
        )
        assert (await controller._capture_scope()).database is database
        assert not worker_leases(database)
    finally:
        if queued and not delivered:
            queued[0]()
        monkeypatch.setattr(loop, "call_soon_threadsafe", schedule)
        await asyncio.gather(pending, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_character_midpoint_closed_event_loop_retires_native_handle(tmp_path):
    """Shutting down the UI loop must not leave the finite worker lease held."""
    from tldw_chatbook.DB.base_db import run_owned_db_call

    database = CharactersRAGDB(tmp_path / "notes.db", "closed-loop")
    controller = _controller(database_accessor=lambda: database)
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()
    try:
        with pytest.raises(RuntimeError, match="Event loop is closed"):
            await run_owned_db_call(
                database,
                controller._read_database_scope_metadata_pair,
                database,
                None,
                "",
                closed_loop,
            )
        assert not worker_leases(database)
    finally:
        database.close()


@pytest.mark.asyncio
async def test_workspace_snapshot_preserves_custom_folder_status_contract(
    tmp_path, monkeypatch
):
    """An adapter override's folder status must not be replaced by local probing."""
    from dataclasses import replace

    class CustomRegistry(LocalWorkspaceRegistryService):
        def list_folder_bindings(self, workspace_id):
            return tuple(
                replace(binding, status=RuntimeBindingStatus.MISSING)
                for binding in super().list_folder_bindings(workspace_id)
            )

    database = WorkspaceDB(tmp_path / "workspaces.db")
    registry = CustomRegistry(database)
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="named", name="Named")
    folder = tmp_path / "project"
    folder.mkdir()
    local = registry.add_folder_binding("named", folder)
    controller = object.__new__(ConsoleWorkspaceController)
    controller.app_instance = SimpleNamespace(workspace_registry_service=registry)
    try:
        availability, by_id = await asyncio.to_thread(
            controller._capture_workspace_files_availability, ("named",)
        )
        assert availability == {"named": False}
        assert by_id["named"][0].binding_id == local.binding_id
        assert by_id["named"][0].status is RuntimeBindingStatus.MISSING
        assert not worker_leases(database)
    finally:
        database.close()
