"""Finite offloads retire real native handles on their executing threads."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.db_offload import run_db_off_loop
from tldw_chatbook.Workspaces import change_retention


def worker_leases(db):
    """Observe native ownership without borrowing the connection."""
    with storage._lock:
        return [
            lease for lease in storage._live_leases
            if lease.resource_path == db.db_path
            and lease.resource_thread is not threading.current_thread()
        ]


@pytest.fixture(params=["subscriptions", "conversations", "owned_notes", "owned_runs", "owned_workspaces"])
def offload(request, tmp_path, local_root, monkeypatch):
    if request.param.startswith("owned_"):
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.DB.base_db import run_owned_db_call
        from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

        if request.param == "owned_notes":
            db = CharactersRAGDB(tmp_path / "notes.db", "test")
            read = db.list_all_active_conversations
        elif request.param == "owned_runs":
            db = AgentRunsDB(tmp_path / "runs.db")
            read = db.pending_wake_conversation_ids
        else:
            db = WorkspaceDB(tmp_path / "workspaces.db")
            def read():
                with db.connection() as connection:
                    return connection.execute("SELECT 1").fetchone()[0]

        async def call(body):
            return await run_owned_db_call(db, body)
    elif request.param == "subscriptions":
        db = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
        row_id = db.add_subscription("retained", "rss", "https://example.invalid/feed", auto_pause_threshold=3)

        def read():
            return db.get_subscription(row_id)["name"]

        async def call(body):
            return await run_db_off_loop(db, body)
    else:
        db = CharactersRAGDB(tmp_path / "conversations.db", "test")
        service = ChatConversationService(db)
        service.create_conversation(title="retained")
        scope = ChatConversationScopeService(local_service=service, server_service=None)
        original = service.list_conversations

        def read():
            return original()["items"][0]["title"]

        async def call(body):
            monkeypatch.setattr(service, "list_conversations", body)
            return await scope.list_conversations()
    try:
        yield db, call, read
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_finite_read_retires_native_handle_and_owner_can_be_reused(offload, fail):
    db, call, read = offload
    expected = read()
    error = ValueError("callback failed")

    def body():
        result = read()
        if fail:
            raise error
        return result

    if fail:
        with pytest.raises(ValueError) as caught:
            await call(body)
        assert caught.value is error
    else:
        assert await call(body) == expected
    assert not worker_leases(db)
    assert await call(read) == expected
    assert not worker_leases(db)
    assert read() == expected


@pytest.mark.asyncio
async def test_cancelled_awaiter_does_not_retire_running_native_work(offload):
    db, call, read = offload
    entered, finish, exited = threading.Event(), threading.Event(), threading.Event()

    def body():
        try:
            result = read()
            entered.set()
            assert finish.wait(3)
            return result
        finally:
            exited.set()

    task = asyncio.create_task(call(body))
    try:
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert worker_leases(db), "a cancelled awaiter cannot retire a running callback"
        finish.set()
        for _ in range(200):
            if not worker_leases(db):
                break
            await asyncio.sleep(0.005)
        assert not worker_leases(db)
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
        # Wait for the actual finite callback even when an assertion fails.
        assert await asyncio.to_thread(exited.wait, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_app_retention_retires_its_constructed_database(tmp_path, local_root, monkeypatch, fail):
    observed = []
    run_ids = []
    report = change_retention.PruneReport(rows_pruned=7)

    def prune(db, service):
        observed.append(db)
        run_ids.append(db.create_run(conversation_id="kept", agent_kind="primary"))
        if fail:
            raise ValueError("prune failed")
        return report

    monkeypatch.setattr(change_retention, "prune_change_history", prune)
    # The pass prunes an existing runs DB only (TASK-34100.16.1).
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    AgentRunsDB(tmp_path / "agent_runs.db").close()
    result = await asyncio.to_thread(
        change_retention.run_retention_for_app,
        tmp_path / "chachanotes.db",
        service=SimpleNamespace(available=True),
    )
    assert result is (None if fail else report)
    assert not worker_leases(observed[0])
    # Retirement leaves committed rows and the owner usable on a later call.
    try:
        assert observed[0].get_run(run_ids[0])["conversation_id"] == "kept"
    finally:
        observed[0].close()


def test_app_retention_never_creates_an_absent_runs_db(tmp_path, local_root):
    """No runs DB means nothing to prune; the pass must not create one.

    TASK-34100.16.1: the startup pass ran 65 s after launch and created
    ``agent_runs.db`` on a profile that had none. Backup Review lists that
    store as unused, so a Review taken in the first minute was voided at
    Create with ``scope_changed``.
    """
    runs = tmp_path / "agent_runs.db"

    result = change_retention.run_retention_for_app(
        tmp_path / "chachanotes.db", service=SimpleNamespace(available=True)
    )

    assert result is None
    assert not runs.exists(), "the retention pass created the runs store"
    assert not list(tmp_path.glob("agent_runs.db*"))


@pytest.mark.asyncio
async def test_failed_sync_log_prune_still_retires_its_worker_connection(
    tmp_path, local_root, monkeypatch
):
    """A prune that raises after its DELETE still retires its pool-thread handle.

    TASK-38955 runs each prune inside ``operation_owned_connection``; its own
    test covers the success path. TASK-34100.16.1 review: keep the failure
    path too, since a kept lease ends every later Create in admission_timeout.
    """
    from loguru import logger

    from tldw_chatbook import app_lifecycle
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    chat = CharactersRAGDB(tmp_path / "chat.db", "test")
    media = MediaDatabase(tmp_path / "media.db", client_id="test")
    ran = []
    original = chat.delete_sync_log_entries_older_than

    def prune(cutoff):
        ran.append(threading.current_thread())
        original(cutoff)
        raise ValueError("prune failed after its delete")

    monkeypatch.setattr(chat, "delete_sync_log_entries_older_than", prune)
    monkeypatch.setattr(app_lifecycle, "get_cli_setting", lambda *args, **kwargs: 90)
    host = SimpleNamespace(chachanotes_db=chat, media_db=media, loguru_logger=logger)
    try:
        await app_lifecycle.LifecycleMixin._perform_sync_log_retention(host)

        assert ran and all(thread is not threading.current_thread() for thread in ran)
        assert not worker_leases(chat), "the failed prune kept its connection"
    finally:
        chat.close_connection()
        media.close_connection()


@pytest.mark.asyncio
async def test_note_backlinks_lookup_retires_its_worker_connection(tmp_path, local_root):
    """Opening a note's Info leaves no pool-thread handle holding storage.

    TASK-34100.16.1: Library's Info lists the notes that link to the open one
    through ``NotesScopeService.list_note_backlinks``, which ran the read on an
    executor thread and left that thread's ChaChaNotes connection cached. Its
    storage lease is one a backup's local pause never revokes, so every Create
    backup after a note was opened ended in ``admission_timeout`` (live drain
    residue: one lease, created in ``get_notes_linking_to`` on ``asyncio_1``).
    """
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

    db = CharactersRAGDB(tmp_path / "notes.db", "test")
    target = db.add_note("Target", "hub")
    linker = db.add_note("Linker", f"See [t](note://{target}).")
    notes = NotesInteropService(
        base_db_directory=tmp_path, api_client_id="test", global_db_to_use=db
    )
    # The app reuses one cached per-user handle after its first use; seed it so
    # the test does not construct one (that path reads config, not storage).
    notes._db_instances["test-user"] = db
    service = NotesScopeService(
        local_notes_service=notes, server_service=None, policy_enforcer=None
    )
    try:
        backlinks = await service.list_note_backlinks(
            scope="local_note", note_id=target, user_id="test-user"
        )

        assert [row["id"] for row in backlinks] == [linker]
        assert not worker_leases(db), "the backlinks worker kept its connection"
        assert [row["id"] for row in db.get_notes_linking_to(target)] == [linker]
    finally:
        db.close_connection()


def _local_notes(tmp_path):
    """A real notes DB behind the service the app uses, cache seeded as above."""
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService

    db = CharactersRAGDB(tmp_path / "notes.db", "test")
    target = db.add_note("Target", "hub")
    other = db.add_note("Abacus", "body")
    notes = NotesInteropService(
        base_db_directory=tmp_path, api_client_id="test", global_db_to_use=db
    )
    notes._db_instances["test-user"] = db
    return db, notes, target, other


@pytest.mark.asyncio
async def test_note_link_picker_lookup_retires_its_worker_connection(tmp_path, local_root):
    """Typing ``[[`` in a note leaves no pool-thread handle holding storage.

    TASK-34100.16.1 review: the ``[[`` picker (TASK-34000.11) read its
    candidates on a bare executor thread, the shape the backlinks fix removed.
    Live, one use of the picker made every later idle Create backup end in
    ``admission_timeout`` (g161v5-rt/27-28).
    """
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

    db, notes, target, other = _local_notes(tmp_path)
    service = NotesScopeService(
        local_notes_service=notes, server_service=None, policy_enforcer=None
    )
    try:
        rows = await service.list_note_link_candidates(
            scope="local_note", query="", exclude_note_id=target, user_id="test-user"
        )

        assert [row["id"] for row in rows] == [other]
        assert not worker_leases(db), "the link-picker worker kept its connection"
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_note_count_fallback_retires_its_worker_connection(tmp_path, local_root):
    """``count_notes``' general path retires its worker handle too.

    Only the exact production types take the self-closing fast path; any other
    service shape ran ``count`` on a bare executor thread and kept its lease.
    """
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService

    class _Scope(NotesScopeService):
        pass

    db, notes, _, _ = _local_notes(tmp_path)
    service = _Scope(local_notes_service=notes, server_service=None, policy_enforcer=None)
    try:
        assert await service.count_notes(scope="local_note", user_id="test-user") == 2
        assert not worker_leases(db), "the count worker kept its connection"
    finally:
        db.close_connection()


def test_every_notes_scope_offload_retires_its_worker_handles():
    """Each executor offload in NotesScopeService runs a self-retiring callable.

    Structural guard for the TASK-34100.16.1 class: a bare ``asyncio.to_thread``
    of a local Notes read keeps a storage lease a backup pause cannot revoke.
    ``count_in_worker`` closes its own handle.
    """
    import ast
    import inspect

    from tldw_chatbook.Notes import notes_scope_service

    tree = ast.parse(inspect.getsource(notes_scope_service))
    offloads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"to_thread", "run_in_executor"}
    ]
    bare = [
        (node.lineno, ast.unparse(node.args[0]) if node.args else "")
        for node in offloads
        if not (
            node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in {"run_finite_local_worker", "count_in_worker"}
        )
    ]

    assert len(offloads) >= 5
    assert bare == []


@pytest.mark.asyncio
async def test_note_link_follow_read_retires_its_worker_connection(tmp_path, local_root):
    """Following a ``note://`` link leaves no pool-thread handle holding storage.

    TASK-34100.16.1 review: Library resolved the link target with a bare
    ``asyncio.to_thread(notes_service.get_note_by_id, ...)`` (TASK-34000.11).
    """
    from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
        LibraryNotesController,
    )

    db, notes, target, _ = _local_notes(tmp_path)
    opened, notices = [], []

    async def open_note(note_id, *, link_origin):
        opened.append((note_id, link_origin))

    controller = SimpleNamespace(
        _close_library_note_link_picker=lambda body: None,
        _library_note_body_or_none=lambda: None,
        _library_note_session=SimpleNamespace(snapshot=None),
        app=SimpleNamespace(notify=lambda *args, **kwargs: notices.append(args)),
        app_instance=SimpleNamespace(notes_service=notes),
        _library_notes_user_id=lambda: "test-user",
        _library_notes_mutation_fenced=lambda: False,
        _open_library_note_from_link=open_note,
    )
    try:
        await LibraryNotesController._open_library_note_link_target(controller, target)

        assert (opened, notices) == ([(target, True)], [])
        assert not worker_leases(db), "the link-follow worker kept its connection"
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["custom", "subclass", "memory"])
async def test_subscription_unqualified_owner_keeps_previous_route(shape):
    class CustomDB:
        is_memory_db = False

        def close(self):
            raise AssertionError("unqualified close")

    class DerivedDB(SubscriptionsDB):
        def close(self):
            raise AssertionError("subclass close")

    db = object.__new__(DerivedDB) if shape == "subclass" else CustomDB()
    db.is_memory_db = shape == "memory"
    origin = threading.current_thread()
    result = await run_db_off_loop(db, threading.current_thread)
    assert (result is origin) is (shape == "memory")


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["custom", "service_subclass", "db_subclass", "memory", "coroutine", "server"])
async def test_conversation_unqualified_route_never_closes(shape):
    class DerivedService(ChatConversationService):
        pass

    class DerivedDB(CharactersRAGDB):
        pass

    def forbidden_close():
        raise AssertionError("unqualified close")

    db_type = DerivedDB if shape == "db_subclass" else CharactersRAGDB
    db = object.__new__(db_type)
    db.is_memory_db = shape == "memory"
    db.close_connection = forbidden_close
    service_type = DerivedService if shape == "service_subclass" else ChatConversationService
    service = object.__new__(service_type)
    service.db = db
    if shape == "custom":
        service = SimpleNamespace(db=db)
    origin = threading.current_thread()
    if shape == "coroutine":
        async def listing():
            return threading.current_thread()
    else:
        def listing():
            return threading.current_thread()
    service.list_conversations = listing
    scope = ChatConversationScopeService(local_service=service, server_service=service)
    result = await scope.list_conversations(mode="server" if shape == "server" else "local")
    assert (result is origin) is (shape in {"memory", "coroutine", "server"})


@pytest.mark.asyncio
async def test_conversation_reassignment_does_not_redirect_worker_cleanup(tmp_path, local_root):
    db = CharactersRAGDB(tmp_path / "original.db", "test")
    service = ChatConversationService(db)
    scope = ChatConversationScopeService(local_service=service, server_service=None)
    service.create_conversation(title="retained")
    listing = service.list_conversations
    entered, release = threading.Event(), threading.Event()
    replacement_closes = []

    def read_then_wait():
        result = listing()
        entered.set()
        assert release.wait(3)
        return result

    service.list_conversations = read_then_wait
    task = asyncio.create_task(scope.list_conversations())
    try:
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set()
        service.db = SimpleNamespace(close_connection=lambda: replacement_closes.append(True))
        release.set()
        result = await task
        assert result["items"][0]["title"] == "retained"
        assert not replacement_closes
        assert not worker_leases(db)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("offload", ["owned_notes", "owned_runs", "owned_workspaces"], indirect=True)
async def test_owned_callback_remains_admitted_after_last_repository_read(offload):
    db, call, read = offload
    observed = []

    def body():
        result = read()
        with storage._lock:
            observed.append(any(
                operation.thread is threading.current_thread()
                and operation.participant.repository() is db
                for operation in storage._operations
            ))
        return result

    await call(body)
    assert observed == [True], "the complete callback must own admission"
    assert not worker_leases(db)


def _console_stores(tmp_path):
    """Real ChaChaNotes and Media stores, as Console's app attributes hold them."""
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    db = CharactersRAGDB(tmp_path / "notes.db", "test")
    media = MediaDatabase(tmp_path / "media.db", client_id="test")
    return db, media


def _local_pause_drains():
    """What a backup's local pause needs: no live lease on any thread."""
    import time

    pause = storage._begin_local_pause()
    try:
        return pause.drain(time.monotonic() + 1.0)
    finally:
        pause.resume()


# TASK-34100.16.1 review v5r2 (R2-1): each read Console's retrieval-scope picker
# makes when it opens or filters, by the lister method that issues it.
_SCOPE_PICKER_READS = {
    "notes_list_notes": lambda spl, app: spl.build_notes_source_lister(
        app, user_id="test-user"
    )._candidate_rows(text=""),
    "notes_keywords_batch": lambda spl, app: spl.build_notes_source_lister(
        app, user_id="test-user"
    )._tags_for_batch(("1",)),
    "media_keywords_batch": lambda spl, app: spl.build_media_source_lister(
        app
    )._keywords_for_media(("1",)),
    "tag_vocabulary_both_stores": lambda spl, app: spl.build_keyword_tag_lister(app)(""),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("read", sorted(_SCOPE_PICKER_READS))
async def test_console_scope_picker_reads_retire_their_worker_connections(
    tmp_path, local_root, read
):
    """Opening Console's retrieval-scope picker leaves no pool-thread handle.

    TASK-34100.16.1 review v5r2 (R2-1): the picker's listers ran bare
    ``asyncio.to_thread`` reads on the app's ChaChaNotes and Media stores
    (``list_notes``, ``get_keywords_for_notes_batch``,
    ``fetch_keywords_for_media_batch``, ``get_keyword_usage_stats``,
    ``_notes_tag_usage``). Each kept its connection, so the local pause could
    not drain and an idle Create backup ended in ``admission_timeout``.
    """
    from tldw_chatbook.Chat import scope_picker_listers as spl

    db, media = _console_stores(tmp_path)
    db.add_note("Alpha", "body")
    db.close_connection()  # setup's own main-thread handle is not under test
    media.close_connection()
    app = SimpleNamespace(chachanotes_db=db, media_db=media, notes_scope_service=None)
    try:
        await _SCOPE_PICKER_READS[read](spl, app)

        leaked = [str(lease.resource_path) for lease in worker_leases(db) + worker_leases(media)]
        assert leaked == [], f"{read} kept its worker connection"
        assert _local_pause_drains(), "a backup's local pause could not drain"
    finally:
        db.close_connection()
        media.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "worker",
    [
        "_console_dictionary_attach_worker",
        "_console_dictionary_detach_worker",
        "_console_worldbook_attach_worker",
        "_console_worldbook_detach_worker",
    ],
)
async def test_console_rail_pickers_retire_their_worker_connections(
    tmp_path, local_root, worker
):
    """Console's dictionary and world-book pickers leave no pool-thread handle.

    TASK-34100.16.1 review v5r2 (R2-1): the dictionary attach/detach pickers
    listed their rows with a bare ``asyncio.to_thread`` on the app's
    ChaChaNotes store, and the world-book pickers beside them did the same for
    their lists and for the attach/detach write. The world-book cases pick a
    book, so the write runs too.
    """
    from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    db = CharactersRAGDB(tmp_path / "notes.db", "test")
    conversation = db.add_conversation({"title": "Rail"})
    books = WorldBookManager(db)
    book = books.create_world_book("Lore")
    if worker == "_console_worldbook_detach_worker":
        books.associate_world_book_with_conversation(conversation, book)
    db.close_connection()  # setup's own main-thread handle is not under test
    notices = []

    async def push_screen_wait(screen):
        return book if "worldbook" in worker else None

    async def refreshed():
        return None

    screen = SimpleNamespace(
        _character=SimpleNamespace(
            _current_console_rail_conversation_id=lambda: conversation
        ),
        app_instance=SimpleNamespace(
            chachanotes_db=db,
            notify=lambda *args, **kwargs: notices.append(args),
            push_screen_wait=push_screen_wait,
        ),
        _retrieval=SimpleNamespace(
            refresh_active_world_books_summary=refreshed,
            refresh_active_dictionaries_summary=refreshed,
        ),
        _console_dictionary_dialog_active=True,
        _console_worldbook_dialog_active=True,
    )
    try:
        await getattr(ChatScreen, worker)(screen)

        assert not [n for n in notices if "failed" in str(n).lower()], notices
        assert not worker_leases(db), f"{worker} kept its worker connection"
        assert _local_pause_drains(), "a backup's local pause could not drain"
        if worker == "_console_worldbook_attach_worker":
            attached = WorldBookManager(db).get_world_books_for_conversation(
                conversation, enabled_only=False
            )
            assert [row["id"] for row in attached] == [book]
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "leg", ["search_media_fts5", "search_conversations_fts5", "search_notes_fts5"]
)
async def test_console_rag_search_legs_retire_their_worker_connections(
    tmp_path, local_root, leg
):
    """Console's RAG search legs leave no pool-thread handle on the app stores.

    TASK-34100.16.1 review v5r2 (R2-1): the reviewer's "same shape, not
    probed" path. ``pipeline_functions_simple``'s FTS legs, which Console's
    RAG pipeline runs, searched the app's live stores on bare executor threads.
    """
    from tldw_chatbook.RAG_Search import pipeline_functions_simple as legs

    db, media = _console_stores(tmp_path)
    db.add_note("Alpha", "hello world")
    conversation = db.add_conversation({"title": "hello"})
    db.add_message({"conversation_id": conversation, "sender": "user", "content": "hello world"})
    media.add_media_with_keywords(
        title="hello", media_type="document", content="hello world", keywords=["k"]
    )
    db.close_connection()  # setup's own main-thread handle is not under test
    media.close_connection()
    app = SimpleNamespace(chachanotes_db=db, media_db=media)
    try:
        if leg == "search_media_fts5":
            results = await legs.search_media_fts5(app, "hello", keyword_filter=["k"])
        else:
            results = await getattr(legs, leg)(app, "hello")

        assert results, f"{leg} found nothing; the read under test did not run"
        leaked = [str(lease.resource_path) for lease in worker_leases(db) + worker_leases(media)]
        assert leaked == [], f"{leg} kept its worker connection"
        assert _local_pause_drains(), "a backup's local pause could not drain"
    finally:
        db.close_connection()
        media.close_connection()


def _bare_offloads(tree, allowed):
    """Executor offloads whose callable is not a self-retiring wrapper."""
    import ast

    offloads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"to_thread", "run_in_executor"}
    ]
    bare = [
        (node.lineno, ast.unparse(node.args[0]) if node.args else "")
        for node in offloads
        if not (
            node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in {"run_finite_local_worker", *allowed}
        )
    ]
    return offloads, bare


@pytest.mark.parametrize(
    "module,functions,allowed,minimum",
    [
        ("tldw_chatbook.Chat.scope_picker_listers", None, (), 5),
        # rerank_results loads and runs a model; it touches no store.
        ("tldw_chatbook.RAG_Search.pipeline_functions_simple", None, ("rerank_results",), 7),
        (
            "tldw_chatbook.UI.Screens.chat_screen",
            (
                "_console_dictionary_attach_worker",
                "_console_dictionary_detach_worker",
                "_console_worldbook_attach_worker",
                "_console_worldbook_detach_worker",
            ),
            (),
            6,
        ),
    ],
)
def test_console_store_offloads_retire_their_worker_handles(
    module, functions, allowed, minimum
):
    """Structural guard for R2-1: Console's picker and RAG store offloads retire.

    TASK-34100.16.1 review v5r2 widened the notes-only guard above to the
    Console surfaces the review found: the retrieval-scope picker, the RAG FTS
    legs it scopes, and the dictionary and world-book rail pickers.
    """
    import ast
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(module))
    tree = ast.parse(source)
    if functions is not None:
        found = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in functions
        }
        assert set(found) == set(functions)
        tree = ast.Module(body=list(found.values()), type_ignores=[])
    offloads, bare = _bare_offloads(tree, allowed)

    assert len(offloads) >= minimum
    assert bare == []
