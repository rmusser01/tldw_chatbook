"""Storage-side archive eligibility tests."""

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def test_storage_filter_precedes_page_limit(tmp_path):
    import time

    from tldw_chatbook.DB.conversation_archive_queries import (
        query_archive_recovery_page,
    )

    db = CharactersRAGDB(tmp_path / "query.sqlite", client_id="recovery")
    try:
        for i in range(25):
            db.add_conversation(
                {
                    "title": f"hidden {i}",
                    "scope_type": "workspace",
                    "workspace_id": "archived",
                }
            )
        page = query_archive_recovery_page(
            db,
            {"query": "", "limit": 20},
            (
                {
                    "workspace_id": "archived",
                    "name": "Archived",
                    "archived": True,
                    "archived_at": None,
                    "archive_revision": 1,
                },
            ),
            after=None,
            deadline=time.monotonic() + 5,
            monotonic=time.monotonic,
        )
        assert len(page["items"]) == 20
        assert page["next_anchor"] is not None
    finally:
        db.close_connection()


def test_sql_progress_deadline_interrupts_and_clears_handler(tmp_path):
    import time

    import pytest

    from tldw_chatbook.DB.conversation_archive_queries import (
        query_archive_recovery_page,
    )

    db = CharactersRAGDB(tmp_path / "interrupt.sqlite", client_id="recovery")
    try:
        cid = db.add_conversation({"title": "needle"})
        db.set_conversations_archived([cid], archived=True, expected_versions={cid: 1})
        for i in range(100):
            db.add_message(
                {
                    "conversation_id": cid,
                    "sender": "user",
                    "content": "x" * 1000 + str(i),
                }
            )
        calls = 0

        def clock():
            nonlocal calls
            calls += 1
            return 0.0 if calls < 15 else 6.0

        conn = db.get_connection()
        old = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        with pytest.raises(TimeoutError):
            query_archive_recovery_page(
                db,
                {"query": "absent", "limit": 20},
                (),
                after=None,
                deadline=5,
                monotonic=clock,
            )
        assert calls >= 15
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 100
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == old
        assert query_archive_recovery_page(
            db,
            {"query": "needle", "limit": 10},
            (),
            after=None,
            deadline=time.monotonic() + 5,
            monotonic=time.monotonic,
        )["items"]
    finally:
        db.close_connection()


def test_query_plan_and_bounded_projection_on_representative_rows(tmp_path, capsys):
    import time

    from tldw_chatbook.DB.conversation_archive_queries import (
        normalized_archive_time,
        query_archive_recovery_page,
    )

    db = CharactersRAGDB(tmp_path / "plan.sqlite", client_id="recovery")
    try:
        for i in range(128):
            cid = db.add_conversation(
                {
                    "title": f"Conversation {i}",
                    "scope_type": "workspace",
                    "workspace_id": "archived",
                }
            )
            db.add_message(
                {
                    "conversation_id": cid,
                    "sender": "user",
                    "content": "needle " + "z" * 10000,
                }
            )
        conn = db.get_connection()
        statements = []
        conn.set_trace_callback(statements.append)
        started = time.monotonic()
        try:
            result = query_archive_recovery_page(
                db,
                {"query": "needle", "limit": 10},
                ({"workspace_id": "archived", "archived": True, "archived_at": None},),
                after=None,
                deadline=started + 5,
                monotonic=time.monotonic,
            )
        finally:
            conn.set_trace_callback(None)
        elapsed = time.monotonic() - started
        assert len(result["items"]) == 10
        assert result["next_anchor"] is not None
        assert all(len(i["excerpt"]) <= 320 for i in result["items"])
        assert elapsed < 5
        conn.create_function(
            "archive_recovery_utc", 1, normalized_archive_time, deterministic=True
        )
        page_sql = next(sql for sql in statements if "SELECT * FROM ordered" in sql)
        plan = [r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + page_sql)]
        conn.create_function("archive_recovery_utc", 1, None)
        assert any("INDEX" in detail and "messages" in detail for detail in plan), plan
        assert any("json_each" in detail for detail in plan), plan
        print(
            f"Recovery page: 128 chats, 1.28M message chars, 10 projected rows, {elapsed:.4f}s; plan={plan}"
        )
    finally:
        db.close_connection()


def test_lock_wait_uses_remaining_budget_and_restores_timeout(tmp_path):
    import sqlite3
    import time

    import pytest

    from tldw_chatbook.DB.conversation_archive_queries import (
        query_archive_recovery_page,
    )

    path = tmp_path / "locked.sqlite"
    db = CharactersRAGDB(path, client_id="recovery")
    try:
        conn = db.get_connection()
        conn.execute("PRAGMA journal_mode=DELETE")
        old = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        blocker = sqlite3.connect(path)
        blocker.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError):
                query_archive_recovery_page(
                    db,
                    {"query": "", "limit": 10},
                    (),
                    after=None,
                    deadline=started + 0.15,
                    monotonic=time.monotonic,
                )
        finally:
            blocker.rollback()
            blocker.close()
        elapsed = time.monotonic() - started
        assert 0.10 <= elapsed < 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == old
    finally:
        db.close_connection()


def test_cold_locked_chat_connection_respects_budget_and_can_reuse(
    tmp_path, monkeypatch
):
    import sqlite3
    import threading
    import time

    import pytest

    import tldw_chatbook.DB.ChaChaNotes_DB as module
    from tldw_chatbook.DB.conversation_archive_queries import (
        query_archive_recovery_page,
    )

    path = tmp_path / "cold-chat.sqlite"
    db = CharactersRAGDB(path, client_id="recovery")
    conn = db.get_connection()
    conn.execute("PRAGMA journal_mode=DELETE")
    db.close_connection()
    opened = []
    real_connect = module.connect_private_sqlite

    def recording_connect(*args, **kwargs):
        result = real_connect(*args, **kwargs)
        opened.append(result)
        return result

    monkeypatch.setattr(module, "connect_private_sqlite", recording_connect)
    blocker = sqlite3.connect(path, check_same_thread=False)
    blocker.execute("BEGIN EXCLUSIVE")
    release = threading.Timer(1.0, blocker.rollback)
    release.start()
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            query_archive_recovery_page(
                db,
                {"query": "", "limit": 10},
                (),
                after=None,
                deadline=started + 0.15,
                monotonic=time.monotonic,
            )
        elapsed = time.monotonic() - started
        print(
            f"Cold chat acquisition: {elapsed:.4f}s; {len(opened)} opened native handle(s), all closed on failure"
        )
        assert elapsed < 0.75, (
            f"cold chat acquisition spent {elapsed:.3f}s outside its 0.15s budget"
        )
        assert db.registered_connection_count() == 0
        for failed in opened:
            with pytest.raises(sqlite3.ProgrammingError):
                failed.execute("SELECT 1")
        release.join()
        reused = db.get_connection()
        assert reused.execute("SELECT 1").fetchone()[0] == 1
        assert db.registered_connection_count() == 1
        assert reused.execute("PRAGMA busy_timeout").fetchone()[0] == 15000
        assert (
            query_archive_recovery_page(
                db,
                {"query": "", "limit": 10},
                (),
                after=None,
                deadline=time.monotonic() + 5,
                monotonic=time.monotonic,
            )["items"]
            == []
        )
    finally:
        release.join()
        blocker.close()
        db.close_connection()


def test_cold_locked_workspace_connection_respects_budget_and_can_reuse(
    tmp_path, monkeypatch
):
    import sqlite3
    import threading
    import time

    import pytest

    import tldw_chatbook.DB.base_db as module
    from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveRecoveryError
    from tldw_chatbook.Chat.archive_recovery.search_service import ArchiveSearchService
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    workspace_path = tmp_path / "cold-workspace.sqlite"
    workspace_db = WorkspaceDB(workspace_path)
    chat_db = CharactersRAGDB(tmp_path / "chat.sqlite", client_id="recovery")
    with workspace_db.connection() as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
    workspace_db.close()
    service = ArchiveSearchService(
        ChatConversationService(chat_db), LocalWorkspaceRegistryService(workspace_db)
    )
    opened = []
    real_connect = module.connect_private_sqlite

    def recording_connect(*args, **kwargs):
        result = real_connect(*args, **kwargs)
        opened.append(result)
        return result

    monkeypatch.setattr(module, "connect_private_sqlite", recording_connect)
    blocker = sqlite3.connect(workspace_path, check_same_thread=False)
    blocker.execute("BEGIN EXCLUSIVE")
    release = threading.Timer(1.0, blocker.rollback)
    release.start()
    try:
        started = time.monotonic()
        with pytest.raises(ArchiveRecoveryError) as error:
            service._workspace_snapshot(started + 0.15)
        elapsed = time.monotonic() - started
        assert error.value.code == "search_incomplete"
        print(
            f"Cold workspace acquisition: {elapsed:.4f}s; {len(opened)} opened native handle(s), all closed on failure"
        )
        assert elapsed < 0.75, (
            f"cold workspace acquisition spent {elapsed:.3f}s outside its 0.15s budget"
        )
        assert getattr(workspace_db._thread_local, "conn", None) is None
        for failed in opened:
            with pytest.raises(sqlite3.ProgrammingError):
                failed.execute("SELECT 1")
        release.join()
        with workspace_db.connection() as reused:
            assert reused.execute("SELECT 1").fetchone()[0] == 1
            assert reused.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert service._workspace_snapshot(time.monotonic() + 5) == ()
    finally:
        release.join()
        blocker.close()
        workspace_db.close()
        chat_db.close_connection()


def test_deadline_aware_cold_chat_success_preserves_private_guards_and_quiescence(
    tmp_path,
):
    import time

    import pytest

    db = CharactersRAGDB(tmp_path / "cold-success.sqlite", client_id="recovery")
    try:
        db.close_connection()
        conn = db.get_connection(operation_deadline=time.monotonic() + 5)
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 15000
        assert conn.isolation_level is None
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert (
            conn.execute(
                "SELECT console_semantic_mutation_authorized('unowned', 'update')"
            ).fetchone()[0]
            == 0
        )
        assert db.get_connection() is conn
        assert db.registered_connection_count() == 1
        with (
            db.quiesce_connections(timeout_seconds=1),
            pytest.raises(Exception, match="maintenance"),
        ):
            db.get_connection(operation_deadline=time.monotonic() + 1)
        assert (
            db.get_connection(operation_deadline=time.monotonic() + 5)
            .execute("SELECT 1")
            .fetchone()[0]
            == 1
        )
    finally:
        db.close_connection()
