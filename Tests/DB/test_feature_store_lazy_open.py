"""Per-store pins for open-on-first-use feature databases (TASK-21105).

Six feature stores used to create their SQLite file and run their full
schema DDL inside their constructors, which ``TldwCli.__init__`` invokes
for every boot -- including boots that never touch the feature. Each store
now defers file creation + schema to its first real operation.

Every store gets the same three pins:

1. construction creates NO database file (this is the red-first pin --
   it fails against the eager constructors);
2. the first feature operation creates the file and works end to end;
3. ``close()`` before any use is safe (shutdown must never construct a
   store just to close it).

``:memory:`` instances deliberately keep the old eager initialization
(no disk cost, and their single cached connection must stay bound to the
constructing thread); that contract is pinned by
``Tests/DB/test_private_sqlite_interop_owners.py`` and is not repeated
here.
"""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Kanban_Interop.local_kanban_service import LocalKanbanService
from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Notifications.event_state_repository import EventStateRepository
from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService


def _assert_no_store_files(db_path: Path) -> None:
    """Assert neither the database file nor its WAL sidecars exist."""
    for candidate in (
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ):
        assert not candidate.exists(), f"{candidate.name} created before first use"


class TestLocalWritingServiceLazyOpen:
    def test_construction_creates_no_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "writing.db"
        LocalWritingService(db_path)
        _assert_no_store_files(db_path)

    def test_first_use_creates_file_and_works(self, tmp_path: Path) -> None:
        db_path = tmp_path / "writing.db"
        service = LocalWritingService(db_path)
        project = service.create_project(title="First novel")
        assert db_path.exists()
        assert [row["id"] for row in service.list_projects()] == [project["id"]]
        service.close()

    def test_close_before_use_is_safe(self, tmp_path: Path) -> None:
        db_path = tmp_path / "writing.db"
        LocalWritingService(db_path).close()
        _assert_no_store_files(db_path)


class TestLocalResearchServiceLazyOpen:
    def test_construction_creates_no_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "research.db"
        LocalResearchService(db_path)
        _assert_no_store_files(db_path)

    def test_first_use_creates_file_and_works(self, tmp_path: Path) -> None:
        db_path = tmp_path / "research.db"
        service = LocalResearchService(db_path)
        session = service.create_session(title="Session", query="q")
        assert db_path.exists()
        assert service.get_session(session["id"])["title"] == "Session"
        # Migrations applied on first use, not construction.
        with service._connect() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] >= 1
        service.close()

    def test_close_before_use_is_safe(self, tmp_path: Path) -> None:
        db_path = tmp_path / "research.db"
        LocalResearchService(db_path).close()
        _assert_no_store_files(db_path)


class TestLocalKanbanServiceLazyOpen:
    def test_construction_creates_no_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "kanban.db"
        LocalKanbanService(db_path=db_path)
        _assert_no_store_files(db_path)

    def test_first_use_creates_file_and_works(self, tmp_path: Path) -> None:
        db_path = tmp_path / "kanban.db"
        service = LocalKanbanService(db_path=db_path)
        status = service.get_storage_status()
        assert db_path.exists()
        assert status["schema_version"] >= 1  # schema meta readable


class TestClientNotificationsDBLazyOpen:
    def test_construction_creates_no_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "notifications.db"
        ClientNotificationsDB(db_path)
        _assert_no_store_files(db_path)

    def test_first_use_creates_file_and_works(self, tmp_path: Path) -> None:
        db_path = tmp_path / "notifications.db"
        store = ClientNotificationsDB(db_path)
        row = store.insert_notification(
            category="watchlist", title="hello", message="body"
        )
        assert db_path.exists()
        assert store.get_notification(row["id"])["title"] == "hello"
        assert len(store.list_notifications()) == 1
        store.close()

    def test_close_before_use_is_safe(self, tmp_path: Path) -> None:
        db_path = tmp_path / "notifications.db"
        ClientNotificationsDB(db_path).close()
        _assert_no_store_files(db_path)


class TestEventStateRepositoryLazyOpen:
    def test_construction_creates_no_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "event_state.db"
        EventStateRepository(db_path)
        _assert_no_store_files(db_path)

    def test_first_use_creates_file_and_works(self, tmp_path: Path) -> None:
        db_path = tmp_path / "event_state.db"
        repository = EventStateRepository(db_path)
        cursor = repository.get_cursor(
            source_authority="server",
            server_profile_id="profile",
            stream_name="notifications",
            stream_instance_id="instance",
        )
        assert db_path.exists()
        assert cursor.cursor is None
        repository.close()

    def test_close_before_use_is_safe(self, tmp_path: Path) -> None:
        db_path = tmp_path / "event_state.db"
        EventStateRepository(db_path).close()
        _assert_no_store_files(db_path)


class TestSyncStateRepositoryLazyOpen:
    def test_construction_creates_no_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sync_state.db"
        SyncStateRepository(db_path)
        _assert_no_store_files(db_path)

    def test_first_use_creates_file_and_works(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sync_state.db"
        repository = SyncStateRepository(db_path)
        assert repository.list_identity_mappings() == []
        assert db_path.exists()
        assert repository.is_durable is True
        repository.close()

    def test_close_before_use_is_safe(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sync_state.db"
        SyncStateRepository(db_path).close()
        _assert_no_store_files(db_path)


class TestLazyOpenSingleFlight:
    def test_concurrent_first_use_initializes_schema_once(
        self, tmp_path: Path
    ) -> None:
        """Racing first operations must not double-run the executescript."""
        import threading

        db_path = tmp_path / "notifications.db"
        store = ClientNotificationsDB(db_path)
        errors: list[BaseException] = []
        barrier = threading.Barrier(4)

        def first_use() -> None:
            try:
                barrier.wait(timeout=10)
                store.insert_notification(
                    category="watchlist", title="t", message="m"
                )
            except BaseException as exc:  # noqa: BLE001 - collected for assert
                errors.append(exc)

        threads = [threading.Thread(target=first_use) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert errors == []
        assert len(store.list_notifications()) == 4
        store.close()
