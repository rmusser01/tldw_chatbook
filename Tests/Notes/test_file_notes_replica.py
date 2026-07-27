from __future__ import annotations

import hashlib
import sqlite3
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

# Avoid importing the unrelated optional MLX stack during focused Notes tests.
sys.modules.setdefault("parakeet_mlx", types.ModuleType("parakeet_mlx"))

from tldw_chatbook.DB.private_sqlite import connect_private_sqlite  # noqa: E402
import tldw_chatbook.Notes.file_notes_replica as file_notes_replica  # noqa: E402
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica  # noqa: E402
from tldw_chatbook.Notes.file_notes_replica import ReplicaFileInfo  # noqa: E402


def _digest(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def _upsert(
    replica: FileNotesReplica,
    root: str,
    relative_path: str,
    raw_bytes: bytes,
    *,
    decoded_text: str | None = None,
    mtime_ns: int = 1,
) -> None:
    if decoded_text is None:
        decoded_text = raw_bytes.decode("utf-8")
    replica.upsert_file(
        root,
        relative_path,
        raw_bytes,
        content_hash=_digest(raw_bytes),
        decoded_text=decoded_text,
        size=len(raw_bytes),
        mtime_ns=mtime_ns,
    )


@pytest.fixture
def replica() -> FileNotesReplica:
    value = FileNotesReplica(":memory:")
    yield value
    value.close()


def test_connections_use_registered_private_sqlite_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    def recording_connect(
        owner_id: str,
        database: str | Path,
        **kwargs: object,
    ) -> sqlite3.Connection:
        calls.append((owner_id, str(database), kwargs))
        return connect_private_sqlite(owner_id, database, **kwargs)

    monkeypatch.setattr(
        file_notes_replica,
        "connect_private_sqlite",
        recording_connect,
    )
    file_path = tmp_path / "file_notes.sqlite"
    memory_replica = FileNotesReplica(":memory:")
    file_replica = FileNotesReplica(file_path)
    memory_replica.close()
    file_replica.close()

    assert calls == [
        (
            "notes.file_notes_replica",
            ":memory:",
            {"isolation_level": None, "check_same_thread": False},
        ),
        (
            "notes.file_notes_replica",
            str(file_path),
            {"isolation_level": None, "check_same_thread": False},
        ),
    ]


def test_connection_operations_are_serialized_across_worker_threads(
    replica: FileNotesReplica,
) -> None:
    root = "/notes"
    _upsert(replica, root, "worker.md", b"thread-safe bytes")
    transaction_started = Event()
    release_transaction = Event()
    read_started = Event()
    read_finished = Event()

    def hold_transaction() -> None:
        with replica._transaction():
            transaction_started.set()
            assert release_transaction.wait(timeout=2)

    def read_from_worker() -> bytes | None:
        read_started.set()
        try:
            return replica.get_bytes(root, "worker.md")
        finally:
            read_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        holder = executor.submit(hold_transaction)
        assert transaction_started.wait(timeout=1)
        reader = executor.submit(read_from_worker)
        assert read_started.wait(timeout=1)
        try:
            assert not read_finished.wait(timeout=0.05)
        finally:
            release_transaction.set()
        holder.result(timeout=1)
        assert reader.result(timeout=1) == b"thread-safe bytes"


def test_commit_failure_rolls_back_and_leaves_connection_usable(
    replica: FileNotesReplica,
) -> None:
    with replica._lock:
        replica._connection.execute("PRAGMA foreign_keys = ON")
        replica._connection.executescript(
            """
            CREATE TEMP TABLE commit_parent (
                id INTEGER PRIMARY KEY
            );
            CREATE TEMP TABLE commit_child (
                parent_id INTEGER,
                FOREIGN KEY(parent_id) REFERENCES commit_parent(id)
                    DEFERRABLE INITIALLY DEFERRED
            );
            """
        )

    insert_completed = False
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
        with replica._transaction() as cursor:
            cursor.execute(
                "INSERT INTO commit_child (parent_id) VALUES (?)",
                (1,),
            )
            insert_completed = True

    assert insert_completed
    with replica._lock:
        assert not replica._connection.in_transaction
        assert (
            replica._connection.execute(
                "SELECT COUNT(*) FROM commit_child"
            ).fetchone()[0]
            == 0
        )

    _upsert(replica, "/notes", "after-failure.md", b"still usable")
    assert replica.get_bytes("/notes", "after-failure.md") == b"still usable"


def test_schema_has_only_required_tables_and_no_triggers(
    replica: FileNotesReplica,
) -> None:
    rows = replica._connection.execute(
        """
        SELECT name, sql
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
          AND name NOT LIKE 'files_fts_%'
        """
    ).fetchall()

    assert {row["name"] for row in rows} == {
        "files",
        "revisions",
        "protected_paths",
        "files_fts",
    }
    assert next(row["sql"] for row in rows if row["name"] == "files_fts").startswith(
        "CREATE VIRTUAL TABLE"
    )
    assert (
        replica._connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger'"
        ).fetchone()[0]
        == 0
    )


def test_upsert_is_root_namespaced_and_manually_replaces_fts_content(
    replica: FileNotesReplica,
) -> None:
    root_a = "/notes/a"
    root_b = "/notes/b"
    _upsert(replica, root_a, "shared.md", b"alpha current")
    _upsert(replica, root_b, "shared.md", b"beta current")

    assert replica.get_bytes(root_a, "shared.md") == b"alpha current"
    assert replica.get_bytes(root_b, "shared.md") == b"beta current"
    assert replica.search(root_a, "alpha") == ["shared.md"]
    assert replica.search(root_b, "alpha") == []

    _upsert(
        replica,
        root_a,
        "shared.md",
        b"gamma replacement",
        mtime_ns=2,
    )

    assert replica.search(root_a, "alpha") == []
    assert replica.search(root_a, "gamma") == ["shared.md"]
    assert (
        replica._connection.execute(
            """
            SELECT COUNT(*)
            FROM files
            WHERE root = ? AND relative_path = ?
            """,
            (root_a, "shared.md"),
        ).fetchone()[0]
        == 1
    )


def test_active_inventory_is_root_scoped_and_excludes_tombstones(
    replica: FileNotesReplica,
) -> None:
    root_a = "/notes/a"
    root_b = "/notes/b"
    _upsert(replica, root_a, "b.md", b"second", mtime_ns=22)
    _upsert(replica, root_a, "a.md", b"first", mtime_ns=11)
    _upsert(replica, root_a, "gone.md", b"deleted", mtime_ns=33)
    _upsert(replica, root_b, "a.md", b"other root", mtime_ns=44)
    replica.mark_deleted(root_a, "gone.md")

    assert replica.list_active_files(root_a) == [
        ReplicaFileInfo(
            relative_path="a.md",
            content_hash=_digest(b"first"),
            size=5,
            mtime_ns=11,
        ),
        ReplicaFileInfo(
            relative_path="b.md",
            content_hash=_digest(b"second"),
            size=6,
            mtime_ns=22,
        ),
    ]
    assert replica.list_active_files(root_b) == [
        ReplicaFileInfo(
            relative_path="a.md",
            content_hash=_digest(b"other root"),
            size=10,
            mtime_ns=44,
        )
    ]


def test_search_ignores_undecodable_and_deleted_rows_and_quotes_user_input(
    replica: FileNotesReplica,
) -> None:
    root = "/notes"
    _upsert(replica, root, "active.md", b"literal OR searchable")
    replica.upsert_file(
        root,
        "unsafe.txt",
        b"\xff\xfe",
        content_hash=_digest(b"\xff\xfe"),
        decoded_text=None,
        size=2,
        mtime_ns=1,
    )
    _upsert(replica, root, "gone.md", b"searchable but deleted")
    assert replica.mark_deleted(root, "gone.md", deleted_at="2026-07-27T10:00:00Z")

    assert replica.search(root, "searchable") == ["active.md"]
    for unsafe_query in ('"', 'searchable OR "', "') OR 1=1 --", "\x00"):
        assert isinstance(replica.search(root, unsafe_query), list)
    assert replica.search(root, "searchable") == ["active.md"]


def test_mark_deleted_retains_bytes_and_clear_tombstone_restores_search(
    replica: FileNotesReplica,
) -> None:
    root = "/notes"
    raw_bytes = b"recover this exact payload\r\n"
    _upsert(replica, root, "folder/gone.md", raw_bytes)

    assert replica.mark_deleted(
        root,
        "folder/gone.md",
        deleted_at="2026-07-27T10:00:00Z",
    )
    assert replica.get_bytes(root, "folder/gone.md") == raw_bytes
    assert replica.get_restore_bytes(root, "folder/gone.md") == raw_bytes
    assert replica.list_deleted(root) == ["folder/gone.md"]
    assert replica.search(root, "recover") == []
    assert not replica.mark_deleted(root, "missing.md")

    assert replica.clear_tombstone(root, "folder/gone.md")
    assert replica.list_deleted(root) == []
    assert replica.search(root, "recover") == ["folder/gone.md"]
    assert not replica.clear_tombstone(root, "folder/gone.md")


def test_protection_matches_exact_files_and_component_bounded_prefixes(
    replica: FileNotesReplica,
) -> None:
    root = "/notes"
    replica.protect(root, "important.md")
    replica.protect(root, "archive", is_prefix=True)
    replica.protect(root, "team%", is_prefix=True)

    assert replica.is_protected(root, "important.md")
    assert not replica.is_protected(root, "folder/important.md")
    assert replica.is_protected(root, "archive")
    assert replica.is_protected(root, "archive/2026/note.md")
    assert not replica.is_protected(root, "archives/note.md")
    assert replica.is_protected(root, "team%/note.md")
    assert not replica.is_protected(root, "teamX/note.md")
    assert not replica.is_protected("/other", "important.md")

    assert replica.unprotect(root, "important.md")
    assert replica.unprotect(root, "archive", is_prefix=True)
    assert not replica.is_protected(root, "important.md")
    assert not replica.is_protected(root, "archive/2026/note.md")
    assert not replica.unprotect(root, "archive", is_prefix=True)


def test_checkpoint_coalesces_exact_bytes_once_per_session_key(
    replica: FileNotesReplica,
) -> None:
    root = "/notes"
    relative_path = "important.md"

    assert replica.checkpoint(
        root,
        relative_path,
        b"first exact bytes",
        content_hash=_digest(b"first exact bytes"),
        session_key="session-1",
        created_at="2026-07-27T10:00:00Z",
    )
    assert not replica.checkpoint(
        root,
        relative_path,
        b"must not replace first",
        content_hash=_digest(b"must not replace first"),
        session_key="session-1",
        created_at="2026-07-27T10:01:00Z",
    )
    assert replica.checkpoint(
        root,
        relative_path,
        b"second session bytes",
        content_hash=_digest(b"second session bytes"),
        session_key="session-2",
        created_at="2026-07-27T10:02:00Z",
    )

    rows = replica._connection.execute(
        """
        SELECT raw_bytes, session_key
        FROM revisions
        WHERE root = ? AND relative_path = ? AND kind = 'pre_edit'
        ORDER BY session_key
        """,
        (root, relative_path),
    ).fetchall()
    assert [(row["raw_bytes"], row["session_key"]) for row in rows] == [
        (b"first exact bytes", "session-1"),
        (b"second session bytes", "session-2"),
    ]


def test_prepare_deletion_rolls_back_snapshot_when_tombstone_write_fails(
    replica: FileNotesReplica,
) -> None:
    root = "/notes"
    relative_path = "keep.md"
    _upsert(replica, root, relative_path, b"still present")
    replica._connection.execute(
        """
        CREATE TEMP TRIGGER fail_tombstone
        BEFORE UPDATE OF deleted_at ON files
        WHEN NEW.deleted_at IS NOT NULL
        BEGIN
            SELECT RAISE(ABORT, 'forced tombstone failure');
        END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="forced tombstone failure"):
        replica.prepare_deletion(
            root,
            relative_path,
            b"deletion snapshot",
            content_hash=_digest(b"deletion snapshot"),
            decoded_text="deletion snapshot",
            deleted_at="2026-07-27T10:00:00Z",
        )

    assert replica.get_bytes(root, relative_path) == b"still present"
    assert replica.list_deleted(root) == []
    assert (
        replica._connection.execute(
            "SELECT COUNT(*) FROM revisions WHERE kind = 'delete'"
        ).fetchone()[0]
        == 0
    )
    assert replica.search(root, "present") == [relative_path]


def test_clear_tombstone_reindexes_the_prepared_snapshot_text(
    replica: FileNotesReplica,
) -> None:
    root = "/notes"
    _upsert(replica, root, "changed.md", b"old searchable text")
    _upsert(replica, root, "undecodable.txt", b"old indexed text")

    replica.prepare_deletion(
        root,
        "changed.md",
        b"replacement searchable text",
        content_hash=_digest(b"replacement searchable text"),
        decoded_text="replacement searchable text",
    )
    replica.prepare_deletion(
        root,
        "undecodable.txt",
        b"\xff\xfe",
        content_hash=_digest(b"\xff\xfe"),
        decoded_text=None,
    )

    assert replica.clear_tombstone(root, "changed.md")
    assert replica.get_bytes(root, "changed.md") == b"replacement searchable text"
    assert replica.search(root, "replacement") == ["changed.md"]
    assert replica.search(root, "old") == []

    assert replica.clear_tombstone(root, "undecodable.txt")
    assert replica.get_bytes(root, "undecodable.txt") == b"\xff\xfe"
    assert replica.search(root, "indexed") == []


def test_prepared_deletion_persists_and_returns_exact_restore_bytes(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "nested" / "file_notes.sqlite"
    root = "/notes"
    relative_path = "binary/data.text"
    initial_bytes = b"old replica"
    restore_bytes = b"\xef\xbb\xbf---\r\ntitle: Exact\r\n---\r\nbody\x00\r\n"

    first = FileNotesReplica(db_path)
    _upsert(first, root, relative_path, initial_bytes)
    first.prepare_deletion(
        root,
        relative_path,
        restore_bytes,
        content_hash=_digest(restore_bytes),
        decoded_text=restore_bytes.decode("utf-8"),
        deleted_at="2026-07-27T10:00:00Z",
        created_at="2026-07-27T10:00:00Z",
    )
    assert first.get_restore_bytes(root, relative_path) == restore_bytes
    deletion_revision = first._connection.execute(
        """
        SELECT raw_bytes, content_hash, session_key
        FROM revisions
        WHERE root = ? AND relative_path = ? AND kind = 'delete'
        """,
        (root, relative_path),
    ).fetchone()
    assert deletion_revision["raw_bytes"] == restore_bytes
    assert deletion_revision["content_hash"] == _digest(restore_bytes)
    assert deletion_revision["session_key"] is None
    first.close()

    assert db_path.parent.is_dir()
    second = FileNotesReplica(db_path)
    try:
        assert second.list_deleted(root) == [relative_path]
        assert second.list_deleted("/other") == []
        assert second.get_bytes(root, relative_path) == restore_bytes
        assert second.get_restore_bytes(root, relative_path) == restore_bytes
    finally:
        second.close()


def test_database_path_expands_user_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)
    expected_path = home / "state" / "file_notes.sqlite"

    replica = FileNotesReplica("~/state/file_notes.sqlite")
    try:
        database_path = replica._connection.execute(
            "SELECT file FROM pragma_database_list WHERE name = 'main'"
        ).fetchone()["file"]
    finally:
        replica.close()

    assert Path(database_path) == expected_path
    assert expected_path.is_file()
    assert not (tmp_path / "~").exists()
