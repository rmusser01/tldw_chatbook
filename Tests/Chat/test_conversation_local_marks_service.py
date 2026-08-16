from __future__ import annotations

import pytest

from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version


def _db(tmp_path):
    return CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="test-client")


def _assert_local_marks_schema(db):
    conn = db.get_connection()
    columns = conn.execute("PRAGMA table_info(conversation_local_marks)").fetchall()
    assert [
        (row["name"], row["type"], row["notnull"], row["pk"]) for row in columns
    ] == [
        ("conversation_id", "TEXT", 1, 1),
        ("mark_type", "TEXT", 1, 2),
        ("created_at", "TEXT", 1, 0),
        ("updated_at", "TEXT", 1, 0),
    ]

    indexes = conn.execute("PRAGMA index_list(conversation_local_marks)").fetchall()
    index_names = {row["name"] for row in indexes}
    assert "idx_conversation_local_marks_type" in index_names

    index_columns = conn.execute(
        "PRAGMA index_info(idx_conversation_local_marks_type)"
    ).fetchall()
    assert [row["name"] for row in index_columns] == [
        "mark_type",
        "updated_at",
        "conversation_id",
    ]
    index_xinfo_columns = conn.execute(
        "PRAGMA index_xinfo(idx_conversation_local_marks_type)"
    ).fetchall()
    indexed_columns = [row for row in index_xinfo_columns if row["key"]]
    assert [(row["name"], row["desc"]) for row in indexed_columns] == [
        ("mark_type", 0),
        ("updated_at", 1),
        ("conversation_id", 0),
    ]


def test_local_marks_table_exists_on_fresh_schema_with_expected_shape(tmp_path):
    db = _db(tmp_path)
    conn = db.get_connection()

    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("conversation_local_marks",),
    ).fetchone()

    assert row is not None
    _assert_local_marks_schema(db)


def test_local_marks_migrate_from_v16_to_v17_with_expected_schema(tmp_path):
    db_path = tmp_path / "chacha.sqlite"
    # Build a genuinely V16-shaped DB: the production migration chain itself,
    # run under a patched _CURRENT_SCHEMA_VERSION, stops and stamps at 16
    # (task-16840; replaces the retired shared rollback registry). One caveat:
    # the v4 base schema ships the marks table baked in (the V16->V17
    # migration replays IF NOT EXISTS over it on the fresh path), so drop the
    # migration-under-test's own two artifacts to prove the V16->V17 step
    # genuinely creates them on replay. That is knowledge about the single
    # migration this test pins — no future schema bump can invalidate it.
    with chachanotes_db_at_version(db_path, 16, client_id="test-client") as db:
        conn = db.get_connection()
        conn.execute("DROP INDEX IF EXISTS idx_conversation_local_marks_type")
        conn.execute("DROP TABLE IF EXISTS conversation_local_marks")
        conn.commit()

        # Guard the replay preconditions before reopening: the marks table
        # the V16->V17 migration must create is absent, and — genuine-shape
        # facts a rolled-back fixture could never assert — so is every
        # later migration's table (note_folders was the artifact that broke
        # the registry-era fixture in task-16197).
        table_names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "conversation_local_marks" not in table_names
        assert "note_folders" not in table_names
        assert "note_folder_memberships" not in table_names
        version_before = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (db._SCHEMA_NAME,),
        ).fetchone()
        assert version_before["version"] == 16

    migrated = CharactersRAGDB(str(db_path), client_id="test-client")

    version = (
        migrated.get_connection()
        .execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (migrated._SCHEMA_NAME,),
        )
        .fetchone()
    )
    assert version["version"] == migrated._CURRENT_SCHEMA_VERSION
    _assert_local_marks_schema(migrated)


def test_star_unstar_is_idempotent_and_ordered(tmp_path):
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)

    service.star_conversation("conv-a")
    service.star_conversation("conv-b")
    service.star_conversation("conv-a")

    assert service.is_starred("conv-a") is True
    assert service.is_starred("conv-b") is True
    assert service.list_marked_conversation_ids() == ("conv-a", "conv-b")

    service.unstar_conversation("conv-a")
    service.unstar_conversation("conv-a")

    assert service.is_starred("conv-a") is False
    assert service.list_marked_conversation_ids() == ("conv-b",)


def test_local_marks_tolerate_missing_conversations(tmp_path):
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)

    service.star_conversation("missing-conversation")

    assert service.is_starred("missing-conversation") is True
    assert service.list_marked_conversation_ids() == ("missing-conversation",)


@pytest.mark.parametrize("mark_type", ["", "   ", "archived"])
def test_local_marks_reject_blank_and_unsupported_mark_types(tmp_path, mark_type):
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)

    with pytest.raises(ValueError, match="Unsupported conversation mark_type"):
        service.set_mark("conv-a", mark_type)


@pytest.mark.parametrize("conversation_id", ["", "   ", None])
def test_local_marks_reject_blank_conversation_ids(tmp_path, conversation_id):
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)

    with pytest.raises(ValueError, match="conversation_id is required"):
        service.star_conversation(conversation_id)


@pytest.mark.parametrize("limit", [0, -1])
def test_list_marked_conversation_ids_rejects_non_positive_limits(tmp_path, limit):
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)

    with pytest.raises(ValueError, match="limit must be positive"):
        service.list_marked_conversation_ids(limit=limit)


def test_local_marks_do_not_create_sync_log_entries(tmp_path):
    db = _db(tmp_path)
    conversations = ChatConversationService(db)
    conversation_id = conversations.create_conversation(title="Sync Boundary")
    db.get_connection().execute("DELETE FROM sync_log")
    db.get_connection().commit()

    ConversationLocalMarksService(db).star_conversation(conversation_id)

    rows = (
        db.get_connection()
        .execute("SELECT entity, entity_id, operation, payload FROM sync_log")
        .fetchall()
    )
    assert rows == []


def test_conversation_metadata_does_not_include_local_marks(tmp_path):
    db = _db(tmp_path)
    conversations = ChatConversationService(db)
    marks = ConversationLocalMarksService(db)
    conversation_id = conversations.create_conversation(title="Plain Metadata")

    marks.star_conversation(conversation_id)
    metadata = conversations.get_conversation_metadata(conversation_id)

    assert metadata is not None
    assert "starred" not in metadata
    assert "marks" not in metadata
    assert "local_marks" not in metadata


# ---------------------------------------------------------------------------
# PR 3a-2 Task 4: the FLEET_UNSEEN mark type (background sub-agent
# completion the user has not seen).
# ---------------------------------------------------------------------------


def test_fleet_unseen_mark_type_is_allowed_and_independent_of_starring(tmp_path):
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)

    service.set_mark("conv-a", ConversationLocalMarksService.FLEET_UNSEEN)
    assert service.has_mark("conv-a", ConversationLocalMarksService.FLEET_UNSEEN)
    # The two mark types never bleed into each other.
    assert not service.is_starred("conv-a")
    assert service.list_marked_conversation_ids(
        ConversationLocalMarksService.FLEET_UNSEEN
    ) == ("conv-a",)
    assert service.list_marked_conversation_ids() == ()

    service.clear_mark("conv-a", ConversationLocalMarksService.FLEET_UNSEEN)
    assert not service.has_mark("conv-a", ConversationLocalMarksService.FLEET_UNSEEN)


def test_fleet_unseen_mark_survives_into_a_fresh_service_handle(tmp_path):
    """Restart-proof by construction: the mark written through one service
    handle is read back through a FRESH service over a FRESH DB handle on
    the same file -- the exact shape of an app restart (PR3a-2 Task 4)."""
    path = str(tmp_path / "chacha.sqlite")
    db = CharactersRAGDB(path, client_id="writer")
    ConversationLocalMarksService(db).set_mark(
        "conv-restart", ConversationLocalMarksService.FLEET_UNSEEN
    )
    db.close_connection()

    fresh_db = CharactersRAGDB(path, client_id="reader")
    fresh = ConversationLocalMarksService(fresh_db)
    assert fresh.has_mark("conv-restart", ConversationLocalMarksService.FLEET_UNSEEN)
    assert fresh.list_marked_conversation_ids(
        ConversationLocalMarksService.FLEET_UNSEEN
    ) == ("conv-restart",)
    fresh_db.close_connection()


def test_get_mark_returns_timestamps_with_created_at_stable_across_refreshes(
    tmp_path,
):
    """PR3a-2 Task 5: the auto-wake mount-claim reads ``created_at`` as the
    since-when boundary for undelivered completions, so a re-delivered
    drain's refresh (``set_mark`` on an existing row) must bump ONLY
    ``updated_at`` -- a moving ``created_at`` would silently exclude the
    first drain's own runs from the claim."""
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)
    assert (
        service.get_mark("conv-a", ConversationLocalMarksService.FLEET_UNSEEN) is None
    )

    service.set_mark("conv-a", ConversationLocalMarksService.FLEET_UNSEEN)
    first = service.get_mark("conv-a", ConversationLocalMarksService.FLEET_UNSEEN)
    assert first is not None
    assert first.conversation_id == "conv-a"
    assert first.mark_type == ConversationLocalMarksService.FLEET_UNSEEN
    assert first.created_at and first.updated_at

    service.set_mark("conv-a", ConversationLocalMarksService.FLEET_UNSEEN)
    refreshed = service.get_mark("conv-a", ConversationLocalMarksService.FLEET_UNSEEN)
    assert refreshed is not None
    assert refreshed.created_at == first.created_at
    assert refreshed.updated_at >= first.updated_at

    with pytest.raises(ValueError):
        service.get_mark("", ConversationLocalMarksService.FLEET_UNSEEN)
    with pytest.raises(ValueError):
        service.get_mark("conv-a", "not-a-mark")


def test_list_marked_ids_cache_never_serves_stale_results_across_writes(tmp_path):
    """`list_marked_conversation_ids` may cache, but writes must invalidate.

    task-15471 made the list read cached (Console's browser refresh calls it
    on the event loop from every repaint path). The cache is only sound if
    every writer path -- star AND unstar, and the independent fleet mark
    type -- drops it, so a toggle is never followed by a stale repaint.
    """
    service = ConversationLocalMarksService(_db(tmp_path))

    assert service.list_marked_conversation_ids() == ()

    service.star_conversation("conv-a")
    assert service.list_marked_conversation_ids() == ("conv-a",)

    # Repeat read (cache hit) must equal the first.
    assert service.list_marked_conversation_ids() == ("conv-a",)

    service.star_conversation("conv-b")
    assert set(service.list_marked_conversation_ids()) == {"conv-a", "conv-b"}

    service.unstar_conversation("conv-a")
    assert service.list_marked_conversation_ids() == ("conv-b",)

    # Mark types are cached independently: writing a fleet mark must not
    # bleed into the starred list, and its own list must see the write.
    service.set_mark("conv-c", ConversationLocalMarksService.FLEET_UNSEEN)
    assert service.list_marked_conversation_ids() == ("conv-b",)
    assert service.list_marked_conversation_ids(
        ConversationLocalMarksService.FLEET_UNSEEN
    ) == ("conv-c",)
    service.clear_mark("conv-c", ConversationLocalMarksService.FLEET_UNSEEN)
    assert (
        service.list_marked_conversation_ids(ConversationLocalMarksService.FLEET_UNSEEN)
        == ()
    )

    # Different limits are distinct cache keys and must both reflect writes.
    assert service.list_marked_conversation_ids(limit=1) == ("conv-b",)
    service.star_conversation("conv-d")
    assert service.list_marked_conversation_ids(limit=1) == ("conv-d",)


def test_list_cache_is_not_repopulated_with_a_pre_write_snapshot(tmp_path):
    """A cache-missing reader must not store rows a concurrent write outdated.

    task-15471 fix round (review M1): the reader holds its fetched rows
    across the transaction COMMIT — a GIL-releasing sqlite call — before
    storing them. A writer that commits AND invalidates inside that window
    must win: without a generation check the reader re-populates the cache
    with its pre-write snapshot, and the just-starred conversation shows
    unstarred (or a FLEET_UNSEEN badge never appears) until the NEXT mark
    write. Both halves are genuinely cross-thread in production: the star
    toggle writes from a pool thread (workspace.py), the fleet drain from a
    child thread (console_fleet_attention.py), while repaint reads run on
    the loop.

    Deterministic interleave (adapted from the review's probe): the reader
    thread is paused exactly at its transaction exit — after fetch, before
    store — while the writer stars a conversation and invalidates.
    """
    import threading

    db = _db(tmp_path)
    read_committed = threading.Event()
    write_invalidated = threading.Event()
    state: dict = {"reader_ident": None}

    class _HookedTx:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *exc):
            result = self._inner.__exit__(*exc)
            # Pause ONLY the reader, and only right after its COMMIT — the
            # real scheduler-switch point between fetchall() and the store.
            if threading.get_ident() == state["reader_ident"]:
                read_committed.set()
                write_invalidated.wait(5)
            return result

    class _DbProxy:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def transaction(self):
            return _HookedTx(self._inner.transaction())

    service = ConversationLocalMarksService(_DbProxy(db))
    reader_seen: dict = {}

    def _reader():
        state["reader_ident"] = threading.get_ident()
        reader_seen["value"] = service.list_marked_conversation_ids()

    def _writer():
        assert read_committed.wait(5), "reader never reached its commit"
        service.star_conversation("conv-a")  # commits, then invalidates
        write_invalidated.set()

    reader = threading.Thread(target=_reader)
    writer = threading.Thread(target=_writer)
    reader.start()
    writer.start()
    reader.join(10)
    writer.join(10)
    assert not reader.is_alive() and not writer.is_alive()

    # The reader legitimately saw the pre-write world...
    assert reader_seen["value"] == ()
    # ...but the SERVICE must now answer with post-write truth, not the
    # reader's stale snapshot resurrected into the cache.
    assert service.list_marked_conversation_ids() == ("conv-a",)
