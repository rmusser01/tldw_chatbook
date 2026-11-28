"""TASK-41310 PR A: a spawned child run's own trace segment (schema v87).

A child's provider calls belong to the parent's owner and turn, but its
request history is its own segment, linked to the owner by the child's run
identity. The owner guards resolve a linked segment to its owner; GC keeps it
while the owner is rooted and collects it after detachment.
"""

from __future__ import annotations

import sqlite3

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from Tests.Chat.test_console_trace_graph_gc import (
    _finish_legacy_migration,
    _root_graph,
    _terminal_call,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_trace_maintenance import TraceGarbageCollector
from tldw_chatbook.Chat.console_trace_repository import (
    ConsoleTraceRepository,
    TraceIdentityConflict,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

OWNER_GUARDS = (
    "console_trace_calls_owner_lineage",
    "console_trace_events_owner_guard",
    "console_trace_surface_nodes_owner_guard",
    "console_trace_surface_replacements_owner_guard",
    "console_trace_response_links_owner_guard",
)


@pytest.fixture
def db():
    database = CharactersRAGDB(":memory:", "trace-child-segments")
    yield database
    database.close_connection()


def _child(db, repository, owner_id, turn_id, run_id=None):
    with db.transaction(immediate=True) as cursor:
        return repository.create_child_segment(
            cursor,
            owner_id=owner_id,
            turn_id=turn_id,
            run_id=run_id or f"run-{turn_id}",
        )


def _rows(db, sql, *args):
    return [tuple(row) for row in db.get_connection().execute(sql, args).fetchall()]


def test_child_segment_is_owned_by_its_parent_owner_and_holds_its_own_calls(db):
    repository = ConsoleTraceRepository()
    _conversation, owner_id, root_id, policy_id, _node = _root_graph(db, repository)
    child = _child(db, repository, owner_id, "turn-1")

    with db.transaction(immediate=True) as cursor:
        owner = repository.get_effective_owner(cursor, child.segment_id)
        call_id = _terminal_call(
            cursor,
            repository,
            owner_id=owner_id,
            segment_id=child.segment_id,
            policy_id=policy_id,
            turn_id="turn-1",
            event_sequence=0,
        )
        found = repository.get_child_segment(cursor, child.run_id)

    assert owner is not None and owner.owner_id == owner_id
    assert found == child
    assert child.segment_id != root_id
    assert _rows(
        db,
        "SELECT owner_id, segment_id FROM console_trace_calls WHERE call_id = ?",
        call_id,
    ) == [(owner_id, child.segment_id)]
    # The parent's own surface is untouched: the child wrote only its segment.
    assert _rows(
        db, "SELECT COUNT(*) FROM console_trace_events WHERE segment_id = ?", root_id
    ) == [(1,)]


def test_unlinked_segment_and_foreign_owner_are_refused(db):
    repository = ConsoleTraceRepository()
    _conversation, owner_id, _root, policy_id, _node = _root_graph(db, repository)
    child = _child(db, repository, owner_id, "turn-1")
    other_id = db.add_conversation({"title": "other"})
    with db.transaction(immediate=True) as cursor:
        other_root = repository.create_segment(cursor)
        other = repository.attach_owner(
            cursor, conversation_id=other_id, root_segment_id=other_root.segment_id
        )
        bare = repository.create_segment(cursor)

    for segment_id, claimed in (
        (bare.segment_id, owner_id),
        (child.segment_id, other.owner_id),
    ):
        with (
            pytest.raises(TraceIdentityConflict, match="call_owner"),
            db.transaction(immediate=True) as cursor,
        ):
            repository.reserve_call(
                cursor,
                owner_id=claimed,
                segment_id=segment_id,
                turn_id="turn-1",
                run_id="foreign",
                call_sequence=0,
                idempotency_key="foreign-" + segment_id,
                policy_id=policy_id,
            )
        # The database trigger refuses it even past the repository's check.
        with pytest.raises(sqlite3.IntegrityError, match="active effective owner"):
            db.get_connection().execute(
                "INSERT INTO console_trace_calls(call_id, owner_id, segment_id, turn_id,"
                " run_id, call_sequence, idempotency_key, policy_id)"
                " VALUES ('raw-' || ?, ?, ?, 't', 'r', 0, 'raw-' || ?, ?)",
                (segment_id, claimed, segment_id, segment_id, policy_id),
            )


def test_child_link_needs_an_attached_owner_and_an_empty_root_less_segment(db):
    repository = ConsoleTraceRepository()
    conversation_id, owner_id, root_id, _policy, _node = _root_graph(db, repository)
    with pytest.raises(sqlite3.IntegrityError, match="root-less segment"):
        db.get_connection().execute(
            "INSERT INTO console_trace_child_segments(segment_id, owner_id, turn_id, run_id)"
            " VALUES (?, ?, 'turn-1', 'run-root')",
            (root_id, owner_id),
        )
    with db.transaction(immediate=True) as cursor:
        repository.detach_owner(
            cursor, owner_id=owner_id, detached_at="2026-11-17T12:00:00Z"
        )
    with pytest.raises(sqlite3.IntegrityError, match="active owner"):
        _child(db, repository, owner_id, "turn-1")
    assert conversation_id


def test_gc_keeps_child_segments_while_attached_and_collects_them_after_detach(db):
    repository = ConsoleTraceRepository()
    conversation_id, owner_id, _root, policy_id, _node = _root_graph(db, repository)
    child = _child(db, repository, owner_id, "turn-1")
    with db.transaction(immediate=True) as cursor:
        call_id = _terminal_call(
            cursor,
            repository,
            owner_id=owner_id,
            segment_id=child.segment_id,
            policy_id=policy_id,
            turn_id="turn-1",
            event_sequence=0,
        )
    _finish_legacy_migration(db)

    TraceGarbageCollector(db).collect(request_id="child-gc-attached")
    assert _rows(
        db,
        "SELECT COUNT(*) FROM console_trace_child_segments WHERE segment_id = ?",
        child.segment_id,
    ) == [(1,)]
    assert _rows(
        db, "SELECT COUNT(*) FROM console_trace_calls WHERE call_id = ?", call_id
    ) == [(1,)]

    ChatPersistenceService(db).purge_console_trace(
        conversation_id=conversation_id,
        request_id="child-gc-detached",
        detached_at="2026-11-17T12:01:00Z",
    )
    for table, column, value in (
        ("console_trace_child_segments", "segment_id", child.segment_id),
        ("console_trace_segments", "segment_id", child.segment_id),
        ("console_trace_calls", "call_id", call_id),
    ):
        assert _rows(db, f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", value) == [
            (0,)
        ], table


def test_a_v86_database_upgrades_with_its_trace_intact(tmp_path):
    path = tmp_path / "v86.sqlite"
    repository = ConsoleTraceRepository()
    with chachanotes_db_at_version(path, 86) as old:
        _conversation, owner_id, root_id, _policy, _node = _root_graph(old, repository)
        assert not _rows(
            old,
            "SELECT 1 FROM sqlite_schema WHERE name = 'console_trace_child_segments'",
        )
    upgraded = CharactersRAGDB(str(path), "trace-child-segments-upgrade")
    try:
        assert _rows(upgraded, "SELECT version FROM db_schema_version") == [
            (CharactersRAGDB._CURRENT_SCHEMA_VERSION,)
        ]
        guards = dict(
            _rows(
                upgraded,
                "SELECT name, sql FROM sqlite_schema WHERE type = 'trigger'"
                " AND name IN (?, ?, ?, ?, ?)",
                *OWNER_GUARDS,
            )
        )
        assert sorted(guards) == sorted(OWNER_GUARDS)
        assert all("console_trace_child_segments" in sql for sql in guards.values())
        with upgraded.transaction(immediate=True) as cursor:
            owner = repository.get_effective_owner(cursor, root_id)
        assert owner is not None and owner.owner_id == owner_id
        child = _child(upgraded, repository, owner_id, "turn-1")
        with upgraded.transaction(immediate=True) as cursor:
            assert repository.get_effective_owner(cursor, child.segment_id) == owner
    finally:
        upgraded.close_connection()


def test_a_failing_v87_statement_leaves_the_database_at_v86(tmp_path, monkeypatch):
    path = tmp_path / "v86-atomic.sqlite"
    with chachanotes_db_at_version(path, 86):
        pass
    original = CharactersRAGDB._execute_migration_statements

    def fail_last(self, cursor, script, label):
        return original(self, cursor, script + "\nSELECT no_such_function();", label)

    monkeypatch.setattr(CharactersRAGDB, "_execute_migration_statements", fail_last)
    with pytest.raises(Exception, match="V86"):
        CharactersRAGDB(str(path), "trace-child-segments-atomic")
    monkeypatch.undo()

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("SELECT version FROM db_schema_version").fetchall() == [
            (86,)
        ]
        assert not raw.execute(
            "SELECT 1 FROM sqlite_schema WHERE name LIKE 'console_trace_child_segments%'"
        ).fetchall()
        for guard in (*OWNER_GUARDS, "console_trace_owners_empty_root_guard"):
            (sql,) = raw.execute(
                "SELECT sql FROM sqlite_schema WHERE name = ?", (guard,)
            ).fetchone()
            assert "console_trace_child_segments" not in sql
    finally:
        raw.close()


def test_the_v87_step_refuses_a_wrong_entry_version(db):
    with pytest.raises(Exception, match="requires schema version"):
        db._migrate_from_v86_to_v87(db.get_connection())


def test_a_fork_does_not_carry_pre_fork_child_calls(db):
    repository = ConsoleTraceRepository()
    _conv, owner_id, root_id, policy_id, head = _root_graph(db, repository)
    child = _child(db, repository, owner_id, "turn-1")
    with db.transaction(immediate=True) as cursor:
        call_id = _terminal_call(
            cursor,
            repository,
            owner_id=owner_id,
            segment_id=child.segment_id,
            policy_id=policy_id,
            turn_id="turn-1",
            event_sequence=0,
        )
    fork_conversation = db.add_conversation({"title": "fork"})
    with db.transaction(immediate=True) as cursor:
        fork_segment = repository.create_segment(
            cursor,
            parent_segment_id=root_id,
            inherited_through_sequence=0,
            inherited_surface_head_id=head,
        )
        fork_owner = repository.attach_owner(
            cursor,
            conversation_id=fork_conversation,
            root_segment_id=fork_segment.segment_id,
        )
    lineage = {
        row[0]
        for row in _rows(
            db,
            "WITH RECURSIVE lineage(segment_id) AS (SELECT ?"
            " UNION SELECT s.parent_segment_id FROM console_trace_segments AS s"
            " JOIN lineage ON s.segment_id = lineage.segment_id"
            " WHERE s.parent_segment_id IS NOT NULL) SELECT segment_id FROM lineage",
            fork_segment.segment_id,
        )
    }
    assert lineage == {fork_segment.segment_id, root_id}
    assert child.segment_id not in lineage
    assert _rows(
        db,
        "SELECT COUNT(*) FROM console_trace_calls WHERE owner_id = ?",
        fork_owner.owner_id,
    ) == [(0,)]
    # The child's call stays with the parent's owner.
    assert _rows(
        db, "SELECT owner_id FROM console_trace_calls WHERE call_id = ?", call_id
    ) == [(owner_id,)]


def test_a_linked_child_segment_never_becomes_a_root_or_a_fork_parent(db):
    repository = ConsoleTraceRepository()
    _conv, owner_id, _root, _policy, head = _root_graph(db, repository)
    child = _child(db, repository, owner_id, "turn-1")
    other = db.add_conversation({"title": "other"})
    with pytest.raises(sqlite3.IntegrityError, match="distinct empty root"):
        with db.transaction(immediate=True) as cursor:
            repository.attach_owner(
                cursor, conversation_id=other, root_segment_id=child.segment_id
            )
    with pytest.raises(sqlite3.IntegrityError, match="detached owner prefix"):
        with db.transaction(immediate=True) as cursor:
            repository.create_segment(
                cursor,
                parent_segment_id=child.segment_id,
                inherited_through_sequence=0,
                inherited_surface_head_id=head,
            )


def test_a_call_on_a_child_segment_must_carry_the_links_turn_and_run(db):
    repository = ConsoleTraceRepository()
    _conv, owner_id, _root, policy_id, _node = _root_graph(db, repository)
    child = _child(db, repository, owner_id, "turn-1", run_id="run-1")
    for turn, run in (("turn-2", "run-1"), ("turn-1", "run-2")):
        with pytest.raises(sqlite3.IntegrityError, match="active effective owner"):
            db.get_connection().execute(
                "INSERT INTO console_trace_calls(call_id, owner_id, segment_id,"
                " turn_id, run_id, call_sequence, idempotency_key, policy_id)"
                " VALUES ('c-' || ?, ?, ?, ?, ?, 0, 'k-' || ?, ?)",
                (turn + run, owner_id, child.segment_id, turn, run, turn + run, policy_id),
            )
    db.get_connection().execute(
        "INSERT INTO console_trace_calls(call_id, owner_id, segment_id,"
        " turn_id, run_id, call_sequence, idempotency_key, policy_id)"
        " VALUES ('ok', ?, ?, 'turn-1', 'run-1', 0, 'k-ok', ?)",
        (owner_id, child.segment_id, policy_id),
    )


def test_a_live_retention_root_keeps_the_link_and_segment_through_gc(db):
    repository = ConsoleTraceRepository()
    conversation_id, owner_id, _root, policy_id, _node = _root_graph(db, repository)
    child = _child(db, repository, owner_id, "turn-1")
    with db.transaction(immediate=True) as cursor:
        call_id = _terminal_call(
            cursor,
            repository,
            owner_id=owner_id,
            segment_id=child.segment_id,
            policy_id=policy_id,
            turn_id="turn-1",
            event_sequence=0,
        )
    collector = TraceGarbageCollector(db)
    collector.retain_conversation(
        conversation_id=conversation_id,
        retain_until="2999-01-01T00:00:00Z",
        reason_code="legal_hold",
    )
    with db.transaction(immediate=True) as cursor:
        repository.detach_owner(
            cursor, owner_id=owner_id, detached_at="2026-11-17T12:00:00Z"
        )
    _finish_legacy_migration(db)

    collector.collect(request_id="child-gc-retained")

    for table, column, value in (
        ("console_trace_child_segments", "segment_id", child.segment_id),
        ("console_trace_segments", "segment_id", child.segment_id),
        ("console_trace_calls", "call_id", call_id),
    ):
        assert _rows(db, f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", value) == [
            (1,)
        ], table


def test_an_unresolved_child_call_never_blocks_the_parents_turn(db):
    """Owner decision 2026-11-17: a child's open call is not the parent's."""
    repository = ConsoleTraceRepository()
    _conversation, owner_id, root_id, policy_id, _node = _root_graph(db, repository)
    child = _child(db, repository, owner_id, "turn-1")

    def reserve(segment_id, run_id):
        with db.transaction(immediate=True) as cursor:
            repository.reserve_call(
                cursor,
                owner_id=owner_id,
                segment_id=segment_id,
                turn_id="turn-1",
                run_id=run_id,
                call_sequence=0,
                idempotency_key="open-" + run_id,
                policy_id=policy_id,
            )

    def blocked():
        with db.transaction(immediate=True) as cursor:
            return repository.has_unresolved_call_for_turn(
                cursor, owner_id=owner_id, turn_id="turn-1"
            )

    reserve(child.segment_id, child.run_id)
    assert blocked() is False
    reserve(root_id, "parent-run")
    assert blocked() is True


def test_failed_mobile_v88_step_preserves_populated_native_v87(tmp_path, monkeypatch):
    path = tmp_path / "native87-mobile88-atomic.sqlite"
    repository = ConsoleTraceRepository()
    with chachanotes_db_at_version(path, 87) as old:
        _, owner_id, _, policy_id, _ = _root_graph(old, repository)
        child = _child(old, repository, owner_id, "migration-turn")
        with old.transaction(immediate=True) as cursor:
            _terminal_call(
                cursor,
                repository,
                owner_id=owner_id,
                segment_id=child.segment_id,
                policy_id=policy_id,
                turn_id="migration-turn",
                event_sequence=0,
            )

    def snapshot():
        raw = sqlite3.connect(path)
        try:
            return {
                "schema": raw.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_schema "
                    "WHERE sql IS NOT NULL ORDER BY name"
                ).fetchall(),
                **{
                    table: raw.execute("SELECT * FROM " + table).fetchall()
                    for table in (
                        "console_trace_child_segments",
                        "console_trace_owners",
                        "console_trace_calls",
                        "console_trace_events",
                        "db_schema_version",
                    )
                },
            }
        finally:
            raw.close()

    before = snapshot()
    assert before["db_schema_version"] == [("rag_char_chat_schema", 87)]
    original = CharactersRAGDB._execute_migration_statements
    observed = []

    def fail_mobile(self, cursor, script, label):
        observed.append(label)
        if label == "V87→V88":
            script += "\nSELECT no_such_mobile_function();"
        return original(self, cursor, script, label)

    with monkeypatch.context() as changed:
        changed.setattr(CharactersRAGDB, "_execute_migration_statements", fail_mobile)
        with pytest.raises(Exception, match="V87→V88"):
            CharactersRAGDB(path, "mobile-child-atomic")
    assert observed == ["V87→V88"]
    assert snapshot() == before

    upgraded = CharactersRAGDB(path, "mobile-child-after-rollback")
    try:
        assert _rows(upgraded, "SELECT version FROM db_schema_version") == [(88,)]
        for table, rows in before.items():
            if table not in {"schema", "db_schema_version"}:
                assert _rows(upgraded, "SELECT * FROM " + table) == rows
        assert _rows(upgraded, "SELECT count(*) FROM mobile_consolidation_batches") == [
            (0,)
        ]
        assert not _rows(upgraded, "PRAGMA foreign_key_check")
    finally:
        upgraded.close_connection()
