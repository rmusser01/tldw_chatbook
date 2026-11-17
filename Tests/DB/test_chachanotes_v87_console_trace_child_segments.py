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


def _child(db, repository, owner_id, turn_id, run_id="child-actor:child-chain"):
    with db.transaction(immediate=True) as cursor:
        return repository.create_child_segment(
            cursor, owner_id=owner_id, turn_id=turn_id, run_id=run_id
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
        assert _rows(upgraded, "SELECT version FROM db_schema_version") == [(87,)]
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
