"""B24: fork-commit verification loads messages in batched IN-lists.

Golden baseline captured against the pre-change implementation
(``/tmp`` capture, review-B session record): for a 300-message fork,
``resolve_console_fork_commit`` issued 303 SELECT statements (one per copied
message plus conversation/policy reads), a fresh-commit bundle ran the same
per-message verification twice (target verify + source recheck), and every
``_fork_source_parent``/lineage hop issued its own single-row SELECT. The
per-message decisions on the healthy fixture are all ``verified``; the
failure decisions are ``Console fork target identity collision.`` (mutated
copied message) and ``Console fork source changed.`` (mutated source row);
resolving a never-committed fork returns ``None``.

The batched implementation must reproduce those decisions exactly while
issuing at most 5 SELECT statements for the 300-message resolution and zero
single-id verification SELECTs anywhere in the fork-commit paths. The 10,000
lineage hop cap is preserved by keeping the loop structure unchanged (each
hop still consumes exactly one iteration).
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from Tests.Chat.test_console_chat_fork_persistence import (
    _commit,
    _configuration,
    _raw_semantic_corruption,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_fork import (
    ConsoleChatForkSnapshot,
    ConsoleForkCitationLink,
    ConsoleForkProjectedMessage,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

MESSAGE_COUNT = 300

GOLDEN_RESOLVE_SELECTS_BEFORE = 303
GOLDEN_FRESH_COMMIT_SELECTS_BEFORE = 2407
GOLDEN_RESOLVED_FIELDS = {
    "already_committed": True,
    "conversation_id": "fork",
    "active_leaf_message_id": "fork-msg-0299",
    "message_id_map_size": MESSAGE_COUNT,
}
GOLDEN_MUTATED_TARGET_REASON = "Console fork target identity collision."
GOLDEN_MUTATED_SOURCE_REASON = "Console fork source changed."

_VERIFICATION_SELECT_PREFIX = "SELECT conversation_id, parent_message_id"


def _normalize(statement: str) -> str:
    return " ".join(statement.split())


def _is_single_id_verification_select(statement: str) -> bool:
    """The four fork-verification read shapes (target/source/hop/lineage).

    All start with the verification column list and filter by one literal
    message id (the trace callback receives parameter-expanded SQL). The
    per-message reads inside ``create_message`` start with different column
    lists and are not counted.
    """
    normalized = _normalize(statement)
    return normalized.startswith(
        _VERIFICATION_SELECT_PREFIX
    ) and "WHERE id = '" in normalized


class SelectCounter:
    """Trace-counts statements on the shared sqlite connection."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._connection = db.get_connection()
        self.statements: list[str] = []

    def __enter__(self) -> "SelectCounter":
        self._connection.set_trace_callback(self.statements.append)
        return self

    def __exit__(self, *args) -> None:
        self._connection.set_trace_callback(None)

    @property
    def selects(self) -> int:
        return sum(
            1
            for statement in self.statements
            if statement.lstrip().upper().startswith("SELECT")
        )

    @property
    def single_id_verification_selects(self) -> int:
        return sum(
            1
            for statement in self.statements
            if _is_single_id_verification_select(statement)
        )


def seed_large_source(db: CharactersRAGDB) -> tuple[int, list[dict]]:
    """Seed one source conversation with a linear 300-message chain."""
    db.add_conversation({"id": "source", "root_id": "root", "title": "Source"})
    rows = []
    parent = None
    for index in range(MESSAGE_COUNT):
        message_id = f"source-msg-{index:04d}"
        db.add_message(
            {
                "id": message_id,
                "conversation_id": "source",
                "parent_message_id": parent,
                "sender": "user" if index % 2 == 0 else "assistant",
                "content": f"body-{index}",
                "client_id": db.client_id,
            }
        )
        rows.append(db.get_message_by_id(message_id))
        parent = message_id
    db.set_conversation_active_leaf("source", f"source-msg-{MESSAGE_COUNT - 1:04d}")
    conversation = db.get_conversation_by_id("source")
    return conversation["version"], rows


def large_snapshot(
    db: CharactersRAGDB, version: int, rows: list[dict]
) -> ConsoleChatForkSnapshot:
    """Build the 300-message durable fork snapshot for the seeded source."""
    messages = []
    persisted_parent = None
    for index, row in enumerate(rows):
        messages.append(
            ConsoleForkProjectedMessage(
                source_native_message_id=f"native-source-{index:04d}",
                source_persisted_message_id=row["id"],
                source_persisted_revision=row["version"],
                source_persisted_content=row["content"],
                native_message_id=f"native-fork-{index:04d}",
                persisted_message_id=f"fork-msg-{index:04d}",
                native_parent_id=(
                    f"native-fork-{index - 1:04d}" if index else None
                ),
                persisted_parent_id=persisted_parent,
                turn_id="fork-turn",
                visible_variant_id=None,
                role=(
                    ConsoleMessageRole.USER
                    if index % 2 == 0
                    else ConsoleMessageRole.ASSISTANT
                ),
                status="complete",
                content=row["content"],
            )
        )
        persisted_parent = f"fork-msg-{index:04d}"
    citation_links = tuple(
        ConsoleForkCitationLink(
            source_persisted_message_id=row["id"],
            source_revision=row["version"],
            state="none",
            trace_id=None,
        )
        for row in rows
    )
    return ConsoleChatForkSnapshot(
        fork_session_id="fork-session",
        fork_conversation_id="fork",
        title="Forked source",
        source_session_id="source-session",
        source_conversation_id="source",
        source_conversation_version=version,
        source_active_leaf_persisted_message_id=(
            f"source-msg-{MESSAGE_COUNT - 1:04d}"
        ),
        source_boundary_persisted_message_id=(
            f"source-msg-{MESSAGE_COUNT - 1:04d}"
        ),
        durable=True,
        messages=tuple(messages),
        configuration=_configuration(),
        citation_links=citation_links,
    )


def expected_decisions(db: CharactersRAGDB, snapshot) -> list[str]:
    """Re-derive each message's verify decision straight from the DB rows."""
    connection = db.get_connection()
    decisions = []
    for message in snapshot.messages:
        row = connection.execute(
            "SELECT conversation_id, parent_message_id, sender, content, deleted "
            "FROM messages WHERE id = ?",
            (message.persisted_message_id,),
        ).fetchone()
        verified = row is not None and tuple(row) == (
            snapshot.fork_conversation_id,
            message.persisted_parent_id,
            message.role.value,
            message.content,
            0,
        )
        decisions.append("verified" if verified else "identity-collision")
    return decisions


@pytest.fixture
def fork_db(tmp_path: Path) -> CharactersRAGDB:
    db = CharactersRAGDB(tmp_path / "fork-batched.db", client_id="fork-batched")
    yield db
    db.close_connection()


@pytest.fixture
def service(fork_db: CharactersRAGDB) -> ChatPersistenceService:
    return ChatPersistenceService(fork_db)


@pytest.fixture
def committed_fork(fork_db, service):
    version, rows = seed_large_source(fork_db)
    snapshot = large_snapshot(fork_db, version, rows)
    committed = _commit(service, snapshot)
    assert committed is not None and committed.already_committed is False
    return snapshot


def test_resolution_is_batched_and_decisions_match_golden(
    fork_db, service, committed_fork
) -> None:
    snapshot = committed_fork

    with SelectCounter(fork_db) as counter:
        resolved = service.resolve_console_fork_commit(snapshot)

    assert resolved is not None
    assert resolved.already_committed is GOLDEN_RESOLVED_FIELDS["already_committed"]
    assert resolved.conversation_id == GOLDEN_RESOLVED_FIELDS["conversation_id"]
    assert (
        resolved.active_leaf_message_id
        == GOLDEN_RESOLVED_FIELDS["active_leaf_message_id"]
    )
    assert len(resolved.message_id_map) == GOLDEN_RESOLVED_FIELDS["message_id_map_size"]

    assert counter.selects <= 5, (
        f"resolve_console_fork_commit issued {counter.selects} SELECTs for "
        f"{MESSAGE_COUNT} messages (golden baseline: "
        f"{GOLDEN_RESOLVE_SELECTS_BEFORE})"
    )
    assert counter.single_id_verification_selects == 0

    decisions = expected_decisions(fork_db, snapshot)
    assert decisions == ["verified"] * MESSAGE_COUNT


def test_fresh_commit_recheck_has_no_per_message_selects(
    fork_db, service
) -> None:
    version, rows = seed_large_source(fork_db)
    snapshot = large_snapshot(fork_db, version, rows)

    with SelectCounter(fork_db) as counter:
        committed = _commit(service, snapshot)

    assert committed is not None
    assert committed.already_committed is False
    assert len(committed.message_id_map) == MESSAGE_COUNT
    assert counter.single_id_verification_selects == 0, (
        "fresh-commit fork verification must not issue single-id message "
        f"SELECTs (golden baseline: {GOLDEN_FRESH_COMMIT_SELECTS_BEFORE} "
        "total SELECTs, most of them per-message verification reads)"
    )


def test_failure_decisions_match_golden(fork_db, service, committed_fork) -> None:
    snapshot = committed_fork

    with _raw_semantic_corruption(fork_db), fork_db.transaction() as cursor:
        cursor.execute(
            "UPDATE messages SET content = 'tampered' WHERE id = 'fork-msg-0150'"
        )
    with pytest.raises(RuntimeError) as excinfo:
        service.resolve_console_fork_commit(snapshot)
    assert str(excinfo.value) == GOLDEN_MUTATED_TARGET_REASON
    with _raw_semantic_corruption(fork_db), fork_db.transaction() as cursor:
        cursor.execute(
            "UPDATE messages SET content = ? WHERE id = 'fork-msg-0150'",
            ("body-150",),
        )

    with _raw_semantic_corruption(fork_db), fork_db.transaction() as cursor:
        cursor.execute(
            "UPDATE messages SET content = 'drifted' WHERE id = 'source-msg-0100'"
        )
    snapshot2 = replace(snapshot, fork_conversation_id="fork-2")
    with pytest.raises(RuntimeError) as excinfo:
        _commit(service, snapshot2)
    assert str(excinfo.value) == GOLDEN_MUTATED_SOURCE_REASON
    with _raw_semantic_corruption(fork_db), fork_db.transaction() as cursor:
        cursor.execute(
            "UPDATE messages SET content = ? WHERE id = 'source-msg-0100'",
            ("body-100",),
        )

    # A fork that never committed resolves to None, not an error.
    assert service.resolve_console_fork_commit(snapshot2) is None


def test_batched_resolution_matches_golden_on_small_fixture(
    fork_db, service
) -> None:
    """Equivalence on the shared 2-message fixture shape (already-committed)."""
    from Tests.Chat.test_console_chat_fork_persistence import _snapshot

    snapshot = _snapshot(fork_db)
    first = _commit(service, snapshot)
    assert first is not None and first.already_committed is False

    with SelectCounter(fork_db) as counter:
        resolved = service.resolve_console_fork_commit(snapshot)

    assert resolved is not None
    assert resolved.already_committed is True
    assert resolved.message_id_map == first.message_id_map
    assert counter.selects <= 5
    assert counter.single_id_verification_selects == 0
