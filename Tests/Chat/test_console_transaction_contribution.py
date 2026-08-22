from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchReconstructability,
    ConsoleDurableTurnAcceptance,
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_transaction_contribution import (
    ConsoleTransactionContribution,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@dataclass
class RecordingContribution:
    fail: bool = False
    seen_cursor: sqlite3.Cursor | None = None
    seen_conversation_id: str | None = None
    seen_message_ids: Mapping[str, str] | None = None

    def write(
        self,
        *,
        cursor: sqlite3.Cursor,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        self.seen_cursor = cursor
        self.seen_conversation_id = conversation_id
        self.seen_message_ids = dict(message_ids)
        cursor.execute(
            "INSERT INTO contribution_probe(conversation_id, user_id, assistant_id) "
            "VALUES (?, ?, ?)",
            (conversation_id, message_ids["user"], message_ids["assistant"]),
        )
        if self.fail:
            raise RuntimeError("injected contribution failure")


@dataclass
class EarlyCommitContribution:
    def write(
        self,
        *,
        cursor: sqlite3.Cursor,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        cursor.execute(
            "INSERT INTO contribution_probe(conversation_id, user_id, assistant_id) "
            "VALUES (?, ?, ?)",
            (conversation_id, message_ids["user"], message_ids["assistant"]),
        )
        cursor.connection.commit()


@dataclass
class AttachDatabaseContribution:
    def write(
        self,
        *,
        cursor: sqlite3.Cursor,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        del conversation_id, message_ids
        cursor.execute("ATTACH DATABASE ':memory:' AS escaped")


def _acceptance(
    conversation_id: str,
    contribution: ConsoleTransactionContribution,
) -> ConsoleDurableTurnAcceptance:
    authority = ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            ConsoleAutoRetrieve.NEVER,
            ConsoleAssistantLibraryAccess.BLOCKED,
            1,
            "durable",
        ),
        direct_library_tools=False,
        source_types=("notes", "media", "conversations"),
        scope_snapshot=ConsoleLibraryItemScopeSnapshot((), (), True),
        provider_intent=ConsoleProviderIntent("openai", "model", None),
        attempt_id="attempt",
    )
    return ConsoleDurableTurnAcceptance(
        conversation_id=conversation_id,
        user_message_id="user",
        assistant_message_id="assistant",
        parent_message_id=None,
        user_content="hello",
        attachments=(),
        preparation_id="preparation",
        attempt_id="attempt",
        origin="manual",
        queue_entry_id=None,
        frozen_authority=authority,
        resolved_destination=ConsoleResolvedDestination(
            "openai", "model", "https://api.example.test:443", ConsoleEgressClass.UNKNOWN
        ),
        reconstructability=ConsoleDispatchReconstructability(True, True, True, None),
        contributions=(contribution,),
    )


def _service(path: Path) -> tuple[ChatPersistenceService, str]:
    db = CharactersRAGDB(path, client_id="contribution-test")
    conversation_id = db.add_conversation({"title": "contribution"})
    assert conversation_id is not None
    connection = db.get_connection()
    connection.execute(
        """
        CREATE TABLE contribution_probe(
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            user_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            assistant_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE
        )
        """
    )
    connection.commit()
    return ChatPersistenceService(db), conversation_id


def test_generic_contribution_receives_only_caller_cursor_and_committed_id_map(
    tmp_path: Path,
) -> None:
    service, conversation_id = _service(tmp_path / "contribution.sqlite")
    contribution = RecordingContribution()
    acceptance = _acceptance(conversation_id, contribution)

    with service.db.transaction(immediate=True) as cursor:
        checkpoint = service.console_dispatch_repository.insert_with_messages(
            cursor, acceptance
        )

    assert checkpoint.conversation_id == conversation_id
    assert isinstance(contribution.seen_cursor, sqlite3.Cursor)
    assert contribution.seen_conversation_id == conversation_id
    assert contribution.seen_message_ids == {
        "user": "user",
        "assistant": "assistant",
    }
    row = service.db.get_connection().execute(
        "SELECT conversation_id, user_id, assistant_id FROM contribution_probe"
    ).fetchone()
    assert tuple(row) == (conversation_id, "user", "assistant")


def test_contribution_error_propagates_and_rolls_back_every_write(tmp_path: Path) -> None:
    service, conversation_id = _service(tmp_path / "rollback.sqlite")
    contribution = RecordingContribution(fail=True)

    with pytest.raises(RuntimeError, match="injected contribution failure"):
        with service.db.transaction(immediate=True) as cursor:
            service.console_dispatch_repository.insert_with_messages(
                cursor, _acceptance(conversation_id, contribution)
            )

    connection = service.db.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM console_dispatch_checkpoints"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM contribution_probe"
    ).fetchone()[0] == 0


def test_contribution_protocol_exposes_no_repository_or_publication_capability() -> None:
    annotations = ConsoleTransactionContribution.write.__annotations__

    assert tuple(annotations) == (
        "cursor",
        "conversation_id",
        "message_ids",
        "return",
    )
    assert "db" not in annotations
    assert "repository" not in annotations
    assert "holder" not in annotations
    assert "session" not in annotations


@pytest.mark.parametrize(
    "contribution",
    [EarlyCommitContribution(), AttachDatabaseContribution()],
    ids=["early_commit", "attach_database"],
)
def test_contribution_cannot_escape_the_caller_owned_transaction(
    tmp_path: Path,
    contribution: ConsoleTransactionContribution,
) -> None:
    service, conversation_id = _service(tmp_path / "transaction-escape.sqlite")

    with pytest.raises(sqlite3.DatabaseError):
        with service.db.transaction(immediate=True) as cursor:
            service.console_dispatch_repository.insert_with_messages(
                cursor,
                _acceptance(conversation_id, contribution),
            )

    connection = service.db.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM console_dispatch_checkpoints"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM contribution_probe"
    ).fetchone()[0] == 0
