from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
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
    ConsoleTransactionWriter,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@dataclass
class RecordingContribution:
    fail: bool = False
    seen_writer: ConsoleTransactionWriter | None = None
    seen_conversation_id: str | None = None
    seen_message_ids: Mapping[str, str] | None = None

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        self.seen_writer = writer
        self.seen_conversation_id = conversation_id
        self.seen_message_ids = dict(message_ids)
        writer.execute(
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
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        writer.execute(
            "INSERT INTO contribution_probe(conversation_id, user_id, assistant_id) "
            "VALUES (?, ?, ?)",
            (conversation_id, message_ids["user"], message_ids["assistant"]),
        )
        writer.connection.commit()  # type: ignore[attr-defined]


@dataclass
class ClearAuthorizerAndCommitContribution:
    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        writer.connection.set_authorizer(None)  # type: ignore[attr-defined]
        writer.execute(
            "INSERT INTO contribution_probe(conversation_id, user_id, assistant_id) "
            "VALUES (?, ?, ?)",
            (conversation_id, message_ids["user"], message_ids["assistant"]),
        )
        writer.connection.commit()  # type: ignore[attr-defined]


@dataclass
class AttachDatabaseContribution:
    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        del conversation_id, message_ids
        writer.execute("ATTACH DATABASE ':memory:' AS escaped", ())


@dataclass
class BatchContribution:
    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        writer.executemany(
            "INSERT INTO contribution_probe(conversation_id, user_id, assistant_id) "
            "VALUES (?, ?, ?)",
            (
                (conversation_id, message_ids["user"], message_ids["assistant"]),
                (conversation_id, message_ids["user"], message_ids["assistant"]),
            ),
        )


@dataclass
class StatementContribution:
    statement: str

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        del conversation_id, message_ids
        writer.execute(self.statement, ())


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


def test_generic_contribution_receives_only_writer_and_committed_id_map(
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
    assert contribution.seen_writer is not None
    assert callable(contribution.seen_writer.execute)
    assert callable(contribution.seen_writer.executemany)
    assert not any(
        hasattr(contribution.seen_writer, name)
        for name in (
            "connection",
            "cursor",
            "set_authorizer",
            "commit",
            "rollback",
            "repository",
            "session",
            "publish",
        )
    )
    assert contribution.seen_conversation_id == conversation_id
    assert contribution.seen_message_ids == {
        "user": "user",
        "assistant": "assistant",
    }
    row = service.db.get_connection().execute(
        "SELECT conversation_id, user_id, assistant_id FROM contribution_probe"
    ).fetchone()
    assert tuple(row) == (conversation_id, "user", "assistant")
    with pytest.raises(RuntimeError, match="active contribution"):
        contribution.seen_writer.execute(
            "INSERT INTO contribution_probe(conversation_id, user_id, assistant_id) "
            "VALUES (?, ?, ?)",
            (conversation_id, "user", "assistant"),
        )


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


def test_later_contribution_failure_rolls_back_normal_parameterized_writes(
    tmp_path: Path,
) -> None:
    service, conversation_id = _service(tmp_path / "later-rollback.sqlite")
    first = RecordingContribution()
    later = RecordingContribution(fail=True)
    acceptance = replace(
        _acceptance(conversation_id, first),
        contributions=(first, later),
    )

    with pytest.raises(RuntimeError, match="injected contribution failure"):
        with service.db.transaction(immediate=True) as cursor:
            service.console_dispatch_repository.insert_with_messages(
                cursor,
                acceptance,
            )

    connection = service.db.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM console_dispatch_checkpoints"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM contribution_probe"
    ).fetchone()[0] == 0


def test_writer_executes_parameterized_insert_many_without_exposing_a_cursor(
    tmp_path: Path,
) -> None:
    service, conversation_id = _service(tmp_path / "batch.sqlite")

    with service.db.transaction(immediate=True) as cursor:
        service.console_dispatch_repository.insert_with_messages(
            cursor,
            _acceptance(conversation_id, BatchContribution()),
        )

    assert service.db.get_connection().execute(
        "SELECT COUNT(*) FROM contribution_probe"
    ).fetchone()[0] == 2


def test_contribution_protocol_exposes_no_repository_or_publication_capability() -> None:
    annotations = ConsoleTransactionContribution.write.__annotations__

    assert tuple(annotations) == (
        "writer",
        "conversation_id",
        "message_ids",
        "return",
    )
    assert annotations["writer"] == "ConsoleTransactionWriter"
    assert "cursor" not in annotations
    assert "connection" not in annotations
    assert "db" not in annotations
    assert "repository" not in annotations
    assert "holder" not in annotations
    assert "session" not in annotations


@pytest.mark.parametrize(
    ("contribution", "expected_error"),
    [
        (EarlyCommitContribution(), AttributeError),
        (ClearAuthorizerAndCommitContribution(), AttributeError),
        (AttachDatabaseContribution(), ValueError),
    ],
    ids=["early_commit", "clear_authorizer_and_commit", "attach_database"],
)
def test_contribution_cannot_escape_the_caller_owned_transaction(
    tmp_path: Path,
    contribution: ConsoleTransactionContribution,
    expected_error: type[Exception],
) -> None:
    service, conversation_id = _service(tmp_path / "transaction-escape.sqlite")

    with pytest.raises(expected_error):
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


@pytest.mark.parametrize(
    "statement",
    [
        "BEGIN IMMEDIATE",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT contribution",
        "RELEASE contribution",
        "ATTACH DATABASE ':memory:' AS escaped",
        "DETACH DATABASE escaped",
        "UPDATE contribution_probe SET user_id = 'escaped'",
        "DELETE FROM contribution_probe",
        "INSERT INTO contribution_probe VALUES ('x', 'y', 'z')",
        "INSERT INTO contribution_probe VALUES ('x', 'y', 'z'); COMMIT",
    ],
)
def test_writer_rejects_transaction_control_and_non_insert_sql(
    tmp_path: Path,
    statement: str,
) -> None:
    service, conversation_id = _service(tmp_path / "rejected-statement.sqlite")

    with pytest.raises(ValueError, match="parameterized INSERT"):
        with service.db.transaction(immediate=True) as cursor:
            service.console_dispatch_repository.insert_with_messages(
                cursor,
                _acceptance(conversation_id, StatementContribution(statement)),
            )

    connection = service.db.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM console_dispatch_checkpoints"
    ).fetchone()[0] == 0
