from __future__ import annotations

import sqlite3
import threading
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
    parameters: tuple[object, ...] = ()

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        del conversation_id, message_ids
        writer.execute(self.statement, self.parameters)


@dataclass
class BatchStatementContribution:
    statement: str
    parameter_rows: tuple[tuple[object, ...], ...]

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        del conversation_id, message_ids
        writer.executemany(self.statement, self.parameter_rows)


@dataclass
class TrajectoryContributionFixture:
    """Representative future sidecar contribution without product semantics."""

    event_kinds: tuple[str, ...]
    seen_writer: ConsoleTransactionWriter | None = None
    allocated_sequences: tuple[int, ...] = ()

    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        self.seen_writer = writer
        sequences = tuple(writer.next_trajectory_sequence() for _ in self.event_kinds)
        self.allocated_sequences = sequences
        rows = tuple(
            (
                message_ids["user"],
                conversation_id,
                message_ids["user"],
                sequence,
                event_kind,
                '{"version":1}',
            )
            for event_kind, sequence in zip(self.event_kinds, sequences, strict=True)
        )
        statement = (
            "INSERT INTO message_trajectory_metadata("
            "message_id, conversation_id, turn_id, seq, event_kind, payload_json"
            ") VALUES (?, ?, ?, ?, ?, ?)"
        )
        if len(rows) == 1:
            writer.execute(statement, rows[0])
        else:
            writer.executemany(statement, rows)


@dataclass
class WrongConversationAllocatorContribution:
    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        del conversation_id, message_ids
        writer.next_trajectory_sequence("other-conversation")  # type: ignore[call-arg]


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
    connection.execute(
        """
        CREATE TABLE writer_probe(
            value TEXT UNIQUE,
            other TEXT
        )
        """
    )
    connection.execute("CREATE TABLE single_value_probe(value TEXT UNIQUE)")
    connection.commit()
    return ChatPersistenceService(db), conversation_id


def _seed_trajectory_sequence(
    service: ChatPersistenceService,
    conversation_id: str,
    sequence: object,
) -> None:
    message_id = service.db.add_message(
        {
            "id": f"existing-{sequence}",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "existing trajectory owner",
        }
    )
    assert message_id is not None
    connection = service.db.get_connection()
    connection.execute(
        "INSERT INTO message_trajectory_metadata("
        "message_id, conversation_id, turn_id, seq, event_kind"
        ") VALUES (?, ?, ?, ?, ?)",
        (message_id, conversation_id, message_id, sequence, "existing"),
    )
    connection.commit()


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
    assert callable(contribution.seen_writer.next_trajectory_sequence)
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
    with pytest.raises(RuntimeError, match="active contribution"):
        contribution.seen_writer.next_trajectory_sequence()


def test_real_v45_trajectory_contribution_allocates_after_existing_sequence(
    tmp_path: Path,
) -> None:
    service, conversation_id = _service(tmp_path / "trajectory-next.sqlite")
    schema_version = service.db.get_connection().execute(
        "SELECT version FROM db_schema_version "
        "WHERE schema_name = 'rag_char_chat_schema'"
    ).fetchone()[0]
    assert schema_version == 45
    _seed_trajectory_sequence(service, conversation_id, 7)
    contribution = TrajectoryContributionFixture(("library_preparation",))

    with service.db.transaction(immediate=True) as cursor:
        service.console_dispatch_repository.insert_with_messages(
            cursor,
            _acceptance(conversation_id, contribution),
        )

    assert contribution.allocated_sequences == (8,)
    rows = service.db.get_connection().execute(
        "SELECT seq, event_kind, payload_json "
        "FROM message_trajectory_metadata WHERE event_kind != 'existing'"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        (8, "library_preparation", '{"version":1}')
    ]


def test_trajectory_allocator_uses_only_the_accepted_conversation_maximum(
    tmp_path: Path,
) -> None:
    service, conversation_id = _service(tmp_path / "trajectory-conversation.sqlite")
    _seed_trajectory_sequence(service, conversation_id, 7)
    other_conversation_id = service.db.add_conversation({"title": "other"})
    assert other_conversation_id is not None
    _seed_trajectory_sequence(service, other_conversation_id, 100)
    contribution = TrajectoryContributionFixture(("library_preparation",))

    with service.db.transaction(immediate=True) as cursor:
        service.console_dispatch_repository.insert_with_messages(
            cursor,
            _acceptance(conversation_id, contribution),
        )

    assert contribution.allocated_sequences == (8,)


def test_two_contributions_share_one_writer_and_allocate_consecutive_sequences(
    tmp_path: Path,
) -> None:
    service, conversation_id = _service(tmp_path / "trajectory-shared.sqlite")
    _seed_trajectory_sequence(service, conversation_id, 11)
    first = TrajectoryContributionFixture(("library_preparation",))
    second = TrajectoryContributionFixture(("library_activity",))
    acceptance = replace(
        _acceptance(conversation_id, first),
        contributions=(first, second),
    )

    with service.db.transaction(immediate=True) as cursor:
        service.console_dispatch_repository.insert_with_messages(cursor, acceptance)

    assert first.seen_writer is second.seen_writer
    assert first.allocated_sequences == (12,)
    assert second.allocated_sequences == (13,)
    assert first.seen_writer is not None
    with pytest.raises(RuntimeError, match="active contribution"):
        first.seen_writer.next_trajectory_sequence()


def test_activity_like_batch_allocates_ordered_consecutive_sequences(
    tmp_path: Path,
) -> None:
    service, conversation_id = _service(tmp_path / "trajectory-batch.sqlite")
    contribution = TrajectoryContributionFixture(
        ("library_activity_first", "library_activity_second", "library_activity_third")
    )

    with service.db.transaction(immediate=True) as cursor:
        service.console_dispatch_repository.insert_with_messages(
            cursor,
            _acceptance(conversation_id, contribution),
        )

    rows = service.db.get_connection().execute(
        "SELECT seq, event_kind FROM message_trajectory_metadata ORDER BY seq"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        (1, "library_activity_first"),
        (2, "library_activity_second"),
        (3, "library_activity_third"),
    ]


def test_trajectory_allocation_rolls_back_and_next_transaction_reuses_max_plus_one(
    tmp_path: Path,
) -> None:
    service, conversation_id = _service(tmp_path / "trajectory-rollback.sqlite")
    _seed_trajectory_sequence(service, conversation_id, 20)
    trajectory = TrajectoryContributionFixture(("library_activity",))
    failure = RecordingContribution(fail=True)
    failed_acceptance = replace(
        _acceptance(conversation_id, trajectory),
        contributions=(trajectory, failure),
    )

    with pytest.raises(RuntimeError, match="injected contribution failure"):
        with service.db.transaction(immediate=True) as cursor:
            service.console_dispatch_repository.insert_with_messages(
                cursor,
                failed_acceptance,
            )

    connection = service.db.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM console_dispatch_checkpoints"
    ).fetchone()[0] == 0
    rows = connection.execute(
        "SELECT seq FROM message_trajectory_metadata ORDER BY seq"
    ).fetchall()
    assert [tuple(row) for row in rows] == [(20,)]

    committed = TrajectoryContributionFixture(("library_preparation",))
    with service.db.transaction(immediate=True) as cursor:
        service.console_dispatch_repository.insert_with_messages(
            cursor,
            _acceptance(conversation_id, committed),
        )

    assert committed.allocated_sequences == (21,)


@pytest.mark.parametrize("bad_max", [-1, "corrupt", 9_223_372_036_854_775_807])
def test_trajectory_allocator_rejects_corrupt_or_out_of_range_maxima(
    tmp_path: Path,
    bad_max: object,
) -> None:
    service, conversation_id = _service(tmp_path / "trajectory-bad-max.sqlite")
    _seed_trajectory_sequence(service, conversation_id, bad_max)

    with pytest.raises(sqlite3.DatabaseError, match="trajectory sequence"):
        with service.db.transaction(immediate=True) as cursor:
            service.console_dispatch_repository.insert_with_messages(
                cursor,
                _acceptance(
                    conversation_id,
                    TrajectoryContributionFixture(("library_activity",)),
                ),
            )

    connection = service.db.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM console_dispatch_checkpoints"
    ).fetchone()[0] == 0


def test_trajectory_allocator_cannot_request_another_conversation(
    tmp_path: Path,
) -> None:
    service, conversation_id = _service(tmp_path / "trajectory-bound.sqlite")

    with pytest.raises(TypeError):
        with service.db.transaction(immediate=True) as cursor:
            service.console_dispatch_repository.insert_with_messages(
                cursor,
                _acceptance(conversation_id, WrongConversationAllocatorContribution()),
            )

    assert service.db.get_connection().execute(
        "SELECT COUNT(*) FROM messages"
    ).fetchone()[0] == 0


def test_concurrent_repositories_serialize_and_reject_second_active_owner(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "trajectory-concurrent.sqlite"
    first_service, conversation_id = _service(database_path)
    _seed_trajectory_sequence(first_service, conversation_id, 30)
    second_service = ChatPersistenceService(
        CharactersRAGDB(database_path, client_id="contribution-test-2")
    )
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def accept(
        service: ChatPersistenceService,
        suffix: str,
        contribution: TrajectoryContributionFixture,
    ) -> None:
        base_acceptance = _acceptance(conversation_id, contribution)
        attempt_id = f"attempt-{suffix}"
        acceptance = replace(
            base_acceptance,
            user_message_id=f"user-{suffix}",
            assistant_message_id=f"assistant-{suffix}",
            preparation_id=f"preparation-{suffix}",
            attempt_id=attempt_id,
            frozen_authority=replace(
                base_acceptance.frozen_authority,
                attempt_id=attempt_id,
            ),
        )
        try:
            barrier.wait()
            with service.db.transaction(immediate=True) as cursor:
                service.console_dispatch_repository.insert_with_messages(
                    cursor,
                    acceptance,
                )
        except BaseException as exc:
            errors.append(exc)

    first_contribution = TrajectoryContributionFixture(("library_activity_a",))
    second_contribution = TrajectoryContributionFixture(("library_activity_b",))
    threads = (
        threading.Thread(
            target=accept,
            args=(first_service, "a", first_contribution),
        ),
        threading.Thread(
            target=accept,
            args=(second_service, "b", second_contribution),
        ),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "active dispatch checkpoint" in str(errors[0])
    assert sorted(
        first_contribution.allocated_sequences
        + second_contribution.allocated_sequences
    ) == [31]
    rows = first_service.db.get_connection().execute(
        "SELECT seq FROM message_trajectory_metadata "
        "WHERE event_kind LIKE 'library_activity_%' ORDER BY seq"
    ).fetchall()
    assert [tuple(row) for row in rows] == [(31,)]


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
    ("statement", "parameters"),
    [
        ("BEGIN IMMEDIATE", ()),
        ("COMMIT", ()),
        ("ROLLBACK", ()),
        ("SAVEPOINT contribution", ()),
        ("RELEASE contribution", ()),
        ("ATTACH DATABASE ':memory:' AS escaped", ()),
        ("DETACH DATABASE escaped", ()),
        ("UPDATE contribution_probe SET user_id = 'escaped'", ()),
        ("DELETE FROM contribution_probe", ()),
        ("INSERT INTO single_value_probe VALUES (?)", ("x",)),
        ("INSERT INTO single_value_probe(value) VALUES ('literal ?')", ()),
        ("INSERT INTO single_value_probe(value) VALUES ('literal') -- ?", ()),
        ("INSERT INTO single_value_probe(value) VALUES ('literal') /* ? */", ()),
        ("INSERT OR REPLACE INTO single_value_probe(value) VALUES (?)", ("x",)),
        ("INSERT OR IGNORE INTO single_value_probe(value) VALUES (?)", ("x",)),
        (
            "INSERT INTO single_value_probe(value) VALUES (?) "
            "ON CONFLICT(value) DO UPDATE SET value=excluded.value",
            ("x",),
        ),
        (
            "INSERT INTO single_value_probe(value) VALUES (?) "
            "ON CONFLICT(value) DO NOTHING",
            ("x",),
        ),
        ("INSERT INTO single_value_probe(value) SELECT ?", ("x",)),
        (
            "INSERT INTO single_value_probe(value) VALUES (?) RETURNING value",
            ("x",),
        ),
        ("INSERT INTO single_value_probe(value) VALUES (?), (?)", ("x", "y")),
        ('INSERT INTO "single_value_probe"(value) VALUES (?)', ("x",)),
        ('INSERT INTO single_value_probe("value") VALUES (?)', ("x",)),
        ("INSERT INTO main.single_value_probe(value) VALUES (?)", ("x",)),
        ("INSERT INTO ſingle_value_probe(value) VALUES (?)", ("x",)),
        ("INSERT INTO single_value_probe(value) VALUES (?); COMMIT", ("x",)),
    ],
)
def test_writer_rejects_every_statement_outside_the_exact_insert_values_grammar(
    tmp_path: Path,
    statement: str,
    parameters: tuple[object, ...],
) -> None:
    service, conversation_id = _service(tmp_path / "rejected-statement.sqlite")

    with pytest.raises(ValueError, match="parameterized INSERT"):
        with service.db.transaction(immediate=True) as cursor:
            service.console_dispatch_repository.insert_with_messages(
                cursor,
                _acceptance(
                    conversation_id,
                    StatementContribution(statement, parameters),
                ),
            )

    connection = service.db.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM console_dispatch_checkpoints"
    ).fetchone()[0] == 0


@pytest.mark.parametrize(
    ("statement", "parameters"),
    [
        ("INSERT INTO writer_probe(value, other) VALUES (?)", ("x",)),
        ("INSERT INTO writer_probe(value) VALUES (?, ?)", ("x", "y")),
        ("INSERT INTO writer_probe(value) VALUES (?)", ()),
        ("INSERT INTO writer_probe(value) VALUES (?)", ("x", "y")),
    ],
)
def test_writer_rejects_column_placeholder_and_execute_tuple_arity_mismatches(
    tmp_path: Path,
    statement: str,
    parameters: tuple[object, ...],
) -> None:
    service, conversation_id = _service(tmp_path / "execute-arity.sqlite")

    with pytest.raises(ValueError, match="same non-zero arity"):
        with service.db.transaction(immediate=True) as cursor:
            service.console_dispatch_repository.insert_with_messages(
                cursor,
                _acceptance(
                    conversation_id,
                    StatementContribution(statement, parameters),
                ),
            )

    connection = service.db.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM writer_probe").fetchone()[0] == 0


@pytest.mark.parametrize(
    "parameter_rows",
    [
        (),
        (("x",),),
        (("x", "y"), ("only-one",)),
        (("x", "y", "extra"),),
    ],
)
def test_writer_rejects_empty_or_wrong_arity_executemany_rows(
    tmp_path: Path,
    parameter_rows: tuple[tuple[object, ...], ...],
) -> None:
    service, conversation_id = _service(tmp_path / "executemany-arity.sqlite")
    contribution = BatchStatementContribution(
        "INSERT INTO writer_probe(value, other) VALUES (?, ?)",
        parameter_rows,
    )

    with pytest.raises(ValueError, match="non-empty rows of matching arity"):
        with service.db.transaction(immediate=True) as cursor:
            service.console_dispatch_repository.insert_with_messages(
                cursor,
                _acceptance(conversation_id, contribution),
            )

    connection = service.db.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM writer_probe").fetchone()[0] == 0


def test_writer_accepts_exact_insert_values_grammar_with_ordinary_whitespace(
    tmp_path: Path,
) -> None:
    service, conversation_id = _service(tmp_path / "ordinary-whitespace.sqlite")
    contribution = StatementContribution(
        "insert\ninto writer_probe\n( value, other )\nvalues\n( ?, ? )",
        ("first", "second"),
    )

    with service.db.transaction(immediate=True) as cursor:
        service.console_dispatch_repository.insert_with_messages(
            cursor,
            _acceptance(conversation_id, contribution),
        )

    row = service.db.get_connection().execute(
        "SELECT value, other FROM writer_probe"
    ).fetchone()
    assert tuple(row) == ("first", "second")
