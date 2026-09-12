"""Isolated ancestry decisions without store construction or schema migrations."""

import sqlite3
from contextlib import closing

import pytest

from Tests.Chat.test_console_dispatch_recovery import _acceptance
from tldw_chatbook.Chat.console_chat_models import ConsoleDispatchRecoveryKind
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpoint,
    ConsoleDispatchCheckpointState,
)
from tldw_chatbook.Chat.console_dispatch_repository import (
    ConsoleDispatchRepository,
    _RECONCILE_WRITE_NEEDED,
)


@pytest.mark.parametrize(
    "problem, allow_writes, expected_error",
    [
        (None, False, None),
        (None, True, None),
        ("cycle", True, "invalid_checkpoint_ancestry"),
        ("parent", True, "invalid_checkpoint_ancestry"),
        ("missing", True, "invalid_checkpoint_ancestry"),
        ("competing", True, "duplicate_active_path_owner"),
        ("invalid", True, "invalid_checkpoint"),
    ],
)
def test_stranded_ancestry_decision(monkeypatch, problem, allow_writes, expected_error):
    acceptance = _acceptance("conversation")
    checkpoint = ConsoleDispatchCheckpoint(
        assistant_message_id=acceptance.assistant_message_id,
        user_message_id=acceptance.user_message_id,
        conversation_id=acceptance.conversation_id,
        preparation_id=acceptance.preparation_id,
        attempt_id=acceptance.attempt_id,
        state=ConsoleDispatchCheckpointState.ACCEPTED,
        checkpoint_revision=1,
        user_message_version=1,
        assistant_message_version=1,
        origin="manual",
        queue_entry_id=None,
        frozen_authority=acceptance.frozen_authority,
        resolved_destination=acceptance.resolved_destination,
        reconstructability=acceptance.reconstructability,
    )
    repository = ConsoleDispatchRepository(None)
    # Payload validation and selected-path discovery have separate integration
    # coverage. Isolate the helper's traversal and write/no-write decisions here.
    monkeypatch.setattr(
        repository,
        "_checkpoint_from_row",
        lambda row: (None, "invalid_checkpoint")
        if problem == "invalid"
        else (checkpoint, None),
    )
    monkeypatch.setattr(
        repository,
        "_reconcile_checkpoint_free_owner",
        lambda *_: object() if problem == "competing" else None,
    )
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            connection.execute(
                "CREATE TABLE messages (id TEXT, conversation_id TEXT, deleted INTEGER, "
                "parent_message_id TEXT, provider_continuation_json TEXT, assistant_generation_state TEXT)"
            )
            connection.execute(
                "CREATE TABLE conversations (id TEXT, deleted INTEGER, active_leaf_message_id TEXT, "
                "active_leaf_before_message_id TEXT)"
            )
            connection.execute(
                "INSERT INTO conversations VALUES ('conversation', 0, 'old', 'before')"
            )
            connection.executemany(
                "INSERT INTO messages VALUES (?, 'conversation', 0, ?, NULL, NULL)",
                [
                    ("assistant-1", "wrong" if problem == "parent" else "user-1"),
                    (
                        "user-1",
                        "assistant-1"
                        if problem == "cycle"
                        else "missing"
                        if problem == "missing"
                        else None,
                    ),
                ],
            )
        with connection, closing(connection.cursor()) as cursor:
            result = repository._reconcile_stranded_checkpoint(
                cursor,
                "conversation",
                {"assistant_message_id": "assistant-1"},
                allow_writes=allow_writes,
            )
        actual_cursor = tuple(
            connection.execute(
                "SELECT active_leaf_message_id, active_leaf_before_message_id FROM conversations"
            ).fetchone()
        )
        if expected_error:
            assert result.kind is ConsoleDispatchRecoveryKind.QUARANTINED
            assert result.error_code == expected_error
            assert result.actions == ()
        elif allow_writes:
            assert result.checkpoint == checkpoint
            assert result.kind is ConsoleDispatchRecoveryKind.ACCEPTED
        else:
            assert result is _RECONCILE_WRITE_NEEDED
        assert actual_cursor == (
            ("assistant-1", None)
            if allow_writes and expected_error is None
            else ("old", "before")
        )
