from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    CHECKPOINT_AUTHORITY_MAX_BYTES,
    CHECKPOINT_DESTINATION_MAX_BYTES,
    CHECKPOINT_RECONSTRUCTABILITY_MAX_BYTES,
    ConsoleAssistantSettlement,
    ConsoleContinuationHandoff,
    ConsoleDispatchCheckpoint,
    ConsoleDispatchCheckpointState,
    ConsoleDispatchCheckpointValidationError,
    ConsoleDispatchReconstructability,
    ConsoleDispatchResultStatus,
    ConsoleDispatchTransition,
    ConsoleDurableTurnAcceptance,
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
    dump_console_dispatch_reconstructability_json,
    dump_console_resolved_destination_json,
    dump_console_turn_library_authority_json,
    parse_console_dispatch_reconstructability_json,
    parse_console_resolved_destination_json,
    parse_console_turn_library_authority_json,
)
from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash


def _authority(*, attempt_id: str = "attempt-1") -> ConsoleTurnLibraryAuthority:
    return ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
            policy_revision=7,
            source="durable",
        ),
        direct_library_tools=True,
        source_types=("notes", "media", "conversations"),
        scope_snapshot=ConsoleLibraryItemScopeSnapshot(
            note_ids=("note-1",),
            media_ids=("media-1",),
            conversations_allowed=False,
        ),
        provider_intent=ConsoleProviderIntent(
            provider="openai",
            model="gpt-test",
            endpoint="https://api.example.test/v1",
        ),
        attempt_id=attempt_id,
    )


def _destination() -> ConsoleResolvedDestination:
    return ConsoleResolvedDestination(
        provider="openai",
        model="gpt-test",
        endpoint_identity="https://api.example.test:443",
        egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
    )


def _reconstructability() -> ConsoleDispatchReconstructability:
    return ConsoleDispatchReconstructability(
        attachments_reconstructable=True,
        evidence_reconstructable=False,
        prefill_reconstructable=True,
        opaque_reference="opaque:opaque-1",
    )


def _acceptance(conversation_id: str, *, suffix: str = "1") -> ConsoleDurableTurnAcceptance:
    return ConsoleDurableTurnAcceptance(
        conversation_id=conversation_id,
        user_message_id=f"user-{suffix}",
        assistant_message_id=f"assistant-{suffix}",
        parent_message_id=None,
        user_content="hello",
        attachments=(),
        preparation_id=f"preparation-{suffix}",
        attempt_id=f"attempt-{suffix}",
        origin="manual",
        queue_entry_id=None,
        frozen_authority=_authority(attempt_id=f"attempt-{suffix}"),
        resolved_destination=_destination(),
        reconstructability=_reconstructability(),
        contributions=(),
    )


def _db_and_conversation(path: Path) -> tuple[CharactersRAGDB, str]:
    db = CharactersRAGDB(path, client_id="dispatch-test")
    conversation_id = db.add_conversation({"title": "dispatch"})
    assert conversation_id is not None
    return db, conversation_id


def _insert(
    db: CharactersRAGDB,
    repository: ConsoleDispatchRepository,
    acceptance: ConsoleDurableTurnAcceptance,
):
    with db.transaction(immediate=True) as cursor:
        return repository.insert_with_messages(cursor, acceptance)


def _start_dispatch(
    repository: ConsoleDispatchRepository,
    checkpoint: ConsoleDispatchCheckpoint,
) -> ConsoleDispatchCheckpoint:
    result = repository.cas_state(
        ConsoleDispatchTransition(
            assistant_message_id=checkpoint.assistant_message_id,
            expected_state=ConsoleDispatchCheckpointState.ACCEPTED,
            expected_checkpoint_revision=1,
            expected_user_message_version=1,
            expected_assistant_message_version=1,
            new_state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
            new_attempt_id="attempt-dispatch",
        )
    )
    assert result.status is ConsoleDispatchResultStatus.COMMITTED
    assert result.checkpoint is not None
    return result.checkpoint


def _active_continuation_json() -> str:
    raw = json.dumps(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "active",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": ["private reasoning"],
                    "calls": [
                        {
                            "call_id": "call-1",
                            "name": "calculator",
                            "arguments": "{\"expression\":\"2+2\"}",
                            "state": "pending",
                        }
                    ],
                }
            ],
        }
    )
    canonical = dump_provider_continuation_json(parse_provider_continuation_json(raw))
    assert canonical is not None
    return canonical


def test_checkpoint_codecs_pin_exact_json_keys_types_and_order() -> None:
    authority_json = dump_console_turn_library_authority_json(_authority())
    destination_json = dump_console_resolved_destination_json(_destination())
    reconstructability_json = dump_console_dispatch_reconstructability_json(
        _reconstructability()
    )

    assert authority_json == (
        '{"policy":{"auto_retrieve":"automatic","assistant_access":"allowed",'
        '"policy_revision":7,"source":"durable","error_code":null},'
        '"direct_library_tools":true,"source_types":["notes","media","conversations"],'
        '"scope_snapshot":{"note_ids":["note-1"],"media_ids":["media-1"],'
        '"conversations_allowed":false},"provider_intent":{"provider":"openai",'
        '"model":"gpt-test","endpoint":"https://api.example.test/v1"},'
        '"attempt_id":"attempt-1"}'
    )
    assert destination_json == (
        '{"provider":"openai","model":"gpt-test",'
        '"endpoint_identity":"https://api.example.test:443",'
        '"egress_class":"public_network"}'
    )
    assert reconstructability_json == (
        '{"attachments_reconstructable":true,"evidence_reconstructable":false,'
        '"prefill_reconstructable":true,"opaque_reference":"opaque:opaque-1"}'
    )
    assert parse_console_turn_library_authority_json(authority_json) == _authority()
    assert parse_console_resolved_destination_json(destination_json) == _destination()
    assert (
        parse_console_dispatch_reconstructability_json(reconstructability_json)
        == _reconstructability()
    )


@pytest.mark.parametrize(
    ("parser", "payload"),
    [
        (
            parse_console_turn_library_authority_json,
            '{"policy":{},"direct_library_tools":true,"source_types":[],'
            '"scope_snapshot":{},"provider_intent":{},"attempt_id":"a",'
            '"request_text":"secret draft"}',
        ),
        (
            parse_console_dispatch_reconstructability_json,
            '{"attachments_reconstructable":true,"evidence_reconstructable":true,'
            '"prefill_reconstructable":true,"opaque_reference":null,'
            '"source_snippets":["private body"]}',
        ),
        (
            parse_console_resolved_destination_json,
            '{"provider":"openai","model":"m","endpoint_identity":"e",'
            '"egress_class":"public_network","api_key":"secret"}',
        ),
        (
            parse_console_resolved_destination_json,
            '{"provider":"openai","model":"m",'
            '"endpoint_identity":"https://api.example.test/v1?api_key=secret",'
            '"egress_class":"public_network"}',
        ),
    ],
)
def test_checkpoint_codecs_reject_request_text_source_snippets_and_credentials(
    parser: object, payload: str
) -> None:
    with pytest.raises(ConsoleDispatchCheckpointValidationError):
        parser(payload)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("dumper", "value", "cap"),
    [
        (
            dump_console_turn_library_authority_json,
            _authority(attempt_id="x" * CHECKPOINT_AUTHORITY_MAX_BYTES),
            CHECKPOINT_AUTHORITY_MAX_BYTES,
        ),
        (
            dump_console_resolved_destination_json,
            replace(
                _destination(),
                endpoint_identity="x" * CHECKPOINT_DESTINATION_MAX_BYTES,
            ),
            CHECKPOINT_DESTINATION_MAX_BYTES,
        ),
        (
            dump_console_dispatch_reconstructability_json,
            replace(
                _reconstructability(),
                opaque_reference="opaque:"
                + "x" * CHECKPOINT_RECONSTRUCTABILITY_MAX_BYTES,
            ),
            CHECKPOINT_RECONSTRUCTABILITY_MAX_BYTES,
        ),
    ],
)
def test_checkpoint_codecs_enforce_utf8_byte_caps(
    dumper: object, value: object, cap: int
) -> None:
    assert cap in {4096, 2048}
    with pytest.raises(ConsoleDispatchCheckpointValidationError):
        dumper(value)  # type: ignore[operator]


@pytest.mark.parametrize(
    "authority",
    [
        replace(
            _authority(),
            policy=ConsoleLibraryPolicySnapshot(
                ConsoleAutoRetrieve.AUTOMATIC,
                ConsoleAssistantLibraryAccess.ALLOWED,
                None,
                "unavailable",
                "policy_read_error",
            ),
        ),
        replace(
            _authority(),
            policy=ConsoleLibraryPolicySnapshot(
                ConsoleAutoRetrieve.AUTOMATIC,
                ConsoleAssistantLibraryAccess.ALLOWED,
                None,
                "durable",
            ),
        ),
        replace(_authority(), source_types=("notes", "private_source")),
        replace(
            _authority(),
            scope_snapshot=ConsoleLibraryItemScopeSnapshot(
                note_ids=("private note body with spaces",),
                media_ids=(),
                conversations_allowed=False,
            ),
        ),
        replace(
            _authority(),
            policy=ConsoleLibraryPolicySnapshot(
                ConsoleAutoRetrieve.NEVER,
                ConsoleAssistantLibraryAccess.BLOCKED,
                None,
                "unavailable",
                "API key sk-secret",
            ),
        ),
    ],
)
def test_authority_codec_rejects_fail_open_and_free_form_allowed_fields(
    authority: ConsoleTurnLibraryAuthority,
) -> None:
    with pytest.raises(ConsoleDispatchCheckpointValidationError):
        dump_console_turn_library_authority_json(authority)


def test_reconstructability_codec_requires_an_explicit_opaque_reference_format() -> None:
    with pytest.raises(ConsoleDispatchCheckpointValidationError):
        dump_console_dispatch_reconstructability_json(
            ConsoleDispatchReconstructability(
                attachments_reconstructable=True,
                evidence_reconstructable=False,
                prefill_reconstructable=False,
                opaque_reference="raw source snippet with spaces",
            )
        )


def test_insert_and_read_validate_roles_conversation_versions_and_state(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "ownership.sqlite")
    repository = ConsoleDispatchRepository(db)
    inserted = _insert(db, repository, _acceptance(conversation_id))

    assert inserted.state is ConsoleDispatchCheckpointState.ACCEPTED
    assert inserted.checkpoint_revision == 1
    assert inserted.user_message_version == 1
    assert inserted.assistant_message_version == 1
    read = repository.read_for_session(conversation_id)
    assert read.status is ConsoleDispatchResultStatus.COMMITTED
    assert read.checkpoint == inserted

    rows = db.get_connection().execute(
        "SELECT id, role, assistant_generation_state, version, deleted "
        "FROM messages ORDER BY id"
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("assistant-1", "assistant", "accepted", 1, 0),
        ("user-1", "user", None, 1, 0),
    ]


@pytest.mark.parametrize("corruption", ["bad_role", "cross_conversation", "bad_state"])
def test_read_quarantines_invalid_ownership(tmp_path: Path, corruption: str) -> None:
    db, conversation_id = _db_and_conversation(
        tmp_path / f"ownership-{corruption}.sqlite"
    )
    repository = ConsoleDispatchRepository(db)
    _insert(db, repository, _acceptance(conversation_id))
    connection = db.get_connection()
    if corruption == "bad_role":
        connection.execute(
            "UPDATE messages SET role = ? WHERE id = ?", ("tool", "assistant-1")
        )
    elif corruption == "cross_conversation":
        other_id = db.add_conversation({"title": "other"})
        assert other_id is not None
        connection.execute(
            "UPDATE messages SET conversation_id = ? WHERE id = ?",
            (other_id, "user-1"),
        )
    else:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE console_dispatch_checkpoints SET state = ?", ("invented",)
        )
    connection.commit()

    result = repository.read_for_session(conversation_id)

    assert result.status is ConsoleDispatchResultStatus.QUARANTINED
    assert result.checkpoint is None
    assert result.error_code is not None


def test_read_quarantines_duplicate_active_path_owners(tmp_path: Path) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "duplicates.sqlite")
    repository = ConsoleDispatchRepository(db)
    first = _insert(db, repository, _acceptance(conversation_id, suffix="1"))
    second_acceptance = replace(
        _acceptance(conversation_id, suffix="2"),
        parent_message_id=first.assistant_message_id,
    )
    second = _insert(db, repository, second_acceptance)
    db.set_conversation_active_leaf(conversation_id, second.assistant_message_id)

    result = repository.read_for_session(conversation_id)

    assert result.status is ConsoleDispatchResultStatus.QUARANTINED
    assert result.checkpoint is None
    assert result.error_code == "duplicate_active_path_owner"


def test_read_considers_only_checkpoint_owners_on_the_selected_active_lineage(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "active-lineage.sqlite")
    repository = ConsoleDispatchRepository(db)
    first = _insert(db, repository, _acceptance(conversation_id, suffix="1"))
    second = _insert(db, repository, _acceptance(conversation_id, suffix="2"))

    db.set_conversation_active_leaf(conversation_id, first.assistant_message_id)
    first_read = repository.read_for_session(conversation_id)
    db.set_conversation_active_leaf(conversation_id, second.assistant_message_id)
    second_read = repository.read_for_session(conversation_id)

    assert first_read.status is ConsoleDispatchResultStatus.COMMITTED
    assert first_read.checkpoint == first
    assert second_read.status is ConsoleDispatchResultStatus.COMMITTED
    assert second_read.checkpoint == second


def test_insert_is_not_a_generic_upsert(tmp_path: Path) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "no-upsert.sqlite")
    repository = ConsoleDispatchRepository(db)
    acceptance = _acceptance(conversation_id)
    original = _insert(db, repository, acceptance)

    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            db,
            repository,
            replace(
                acceptance,
                attempt_id="overwritten-attempt",
                frozen_authority=replace(
                    acceptance.frozen_authority,
                    attempt_id="overwritten-attempt",
                ),
            ),
        )

    assert repository.read_for_session(conversation_id).checkpoint == original


def test_acceptance_requires_the_checkpoint_and_frozen_authority_attempt_to_match(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "attempt-mismatch.sqlite")
    repository = ConsoleDispatchRepository(db)
    acceptance = replace(_acceptance(conversation_id), attempt_id="other-attempt")

    with pytest.raises(ConsoleDispatchCheckpointValidationError):
        _insert(db, repository, acceptance)

    assert db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


@pytest.mark.parametrize(
    "corruption",
    ["preparation_id", "attempt_id", "authority_attempt", "assistant_message_id"],
)
def test_read_quarantines_malformed_or_mismatched_checkpoint_identity(
    tmp_path: Path,
    corruption: str,
) -> None:
    db, conversation_id = _db_and_conversation(
        tmp_path / f"identity-{corruption}.sqlite"
    )
    repository = ConsoleDispatchRepository(db)
    inserted = _insert(db, repository, _acceptance(conversation_id))
    connection = db.get_connection()
    db.set_conversation_active_leaf(conversation_id, inserted.assistant_message_id)
    if corruption == "preparation_id":
        connection.execute(
            "UPDATE console_dispatch_checkpoints SET preparation_id = ?",
            ("private draft text",),
        )
    elif corruption == "attempt_id":
        connection.execute(
            "UPDATE console_dispatch_checkpoints SET attempt_id = ?",
            ("",),
        )
    elif corruption == "authority_attempt":
        connection.execute(
            "UPDATE console_dispatch_checkpoints SET attempt_id = ?",
            ("other-attempt",),
        )
    else:
        malformed_id = "assistant private body"
        connection.commit()
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "UPDATE console_dispatch_checkpoints SET assistant_message_id = ?",
            (malformed_id,),
        )
        connection.execute(
            "UPDATE messages SET id = ? WHERE id = ?",
            (malformed_id, inserted.assistant_message_id),
        )
        connection.execute(
            "UPDATE conversations SET active_leaf_message_id = ? WHERE id = ?",
            (malformed_id, conversation_id),
        )
    connection.commit()

    result = repository.read_for_session(conversation_id)

    assert result.status is ConsoleDispatchResultStatus.QUARANTINED
    assert result.checkpoint is None


@pytest.mark.parametrize("operation", ["read", "cas", "settle", "handoff"])
def test_soft_deleted_conversation_cannot_recover_or_mutate_dispatch_ownership(
    tmp_path: Path,
    operation: str,
) -> None:
    db, conversation_id = _db_and_conversation(
        tmp_path / f"deleted-{operation}.sqlite"
    )
    repository = ConsoleDispatchRepository(db)
    inserted = _insert(db, repository, _acceptance(conversation_id))
    owner = (
        _start_dispatch(repository, inserted) if operation == "handoff" else inserted
    )
    assert db.soft_delete_conversation(conversation_id, expected_version=1) is True

    if operation == "read":
        result = repository.read_for_session(conversation_id)
        assert result.status is ConsoleDispatchResultStatus.NOT_FOUND
    elif operation == "cas":
        result = repository.cas_state(
            ConsoleDispatchTransition(
                assistant_message_id=owner.assistant_message_id,
                expected_state=ConsoleDispatchCheckpointState.ACCEPTED,
                expected_checkpoint_revision=1,
                expected_user_message_version=1,
                expected_assistant_message_version=1,
                new_state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
                new_attempt_id="attempt-after-delete",
            )
        )
        assert result.status is ConsoleDispatchResultStatus.CONFLICT
    elif operation == "settle":
        result = repository.settle_with_assistant(
            ConsoleAssistantSettlement(
                assistant_message_id=owner.assistant_message_id,
                expected_checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
                expected_checkpoint_revision=1,
                expected_user_message_version=1,
                expected_assistant_message_version=1,
                terminal_state="discarded",
                content="discarded",
                metadata_json=None,
            )
        )
        assert result.status is ConsoleDispatchResultStatus.CONFLICT
    else:
        result = repository.handoff_to_provider_continuation(
            ConsoleContinuationHandoff(
                assistant_message_id=owner.assistant_message_id,
                expected_checkpoint_revision=2,
                expected_user_message_version=1,
                expected_assistant_message_version=2,
                provider_continuation_json=_active_continuation_json(),
            )
        )
        assert result.status is ConsoleDispatchResultStatus.CONFLICT

    assistant = db.get_connection().execute(
        "SELECT assistant_generation_state, version, deleted FROM messages WHERE id = ?",
        (owner.assistant_message_id,),
    ).fetchone()
    assert tuple(assistant) == (
        owner.state.value,
        owner.assistant_message_version,
        0,
    )
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM console_dispatch_checkpoints"
    ).fetchone()[0] == 1


@pytest.mark.parametrize("boundary", ["user", "assistant", "checkpoint"])
def test_insert_failure_at_each_write_boundary_rolls_back(
    tmp_path: Path, boundary: str
) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / f"insert-{boundary}.sqlite")
    repository = ConsoleDispatchRepository(db)
    connection = db.get_connection()
    if boundary in {"user", "assistant"}:
        connection.execute(
            f"""
            CREATE TRIGGER fail_{boundary} BEFORE INSERT ON messages
            WHEN NEW.role = '{boundary if boundary == 'user' else 'assistant'}'
            BEGIN SELECT RAISE(ABORT, 'injected {boundary} failure'); END
            """
        )
    else:
        connection.execute(
            """
            CREATE TRIGGER fail_checkpoint BEFORE INSERT ON console_dispatch_checkpoints
            BEGIN SELECT RAISE(ABORT, 'injected checkpoint failure'); END
            """
        )
    connection.commit()

    with pytest.raises(sqlite3.IntegrityError):
        _insert(db, repository, _acceptance(conversation_id))

    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM console_dispatch_checkpoints"
    ).fetchone()[0] == 0


def test_acceptance_persists_the_full_user_attachment_set_atomically(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "attachments.sqlite")
    repository = ConsoleDispatchRepository(db)
    acceptance = replace(
        _acceptance(conversation_id),
        attachments=(
            {
                "position": 0,
                "data": b"first-image",
                "mime_type": "image/png",
                "display_name": "first.png",
            },
            {
                "position": 1,
                "data": b"second-image",
                "mime_type": "image/webp",
                "display_name": "second.webp",
            },
        ),
    )

    checkpoint = _insert(db, repository, acceptance)

    user = db.get_message_by_id(checkpoint.user_message_id)
    assert user is not None
    assert (user["image_data"], user["image_mime_type"]) == (
        b"first-image",
        "image/png",
    )
    extras = db.get_attachments_for_messages([checkpoint.user_message_id])
    assert extras[checkpoint.user_message_id] == [
        {
            "position": 1,
            "data": b"second-image",
            "mime_type": "image/webp",
            "display_name": "second.webp",
        }
    ]


def test_attachment_sidecar_failure_rolls_back_the_entire_acceptance(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "attachment-rollback.sqlite")
    repository = ConsoleDispatchRepository(db)
    connection = db.get_connection()
    connection.execute(
        """
        CREATE TRIGGER fail_dispatch_attachment
        BEFORE INSERT ON message_attachments
        BEGIN SELECT RAISE(ABORT, 'injected attachment failure'); END
        """
    )
    connection.commit()
    acceptance = replace(
        _acceptance(conversation_id),
        attachments=(
            {
                "position": 1,
                "data": b"second-image",
                "mime_type": "image/webp",
                "display_name": "second.webp",
            },
        ),
    )

    with pytest.raises(sqlite3.IntegrityError):
        _insert(db, repository, acceptance)

    assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM message_attachments"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM console_dispatch_checkpoints"
    ).fetchone()[0] == 0


def test_nonempty_attachments_cannot_claim_unreconstructable_retry_state(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "attachment-flags.sqlite")
    repository = ConsoleDispatchRepository(db)
    acceptance = replace(
        _acceptance(conversation_id),
        attachments=(
            {
                "position": 0,
                "data": b"image",
                "mime_type": "image/png",
            },
        ),
        reconstructability=replace(
            _reconstructability(),
            attachments_reconstructable=False,
        ),
    )

    with pytest.raises(ConsoleDispatchCheckpointValidationError):
        _insert(db, repository, acceptance)

    assert db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


@pytest.mark.parametrize("boundary", ["assistant_state", "checkpoint_state"])
def test_state_cas_failure_rolls_back_both_owner_and_checkpoint(
    tmp_path: Path, boundary: str
) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / f"cas-{boundary}.sqlite")
    repository = ConsoleDispatchRepository(db)
    inserted = _insert(db, repository, _acceptance(conversation_id))
    connection = db.get_connection()
    table = "messages" if boundary == "assistant_state" else "console_dispatch_checkpoints"
    connection.execute(
        f"""
        CREATE TRIGGER fail_{boundary} BEFORE UPDATE ON {table}
        BEGIN SELECT RAISE(ABORT, 'injected state failure'); END
        """
    )
    connection.commit()
    transition = ConsoleDispatchTransition(
        assistant_message_id=inserted.assistant_message_id,
        expected_state=ConsoleDispatchCheckpointState.ACCEPTED,
        expected_checkpoint_revision=1,
        expected_user_message_version=1,
        expected_assistant_message_version=1,
        new_state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
        new_attempt_id="attempt-2",
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.cas_state(transition)

    checkpoint = repository.read_for_session(conversation_id).checkpoint
    assert checkpoint == inserted
    assistant = db.get_message_by_id(inserted.assistant_message_id)
    assert assistant is not None
    assert (assistant["assistant_generation_state"], assistant["version"]) == (
        "accepted",
        1,
    )


@pytest.mark.parametrize(
    "mismatch",
    ["checkpoint_revision", "user_version", "assistant_version", "assistant_state", "deleted"],
)
def test_state_cas_requires_every_expected_owner_predicate(
    tmp_path: Path, mismatch: str
) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / f"cas-{mismatch}.sqlite")
    repository = ConsoleDispatchRepository(db)
    inserted = _insert(db, repository, _acceptance(conversation_id))
    transition = ConsoleDispatchTransition(
        assistant_message_id=inserted.assistant_message_id,
        expected_state=ConsoleDispatchCheckpointState.ACCEPTED,
        expected_checkpoint_revision=2 if mismatch == "checkpoint_revision" else 1,
        expected_user_message_version=2 if mismatch == "user_version" else 1,
        expected_assistant_message_version=2 if mismatch == "assistant_version" else 1,
        new_state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
        new_attempt_id="attempt-2",
    )
    connection = db.get_connection()
    if mismatch == "assistant_state":
        connection.execute(
            "UPDATE messages SET assistant_generation_state = ? WHERE id = ?",
            ("dispatch_started", inserted.assistant_message_id),
        )
        connection.commit()
    elif mismatch == "deleted":
        connection.execute(
            "UPDATE messages SET deleted = 1 WHERE id = ?", (inserted.user_message_id,)
        )
        connection.commit()

    result = repository.cas_state(transition)

    assert result.status is ConsoleDispatchResultStatus.CONFLICT
    row = connection.execute(
        "SELECT state, checkpoint_revision FROM console_dispatch_checkpoints"
    ).fetchone()
    assert tuple(row) == ("accepted", 1)


def test_state_cas_commits_versions_state_hash_and_sync_intent(tmp_path: Path) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "cas-success.sqlite")
    repository = ConsoleDispatchRepository(db)
    inserted = _insert(db, repository, _acceptance(conversation_id))

    result = repository.cas_state(
        ConsoleDispatchTransition(
            assistant_message_id=inserted.assistant_message_id,
            expected_state=ConsoleDispatchCheckpointState.ACCEPTED,
            expected_checkpoint_revision=1,
            expected_user_message_version=1,
            expected_assistant_message_version=1,
            new_state=ConsoleDispatchCheckpointState.DISPATCH_STARTED,
            new_attempt_id="attempt-2",
        )
    )

    expected_hash = canonical_payload_hash(
        {
            "assistant_generation_state": "dispatch_started",
            "content": "",
            "role": "assistant",
        }
    )
    assert result.status is ConsoleDispatchResultStatus.COMMITTED
    assert result.checkpoint is not None
    assert result.checkpoint.checkpoint_revision == 2
    assert result.checkpoint.assistant_message_version == 2
    assert result.committed_message_version == 2
    assert result.committed_payload_hash == expected_hash
    assert db.read_committed_chat_sync_intent(
        message_id=inserted.assistant_message_id,
        message_version=2,
        payload_hash=expected_hash,
    ) is not None


@pytest.mark.parametrize("boundary", ["terminal_content", "sync_intent", "checkpoint_delete"])
def test_terminal_settlement_failure_at_each_write_boundary_rolls_back(
    tmp_path: Path, boundary: str
) -> None:
    db, conversation_id = _db_and_conversation(
        tmp_path / f"settle-{boundary}.sqlite"
    )
    repository = ConsoleDispatchRepository(db)
    inserted = _insert(db, repository, _acceptance(conversation_id))
    connection = db.get_connection()
    if boundary == "terminal_content":
        trigger = """
            CREATE TRIGGER fail_terminal_content BEFORE UPDATE ON messages
            WHEN NEW.assistant_generation_state = 'complete'
            BEGIN SELECT RAISE(ABORT, 'injected terminal failure'); END
        """
    elif boundary == "sync_intent":
        trigger = """
            CREATE TRIGGER fail_sync_intent BEFORE INSERT ON sync_log
            WHEN NEW.entity = 'messages' AND NEW.version = 2
            BEGIN SELECT RAISE(ABORT, 'injected sync failure'); END
        """
    else:
        trigger = """
            CREATE TRIGGER fail_checkpoint_delete BEFORE DELETE ON console_dispatch_checkpoints
            BEGIN SELECT RAISE(ABORT, 'injected delete failure'); END
        """
    connection.execute(trigger)
    connection.commit()

    with pytest.raises(sqlite3.IntegrityError):
        repository.settle_with_assistant(
            ConsoleAssistantSettlement(
                assistant_message_id=inserted.assistant_message_id,
                expected_checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
                expected_checkpoint_revision=1,
                expected_user_message_version=1,
                expected_assistant_message_version=1,
                terminal_state="complete",
                content="finished",
                metadata_json='{"finish_reason":"stop"}',
            )
        )

    assistant = db.get_message_by_id(inserted.assistant_message_id)
    assert assistant is not None
    assert (
        assistant["content"],
        assistant["assistant_generation_state"],
        assistant["version"],
    ) == ("", "accepted", 1)
    assert repository.read_for_session(conversation_id).checkpoint == inserted


def test_terminal_settlement_is_atomic_and_returns_committed_proof(tmp_path: Path) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "settle.sqlite")
    repository = ConsoleDispatchRepository(db)
    inserted = _insert(db, repository, _acceptance(conversation_id))

    result = repository.settle_with_assistant(
        ConsoleAssistantSettlement(
            assistant_message_id=inserted.assistant_message_id,
            expected_checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
            expected_checkpoint_revision=1,
            expected_user_message_version=1,
            expected_assistant_message_version=1,
            terminal_state="complete",
            content="finished",
            metadata_json='{"finish_reason":"stop"}',
        )
    )

    expected_hash = canonical_payload_hash(
        {
            "assistant_generation_state": "complete",
            "content": "finished",
            "role": "assistant",
        }
    )
    assert result.status is ConsoleDispatchResultStatus.COMMITTED
    assert result.checkpoint is None
    assert result.committed_message_version == 2
    assert result.committed_payload_hash == expected_hash
    assert repository.read_for_session(conversation_id).status is (
        ConsoleDispatchResultStatus.NOT_FOUND
    )
    assistant = db.get_message_by_id(inserted.assistant_message_id)
    assert assistant is not None
    assert (
        assistant["content"],
        assistant["metadata_json"],
        assistant["assistant_generation_state"],
        assistant["version"],
        assistant["deleted"],
    ) == ("finished", '{"finish_reason":"stop"}', "complete", 2, 0)
    assert db.read_committed_chat_sync_intent(
        message_id=inserted.assistant_message_id,
        message_version=2,
        payload_hash=expected_hash,
    ) is not None


@pytest.mark.parametrize("boundary", ["continuation_write", "handoff_delete"])
def test_continuation_handoff_failure_rolls_back_both_owners(
    tmp_path: Path, boundary: str
) -> None:
    db, conversation_id = _db_and_conversation(
        tmp_path / f"handoff-{boundary}.sqlite"
    )
    repository = ConsoleDispatchRepository(db)
    inserted = _insert(db, repository, _acceptance(conversation_id))
    started = _start_dispatch(repository, inserted)
    connection = db.get_connection()
    table = "messages" if boundary == "continuation_write" else "console_dispatch_checkpoints"
    operation = "UPDATE" if boundary == "continuation_write" else "DELETE"
    connection.execute(
        f"""
        CREATE TRIGGER fail_{boundary} BEFORE {operation} ON {table}
        BEGIN SELECT RAISE(ABORT, 'injected handoff failure'); END
        """
    )
    connection.commit()

    with pytest.raises(sqlite3.IntegrityError):
        repository.handoff_to_provider_continuation(
            ConsoleContinuationHandoff(
                assistant_message_id=inserted.assistant_message_id,
                expected_checkpoint_revision=2,
                expected_user_message_version=1,
                expected_assistant_message_version=2,
                provider_continuation_json=_active_continuation_json(),
            )
        )

    assistant = db.get_message_by_id(inserted.assistant_message_id)
    assert assistant is not None
    assert (
        assistant["provider_continuation_json"],
        assistant["assistant_generation_state"],
        assistant["version"],
    ) == (None, "dispatch_started", 2)
    assert repository.read_for_session(conversation_id).checkpoint == started


def test_continuation_handoff_rejects_an_owner_that_has_not_started_dispatch(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "handoff-accepted.sqlite")
    repository = ConsoleDispatchRepository(db)
    inserted = _insert(db, repository, _acceptance(conversation_id))

    result = repository.handoff_to_provider_continuation(
        ConsoleContinuationHandoff(
            assistant_message_id=inserted.assistant_message_id,
            expected_checkpoint_revision=1,
            expected_user_message_version=1,
            expected_assistant_message_version=1,
            provider_continuation_json=_active_continuation_json(),
        )
    )

    assert result.status is ConsoleDispatchResultStatus.CONFLICT
    assert repository.read_for_session(conversation_id).checkpoint == inserted


def test_continuation_handoff_atomically_transfers_ownership_and_sync_intent(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_and_conversation(tmp_path / "handoff.sqlite")
    repository = ConsoleDispatchRepository(db)
    inserted = _insert(db, repository, _acceptance(conversation_id))
    _start_dispatch(repository, inserted)
    continuation_json = _active_continuation_json()

    result = repository.handoff_to_provider_continuation(
        ConsoleContinuationHandoff(
            assistant_message_id=inserted.assistant_message_id,
            expected_checkpoint_revision=2,
            expected_user_message_version=1,
            expected_assistant_message_version=2,
            provider_continuation_json=continuation_json,
        )
    )

    expected_hash = canonical_payload_hash(
        {
            "assistant_generation_state": "continuation_active",
            "content": "",
            "provider_continuation_json": continuation_json,
            "role": "assistant",
        }
    )
    assert result.status is ConsoleDispatchResultStatus.COMMITTED
    assert result.checkpoint is None
    assert result.committed_message_version == 3
    assert result.committed_payload_hash == expected_hash
    assistant = db.get_message_by_id(inserted.assistant_message_id)
    assert assistant is not None
    assert (
        assistant["content"],
        assistant["provider_continuation_json"],
        assistant["assistant_generation_state"],
        assistant["version"],
        assistant["deleted"],
    ) == ("", continuation_json, "continuation_active", 3, 0)
    assert repository.read_for_session(conversation_id).status is (
        ConsoleDispatchResultStatus.NOT_FOUND
    )
    assert db.read_committed_chat_sync_intent(
        message_id=inserted.assistant_message_id,
        message_version=3,
        payload_hash=expected_hash,
    ) is not None
