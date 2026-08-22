"""V32 -> current local Console context policy and memory ownership."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import (
    open_current_chachanotes_from_legacy,
)

from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextBudgetMode,
    ContextCompactionMode,
    ContextCompactionRepresentation,
)
from tldw_chatbook.Chat.console_context_repository import (
    AuxiliaryAttemptStart,
    AuxiliaryAttemptStatus,
    AuxiliaryPricingProvenance,
    ConsoleContextRepository,
    ConsoleMemoryRecord,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


SCHEMA_NAME = "rag_char_chat_schema"
EXPECTED_TABLES = {
    "console_conversation_context_policy",
    "console_conversation_memories",
    "console_auxiliary_attempts",
}


def _version(db: CharactersRAGDB) -> int:
    row = (
        db.get_connection()
        .execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (SCHEMA_NAME,),
        )
        .fetchone()
    )
    return int(row[0])


def _seed_v32_database(path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    with monkeypatch.context() as v32_patch:
        v32_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 32)
        db = CharactersRAGDB(path, client_id="v32-seed")
        conversation_id = db.add_conversation({"title": "legacy summary"})
        boundary_id = "legacy-boundary"
        timestamp = "2026-01-01T00:00:00+00:00"
        with db.transaction() as transaction:
            transaction.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, sender, content, timestamp,
                    last_modified, client_id, version, deleted, role
                ) VALUES (?, ?, 'user', 'old turn', ?, ?, 'v32-seed', 1, 0, 'user')
                """,
                (boundary_id, conversation_id, timestamp, timestamp),
            )
        db.set_conversation_context_summary(
            conversation_id, "legacy recap", boundary_id
        )
        assert _version(db) == 32
        db.close_connection()
    return conversation_id, boundary_id


def _seed_v33_database(path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    with monkeypatch.context() as v33_patch:
        v33_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 33)
        db = CharactersRAGDB(path, client_id="v33-seed")
        conversation_id = db.add_conversation({"title": "v33 policy"})
        db.get_connection().execute(
            """
            INSERT INTO console_conversation_context_policy(
                conversation_id, budget_mode, custom_budget_tokens,
                compaction_mode, policy_revision
            ) VALUES (?, 'custom', 12000, 'automatic', 1)
            """,
            (conversation_id,),
        )
        db.get_connection().commit()
        assert _version(db) == 33
        db.close_connection()
    return conversation_id


def test_fresh_database_reaches_current_with_local_tables(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="fresh")
    rows = (
        db.get_connection()
        .execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        .fetchall()
    )
    table_names = {str(row[0]) for row in rows}

    assert _version(db) == db._CURRENT_SCHEMA_VERSION
    assert EXPECTED_TABLES <= table_names

    sync_triggers = (
        db.get_connection()
        .execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE '%sync%'"
        )
        .fetchall()
    )
    trigger_sql = "\n".join(str(row[0] or "") for row in sync_triggers)
    for table_name in EXPECTED_TABLES:
        assert table_name not in trigger_sql


def test_migration_preserves_valid_legacy_summary_as_inactive_memory(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "legacy.db"
    conversation_id, boundary_id = _seed_v32_database(path, monkeypatch)

    db = open_current_chachanotes_from_legacy(path, client_id="v33-open")
    row = (
        db.get_connection()
        .execute(
            """
        SELECT conversation_id, boundary_message_id, captured_leaf_message_id,
               summary_text, active, source_kind, summarized_prefix_digest
          FROM console_conversation_memories
         WHERE id = ?
        """,
            (f"legacy-context-summary:{conversation_id}",),
        )
        .fetchone()
    )

    assert _version(db) == db._CURRENT_SCHEMA_VERSION
    assert row is not None
    assert row["conversation_id"] == conversation_id
    assert row["boundary_message_id"] == boundary_id
    assert row["captured_leaf_message_id"] == boundary_id
    assert row["summary_text"] == "legacy recap"
    assert row["active"] == 0
    assert row["source_kind"] == "legacy"
    assert row["summarized_prefix_digest"] is None
    assert db.get_conversation_context_summary(conversation_id) == (
        "legacy recap",
        boundary_id,
    )


def test_v33_policy_migrates_with_inherited_text_representation(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "v33-policy.db"
    conversation_id = _seed_v33_database(path, monkeypatch)

    db = open_current_chachanotes_from_legacy(path, client_id="v34-open")
    result = ConsoleContextRepository(db).load_policy(conversation_id)

    assert _version(db) == db._CURRENT_SCHEMA_VERSION
    assert result.error is None
    assert result.overrides.custom_budget_tokens == 12_000
    assert result.overrides.compaction_mode is ContextCompactionMode.AUTOMATIC
    assert result.overrides.compaction_representation is None


def test_policy_repository_round_trip_is_sparse_revisioned_and_local(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "policy.db", client_id="policy")
    conversation_id = db.add_conversation({"title": "policy"})
    repository = ConsoleContextRepository(db)
    sync_before = db.get_latest_sync_log_change_id()
    overrides = ConsoleContextPolicyOverrides(
        budget_mode=ContextBudgetMode.CUSTOM,
        custom_budget_tokens=24_000,
        compaction_mode=ContextCompactionMode.OFF,
        compaction_representation=ContextCompactionRepresentation.HYBRID,
    )

    first_revision = repository.save_policy(conversation_id, overrides)
    second_revision = repository.save_policy(
        conversation_id,
        ConsoleContextPolicyOverrides(
            budget_mode=ContextBudgetMode.CUSTOM,
            custom_budget_tokens=12_000,
        ),
    )
    result = repository.load_policy(conversation_id)

    assert first_revision == 1
    assert second_revision == 2
    assert result.revision == 2
    assert result.error is None
    assert result.overrides.custom_budget_tokens == 12_000
    assert result.overrides.compaction_mode is None
    assert result.overrides.compaction_representation is None
    assert db.get_sync_log_entries(since_change_id=sync_before) == []

    assert (
        repository.save_policy(conversation_id, ConsoleContextPolicyOverrides()) is None
    )
    assert repository.load_policy(conversation_id).overrides.is_empty


def test_corrupt_policy_row_fails_closed_to_inheritance(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "corrupt.db", client_id="corrupt")
    conversation_id = db.add_conversation({"title": "corrupt"})
    repository = ConsoleContextRepository(db)
    repository.save_policy(
        conversation_id,
        ConsoleContextPolicyOverrides(
            budget_mode=ContextBudgetMode.CUSTOM,
            custom_budget_tokens=8_000,
        ),
    )
    connection = db.get_connection()
    connection.execute("PRAGMA ignore_check_constraints = ON")
    connection.execute(
        "UPDATE console_conversation_context_policy SET budget_mode = 'mystery' "
        "WHERE conversation_id = ?",
        (conversation_id,),
    )
    connection.commit()
    connection.execute("PRAGMA ignore_check_constraints = OFF")

    result = repository.load_policy(conversation_id)

    assert result.overrides.is_empty
    assert result.error == "invalid_persisted_context_policy"


def test_generated_memory_repository_preserves_branch_provenance(tmp_path) -> None:
    db = CharactersRAGDB(tmp_path / "memory.db", client_id="memory")
    conversation_id = db.add_conversation({"title": "memory"})
    boundary_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "durable turn",
        }
    )
    repository = ConsoleContextRepository(db)

    record = ConsoleMemoryRecord(
        memory_id="memory-1",
        conversation_id=conversation_id,
        boundary_message_id=boundary_id,
        captured_leaf_message_id=boundary_id,
        lineage_json=f'["{boundary_id}"]',
        summary_text="A derived recap.",
        provider="openai",
        model="gpt-test",
        prompt_id="console.rewind_summarize",
        prompt_revision=3,
        prompt_digest="prompt-sha256",
        selected_units_json=f'[{{"message_id":"{boundary_id}","version":1}}]',
        summarized_prefix_digest="prefix-sha256",
        input_tokens=1_000,
        output_tokens=120,
        before_tokens=8_000,
        after_tokens=2_000,
        created_at="2026-08-10T12:00:00Z",
    )
    repository.insert_memory(record)
    row = (
        db.get_connection()
        .execute("SELECT * FROM console_conversation_memories WHERE id = 'memory-1'")
        .fetchone()
    )

    assert row is not None
    assert row["conversation_id"] == conversation_id
    assert row["boundary_message_id"] == boundary_id
    assert row["captured_leaf_message_id"] == boundary_id
    assert row["summarized_prefix_digest"] == "prefix-sha256"
    assert row["active"] == 1
    assert row["source_kind"] == "generated"
    loaded = repository.list_active_memories(conversation_id)
    assert len(loaded) == 1
    assert loaded[0].memory_id == record.memory_id
    assert loaded[0].summary_text == record.summary_text
    assert loaded[0].summarized_prefix_digest == record.summarized_prefix_digest
    assert repository.deactivate_memory(
        "memory-1",
        expected_revision=1,
        reset_at="2026-08-10T12:01:00Z",
    )
    assert repository.list_active_memories(conversation_id) == ()
    assert repository.reactivate_memory("memory-1", expected_revision=2)
    assert repository.list_active_memories(conversation_id)[0].revision == 3
    assert not repository.reactivate_memory("memory-1", expected_revision=2)
    assert repository.deactivate_memory(
        "memory-1",
        expected_revision=3,
        reset_at="2026-08-10T12:01:30Z",
    )
    assert repository.list_active_memories(conversation_id) == ()
    assert not repository.deactivate_memory(
        "memory-1",
        expected_revision=1,
        reset_at="2026-08-10T12:02:00Z",
    )
    assert not repository.insert_memory_if_current(
        replace(record, memory_id="memory-stale"),
        expected_memory_id="memory-1",
        expected_memory_revision=1,
    )

    repository.insert_memory(replace(record, memory_id="memory-2"))
    assert repository.insert_memory_if_current(
        replace(record, memory_id="memory-guarded"),
        expected_memory_id="memory-2",
        expected_memory_revision=1,
    )
    repository.insert_memory(replace(record, memory_id="memory-3"))
    first_page = repository.list_active_memories(conversation_id, limit=2)
    second_page = repository.list_active_memories(
        conversation_id,
        limit=2,
        offset=2,
    )
    assert [item.memory_id for item in first_page] == [
        "memory-3",
        "memory-guarded",
    ]
    assert [item.memory_id for item in second_page] == ["memory-2"]
    with pytest.raises(ValueError, match="limit"):
        repository.list_active_memories(conversation_id, limit=0)
    with pytest.raises(ValueError, match="offset"):
        repository.list_active_memories(conversation_id, offset=-1)
    assert (
        repository.deactivate_all_memories(
            conversation_id,
            reset_at="2026-08-10T12:03:00Z",
        )
        == 3
    )
    assert repository.list_active_memories(conversation_id) == ()

    other_conversation_id = db.add_conversation({"title": "other"})
    other_message_id = db.add_message(
        {
            "conversation_id": other_conversation_id,
            "sender": "user",
            "content": "foreign branch",
        }
    )
    with pytest.raises(sqlite3.IntegrityError):
        repository.insert_memory(
            replace(
                record,
                memory_id="memory-foreign",
                boundary_message_id=other_message_id,
            )
        )


@pytest.mark.parametrize(
    ("status", "usage"),
    [
        (
            AuxiliaryAttemptStatus.SUCCEEDED,
            ProviderUsage(
                uncached_input=4_000,
                output=200,
                provider="openai",
                model="gpt-test",
            ),
        ),
        (AuxiliaryAttemptStatus.FAILED, None),
        (AuxiliaryAttemptStatus.CANCELLED, None),
        (AuxiliaryAttemptStatus.STALE, None),
    ],
)
def test_auxiliary_attempt_ledger_accepts_usage_but_no_content_fields(
    tmp_path,
    status: AuxiliaryAttemptStatus,
    usage: ProviderUsage | None,
) -> None:
    # "aux" is a reserved Windows device name even with a .db suffix.
    db = CharactersRAGDB(tmp_path / "attempts.db", client_id="aux")
    conversation_id = db.add_conversation({"title": "aux"})
    repository = ConsoleContextRepository(db)
    repository.start_auxiliary_attempt(
        AuxiliaryAttemptStart(
            operation_id="op-1",
            conversation_id=conversation_id,
            purpose="conversation_compaction",
            provider="openai",
            model="gpt-test",
            requested_output_cap=512,
            estimated_input_tokens=4_000,
            started_at="2026-08-10T12:00:00Z",
        )
    )

    assert repository.finish_auxiliary_attempt(
        "op-1",
        status=status,
        finished_at="2026-08-10T12:00:01Z",
        elapsed_ms=1_000,
        usage=usage,
        pricing=(
            AuxiliaryPricingProvenance(
                catalog_revision="2026-08-01",
                source="pricing_catalog",
                estimated=False,
            )
            if usage is not None
            else None
        ),
    )
    row = repository.get_auxiliary_attempt("op-1")
    assert row is not None
    assert row["status"] == status.value
    if usage is None:
        assert row["provider_usage_json"] is None
    else:
        assert '"output": 200' in row["provider_usage_json"]
        assert '"source": "pricing_catalog"' in row["pricing_provenance_json"]
    listed = repository.list_auxiliary_attempts(conversation_id)
    assert len(listed) == 1
    assert listed[0]["purpose"] == "conversation_compaction"
    assert "summary_text" not in listed[0]
    assert "content" not in listed[0]
    assert repository.list_auxiliary_attempts(conversation_id, offset=1) == ()
    with pytest.raises(ValueError, match="offset"):
        repository.list_auxiliary_attempts(conversation_id, offset=-1)
    assert not repository.finish_auxiliary_attempt(
        "op-1",
        status=AuxiliaryAttemptStatus.FAILED,
        finished_at="2026-08-10T12:00:02Z",
    )

    columns = {
        str(info[1])
        for info in db.get_connection()
        .execute("PRAGMA table_info(console_auxiliary_attempts)")
        .fetchall()
    }
    assert (
        not {"content", "prompt", "summary", "request_body", "response_body"} & columns
    )
