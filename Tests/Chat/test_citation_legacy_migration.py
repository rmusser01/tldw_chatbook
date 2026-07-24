from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import json
import os

import pytest

from tldw_chatbook.Chat.citation_legacy_migration import (
    LegacyCitationReadState,
    LegacyMigrationState,
    CitationLegacyMigrationService,
    synthesize_legacy_message,
)
from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_trace_identity import (
    CitationFingerprintCodec,
)
from tldw_chatbook.Chat.citation_trace_models import (
    CitationCompleteness,
    TraceOrigin,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationTraceRepository,
    load_local_citation_identity_context,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
CODEC = CitationFingerprintCodec(b"m" * 32)


def _bundle() -> dict:
    return {
        "bundle_id": "bundle-1",
        "query": "private query",
        "references": [
            {
                "evidence_id": "1",
                "source_id": "note-1",
                "source_type": "note",
                "title": "Legacy title",
                "snippet": "Exact cited text",
                "authority_label": "Local Library",
                "content_ref": "file:///tmp/do-not-open",
            }
        ],
    }


def _record(message_id: str, *, answer: str = "Legacy answer [1].") -> dict:
    return {
        "conversation_id": "conversation-1",
        "message_id": message_id,
        "rag_context": {
            "evidence_bundle": _bundle(),
            "citation_validation": {"valid": True},
        },
        "citations": [{"evidence_id": "1", "source_id": "note-1"}],
        "answer_body": answer,
    }


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(
        tmp_path / "legacy-migration.sqlite",
        client_id="legacy-migration-test",
    )
    yield database
    database.close_connection()


def _repository(db: CharactersRAGDB, *, enabled: bool = True):
    identity = load_local_citation_identity_context(db)
    assert identity is not None
    return CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(
            canonical_writes_enabled=enabled,
        ),
        identity_context=identity,
        fingerprint_codec=CODEC,
    )


def _conversation_with_messages(
    db: CharactersRAGDB,
    count: int,
) -> tuple[str, list[str]]:
    conversation_id = db.add_conversation(
        {"title": "Legacy migration", "character_id": None}
    )
    message_ids: list[str] = []
    for ordinal in range(count):
        message_id = f"message-{ordinal:04d}"
        db.add_message(
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": f"Legacy answer [{(ordinal % 2) + 1}].",
                "timestamp": NOW.isoformat(),
            }
        )
        message_ids.append(message_id)
    return conversation_id, message_ids


def _write_sidecar(path, conversation_id: str, message_ids: list[str]) -> None:
    records = {
        message_id: {
            **_record(
                message_id,
                answer=f"Legacy answer [{(ordinal % 2) + 1}].",
            ),
            "conversation_id": conversation_id,
        }
        for ordinal, message_id in enumerate(message_ids)
    }
    path.write_text(
        json.dumps({"version": 1, "conversations": {conversation_id: records}}),
        encoding="utf-8",
    )


def test_pure_legacy_synthesis_is_bounded_partial_and_deterministic() -> None:
    first = synthesize_legacy_message(
        _record("message-1"),
        conversation_id="conversation-1",
        message_id="message-1",
        answer_body="Legacy answer [1].",
        created_at=NOW,
        fingerprint_codec=CODEC,
    )
    second = synthesize_legacy_message(
        _record("message-1"),
        conversation_id="conversation-1",
        message_id="message-1",
        answer_body="Legacy answer [1].",
        created_at=NOW,
        fingerprint_codec=CODEC,
    )

    assert first == second
    assert first.trace.origin is TraceOrigin.LEGACY_INFERRED
    assert first.trace.completeness_at_seal is CitationCompleteness.PARTIAL
    assert first.trace.completeness_at_seal is not CitationCompleteness.COMPLETE
    assert [item.raw_marker for item in first.trace.answer_attempts[0].occurrences] == [
        "[1]"
    ]
    snapshot = first.evidence_snapshot_payloads[0]
    assert snapshot.locator == {"legacy_free_form": "file:///tmp/do-not-open"}


@pytest.mark.parametrize(
    "record",
    [
        {"citation_validation": {"valid": False}},
        {"chat_rag_context": {"citations": [{"evidence_id": "1"}]}},
        {"evidence_bundle": {"bundle_id": "bad", "references": "not-a-list"}},
        {"citations": [{"evidence_id": True, "source_id": "note"}]},
        {
            "evidence_bundle": {
                **_bundle(),
                "references": [{**_bundle()["references"][0], "score": True}],
            }
        },
    ],
)
def test_malformed_and_partial_legacy_records_fail_closed_to_unavailable(
    record: dict,
) -> None:
    write = synthesize_legacy_message(
        record,
        conversation_id="conversation-1",
        message_id="message-1",
        answer_body="Legacy answer [1].",
        created_at=NOW,
        fingerprint_codec=CODEC,
    )

    assert write.trace.origin is TraceOrigin.LEGACY_INFERRED
    assert write.trace.completeness_at_seal is CitationCompleteness.UNAVAILABLE
    assert not write.evidence_snapshot_payloads


def test_migration_batches_are_hidden_restartable_and_cut_over_atomically(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    conversation_id, message_ids = _conversation_with_messages(db, 205)
    sidecar = tmp_path / "chat_rag_context.json"
    _write_sidecar(sidecar, conversation_id, message_ids)
    service = CitationLegacyMigrationService(
        db=db,
        repository=_repository(db),
        sidecar_path=sidecar,
        fingerprint_codec=CODEC,
    )

    first = service.migrate_next_batch(conversation_id)
    assert first.state is LegacyMigrationState.RUNNING
    assert first.processed_messages == 100
    assert (
        db.get_connection()
        .execute(
            "SELECT count(*) FROM rag_citation_traces WHERE visibility_state='active'"
        )
        .fetchone()[0]
        == 0
    )

    restarted = CitationLegacyMigrationService(
        db=db,
        repository=_repository(db),
        sidecar_path=sidecar,
        fingerprint_codec=CODEC,
    )
    assert restarted.migrate_next_batch(conversation_id).processed_messages == 100
    final = restarted.migrate_next_batch(conversation_id)
    assert final.state is LegacyMigrationState.COMPLETE
    assert final.processed_messages == 5

    connection = db.get_connection()
    assert (
        connection.execute(
            "SELECT count(*) FROM rag_citation_traces WHERE visibility_state='active'"
        ).fetchone()[0]
        == 205
    )
    assert (
        connection.execute("SELECT count(*) FROM rag_message_trace_owners").fetchone()[
            0
        ]
        == 205
    )
    assert (
        connection.execute(
            "SELECT count(DISTINCT trace_id) FROM rag_citation_traces"
        ).fetchone()[0]
        == 205
    )
    journal = service.get_journal(conversation_id)
    assert journal is not None
    assert journal.state is LegacyMigrationState.COMPLETE
    assert journal.next_message_cursor == message_ids[-1]
    assert not sidecar.read_bytes() == b""


def test_changed_sidecar_between_batches_diverges_without_visible_merge(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    conversation_id, message_ids = _conversation_with_messages(db, 101)
    sidecar = tmp_path / "chat_rag_context.json"
    _write_sidecar(sidecar, conversation_id, message_ids)
    service = CitationLegacyMigrationService(
        db=db,
        repository=_repository(db),
        sidecar_path=sidecar,
        fingerprint_codec=CODEC,
    )

    assert (
        service.migrate_next_batch(conversation_id).state
        is LegacyMigrationState.RUNNING
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["conversations"][conversation_id][message_ids[-1]]["citations"] = []
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    result = service.migrate_next_batch(conversation_id)
    assert result.state is LegacyMigrationState.DIVERGED
    assert (
        db.get_connection()
        .execute(
            "SELECT count(*) FROM rag_citation_traces WHERE visibility_state='active'"
        )
        .fetchone()[0]
        == 0
    )


def test_concurrent_batch_callers_converge_without_duplicate_owners(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    conversation_id, message_ids = _conversation_with_messages(db, 100)
    sidecar = tmp_path / "chat_rag_context.json"
    _write_sidecar(sidecar, conversation_id, message_ids)
    repository = _repository(db)

    def migrate() -> LegacyMigrationState:
        service = CitationLegacyMigrationService(
            db=db,
            repository=repository,
            sidecar_path=sidecar,
            fingerprint_codec=CODEC,
        )
        return service.migrate_next_batch(conversation_id).state

    with ThreadPoolExecutor(max_workers=2) as executor:
        states = [
            future.result() for future in (executor.submit(migrate) for _ in range(2))
        ]

    assert LegacyMigrationState.COMPLETE in states
    connection = db.get_connection()
    assert (
        connection.execute("SELECT count(*) FROM rag_citation_traces").fetchone()[0]
        == 100
    )
    assert (
        connection.execute("SELECT count(*) FROM rag_message_trace_owners").fetchone()[
            0
        ]
        == 100
    )


def test_explicit_retry_deletes_hidden_staging_before_rebuild(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    conversation_id, message_ids = _conversation_with_messages(db, 101)
    sidecar = tmp_path / "chat_rag_context.json"
    _write_sidecar(sidecar, conversation_id, message_ids)
    service = CitationLegacyMigrationService(
        db=db,
        repository=_repository(db),
        sidecar_path=sidecar,
        fingerprint_codec=CODEC,
    )
    assert service.migrate_next_batch(conversation_id).processed_messages == 100
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["conversations"][conversation_id][message_ids[-1]][
        "citation_validation"
    ] = {"valid": False}
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        service.migrate_next_batch(conversation_id).state
        is LegacyMigrationState.DIVERGED
    )

    service.retry_diverged(conversation_id)

    connection = db.get_connection()
    assert (
        connection.execute("SELECT count(*) FROM rag_citation_traces").fetchone()[0]
        == 0
    )
    assert (
        connection.execute(
            "SELECT count(*) FROM rag_legacy_migration_journal"
        ).fetchone()[0]
        == 0
    )
    assert service.migrate_next_batch(conversation_id).processed_messages == 100
    assert (
        service.migrate_next_batch(conversation_id).state
        is LegacyMigrationState.COMPLETE
    )
    assert (
        connection.execute("SELECT count(*) FROM rag_citation_traces").fetchone()[0]
        == 101
    )


def test_raw_verification_detects_same_stat_rewrite_after_cutover(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    conversation_id, message_ids = _conversation_with_messages(db, 1)
    sidecar = tmp_path / "chat_rag_context.json"
    _write_sidecar(sidecar, conversation_id, message_ids)
    service = CitationLegacyMigrationService(
        db=db,
        repository=_repository(db),
        sidecar_path=sidecar,
        fingerprint_codec=CODEC,
    )
    assert (
        service.migrate_next_batch(conversation_id).state
        is LegacyMigrationState.COMPLETE
    )
    original = sidecar.read_bytes()
    original_stat = sidecar.stat()
    changed = original.replace(b"Legacy title", b"ChangedTitle")
    assert len(changed) == len(original)
    sidecar.write_bytes(changed)
    os.utime(
        sidecar,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    assert (
        service.read_conversation(conversation_id, verify_canonical=True).state
        is LegacyCitationReadState.DIVERGED
    )
    assert service.get_journal(conversation_id).state is LegacyMigrationState.DIVERGED


def test_reader_is_fallback_until_complete_then_canonical_first(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    conversation_id, message_ids = _conversation_with_messages(db, 1)
    sidecar = tmp_path / "chat_rag_context.json"
    _write_sidecar(sidecar, conversation_id, message_ids)
    service = CitationLegacyMigrationService(
        db=db,
        repository=_repository(db),
        sidecar_path=sidecar,
        fingerprint_codec=CODEC,
    )

    legacy = service.read_conversation(conversation_id)
    assert legacy.state is LegacyCitationReadState.LEGACY_FALLBACK
    assert list(legacy.records) == message_ids

    assert (
        service.migrate_next_batch(conversation_id).state
        is LegacyMigrationState.COMPLETE
    )
    pending = service.read_conversation(conversation_id)
    assert pending.state is LegacyCitationReadState.VERIFICATION_PENDING
    canonical = service.read_conversation(conversation_id, verify_canonical=True)
    assert canonical.state is LegacyCitationReadState.CANONICAL
    assert list(canonical.records) == message_ids
    reference = canonical.records[message_ids[0]]["evidence_bundle"]["references"][0]
    assert reference["title"] == "Legacy title"
    assert reference["snippet"] == "Exact cited text"
    assert "content_ref" not in reference


def test_disabled_policy_does_not_migrate(db: CharactersRAGDB, tmp_path) -> None:
    conversation_id, message_ids = _conversation_with_messages(db, 1)
    sidecar = tmp_path / "chat_rag_context.json"
    _write_sidecar(sidecar, conversation_id, message_ids)
    service = CitationLegacyMigrationService(
        db=db,
        repository=_repository(db, enabled=False),
        sidecar_path=sidecar,
        fingerprint_codec=CODEC,
    )

    result = service.migrate_next_batch(conversation_id)
    assert result.state is LegacyMigrationState.PENDING
    assert (
        db.get_connection()
        .execute("SELECT count(*) FROM rag_legacy_migration_journal")
        .fetchone()[0]
        == 0
    )


def test_chatbook_package_record_stays_legacy_inferred_and_local_bound(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    conversation_id, message_ids = _conversation_with_messages(db, 1)
    sidecar = tmp_path / "chat_rag_context.json"
    service = CitationLegacyMigrationService(
        db=db,
        repository=_repository(db),
        sidecar_path=sidecar,
        fingerprint_codec=CODEC,
    )
    package_record = {
        **_record(
            message_ids[0],
            answer="Legacy answer [1].",
        ),
        "conversation_id": conversation_id,
        "origin": "imported",
        "external_trace_id": "must-remain-inert",
        "locator": "https://example.invalid/do-not-open",
    }

    result = service.persist_package_record(
        conversation_id=conversation_id,
        message_id=message_ids[0],
        record=package_record,
    )

    assert result.state is LegacyMigrationState.COMPLETE
    row = (
        db.get_connection()
        .execute(
            """
        SELECT origin, completeness_at_seal, connection_authority_id,
               import_package_fingerprint, external_trace_id,
               legacy_conversation_id, legacy_message_id, visibility_state
        FROM rag_citation_traces
        """
        )
        .fetchone()
    )
    assert tuple(row) == (
        "legacy_inferred",
        "partial",
        None,
        None,
        None,
        conversation_id,
        message_ids[0],
        "active",
    )
    snapshot = (
        db.get_connection()
        .execute("SELECT locator_json FROM rag_evidence_snapshots")
        .fetchone()
    )
    assert json.loads(snapshot["locator_json"]) == {
        "legacy_free_form": "file:///tmp/do-not-open"
    }


def test_oversized_sidecar_is_rejected_before_json_parse(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    conversation_id, _ = _conversation_with_messages(db, 1)
    sidecar = tmp_path / "chat_rag_context.json"
    with sidecar.open("wb") as handle:
        handle.truncate((32 * 1024 * 1024) + 1)
    service = CitationLegacyMigrationService(
        db=db,
        repository=_repository(db),
        sidecar_path=sidecar,
        fingerprint_codec=CODEC,
    )

    result = service.migrate_next_batch(conversation_id)
    assert result.state is LegacyMigrationState.FAILED
    assert result.reason_code == "legacy_sidecar_too_large"
