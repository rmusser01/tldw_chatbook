from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import json
import os

import pytest

from Tests.Chat.test_citation_trace_repository import (
    TEST_FINGERPRINT_CODEC,
    _sealed_write as _local_sealed_write,
)
from tldw_chatbook.Chat.citation_legacy_migration import (
    LEGACY_FIELD_UTF8_BYTES_MAX,
    LEGACY_JSON_DEPTH_MAX,
    LEGACY_JSON_NODES_MAX,
    LEGACY_KEY_UTF8_BYTES_MAX,
    LEGACY_MAPPING_ITEMS_MAX,
    LEGACY_SEQUENCE_ITEMS_MAX,
    LEGACY_SIDECAR_BYTES_MAX,
    LegacyCitationReadState,
    LegacyMigrationState,
    CitationLegacyMigrationService,
    _validate_json_bounds,
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


def test_mutation_after_final_batch_commit_keeps_every_trace_hidden(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    conversation_id, message_ids = _conversation_with_messages(db, 1)
    sidecar = tmp_path / "chat_rag_context.json"
    _write_sidecar(sidecar, conversation_id, message_ids)

    def mutate_after_staging() -> None:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["conversations"][conversation_id][message_ids[0]][
            "citation_validation"
        ] = {"valid": False}
        sidecar.write_text(json.dumps(payload), encoding="utf-8")

    service = CitationLegacyMigrationService(
        db=db,
        repository=_repository(db),
        sidecar_path=sidecar,
        fingerprint_codec=CODEC,
        before_cutover_hook=mutate_after_staging,
    )

    result = service.migrate_next_batch(conversation_id)

    assert result.state is LegacyMigrationState.DIVERGED
    connection = db.get_connection()
    assert (
        connection.execute(
            "SELECT count(*) FROM rag_citation_traces WHERE visibility_state='active'"
        ).fetchone()[0]
        == 0
    )
    assert (
        connection.execute(
            "SELECT count(*) FROM rag_citation_traces WHERE visibility_state='migrating'"
        ).fetchone()[0]
        == 1
    )
    assert service.get_journal(conversation_id).next_message_cursor == message_ids[0]


def test_non_mapping_message_record_synthesizes_unavailable_without_hiding_siblings(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    conversation_id, message_ids = _conversation_with_messages(db, 2)
    sidecar = tmp_path / "chat_rag_context.json"
    _write_sidecar(sidecar, conversation_id, message_ids)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["conversations"][conversation_id][message_ids[0]] = "malformed"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
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

    rows = (
        db.get_connection()
        .execute(
            """
        SELECT legacy_message_id, completeness_at_seal
        FROM rag_citation_traces ORDER BY legacy_message_id
        """
        )
        .fetchall()
    )
    assert [tuple(row) for row in rows] == [
        (message_ids[0], "unavailable"),
        (message_ids[1], "partial"),
    ]


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


@pytest.mark.parametrize(
    ("origin", "origin_fields"),
    [
        ("local", {}),
        (
            "server",
            {
                "origin_scope_id": "authority-root",
                "connection_authority_id": "authority-1",
                "server_trace_id": "server-trace-1",
                "wire_schema_version": "1",
            },
        ),
        (
            "imported",
            {
                "origin_scope_id": "package-1",
                "import_package_fingerprint": "package-1",
                "external_trace_id": "external-trace-1",
            },
        ),
        (
            "legacy_inferred",
            {
                "legacy_conversation_id": "set-by-test",
                "legacy_message_id": "set-by-test",
            },
        ),
    ],
)
def test_reader_prefers_active_owner_linked_canonical_trace_for_every_origin(
    db: CharactersRAGDB,
    tmp_path,
    origin: str,
    origin_fields: dict[str, str],
) -> None:
    conversation_id = db.add_conversation(
        {"title": "Canonical export", "character_id": None}
    )
    message_id = "canonical-message"
    repository = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
        identity_context=load_local_citation_identity_context(db),
        fingerprint_codec=TEST_FINGERPRINT_CODEC,
    )
    prepared = repository.prepare_write(_local_sealed_write())
    with db.transaction() as cursor:
        db.add_message(
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": "Answer [S1].",
                "timestamp": NOW.isoformat(),
            }
        )
        repository.write_prepared(
            cursor,
            prepared,
            message_id=message_id,
            message_revision=1,
            message_body="Answer [S1].",
        )
        if origin != "local":
            assignments = {
                "origin": origin,
                "aggregate_json": _local_sealed_write()
                .trace.model_copy(update={"origin": TraceOrigin(origin)})
                .model_dump_json(),
                **origin_fields,
            }
            if origin == "legacy_inferred":
                assignments["legacy_conversation_id"] = conversation_id
                assignments["legacy_message_id"] = message_id
            cursor.execute(
                f"""
                UPDATE rag_citation_traces
                SET {", ".join(f"{key}=?" for key in assignments)}
                WHERE trace_id='trace-1'
                """,
                tuple(assignments.values()),
            )

    sidecar = tmp_path / "chat_rag_context.json"
    sidecar.write_text(
        json.dumps(
            {
                "conversations": {
                    conversation_id: {
                        "sidecar-only": {"citations": []},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    service = CitationLegacyMigrationService(
        db=db,
        repository=repository,
        sidecar_path=sidecar,
        fingerprint_codec=TEST_FINGERPRINT_CODEC,
    )

    view = service.read_conversation(conversation_id, verify_canonical=True)

    assert view.state is LegacyCitationReadState.CANONICAL
    assert list(view.records) == [message_id]
    assert view.records[message_id]["provenance_origin"] == origin


def test_disabled_recovery_switch_keeps_active_canonical_reads_available(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    conversation_id = db.add_conversation(
        {"title": "Disabled canonical read", "character_id": None}
    )
    message_id = "disabled-read-message"
    writer = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
        identity_context=load_local_citation_identity_context(db),
        fingerprint_codec=TEST_FINGERPRINT_CODEC,
    )
    prepared = writer.prepare_write(_local_sealed_write())
    with db.transaction() as cursor:
        db.add_message(
            {
                "id": message_id,
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": "Answer [S1].",
            }
        )
        writer.write_prepared(
            cursor,
            prepared,
            message_id=message_id,
            message_revision=1,
            message_body="Answer [S1].",
        )
    reader = CitationTraceRepository(
        db,
        policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=False),
        identity_context=load_local_citation_identity_context(db),
        fingerprint_codec=None,
    )
    service = CitationLegacyMigrationService(
        db=db,
        repository=reader,
        sidecar_path=tmp_path / "missing-sidecar.json",
    )

    view = service.read_conversation(conversation_id, verify_canonical=True)

    assert view.state is LegacyCitationReadState.CANONICAL
    assert list(view.records) == [message_id]


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


@pytest.mark.parametrize(
    ("exact", "over", "reason"),
    [
        (
            [[None] for _ in range(LEGACY_JSON_DEPTH_MAX - 1)],
            [[None] for _ in range(LEGACY_JSON_DEPTH_MAX)],
            "legacy_json_too_deep",
        ),
        (
            [[None, None, None, None] for _ in range(4_000)],
            [[None, None, None, None, None]]
            + [[None, None, None, None] for _ in range(3_999)],
            "legacy_json_too_many_nodes",
        ),
        (
            {f"k{i}": None for i in range(LEGACY_MAPPING_ITEMS_MAX)},
            {f"k{i}": None for i in range(LEGACY_MAPPING_ITEMS_MAX + 1)},
            "legacy_mapping_too_large",
        ),
        (
            [None] * LEGACY_SEQUENCE_ITEMS_MAX,
            [None] * (LEGACY_SEQUENCE_ITEMS_MAX + 1),
            "legacy_sequence_too_large",
        ),
        (
            {"k" * LEGACY_KEY_UTF8_BYTES_MAX: None},
            {"k" * (LEGACY_KEY_UTF8_BYTES_MAX + 1): None},
            "legacy_key_too_large",
        ),
        (
            "x" * LEGACY_FIELD_UTF8_BYTES_MAX,
            "x" * (LEGACY_FIELD_UTF8_BYTES_MAX + 1),
            "legacy_field_too_large",
        ),
    ],
)
def test_each_legacy_json_limit_accepts_exact_and_rejects_one_over(
    exact,
    over,
    reason,
) -> None:
    if reason == "legacy_json_too_deep":
        exact_value = None
        for _ in range(LEGACY_JSON_DEPTH_MAX - 1):
            exact_value = [exact_value]
        over_value = [exact_value]
    elif reason == "legacy_json_too_many_nodes":
        exact_value = exact
        over_value = over
        assert LEGACY_JSON_NODES_MAX == 20_001
    else:
        exact_value = exact
        over_value = over

    _validate_json_bounds(exact_value)
    with pytest.raises(ValueError, match=f"^{reason}$"):
        _validate_json_bounds(over_value)


def test_raw_sidecar_limit_accepts_exact_and_rejects_one_over(
    db: CharactersRAGDB,
    tmp_path,
) -> None:
    conversation_id, _ = _conversation_with_messages(db, 1)
    over_limit_conversation_id, _ = _conversation_with_messages(db, 0)
    prefix = b'{"conversations":{},"padding":['
    suffix = b"]}"
    unit = b'"' + (b"x" * LEGACY_FIELD_UTF8_BYTES_MAX) + b'"'
    available = LEGACY_SIDECAR_BYTES_MAX - len(prefix) - len(suffix)
    unit_count = (available + 1) // (len(unit) + 1)
    parts = [unit] * unit_count
    remaining = available - len(b",".join(parts))
    if remaining:
        content_size = remaining - 3
        assert 0 <= content_size <= LEGACY_FIELD_UTF8_BYTES_MAX
        parts.append(b'"' + (b"y" * content_size) + b'"')
    exact = prefix + b",".join(parts) + suffix
    assert len(exact) == LEGACY_SIDECAR_BYTES_MAX
    sidecar = tmp_path / "chat_rag_context.json"
    sidecar.write_bytes(exact)
    exact_service = CitationLegacyMigrationService(
        db=db,
        repository=_repository(db),
        sidecar_path=sidecar,
        fingerprint_codec=CODEC,
    )
    assert (
        exact_service.migrate_next_batch(conversation_id).state
        is LegacyMigrationState.COMPLETE
    )

    sidecar.write_bytes(exact + b" ")
    over_service = CitationLegacyMigrationService(
        db=db,
        repository=_repository(db),
        sidecar_path=sidecar,
        fingerprint_codec=CODEC,
    )
    result = over_service.migrate_next_batch(over_limit_conversation_id)
    assert result.state is LegacyMigrationState.FAILED
    assert result.reason_code == "legacy_sidecar_too_large"
