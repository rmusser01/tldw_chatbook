from __future__ import annotations

import json

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import DatabaseError, MediaDatabase
from tldw_chatbook.STT.persistence import (
    dump_transcription_provenance_document,
)


def _provenance_document() -> dict[str, object]:
    artifact = {
        "artifact_id": "parakeet-v2",
        "revision": "revision-2",
        "variant": "int8",
    }
    failed_attempt = {
        "attempt_id": "attempt-1",
        "batch_id": "batch-1",
        "job_id": "job-1",
        "provider_id": "parakeet-onnx",
        "model_id": "parakeet-v2",
        "artifact_root": artifact,
        "artifact_dependencies": [],
        "precision": "int8",
        "requested_device": "auto",
        "effective_device": "cpu",
        "requested_language": "en",
        "effective_language": "en",
        "detected_language": None,
        "task": "transcribe",
        "error_code": "inference_failed",
    }
    return {
        "schema_version": 1,
        "attempt_id": "attempt-2",
        "batch_id": "batch-1",
        "job_id": "job-2",
        "retry_of_attempt_id": "attempt-1",
        "retry_of_job_id": "job-1",
        "provider_id": "faster-whisper",
        "model_id": "large-v3",
        "artifact_root": {
            "artifact_id": "faster-whisper-large-v3",
            "revision": "revision-3",
            "variant": "int8",
        },
        "artifact_dependencies": [],
        "precision": "int8",
        "requested_device": "auto",
        "effective_device": "cpu",
        "requested_language": "auto",
        "effective_language": "auto",
        "detected_language": "es",
        "task": "transcribe",
        "produced_capabilities": {
            "timestamps": "segment",
            "punctuation": True,
            "capitalization": True,
            "vad": False,
            "diarization": False,
        },
        "warnings": [],
        "failed_attempt": failed_attempt,
    }


def _stored_provenance(db: MediaDatabase, media_id: int) -> dict[str, object] | None:
    row = db.execute_query(
        "SELECT transcription_provenance_json FROM Media WHERE id = ?",
        (media_id,),
    ).fetchone()
    return json.loads(row["transcription_provenance_json"]) if row[0] else None


def test_new_and_old_media_rows_round_trip_nullable_provenance(file_db) -> None:
    old_id, _, _ = file_db.add_media_with_keywords(
        title="Old transcript",
        media_type="audio",
        content="legacy transcript",
        transcription_model="legacy-whisper",
    )
    document = _provenance_document()
    new_id, _, _ = file_db.add_media_with_keywords(
        title="New transcript",
        media_type="audio",
        content="normalized transcript",
        transcription_model="large-v3",
        transcription_provenance=document,
    )

    assert _stored_provenance(file_db, old_id) is None
    assert _stored_provenance(file_db, new_id) == document
    assert file_db.get_media_by_id(new_id)["transcription_model"] == "large-v3"


def test_identical_overwrite_can_attach_provenance_atomically(file_db) -> None:
    media_id, _, _ = file_db.add_media_with_keywords(
        title="Retry transcript",
        media_type="audio",
        content="same transcript",
        transcription_model="parakeet-v2",
    )
    document = _provenance_document()

    updated_id, _, message = file_db.add_media_with_keywords(
        title="Retry transcript",
        media_type="audio",
        content="same transcript",
        transcription_model="large-v3",
        transcription_provenance=document,
        overwrite=True,
    )

    row = file_db.get_media_by_id(media_id)
    assert updated_id == media_id
    assert "updated" in message
    assert row["transcription_model"] == "large-v3"
    assert json.loads(row["transcription_provenance_json"]) == document


def test_search_projection_preserves_canonical_provenance(file_db) -> None:
    document = _provenance_document()
    media_id, _, _ = file_db.add_media_with_keywords(
        title="Searchable provenance",
        media_type="audio",
        content="search projection",
        transcription_model="large-v3",
        transcription_provenance=document,
    )

    rows, total = file_db.search_media_db(
        search_query=None,
        media_ids_filter=[media_id],
    )

    assert total == 1
    assert rows[0]["transcription_provenance_json"] == (
        dump_transcription_provenance_document(document)
    )


def test_invalid_provenance_is_rejected_before_any_media_write(file_db) -> None:
    invalid = _provenance_document()
    invalid["raw_exception"] = "secret traceback"

    with pytest.raises(ValueError, match="fields"):
        file_db.add_media_with_keywords(
            title="Must not persist",
            media_type="audio",
            content="transcript",
            transcription_model="large-v3",
            transcription_provenance=invalid,
        )

    assert file_db.execute_query("SELECT COUNT(*) FROM Media").fetchone()[0] == 0


def test_writer_failure_rolls_back_transcript_and_provenance(
    file_db,
    monkeypatch,
) -> None:
    def fail_sync_log(*args, **kwargs):
        raise RuntimeError("injected writer failure")

    monkeypatch.setattr(file_db, "_log_sync_event", fail_sync_log)

    with pytest.raises(DatabaseError, match="injected writer failure"):
        file_db.add_media_with_keywords(
            title="Atomic transcript",
            media_type="audio",
            content="must roll back",
            transcription_model="large-v3",
            transcription_provenance=_provenance_document(),
        )

    assert file_db.execute_query("SELECT COUNT(*) FROM Media").fetchone()[0] == 0


def test_v4_to_v5_migration_rolls_back_column_version_and_data(
    file_db,
) -> None:
    media_id, _, _ = file_db.add_media_with_keywords(
        title="Migration survivor",
        media_type="audio",
        content="existing transcript",
    )
    conn = file_db.get_connection()
    conn.execute("ALTER TABLE Media DROP COLUMN transcription_provenance_json")
    conn.execute("UPDATE schema_version SET version = 4")
    conn.commit()
    original_sql = file_db._TRANSCRIPTION_PROVENANCE_MIGRATION_SQL
    file_db._TRANSCRIPTION_PROVENANCE_MIGRATION_SQL = (
        f"{original_sql}\nINSERT INTO table_that_does_not_exist VALUES (1);"
    )
    try:
        with pytest.raises(DatabaseError, match="Migration v4->v5 failed"):
            file_db._apply_migration_v4_to_v5(conn)
    finally:
        file_db._TRANSCRIPTION_PROVENANCE_MIGRATION_SQL = original_sql

    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 4
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(Media)")}
    assert "transcription_provenance_json" not in columns
    assert (
        conn.execute("SELECT title FROM Media WHERE id = ?", (media_id,)).fetchone()[0]
        == "Migration survivor"
    )

    file_db._apply_migration_v4_to_v5(conn)
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 5
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(Media)")}
    assert "transcription_provenance_json" in columns
    assert (
        conn.execute(
            "SELECT transcription_provenance_json FROM Media WHERE id = ?",
            (media_id,),
        ).fetchone()[0]
        is None
    )
