from __future__ import annotations

import json

from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter, ImportStatus
from tldw_chatbook.Chatbooks.chatbook_models import (
    ChatbookContent,
    ChatbookManifest,
    ChatbookVersion,
)
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


def _provenance_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "attempt_id": "attempt-1",
        "batch_id": None,
        "job_id": None,
        "retry_of_attempt_id": None,
        "retry_of_job_id": None,
        "provider_id": "parakeet-onnx",
        "model_id": "parakeet-v2",
        "artifact_root": None,
        "artifact_dependencies": [],
        "precision": "int8",
        "requested_device": "auto",
        "effective_device": "cpu",
        "requested_language": "en",
        "effective_language": "en",
        "detected_language": None,
        "task": "transcribe",
        "produced_capabilities": {
            "timestamps": "none",
            "punctuation": True,
            "capitalization": True,
            "vad": False,
            "diarization": False,
        },
        "warnings": [],
        "failed_attempt": None,
    }


def test_chatbook_media_round_trip_preserves_validated_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "source.sqlite"
    source = MediaDatabase(source_path, client_id="source")
    document = _provenance_document()
    media_id, _, _ = source.add_media_with_keywords(
        title="Portable transcript",
        media_type="audio",
        content="hello from export",
        transcription_model="parakeet-v2",
        transcription_provenance=document,
    )
    source.close_connection()

    export_root = tmp_path / "export"
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="STT provenance",
        description="round trip",
    )
    content = ChatbookContent()
    creator = ChatbookCreator(db_paths={"Media": str(source_path)})
    creator._collect_media(
        [str(media_id)],
        export_root,
        manifest,
        content,
        quality="original",
    )
    metadata_path = (
        export_root / "content" / "media" / "metadata" / f"media_{media_id}.json"
    )
    exported = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert exported["metadata"]["transcription_provenance"] == document

    runtime_root = tmp_path / "runtime"
    monkeypatch.setattr(
        "tldw_chatbook.Chatbooks.chatbook_importer.get_user_data_dir",
        lambda: runtime_root,
    )
    target_path = tmp_path / "target.sqlite"
    target = MediaDatabase(target_path, client_id="target")
    target.close_connection()
    importer = ChatbookImporter(db_paths={"Media": str(target_path)})
    status = ImportStatus()
    importer._import_media(
        export_root,
        manifest,
        [str(media_id)],
        ConflictResolution.SKIP,
        status,
    )

    imported = MediaDatabase(target_path, client_id="verify")
    row = imported.execute_query(
        "SELECT transcription_provenance_json FROM Media"
    ).fetchone()
    assert status.successful_items == 1
    assert json.loads(row["transcription_provenance_json"]) == document
    imported.close_connection()

    malformed = exported
    malformed["metadata"]["transcription_provenance"]["raw_exception"] = (
        "private traceback"
    )
    metadata_path.write_text(json.dumps(malformed), encoding="utf-8")
    rejected_path = tmp_path / "rejected.sqlite"
    rejected = MediaDatabase(rejected_path, client_id="rejected")
    rejected.close_connection()
    rejected_importer = ChatbookImporter(db_paths={"Media": str(rejected_path)})
    rejected_status = ImportStatus()
    rejected_importer._import_media(
        export_root,
        manifest,
        [str(media_id)],
        ConflictResolution.SKIP,
        rejected_status,
    )

    rejected = MediaDatabase(rejected_path, client_id="verify-rejected")
    assert rejected_status.failed_items == 1
    assert rejected.execute_query("SELECT COUNT(*) FROM Media").fetchone()[0] == 0
    rejected.close_connection()
