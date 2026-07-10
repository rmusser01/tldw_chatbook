"""L3b Task 0 smoke: the local ingest seam writes a real MediaDatabase row headlessly.

Proves the inventory-chosen backend seam (``ingest_local_file`` ->
``add_media_with_keywords``) end to end with zero optional dependencies
before any Library ingest UI is built on top of it. The plaintext path is
the guaranteed baseline: no transcription, OCR, or PDF libraries involved.
"""

from pathlib import Path

import pytest

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Local_Ingestion import FileIngestionError, ingest_local_file


def test_ingest_local_text_file_creates_media_row(tmp_path: Path) -> None:
    source = tmp_path / "smoke-note.txt"
    source.write_text("Tides are driven by the moon's gravity.", encoding="utf-8")
    db = MediaDatabase(tmp_path / "smoke_media.db", client_id="l3b-smoke")

    result = ingest_local_file(
        file_path=source,
        media_db=db,
        title="Smoke note",
        author="tester",
        keywords=["smoke"],
        perform_analysis=False,
        chunk_options=None,
    )

    media_id = result["media_id"]
    assert isinstance(media_id, int)
    row = db.get_media_by_id(media_id)
    assert row is not None
    assert row["title"] == "Smoke note"
    assert "moon's gravity" in row["content"]
    assert row["type"] == "plaintext"


def test_ingest_failure_surfaces_as_exception(tmp_path: Path) -> None:
    db = MediaDatabase(tmp_path / "smoke_media.db", client_id="l3b-smoke")
    missing = tmp_path / "does-not-exist.txt"

    with pytest.raises((FileIngestionError, FileNotFoundError)):
        ingest_local_file(file_path=missing, media_db=db, perform_analysis=False)
