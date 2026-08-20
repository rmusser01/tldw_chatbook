"""Stamp + report (spec §8): new chunks carry the engine version; report counts.

Adaptations of the brief's verbatim test code (all forced by the as-built
code, all minimal):
- ``test_new_chunks_stamped`` goes through ``persist_parsed_media`` (the
  stamper lives at the persist seam) and passes ``overlap: 0`` (the
  RAG_Search wrapper's legacy validation rejects the brief's implicit
  default overlap 100 against max_size 5);
- ``test_legacy_rows_read_as_legacy``'s bare UPDATE ABORTs on the
  ``unvectorizedmediachunks_validate_sync_update`` trigger (every UPDATE on
  that table must increment ``version`` by exactly 1 -- same trap Task 11's
  tests hit), so it sets ``version = version + 1`` too.
"""
import pytest


def test_new_chunks_stamped(tmp_path):
    """Brief's test, adapted to exercise the stamper (see module docstring).

    Two minimal adaptations of the brief's verbatim snippet:
    - routed through ``persist_parsed_media`` instead of a direct
      ``add_media_with_keywords`` call: the stamp lands at the persist seam
      (Task 11's design -- top-level chunk dicts stay clean, the DB stamp
      happens at persist), so the direct call cannot exercise the stamper;
    - ``"overlap": 0`` added: the wrapper defaults overlap to 100 and its
      legacy validation rejects ``overlap >= max_size`` (100 >= 5) before
      the chunker ever runs.
    """
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
        persist_parsed_media,
    )
    from tldw_chatbook.RAG_Search.chunking_service import improved_chunking_process
    chunks = improved_chunking_process(
        "One two three four five six. " * 5,
        {"method": "words", "max_size": 5, "overlap": 0},
    )
    assert chunks
    db = MediaDatabase(str(tmp_path / "m.db"), client_id="test")
    persist_parsed_media(
        {
            "file_type": "txt",
            "title": "t",
            "media_type": "document",
            "content": "One two three four five six. " * 5,
            "keywords": [],
            "url": None,
            "analysis_content": None,
            "author": None,
            "chunks": chunks,
            "chunk_options": {},
            "warnings": [],
        },
        db,
        generate_embeddings=False,
    )
    rows = db.get_connection().execute(
        "SELECT chunk_engine_version FROM UnvectorizedMediaChunks WHERE deleted = 0"
    ).fetchall()
    assert rows and all(r["chunk_engine_version"] == "parity-1@385afa95" for r in rows)


def test_legacy_rows_read_as_legacy(tmp_path):
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    db = MediaDatabase(str(tmp_path / "m.db"), client_id="test")
    # insert a row with NULL version (pre-parity)
    db.add_media_with_keywords(
        title="t", media_type="document", content="...", keywords=None, url=None,
        analysis_content=None, author=None, chunks=[{"text": "old", "metadata": {}}],
        chunk_options={},
    )
    db.get_connection().execute(
        # ``version = version + 1`` is required by the
        # unvectorizedmediachunks_validate_sync_update trigger (same trap
        # Task 11's tests hit): every UPDATE on this table must increment
        # version by exactly 1; the brief's bare UPDATE would ABORT.
        "UPDATE UnvectorizedMediaChunks SET chunk_engine_version = NULL, "
        "version = version + 1"
    )
    db.get_connection().commit()
    from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService
    svc = LocalRAGAdminService.__new__(LocalRAGAdminService)  # read-only query path
    # count_chunks_by_engine_version is a small, dependency-light method
    counts = svc.count_chunks_by_engine_version(db)
    assert counts.get("legacy") == 1


def test_persist_parsed_media_stamps_chunks(tmp_path):
    """Task 12: the ingestion persist seam stamps every chunk it writes.

    Goes THROUGH ``persist_parsed_media`` (the real Library ingest writer)
    rather than calling ``add_media_with_keywords`` directly, so the stamper
    at the persist seam is what's under test -- a stamper that only lived in
    the chunker would pass the direct path but never fire for ingestion.
    """
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import persist_parsed_media

    db = MediaDatabase(str(tmp_path / "m.db"), client_id="test")
    payload = {
        "file_type": "txt",
        "title": "t",
        "media_type": "document",
        "content": "One two three four five six. " * 5,
        "keywords": [],
        "url": None,
        "analysis_content": None,
        "author": None,
        "chunks": [
            {"text": "chunk one", "metadata": {"chunk_index": 1}},
            {"text": "chunk two", "metadata": {"chunk_index": 2}},
        ],
        "chunk_options": {},
        "warnings": [],
    }
    persist_parsed_media(payload, db, generate_embeddings=False)
    rows = db.get_connection().execute(
        "SELECT chunk_engine_version FROM UnvectorizedMediaChunks WHERE deleted = 0"
    ).fetchall()
    assert rows and all(r["chunk_engine_version"] == "parity-1@385afa95" for r in rows)


def test_shim_stamps_chunk_metadata():
    """Spec §8: in-memory consumers see the engine version without a DB read.

    The shim stamps ``metadata["chunk_engine_version"]``; the TOP-LEVEL chunk
    dict stays clean of the key (DB stamping happens at the persist seam, per
    Task 11's design).
    """
    from tldw_chatbook.Chunking.Chunk_Lib import (
        ENGINE_VERSION,
        improved_chunking_process,
    )

    assert ENGINE_VERSION == "parity-1@385afa95"
    chunks = improved_chunking_process(
        "One two three four five six. " * 5,
        {"method": "words", "max_size": 5, "overlap": 0},
    )
    assert chunks
    for chunk in chunks:
        assert chunk["metadata"]["chunk_engine_version"] == "parity-1@385afa95"
        # top-level dict stays clean (persist seam owns the DB stamp)
        assert "chunk_engine_version" not in chunk


def test_engine_version_reexported_from_package():
    from tldw_chatbook.Chunking import ENGINE_VERSION
    from tldw_chatbook.Chunking.Chunk_Lib import ENGINE_VERSION as shim_version

    assert ENGINE_VERSION == "parity-1@385afa95"
    assert ENGINE_VERSION is shim_version


def test_batch_writer_keeps_row_and_sync_payload_consistent(tmp_path):
    """Carry-forward from Task 11's review (silent-stamp-drop trap).

    ``process_unvectorized_chunks`` is the OTHER chunk writer: its sync event
    payload is an explicit ``insert_data`` dict, unlike ``_persist_chunks``
    which spreads ``**ch``. Both the row AND the sync event payload must
    carry the stamp -- patching only the SQL would stamp the row while the
    sync event silently drops it.
    """
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    db = MediaDatabase(str(tmp_path / "m.db"), client_id="test")
    db.add_media_with_keywords(
        title="t", media_type="document", content="...", keywords=None, url=None,
        analysis_content=None, author=None,
        chunks=[{"text": "old", "metadata": {}}],
        chunk_options={},
    )
    media_id = db.get_connection().execute(
        "SELECT media_id FROM UnvectorizedMediaChunks WHERE deleted = 0 LIMIT 1"
    ).fetchone()["media_id"]

    db.process_unvectorized_chunks(
        media_id,
        [
            {
                "text": "stamped one",
                "chunk_index": 1,
                "chunk_engine_version": "parity-1@385afa95",
                "metadata": {},
            },
            {"text": "unstamped two", "chunk_index": 2, "metadata": {}},
        ],
    )

    rows = {
        r["chunk_index"]: r
        for r in db.get_connection().execute(
            "SELECT chunk_index, chunk_engine_version FROM UnvectorizedMediaChunks "
            "WHERE deleted = 0 AND media_id = ? ORDER BY chunk_index",
            (media_id,),
        ).fetchall()
    }
    # _persist_chunks writes 0-based indices: the pre-existing row is index 0;
    # process_unvectorized_chunks uses the chunk's own 'chunk_index' (1, 2).
    assert rows[0]["chunk_engine_version"] is None  # pre-existing unstamped row
    assert rows[1]["chunk_engine_version"] == "parity-1@385afa95"
    assert rows[2]["chunk_engine_version"] is None  # unstamped stays NULL

    # the sync event payload must agree with the row (the trap)
    sync_rows = db.get_connection().execute(
        "SELECT payload FROM sync_log WHERE entity = "
        "'UnvectorizedMediaChunks' AND operation = 'create' ORDER BY change_id"
    ).fetchall()
    stamped_events = [
        r for r in sync_rows if '"chunk_engine_version": "parity-1@385afa95"' in r["payload"]
        or '"chunk_engine_version":"parity-1@385afa95"' in r["payload"]
    ]
    assert stamped_events, "sync events dropped the engine-version stamp"


def test_count_chunks_by_engine_version_groups(tmp_path):
    """The report method: NULL → 'legacy', stamped versions keyed verbatim."""
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService

    db = MediaDatabase(str(tmp_path / "m.db"), client_id="test")
    db.add_media_with_keywords(
        title="t", media_type="document", content="...", keywords=None, url=None,
        analysis_content=None, author=None,
        chunks=[
            {"text": "a", "metadata": {}, "chunk_engine_version": "parity-1@385afa95"},
            {"text": "b", "metadata": {}, "chunk_engine_version": "parity-1@385afa95"},
        ],
        chunk_options={},
    )
    svc = LocalRAGAdminService.__new__(LocalRAGAdminService)
    counts = svc.count_chunks_by_engine_version(db)
    assert counts == {"parity-1@385afa95": 2}


def test_legacy_chunk_report_line(tmp_path):
    """The RAG Admin report line: shown only when legacy chunks exist."""
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.RAG_Admin.local_rag_admin_service import LocalRAGAdminService

    db = MediaDatabase(str(tmp_path / "m.db"), client_id="test")
    # chunking_service is a dummy: the report path must not touch it
    svc = LocalRAGAdminService(db, chunking_service=object())

    # fully stamped library (or empty): nothing to report
    assert svc.get_legacy_chunk_report_line() == ""

    # a media row + a legacy (unstamped) chunk: the line appears
    db.add_media_with_keywords(
        title="t", media_type="document", content="...", keywords=None, url=None,
        analysis_content=None, author=None, chunk_options={},
    )
    media_id = db.get_connection().execute(
        "SELECT id FROM Media WHERE deleted = 0 LIMIT 1"
    ).fetchone()["id"]
    db.process_unvectorized_chunks(
        media_id,
        [{"text": "old chunk", "chunk_index": 0, "metadata": {}}],
    )
    assert svc.get_legacy_chunk_report_line() == "Chunked by an older engine: 1 items"

    # the line rides the diagnostics payload (the surviving read-only stats
    # surface -- the legacy UI was deleted with PR #669)
    diagnostics = svc.get_template_diagnostics()
    assert diagnostics["legacy_chunk_report"] == "Chunked by an older engine: 1 items"
