import sqlite3
from dataclasses import replace

import pytest

from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.library_ingest_jobs import (
    LibraryIngestJobRegistry,
    _job_from_row,
)


_GENUINE_V5_SCHEMA = """
CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL);
INSERT INTO schema_version (version) VALUES (5);

CREATE TABLE ingest_jobs (
    seq INTEGER PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL,
    source_path TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '[]',
    perform_analysis INTEGER NOT NULL DEFAULT 0,
    chunk_enabled INTEGER NOT NULL DEFAULT 0,
    chunk_size INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL CHECK (state IN ('queued','parsing','writing','done','failed','cancelled')),
    retry_count INTEGER NOT NULL DEFAULT 0,
    detected_type TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    finished_at_wall TEXT NOT NULL DEFAULT '',
    media_id INTEGER,
    superseded INTEGER NOT NULL DEFAULT 0,
    dismissed INTEGER NOT NULL DEFAULT 0,
    permanent INTEGER NOT NULL DEFAULT 0,
    ingest_options TEXT DEFAULT '{}',
    error_detail TEXT DEFAULT NULL,
    progress TEXT DEFAULT NULL,
    content_hash TEXT DEFAULT NULL,
    origin TEXT NOT NULL DEFAULT 'local' CHECK (origin IN ('local','server')),
    remote_job_id TEXT DEFAULT NULL,
    batch_id TEXT DEFAULT NULL,
    remote_media_id TEXT DEFAULT NULL,
    retry_of_job_id TEXT DEFAULT NULL,
    stt_failure_provenance_json TEXT DEFAULT NULL,
    retry_source_failure_provenance_json TEXT DEFAULT NULL
);
"""


def _db(tmp_path):
    return LibraryIngestJobsDB(tmp_path / "jobs.db")


def test_upsert_and_all_jobs_roundtrip_ordered(tmp_path):
    reg = LibraryIngestJobRegistry()
    j1 = reg.submit(
        source_path="/a.mp3", title="A", keywords=("k1", "k2"), detected_type="audio"
    )
    j2 = reg.submit(source_path="/b.txt", title="B")
    db = _db(tmp_path)
    db.upsert_job(j1)
    db.upsert_job(j2)
    rows = db.all_jobs()
    assert [r["job_id"] for r in rows] == [j1.job_id, j2.job_id]  # seq order
    assert rows[0]["source_path"] == "/a.mp3" and rows[0]["detected_type"] == "audio"
    assert rows[0]["keywords"] == '["k1", "k2"]'
    assert rows[0]["state"] == "queued" and rows[0]["retry_count"] == 0
    db.close()


def test_upsert_is_idempotent_update_in_place(tmp_path):
    reg = LibraryIngestJobRegistry()
    j = reg.submit(source_path="/a.mp3")
    db = _db(tmp_path)
    db.upsert_job(j)
    reg.mark_parsing(j.job_id, detected_type="audio")
    db.upsert_job(reg.jobs()[0])  # same job_id, now PARSING
    rows = db.all_jobs()
    assert len(rows) == 1 and rows[0]["state"] == "parsing"
    db.close()


def test_retry_pair_upsert_rolls_back_both_rows_on_second_write_failure(
    tmp_path,
    monkeypatch,
):
    reg = LibraryIngestJobRegistry()
    original = reg.submit(source_path="/a.mp3")
    db = _db(tmp_path)
    db.upsert_job(original)

    superseded = replace(original, superseded=True)
    retry = replace(
        original,
        job_id="ingest-job-2",
        retry_of_job_id=original.job_id,
    )
    real_upsert = db._upsert_job
    writes = 0

    def fail_second_write(conn, job):
        nonlocal writes
        writes += 1
        real_upsert(conn, job)
        if writes == 2:
            raise RuntimeError("second write failed")

    monkeypatch.setattr(db, "_upsert_job", fail_second_write)

    with pytest.raises(RuntimeError, match="second write failed"):
        db.upsert_retry(superseded, retry)

    rows = db.all_jobs()
    assert [row["job_id"] for row in rows] == [original.job_id]
    assert rows[0]["superseded"] == 0
    db.close()


def test_delete_job(tmp_path):
    reg = LibraryIngestJobRegistry()
    j = reg.submit(source_path="/a.mp3")
    db = _db(tmp_path)
    db.upsert_job(j)
    db.delete_job(j.job_id)
    assert db.all_jobs() == []
    db.close()


def test_state_check_constraint_rejects_bad_state(tmp_path):
    db = _db(tmp_path)
    conn = db._get_connection()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ingest_jobs (seq, job_id, source_path, state) VALUES (1,'x','/p','bogus')"
        )
    db.close()


def test_db_migration_v1_to_v2(tmp_path):
    db_path = tmp_path / "jobs.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL);
        INSERT INTO schema_version (version) VALUES (1);

        CREATE TABLE ingest_jobs (
            seq INTEGER PRIMARY KEY,
            job_id TEXT UNIQUE NOT NULL,
            source_path TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            author TEXT NOT NULL DEFAULT '',
            keywords TEXT NOT NULL DEFAULT '[]',
            perform_analysis INTEGER NOT NULL DEFAULT 0,
            chunk_enabled INTEGER NOT NULL DEFAULT 0,
            chunk_size INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL CHECK (state IN ('queued','parsing','writing','done','failed')),
            retry_count INTEGER NOT NULL DEFAULT 0,
            detected_type TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            finished_at_wall TEXT NOT NULL DEFAULT '',
            media_id INTEGER,
            superseded INTEGER NOT NULL DEFAULT 0,
            dismissed INTEGER NOT NULL DEFAULT 0,
            permanent INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute(
        "INSERT INTO ingest_jobs (seq, job_id, source_path, state) VALUES (1, 'ingest-job-1', '/a.mp3', 'queued')"
    )
    conn.commit()
    conn.close()

    db = LibraryIngestJobsDB(db_path)
    rows = db.all_jobs()
    assert len(rows) == 1
    assert rows[0]["ingest_options"] == "{}"
    assert rows[0]["error_detail"] is None
    assert rows[0]["progress"] is None
    assert rows[0]["content_hash"] is None
    db.close()


def test_job_round_trip_with_json_columns(tmp_path):
    reg = LibraryIngestJobRegistry()
    job = reg.submit(source_path="/a.pdf", title="A")
    job.ingest_options = {"pdf": {"engine": "pymupdf"}}
    job.progress = {"message": "50%"}
    job.error_detail = {"category": "unsupported_file_type", "message": "nope"}
    job.content_hash = "abc123"

    db = _db(tmp_path)
    db.upsert_job(job)
    rows = db.all_jobs()
    assert len(rows) == 1
    row = rows[0]
    assert row["ingest_options"] == '{"pdf": {"engine": "pymupdf"}}'
    assert row["progress"] == '{"message": "50%"}'
    assert (
        row["error_detail"]
        == '{"category": "unsupported_file_type", "message": "nope"}'
    )
    assert row["content_hash"] == "abc123"

    restored = _job_from_row(row)
    assert restored.ingest_options == {"pdf": {"engine": "pymupdf"}}
    assert restored.progress == {"message": "50%"}
    assert restored.error_detail == {
        "category": "unsupported_file_type",
        "message": "nope",
    }
    assert restored.content_hash == "abc123"
    db.close()


def test_v4_to_v5_stt_lineage_migration_is_nullable_and_atomic(tmp_path):
    db = _db(tmp_path)
    reg = LibraryIngestJobRegistry()
    job = reg.submit(source_path="/kept.wav", title="Kept")
    db.upsert_job(job)
    conn = db._get_connection()
    for column in (
        "retry_of_job_id",
        "stt_failure_provenance_json",
        "retry_source_failure_provenance_json",
    ):
        conn.execute(f"ALTER TABLE ingest_jobs DROP COLUMN {column}")
    conn.execute("UPDATE schema_version SET version = 4")
    conn.commit()

    original_columns = db._STT_LINEAGE_COLUMNS
    db._STT_LINEAGE_COLUMNS = (*original_columns, ("broken)", "TEXT"))
    try:
        with pytest.raises(sqlite3.OperationalError):
            db._migrate_v4_to_v5()
    finally:
        db._STT_LINEAGE_COLUMNS = original_columns

    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 4
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(ingest_jobs)")}
    assert "retry_of_job_id" not in columns
    assert "stt_failure_provenance_json" not in columns
    assert "retry_source_failure_provenance_json" not in columns
    assert (
        conn.execute(
            "SELECT title FROM ingest_jobs WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()[0]
        == "Kept"
    )

    db._migrate_v4_to_v5()
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 5
    row = db.all_jobs()[0]
    assert row["retry_of_job_id"] is None
    assert row["stt_failure_provenance_json"] is None
    assert row["retry_source_failure_provenance_json"] is None
    db.close()


# --- schema v3: remote-job columns + cancelled state (task-684.2) ------------


def test_v3_persists_origin_and_remote_ids(tmp_path):
    """A server job needs somewhere to record where it runs and its remote ids.

    Without these the ingest queue can only ever hold local jobs, which is what
    blocked routing a server submission at all (task-684.2).
    """
    reg = LibraryIngestJobRegistry()
    job = reg.submit(source_path="/a.mp3", detected_type="audio", origin="server")
    job = reg.attach_remote(job.job_id, remote_job_id="4171", batch_id="batch-9")

    db = _db(tmp_path)
    db.upsert_job(job)
    row = db.all_jobs()[0]

    assert row["origin"] == "server"
    assert row["remote_job_id"] == "4171"
    assert row["batch_id"] == "batch-9"
    db.close()


def test_v3_defaults_origin_to_local(tmp_path):
    """Every pre-existing job is a local one; nothing has to be backfilled."""
    reg = LibraryIngestJobRegistry()
    job = reg.submit(source_path="/b.txt")

    db = _db(tmp_path)
    db.upsert_job(job)
    row = db.all_jobs()[0]

    assert row["origin"] == "local"
    assert row["remote_job_id"] is None
    assert row["batch_id"] is None
    db.close()


def test_v3_accepts_the_cancelled_state(tmp_path):
    """The server reports cancelled, which the v2 CHECK constraint rejected."""
    db = _db(tmp_path)
    conn = db._get_connection()
    conn.execute(
        "INSERT INTO ingest_jobs (seq, job_id, source_path, state)"
        " VALUES (1,'ingest-job-1','/p','cancelled')"
    )
    conn.commit()
    assert db.all_jobs()[0]["state"] == "cancelled"
    db.close()


def test_v3_still_rejects_a_bogus_state(tmp_path):
    """Relaxing the CHECK for cancelled must not turn it into a free-text column."""
    db = _db(tmp_path)
    conn = db._get_connection()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ingest_jobs (seq, job_id, source_path, state)"
            " VALUES (2,'ingest-job-2','/p','bogus')"
        )
    db.close()


def test_db_migration_v2_to_v3_preserves_existing_rows(tmp_path):
    """Migrating a populated v2 database keeps its jobs and their values.

    The CHECK constraint has to be replaced, which SQLite can only do by
    rebuilding the table -- so this asserts the copy, not just the new columns.
    """
    db_path = tmp_path / "jobs.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL);
        INSERT INTO schema_version (version) VALUES (2);

        CREATE TABLE ingest_jobs (
            seq INTEGER PRIMARY KEY,
            job_id TEXT UNIQUE NOT NULL,
            source_path TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            author TEXT NOT NULL DEFAULT '',
            keywords TEXT NOT NULL DEFAULT '[]',
            perform_analysis INTEGER NOT NULL DEFAULT 0,
            chunk_enabled INTEGER NOT NULL DEFAULT 0,
            chunk_size INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL CHECK (state IN ('queued','parsing','writing','done','failed')),
            retry_count INTEGER NOT NULL DEFAULT 0,
            detected_type TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            finished_at_wall TEXT NOT NULL DEFAULT '',
            media_id INTEGER,
            superseded INTEGER NOT NULL DEFAULT 0,
            dismissed INTEGER NOT NULL DEFAULT 0,
            permanent INTEGER NOT NULL DEFAULT 0,
            ingest_options TEXT DEFAULT '{}',
            error_detail TEXT DEFAULT NULL,
            progress TEXT DEFAULT NULL,
            content_hash TEXT DEFAULT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO ingest_jobs"
        " (seq, job_id, source_path, title, state, media_id, detected_type,"
        "  ingest_options, retry_count, permanent)"
        " VALUES (7, 'ingest-job-7', '/kept.pdf', 'Kept', 'done', 42, 'pdf',"
        '         \'{"pdf": {"ocr": true}}\', 2, 1)'
    )
    conn.commit()
    conn.close()

    db = LibraryIngestJobsDB(db_path)
    rows = db.all_jobs()

    assert len(rows) == 1
    row = rows[0]
    assert row["seq"] == 7 and row["job_id"] == "ingest-job-7"
    assert row["source_path"] == "/kept.pdf" and row["title"] == "Kept"
    assert row["state"] == "done" and row["media_id"] == 42
    assert row["detected_type"] == "pdf" and row["retry_count"] == 2
    assert row["permanent"] == 1
    assert row["ingest_options"] == '{"pdf": {"ocr": true}}'
    # New columns, defaulted for a pre-existing local job.
    assert row["origin"] == "local"
    assert row["remote_job_id"] is None and row["batch_id"] is None
    db.close()


def test_fresh_v6_schema_has_nullable_research_operation_link(tmp_path):
    db = _db(tmp_path)
    columns = {
        row["name"]: row
        for row in db._get_connection().execute("PRAGMA table_info(ingest_jobs)")
    }
    assert columns["research_source_operation_id"]["notnull"] == 0
    assert (
        db._get_connection().execute("SELECT version FROM schema_version").fetchone()[0]
        == db._CURRENT_SCHEMA_VERSION
    )
    db.close()


def test_genuine_v5_migration_preserves_row_and_adds_nullable_operation_link(
    tmp_path,
):
    path = tmp_path / "historical-v5.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(_GENUINE_V5_SCHEMA)
    connection.execute(
        """
        INSERT INTO ingest_jobs
          (seq, job_id, source_path, title, author, keywords, perform_analysis,
           chunk_enabled, chunk_size, state, retry_count, detected_type, error,
           finished_at_wall, media_id, superseded, dismissed, permanent,
           ingest_options, error_detail, progress, content_hash, origin,
           remote_job_id, batch_id, remote_media_id, retry_of_job_id,
           stt_failure_provenance_json, retry_source_failure_provenance_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            7,
            "ingest-job-7",
            "/kept.pdf",
            "Kept",
            "Author",
            '["one"]',
            1,
            1,
            900,
            "done",
            2,
            "pdf",
            "",
            "2026-08-20T00:00:00Z",
            42,
            0,
            0,
            0,
            '{"pdf": {"ocr": true}}',
            None,
            '{"message": "done"}',
            "sha256-kept",
            "local",
            None,
            "batch-kept",
            None,
            "ingest-job-6",
            None,
            None,
        ),
    )
    connection.commit()
    assert "research_source_operation_id" not in {
        row[1] for row in connection.execute("PRAGMA table_info(ingest_jobs)")
    }
    connection.close()

    db = LibraryIngestJobsDB(path)
    row = db.all_jobs()[0]

    assert (
        db._get_connection().execute("SELECT version FROM schema_version").fetchone()[0]
        == db._CURRENT_SCHEMA_VERSION
    )
    assert row["research_source_operation_id"] is None
    assert row["job_id"] == "ingest-job-7"
    assert row["title"] == "Kept"
    assert row["media_id"] == 42
    assert row["retry_of_job_id"] == "ingest-job-6"
    assert row["batch_id"] == "batch-kept"
    db.close()

    reopened = LibraryIngestJobsDB(path)
    reopened_row = reopened.all_jobs()[0]
    assert reopened_row["research_source_operation_id"] is None
    assert reopened_row["job_id"] == "ingest-job-7"
    reopened.close()


def test_operation_link_survives_local_and_server_completion_and_reload(tmp_path):
    db = _db(tmp_path)
    registry = LibraryIngestJobRegistry()
    registry.attach_store(db)

    local = registry.submit(
        source_path="/local.txt",
        research_source_operation_id="operation-local",
    )
    assert registry.mark_parsing(local.job_id) is not None
    assert registry.mark_writing(local.job_id) is not None
    local_done = registry.mark_done(local.job_id, media_id=41)
    assert local_done is not None
    assert local_done.research_source_operation_id == "operation-local"
    assert local_done.media_id == 41
    assert local_done.remote_media_id is None

    server = registry.submit(
        source_path="https://example.test/source",
        origin="server",
        research_source_operation_id="operation-server",
    )
    server = registry.attach_remote(
        server.job_id,
        remote_job_id="remote-job",
        batch_id="remote-batch",
    )
    assert server is not None
    assert server.research_source_operation_id == "operation-server"
    server_done = registry.mark_remote_done(server.job_id, remote_media_id="900")
    assert server_done is not None
    assert server_done.research_source_operation_id == "operation-server"
    assert server_done.media_id is None
    assert server_done.remote_media_id == "900"

    restored = {row["job_id"]: _job_from_row(row) for row in db.all_jobs()}
    assert restored[local.job_id].research_source_operation_id == "operation-local"
    assert restored[server.job_id].research_source_operation_id == "operation-server"
    db.close()


def test_retry_preserves_operation_link_in_memory_and_persisted_rows(tmp_path):
    db = _db(tmp_path)
    registry = LibraryIngestJobRegistry()
    registry.attach_store(db)
    source = registry.submit(
        source_path="https://example.test/retry",
        origin="server",
        research_source_operation_id="operation-retry",
    )
    assert registry.mark_failed(source.job_id, error="retryable") is not None

    retry = registry.requeue(source.job_id)

    assert retry is not None
    assert retry.origin == "server"
    assert retry.research_source_operation_id == "operation-retry"
    rows = {row["job_id"]: row for row in db.all_jobs()}
    assert rows[source.job_id]["research_source_operation_id"] == "operation-retry"
    assert rows[retry.job_id]["research_source_operation_id"] == "operation-retry"
    assert _job_from_row(rows[retry.job_id]).research_source_operation_id == (
        "operation-retry"
    )
    db.close()


def test_local_and_server_completion_id_spaces_remain_disjoint():
    registry = LibraryIngestJobRegistry()
    local = registry.submit(source_path="/local.txt")
    server = registry.submit(
        source_path="https://example.test/source",
        origin="server",
    )

    assert registry.mark_remote_done(local.job_id, remote_media_id="900") is None
    assert registry.mark_done(server.job_id, media_id=900) is None
