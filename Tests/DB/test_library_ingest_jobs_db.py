import sqlite3
import pytest

from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry, _job_from_row


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
    assert row["error_detail"] == '{"category": "unsupported_file_type", "message": "nope"}'
    assert row["content_hash"] == "abc123"

    restored = _job_from_row(row)
    assert restored.ingest_options == {"pdf": {"engine": "pymupdf"}}
    assert restored.progress == {"message": "50%"}
    assert restored.error_detail == {"category": "unsupported_file_type", "message": "nope"}
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
    job = reg.submit(
        source_path="/a.mp3", detected_type="audio", origin="server"
    )
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
        "         '{\"pdf\": {\"ocr\": true}}', 2, 1)"
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
