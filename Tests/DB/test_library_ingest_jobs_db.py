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
