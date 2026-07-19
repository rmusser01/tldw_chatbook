import sqlite3
import pytest

from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry, IngestJobState


def _db(tmp_path):
    return LibraryIngestJobsDB(tmp_path / "jobs.db")


def test_upsert_and_all_jobs_roundtrip_ordered(tmp_path):
    reg = LibraryIngestJobRegistry()
    j1 = reg.submit(source_path="/a.mp3", title="A", keywords=("k1", "k2"), detected_type="audio")
    j2 = reg.submit(source_path="/b.txt", title="B")
    db = _db(tmp_path)
    db.upsert_job(j1)
    db.upsert_job(j2)
    rows = db.all_jobs()
    assert [r["job_id"] for r in rows] == [j1.job_id, j2.job_id]     # seq order
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
    db.upsert_job(reg.jobs()[0])          # same job_id, now PARSING
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
