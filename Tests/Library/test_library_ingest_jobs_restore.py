from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.library_ingest_jobs import (
    LibraryIngestJobRegistry, IngestJobState, plan_restore,
)


def _restore(store, max_persisted=500, now_iso="2026-07-12T09:00:00+00:00"):
    reg = LibraryIngestJobRegistry()
    reg.attach_store(store)
    plan = plan_restore(store.all_jobs(), max_persisted=max_persisted, now_iso=now_iso)
    reg.restore(plan.jobs, plan.next_id)
    for j in plan.upsert:
        store.upsert_job(j)
    for jid in plan.delete_ids:
        store.delete_job(jid)
    return reg


def test_history_survives_restart_interrupted_normalized(tmp_path):
    store = LibraryIngestJobsDB(tmp_path / "jobs.db")
    a = LibraryIngestJobRegistry(); a.attach_store(store)
    done = a.submit(source_path="/done.pdf"); a.mark_parsing(done.job_id); a.mark_writing(done.job_id); a.mark_done(done.job_id, media_id=5)
    interrupted = a.submit(source_path="/x.mp4"); a.mark_parsing(interrupted.job_id)   # left PARSING (quit)
    failed = a.submit(source_path="/y.mp3"); a.mark_parsing(failed.job_id); a.mark_failed(failed.job_id, error="bad codec")
    store.close()

    store2 = LibraryIngestJobsDB(tmp_path / "jobs.db")            # reopen (restart)
    reg = _restore(store2)
    by_id = {j.job_id: j for j in reg.jobs()}
    assert by_id[done.job_id].state == IngestJobState.DONE and by_id[done.job_id].media_id == 5
    assert by_id[interrupted.job_id].state == IngestJobState.FAILED
    assert by_id[interrupted.job_id].error == "Interrupted by app restart"
    assert by_id[failed.job_id].state == IngestJobState.FAILED and by_id[failed.job_id].error == "bad codec"
    # _next_id advanced past the max so a new submit doesn't collide
    fresh = reg.submit(source_path="/z.txt")
    assert fresh.job_id == "ingest-job-4"
    store2.close()


def test_interrupted_retry_after_restart_requeues(tmp_path):
    store = LibraryIngestJobsDB(tmp_path / "jobs.db")
    a = LibraryIngestJobRegistry(); a.attach_store(store)
    j = a.submit(source_path="/x.mp4"); a.mark_parsing(j.job_id)     # interrupted
    store.close()
    reg = _restore(LibraryIngestJobsDB(tmp_path / "jobs.db"))
    restored = reg.jobs()[0]
    assert restored.state == IngestJobState.FAILED
    requeued = reg.requeue(restored.job_id)                          # AC2: retryable
    assert requeued.state == IngestJobState.QUEUED and requeued.retry_count == 1
