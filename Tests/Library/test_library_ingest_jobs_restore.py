from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.Library.library_ingest_jobs import (
    LibraryIngestJobRegistry,
    IngestJobState,
    plan_restore,
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
    a = LibraryIngestJobRegistry()
    a.attach_store(store)
    done = a.submit(source_path="/done.pdf")
    a.mark_parsing(done.job_id)
    a.mark_writing(done.job_id)
    a.mark_done(done.job_id, media_id=5)
    interrupted = a.submit(source_path="/x.mp4")
    a.mark_parsing(interrupted.job_id)  # left PARSING (quit)
    failed = a.submit(source_path="/y.mp3")
    a.mark_parsing(failed.job_id)
    a.mark_failed(failed.job_id, error="bad codec")
    store.close()

    store2 = LibraryIngestJobsDB(tmp_path / "jobs.db")  # reopen (restart)
    reg = _restore(store2)
    by_id = {j.job_id: j for j in reg.jobs()}
    assert (
        by_id[done.job_id].state == IngestJobState.DONE
        and by_id[done.job_id].media_id == 5
    )
    assert by_id[interrupted.job_id].state == IngestJobState.FAILED
    assert by_id[interrupted.job_id].error == "Interrupted by app restart"
    assert (
        by_id[failed.job_id].state == IngestJobState.FAILED
        and by_id[failed.job_id].error == "bad codec"
    )
    # _next_id advanced past the max so a new submit doesn't collide
    fresh = reg.submit(source_path="/z.txt")
    assert fresh.job_id == "ingest-job-4"
    store2.close()


def test_interrupted_retry_after_restart_requeues(tmp_path):
    store = LibraryIngestJobsDB(tmp_path / "jobs.db")
    a = LibraryIngestJobRegistry()
    a.attach_store(store)
    j = a.submit(source_path="/x.mp4")
    a.mark_parsing(j.job_id)  # interrupted
    store.close()
    reg = _restore(LibraryIngestJobsDB(tmp_path / "jobs.db"))
    restored = reg.jobs()[0]
    assert restored.state == IngestJobState.FAILED
    requeued = reg.requeue(restored.job_id)  # AC2: retryable
    assert requeued.state == IngestJobState.QUEUED and requeued.retry_count == 1


# --------------------------------------------------------------------------
# merge_restored: the deferral window opened by TASK-21111(c)
# --------------------------------------------------------------------------


def test_merge_restored_on_an_empty_registry_is_exactly_restore(tmp_path):
    """The normal startup case must be indistinguishable from before."""
    store = LibraryIngestJobsDB(tmp_path / "jobs.db")
    seeder = LibraryIngestJobRegistry()
    seeder.attach_store(store)
    seeder.submit(source_path="/a.pdf")
    seeder.submit(source_path="/b.pdf")
    store.close()

    store2 = LibraryIngestJobsDB(tmp_path / "jobs.db")
    plan = plan_restore(
        store2.all_jobs(), max_persisted=500, now_iso="2026-08-23T00:00:00+00:00"
    )
    merged = LibraryIngestJobRegistry()
    merged.merge_restored(plan.jobs, plan.next_id)
    plain = LibraryIngestJobRegistry()
    plain.restore(plan.jobs, plan.next_id)

    assert [j.job_id for j in merged.jobs()] == [j.job_id for j in plain.jobs()]
    assert merged.submit(source_path="/c.pdf").job_id == plain.submit(
        source_path="/c.pdf"
    ).job_id
    store2.close()


def test_merge_restored_keeps_a_job_submitted_during_the_restore_window(tmp_path):
    """The read is off-thread now; a submit can beat the seeding callback.

    Plain `restore` would delete that job outright.
    """
    store = LibraryIngestJobsDB(tmp_path / "jobs.db")
    seeder = LibraryIngestJobRegistry()
    seeder.attach_store(store)
    seeder.submit(source_path="/persisted-1.pdf")
    seeder.submit(source_path="/persisted-2.pdf")
    store.close()

    store2 = LibraryIngestJobsDB(tmp_path / "jobs.db")
    plan = plan_restore(
        store2.all_jobs(), max_persisted=500, now_iso="2026-08-23T00:00:00+00:00"
    )

    live = LibraryIngestJobRegistry()
    raced = live.submit(source_path="/submitted-during-startup.pdf")
    live.merge_restored(plan.jobs, plan.next_id)

    # The live job survives, ordered last (it is the newest), and the
    # persisted row that happened to share its id (`ingest-job-1`) is dropped
    # rather than duplicated -- two entries under one id would make every
    # id-keyed mutation ambiguous, and the live job is the one with work
    # attached to it.
    assert [j.source_path for j in reversed(live.jobs())] == [
        "/persisted-2.pdf",
        "/submitted-during-startup.pdf",
    ]
    assert sum(1 for j in live.jobs() if j.job_id == raced.job_id) == 1
    # No future allocation can collide with a restored id either.
    assert live.submit(source_path="/next.pdf").job_id == "ingest-job-3"
    store2.close()
