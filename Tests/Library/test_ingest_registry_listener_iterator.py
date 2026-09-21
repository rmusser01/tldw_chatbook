"""Registry listeners must iterate jobs without deep-copying (TASK-32804.5).

The research-source listener called registry.jobs() -- a per-job deep copy of
the whole queue -- on every registry mutation, so a folder import (one listener
fire per file) was O(n^2). iter_jobs_for_listeners yields the same visible jobs
newest-first without copying, for read-only consumers.
"""

from pathlib import Path

import tldw_chatbook.Library.library_ingest_jobs as lij
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry


def _registry_with_jobs(tmp_path, n):
    reg = LibraryIngestJobRegistry()
    for i in range(n):
        reg.submit(source_path=str(tmp_path / f"f{i}.wav"))
    return reg


def test_iter_matches_jobs_content_and_order(tmp_path):
    reg = _registry_with_jobs(tmp_path, 4)
    assert [j.job_id for j in reg.iter_jobs_for_listeners()] == [
        j.job_id for j in reg.jobs()
    ]


def test_iter_yields_registry_own_objects_not_copies(tmp_path):
    reg = _registry_with_jobs(tmp_path, 3)
    live = list(reg.iter_jobs_for_listeners())
    # Identity: the iterator yields the registry's own job objects...
    assert all(any(j is internal for internal in reg._jobs) for j in live)
    # ...while jobs() returns copies (distinct objects with equal ids).
    snapshot = reg.jobs()
    assert all(
        not any(copy is internal for internal in reg._jobs) for copy in snapshot
    )


def test_iter_does_not_call_copy_job(tmp_path, monkeypatch):
    reg = _registry_with_jobs(tmp_path, 5)
    calls = {"n": 0}
    real = lij._copy_job
    monkeypatch.setattr(
        lij, "_copy_job", lambda job: (calls.__setitem__("n", calls["n"] + 1), real(job))[1]
    )
    consumed = list(reg.iter_jobs_for_listeners())
    assert len(consumed) == 5
    assert calls["n"] == 0, "listener iterator must not deep-copy jobs"
