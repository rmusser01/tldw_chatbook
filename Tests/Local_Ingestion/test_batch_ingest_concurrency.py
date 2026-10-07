"""B15: bounded concurrency for the CLI batch-ingest seam (parse stage only).

SAFETY GATE FINDINGS (plan Step 1 -- why the whole call is NOT parallelized):

* ``ingest_local_file(file_path, media_db, ...)`` does NOT construct its own
  DB connection per call. It receives the CALLER's shared ``MediaDatabase``
  instance (``batch_ingest_files``'s ``media_db`` parameter) and runs
  ``persist_parsed_media(payload, media_db)`` -- documented as "the *only*
  place in the ingest pipeline that writes to ``media_db``" and as "meant to
  always run on the single Library ingest writer thread" (see
  ``persist_parsed_media``'s docstring, and the F3 split in
  ``ingest_parse_worker.py``: pool workers "never touch the media db").
* ``MediaDatabase`` connections are thread-local (``Client_Media_DB_v2``
  ``_get_connection``). For FILE databases that means a thread pool would
  serialize on SQLite's single-writer lock (WAL + 10 s busy timeout) --
  correct but contention-prone. For ``:memory:`` databases, thread-local
  connections mean each pool thread gets its OWN private empty database
  (there is no shared-cache URI anywhere in the DB layer), so parallel
  writes would silently vanish instead of failing.

Verdict: the gate FAILS for whole-call parallelization. The documented
partial fix lands instead: the parse/analyze sub-stage
(``parse_local_file_for_ingest`` -- no DB I/O by contract) fans out under a
bounded ThreadPoolExecutor(4); persistence stays SERIAL on the calling
thread, in input order. ``stop_on_error=True`` keeps the historical strictly
serial loop (halt at the first failure so later files are never parsed and
never incur analysis spend).

Tests here stub the two seams (parse records concurrency; persist records
the writing thread) and assert order, error aggregation, the concurrency
bound, and the single-writer contract.
"""

import threading
import time

import pytest

from tldw_chatbook.Local_Ingestion import local_file_ingestion as lfi


class ParseSpy:
    """Fake ``parse_local_file_for_ingest``: records peak concurrency and
    returns a marker payload per file."""

    def __init__(self, delay: float = 0.0, fail_paths: set[str] | None = None):
        self.delay = delay
        self.fail_paths = fail_paths or set()
        self.parsed: list[str] = []
        self.peak = 0
        self._live = 0
        self._lock = threading.Lock()

    def __call__(self, file_path: str, options: dict):
        with self._lock:
            self.parsed.append(file_path)
            self._live += 1
            self.peak = max(self.peak, self._live)
        try:
            if self.delay:
                threading.Event().wait(self.delay)
            if file_path in self.fail_paths:
                raise ValueError(f"parse failed: {file_path}")
            return {
                "media_type": "plaintext",
                "file_type": "plaintext",
                "title": f"title-for-{file_path}",
                "author": "Unknown",
                "content": f"content-for-{file_path}",
                "keywords": [],
                "url": f"file://{file_path}",
                "analysis_content": "",
                "chunks": None,
                "chunk_options": None,
                "metadata": None,
                "file_path": str(file_path),
            }
        finally:
            with self._lock:
                self._live -= 1


@pytest.fixture()
def stubbed_seams(monkeypatch):
    """Stub parse + persist; persist records the writing thread."""
    persist_threads: list[threading.Thread] = []
    persist_order: list[str] = []

    def fake_persist(payload, media_db, **kwargs):
        persist_threads.append(threading.current_thread())
        persist_order.append(payload["file_path"])
        return 101 + len(persist_order), None, "ok"

    monkeypatch.setattr(lfi, "persist_parsed_media", fake_persist)
    return {"persist_threads": persist_threads, "persist_order": persist_order}


def _files(tmp_path, n: int) -> list:
    return [tmp_path / f"file_{i:02d}.txt" for i in range(n)]


def test_parse_stage_concurrent_and_bounded(tmp_path, stubbed_seams, monkeypatch):
    spy = ParseSpy(delay=0.03)
    monkeypatch.setattr(lfi, "parse_local_file_for_ingest", spy)
    files = _files(tmp_path, 12)

    results = lfi.batch_ingest_files(files, media_db=object())

    assert len(results) == 12
    assert spy.peak >= 2, "parse stage is still fully serial"
    assert spy.peak <= 4, f"peak parse concurrency {spy.peak} exceeded the bound of 4"


def test_results_preserve_input_order(tmp_path, stubbed_seams, monkeypatch):
    monkeypatch.setattr(lfi, "parse_local_file_for_ingest", ParseSpy(delay=0.01))
    files = _files(tmp_path, 8)

    results = lfi.batch_ingest_files(files, media_db=object())

    assert [r["title"] for r in results] == [f"title-for-{f}" for f in files]
    assert [r["file_path"] for r in results] == [str(f) for f in files]
    assert all("media_id" in r for r in results)
    # Persistence happened in input order on the single calling thread.
    assert stubbed_seams["persist_order"] == [str(f) for f in files]
    assert stubbed_seams["persist_threads"], "persist never ran"
    assert all(t is threading.main_thread() for t in stubbed_seams["persist_threads"])


def test_one_failing_file_does_not_prevent_others(tmp_path, stubbed_seams, monkeypatch):
    files = _files(tmp_path, 6)
    spy = ParseSpy(fail_paths={str(files[2])})
    monkeypatch.setattr(lfi, "parse_local_file_for_ingest", spy)

    results = lfi.batch_ingest_files(files, media_db=object())

    assert len(results) == 6
    failed = results[2]
    assert failed == {
        "file_path": str(files[2]),
        "error": "parse failed: " + str(files[2]),
        "success": False,
    }
    for i, result in enumerate(results):
        if i != 2:
            assert result["title"] == f"title-for-{files[i]}"


def test_stop_on_error_keeps_strict_serial_halt(tmp_path, stubbed_seams, monkeypatch):
    files = _files(tmp_path, 5)
    spy = ParseSpy(fail_paths={str(files[1])})
    monkeypatch.setattr(lfi, "parse_local_file_for_ingest", spy)

    with pytest.raises(lfi.FileIngestionError, match="Batch ingestion stopped"):
        lfi.batch_ingest_files(files, media_db=object(), stop_on_error=True)

    # Only files 0 and 1 were attempted; files 2-4 were never parsed (no
    # analysis spend past the failure), matching the historical semantics.
    assert spy.parsed == [str(files[0]), str(files[1])]
    assert stubbed_seams["persist_order"] == [str(files[0])]


def test_chunk_options_sentinel_becomes_fresh_dict_per_file(
    tmp_path, stubbed_seams, monkeypatch
):
    seen_options: list[dict] = []

    def spy_parse(file_path, options):
        seen_options.append(options["chunk_options"])
        return ParseSpy()(file_path, options)

    monkeypatch.setattr(lfi, "parse_local_file_for_ingest", spy_parse)
    files = _files(tmp_path, 3)

    results = lfi.batch_ingest_files(files, media_db=object())  # sentinel default

    assert len(results) == 3
    # Every file got an independent defaults dict (processors setdefault into
    # it), and never the shared sentinel object.
    assert all(opts == {} for opts in seen_options)
    assert seen_options[0] is not seen_options[1]


def test_explicit_chunk_options_dict_shared_not_mutated_across_files(
    tmp_path, stubbed_seams, monkeypatch
):
    """A caller's explicit dict is passed per file (copied in the concurrent
    path so concurrent parses cannot contaminate each other)."""
    explicit = {"method": "words", "max_size": 10}
    seen: list[dict] = []

    def spy_parse(file_path, options):
        seen.append(options["chunk_options"])
        return ParseSpy()(file_path, options)

    monkeypatch.setattr(lfi, "parse_local_file_for_ingest", spy_parse)
    files = _files(tmp_path, 4)

    lfi.batch_ingest_files(files, media_db=object(), chunk_options=explicit)

    assert all(opts == explicit for opts in seen)
    # The caller's dict itself is never handed to more than one parse at a
    # time -- the concurrent path copies it per file.
    assert all(opts is not explicit for opts in seen)
    assert explicit == {"method": "words", "max_size": 10}
