# TASK-207 Live Local Ingest Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give long-running Local Library imports truthful, non-blocking stage progress and exact percentages only when the parser exposes a real bounded measurement.

**Architecture:** Add a stdlib-only parse-progress contract beside the spawn-safe worker, then let each parse-pool generation own a bounded queue and coalescing drain thread. The app validates and fences events before transient registry updates; the Library screen consumes a progress-specific listener to patch stable row widgets in place while lifecycle and action-structure changes keep their existing recompose path.

**Tech Stack:** Python 3.11+, `multiprocessing` spawn pools, Textual 8.x, dataclasses, SQLite-backed ingest registry, pytest.

## Global Constraints

- Keep existing `queued -> parsing -> writing -> terminal` lifecycle states; add no state or database schema.
- Progress is non-authoritative telemetry. Queue saturation, channel failure, or shutdown may drop ticks but must never block or change a parse result.
- Accept a percentage only when it is finite and inside 0-100. Never interpolate, weight phases, clamp invalid values into plausibility, or synthesize overall completion.
- Phase changes remove the previous phase's percentage. Local live ticks are memory-only; lifecycle, terminal, and existing Server progress persistence remain authoritative.
- Coalesce ordinary progress updates to approximately one batch per 250 ms and expose a deterministic clock/flush seam in tests.
- All cross-process values must be bounded picklable primitives before IPC and revalidated after IPC. Messages are markup-disabled, single-line, and length-bounded.
- Never join or wait for a pool, queue, or progress thread on the Textual event-loop thread.
- Preserve the current Windows real-stderr/resource-tracker workaround, worker-sentinel recovery, generation fencing, payload-ready protection, local-STT cancellation actions, and import-weight guard.
- Add no runtime dependency. Keep `ingest_parse_worker` and its new progress-contract import free of heavy ingestion modules at module scope.
- Follow ADR-061 and ADR-014. Source-of-truth design: `Docs/superpowers/specs/2026-08-12-task-207-live-ingest-progress-design.md`.

---

## File structure

- Create `tldw_chatbook/Local_Ingestion/ingest_parse_progress.py`: stdlib-only event normalization, worker queue sink, and deterministic latest-per-job coalescer.
- Create `Tests/Local_Ingestion/test_ingest_parse_progress.py`: pure contract, non-blocking emission, and coalescing tests.
- Modify `tldw_chatbook/Local_Ingestion/ingest_parse_worker.py`: combined worker initializer and per-job progress callback binding.
- Modify `tldw_chatbook/Local_Ingestion/local_file_ingestion.py`: optional best-effort progress callback and truthful observable stage instrumentation.
- Modify `Tests/Local_Ingestion/test_ingest_parse_worker.py`: parser/worker propagation, import-weight, and real spawn coverage.
- Modify `Tests/Local_Ingestion/test_local_file_ingestion.py`: audio/video transcription callback forwarding.
- Modify `tldw_chatbook/Library/library_ingest_jobs.py`: progress-specific listeners, transient update option, and writing-stage reset.
- Modify `tldw_chatbook/Library/library_ingest_state.py`: one pure progress-line formatter with readable phase fallbacks.
- Modify `Tests/Library/test_library_ingest_jobs.py` and `Tests/Library/test_library_ingest_state.py`: registry, persistence, and formatting contracts.
- Modify `tldw_chatbook/app.py`: parse-pool resource bundle, queue/drain ownership, event fencing, submission context, broken-pool cleanup, and shutdown cleanup.
- Modify `Tests/Library/test_library_ingest_runner.py`: coordinator, coalescing, fencing, resource lifecycle, and Windows spawn tests.
- Modify `tldw_chatbook/Widgets/Library/library_ingest_canvas.py`: stable progress widget composition and shared formatter use.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`: progress listener lifecycle and in-place patcher with structural fallback.
- Modify `Tests/UI/test_library_ingest_canvas.py` and `Tests/UI/test_library_shell.py`: copy, action, identity, focus, and scroll behavior.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss`: dedicated muted progress styling.
- Regenerate `tldw_chatbook/css/tldw_cli_modular.tcss` with the repository CSS builder; never edit the bundle by hand.
- Modify `Docs/User_Guide/library.md`: describe truthful Local progress and indeterminate stages.
- Modify `backlog/tasks/task-207 - Live-parse-progress-for-ingest-jobs-progress_percent-progress_message.md`: track plan, evidence, acceptance criteria, and closeout.

---

### Task 1: Spawn-safe progress event and coalescer

**Files:**
- Create: `tldw_chatbook/Local_Ingestion/ingest_parse_progress.py`
- Create: `Tests/Local_Ingestion/test_ingest_parse_progress.py`

**Interfaces:**
- Produces: `ParseProgressEvent(generation: int, job_id: str, phase: str, message: str, percent: float | None)`.
- Produces: `make_parse_progress_event(...)->ParseProgressEvent | None`, which rejects unknown phases, bounds/sanitizes text, and omits invalid percentages.
- Produces: `install_parse_progress_sink(progress_queue: Any | None) -> None` and `emit_parse_progress(...) -> None`.
- Produces: `ParseProgressCoalescer(*, interval: float, started_at: float)`, `.accept(event)`, and `.take_due(now: float, *, force: bool = False) -> tuple[ParseProgressEvent, ...]`.

- [ ] **Step 1: Write RED normalization tests**

```python
def test_progress_event_is_bounded_plain_data_and_invalid_percent_is_omitted():
    event = make_parse_progress_event(
        generation=4,
        job_id="ingest-job-7",
        phase="extracting",
        message="Extracting page 2\nof 5\x00",
        percent=float("inf"),
    )
    assert event == ParseProgressEvent(
        generation=4,
        job_id="ingest-job-7",
        phase="extracting",
        message="Extracting page 2 of 5",
        percent=None,
    )
    assert make_parse_progress_event(
        generation=4,
        job_id="ingest-job-7",
        phase="provider-private-stage",
        message="raw",
    ) is None
```

- [ ] **Step 2: Run the normalization tests and confirm RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest Tests/Local_Ingestion/test_ingest_parse_progress.py -q`

Expected: collection/import failure because `ingest_parse_progress` does not exist.

- [ ] **Step 3: Implement the stdlib-only event boundary**

```python
INGEST_PARSE_PROGRESS_MESSAGE_MAX_CHARS = 160
INGEST_PARSE_PROGRESS_FLUSH_SECONDS = 0.25
INGEST_PARSE_PROGRESS_QUEUE_MAXSIZE = 64
INGEST_PARSE_PROGRESS_PHASES = frozenset({
    "inspecting", "extracting", "processing", "transcribing",
    "chunking", "analyzing", "preparing", "loading",
    "post-processing", "writing",
})

@dataclass(frozen=True, slots=True)
class ParseProgressEvent:
    generation: int
    job_id: str
    phase: str
    message: str
    percent: float | None = None
```

Normalize strings before constructing the dataclass. Convert an accepted numeric percentage to `float`; use `None` for booleans, non-numeric, non-finite, or out-of-range values. Return `None` for blank identity or unknown phase.

- [ ] **Step 4: Write RED non-blocking sink and coalescer tests**

```python
class _FullQueue:
    def put_nowait(self, _event):
        raise queue.Full

def test_full_progress_queue_is_best_effort():
    install_parse_progress_sink(_FullQueue())
    emit_parse_progress(1, "ingest-job-1", "extracting", "Extracting")

def test_coalescer_keeps_latest_event_per_job_until_due():
    coalescer = ParseProgressCoalescer(interval=0.25, started_at=10.0)
    coalescer.accept(ParseProgressEvent(1, "a", "extracting", "first", 10.0))
    coalescer.accept(ParseProgressEvent(1, "a", "extracting", "latest", 30.0))
    assert coalescer.take_due(10.24) == ()
    assert coalescer.take_due(10.25) == (
        ParseProgressEvent(1, "a", "extracting", "latest", 30.0),
    )
```

- [ ] **Step 5: Implement best-effort emission and deterministic coalescing**

Catch `queue.Full`, `BrokenPipeError`, `EOFError`, `OSError`, and `ValueError` around `put_nowait`. Keep only the newest event per job. Sort flushed events by job id for deterministic tests; reset the next deadline from the supplied `now` value.

- [ ] **Step 6: Run Task 1 tests and static checks**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Local_Ingestion/test_ingest_parse_progress.py -q
..\..\.venv\Scripts\python.exe -m ruff check tldw_chatbook/Local_Ingestion/ingest_parse_progress.py Tests/Local_Ingestion/test_ingest_parse_progress.py
```

Expected: all Task 1 tests pass and Ruff reports no findings.

- [ ] **Step 7: Commit Task 1**

```powershell
git add tldw_chatbook/Local_Ingestion/ingest_parse_progress.py Tests/Local_Ingestion/test_ingest_parse_progress.py
git commit -m "feat: add spawn-safe ingest progress contract"
```

---

### Task 2: Registry projection, persistence policy, and display formatter

**Files:**
- Modify: `tldw_chatbook/Library/library_ingest_jobs.py`
- Modify: `tldw_chatbook/Library/library_ingest_state.py`
- Modify: `Tests/Library/test_library_ingest_jobs.py`
- Modify: `Tests/Library/test_library_ingest_state.py`

**Interfaces:**
- Consumes: controlled progress dictionaries produced from `ParseProgressEvent`.
- Produces: `add_progress_listener(callback)`, `remove_progress_listener(callback)`, and `update_progress(job_id, *, progress, persist: bool = True)`.
- Produces: `format_ingest_progress_line(progress, *, state) -> str` for the canvas and screen patcher.
- Produces: `ingest_progress_action_signature(job) -> tuple[bool, bool]` returning `(can_cancel, can_force_stop)` from the same Local-STT rules used by queue-row construction.

- [ ] **Step 1: Write RED registry-listener and transient-persistence tests**

```python
def test_transient_progress_uses_progress_listener_without_persisting():
    registry = LibraryIngestJobRegistry()
    store = _FakeStore()
    registry.attach_store(store)
    job = registry.submit(source_path="/tmp/report.txt")
    registry.mark_parsing(job.job_id)
    lifecycle_calls = []
    progress_calls = []
    registry.add_listener(lambda: lifecycle_calls.append("lifecycle"))
    registry.add_progress_listener(lambda before, after: progress_calls.append((before, after)))
    persisted_before = len(store.upserts)

    updated = registry.update_progress(
        job.job_id,
        progress={"phase": "extracting", "message": "Extracting"},
        persist=False,
    )

    assert updated.progress == {"phase": "extracting", "message": "Extracting"}
    assert lifecycle_calls == []
    assert progress_calls[0][0].progress is None
    assert progress_calls[0][1].progress == updated.progress
    assert len(store.upserts) == persisted_before
```

Also test removal, listener exception isolation, default `persist=True` for Server reconciliation, terminal rejection, and no notification on a no-op.

- [ ] **Step 2: Run focused registry tests and confirm RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest Tests/Library/test_library_ingest_jobs.py -k "progress or mark_writing" -q`

Expected: failure because progress listeners and `persist` do not exist and `mark_writing` preserves parse progress.

- [ ] **Step 3: Implement progress-specific listeners and writing reset**

```python
ProgressListener = Callable[[LibraryIngestJob, LibraryIngestJob], None]

def update_progress(self, job_id: str, *, progress: dict[str, Any] | None,
                    persist: bool = True) -> LibraryIngestJob | None:
    ...
    if current.progress == progress:
        return _copy_job(current)
    before = _copy_job(current)
    updated = replace(current, progress=progress)
    self._jobs[index] = updated
    self._notify_progress_listeners(before, _copy_job(updated))
    if persist:
        self._persist(updated)
    return _copy_job(updated)
```

Change `_copy_job` to deep-copy `progress` so listener consumers cannot mutate registry-owned payloads through before/after snapshots. Change `mark_writing` to `replace(current, state=IngestJobState.WRITING, progress={"phase": "writing", "message": "Saving to Library"})`. Keep normal lifecycle listener and persistence behavior for that state transition.

- [ ] **Step 4: Write RED formatter tests**

```python
@pytest.mark.parametrize(
    ("progress", "state", "expected"),
    [
        ({"phase": "extracting", "message": "Extracting page 21 of 50", "percent": 42.0}, IngestJobState.PARSING, "42% · Extracting page 21 of 50"),
        ({"phase": "transcribing"}, IngestJobState.PARSING, "Transcribing audio"),
        (None, IngestJobState.PARSING, "Preparing import"),
        ({"phase": "writing", "message": "Saving to Library"}, IngestJobState.WRITING, "Saving to Library"),
        ({"message": "Imported report.txt"}, IngestJobState.DONE, "Imported report.txt"),
    ],
)
def test_format_ingest_progress_line(progress, state, expected):
    assert format_ingest_progress_line(progress, state=state) == expected
```

- [ ] **Step 5: Implement the one pure formatter**

Use a fixed phase-label mapping for `preparing`, `loading`, `transcribing`, `post-processing`, `inspecting`, `extracting`, `processing`, `chunking`, `analyzing`, and `writing`. Prefer a sanitized non-blank message, fall back to the phase label, use `Preparing import` when a `PARSING` job has no payload yet, and prefix only a valid in-range finite percentage rounded to the nearest integer. Do not prefix lifecycle state text. Extract the existing Local-STT action predicate into `ingest_progress_action_signature` and have both `_build_queue_row` and the screen use it.

- [ ] **Step 6: Run Task 2 suites and static checks**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Library/test_library_ingest_jobs.py Tests/Library/test_library_ingest_state.py -q
..\..\.venv\Scripts\python.exe -m pytest Tests/Library/test_server_ingest_reconcile.py -q
..\..\.venv\Scripts\python.exe -m ruff check tldw_chatbook/Library/library_ingest_jobs.py tldw_chatbook/Library/library_ingest_state.py Tests/Library/test_library_ingest_jobs.py Tests/Library/test_library_ingest_state.py
```

Expected: registry/state/server reconciliation suites pass; Ruff is clean.

- [ ] **Step 7: Commit Task 2**

```powershell
git add tldw_chatbook/Library/library_ingest_jobs.py tldw_chatbook/Library/library_ingest_state.py Tests/Library/test_library_ingest_jobs.py Tests/Library/test_library_ingest_state.py
git commit -m "feat: add transient ingest progress projection"
```

---

### Task 3: Truthful parser and transcription instrumentation

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/ingest_parse_worker.py`
- Modify: `tldw_chatbook/Local_Ingestion/local_file_ingestion.py`
- Modify: `Tests/Local_Ingestion/test_ingest_parse_worker.py`
- Modify: `Tests/Local_Ingestion/test_local_file_ingestion.py`

**Interfaces:**
- Consumes: `emit_parse_progress` and `install_parse_progress_sink` from Task 1.
- Produces: `initialize_ingest_parse_worker(progress_queue: Any | None = None) -> None`.
- Extends: `run_parse_job(file_path, options, progress_context: tuple[int, str] | None = None)`.
- Extends: `parse_local_file_for_ingest(..., progress_callback: Callable[[str, str, float | None], None] | None = None)`.

- [ ] **Step 1: Write RED worker-binding and callback-safety tests**

```python
def test_run_parse_job_emits_bound_progress_without_changing_result(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(ingest_parse_worker, "emit_parse_progress", lambda *args: events.append(args))
    source = tmp_path / "note.txt"
    source.write_text("hello", encoding="utf-8")

    result = run_parse_job(str(source), {}, (3, "ingest-job-9"))

    assert result["ok"] is True
    assert events[0][:3] == (3, "ingest-job-9", "inspecting")
    assert any(event[2] == "processing" for event in events)

def test_progress_callback_exception_never_fails_parse(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("hello", encoding="utf-8")
    payload = parse_local_file_for_ingest(
        str(source), {}, progress_callback=lambda *_args: (_ for _ in ()).throw(RuntimeError("telemetry failed"))
    )
    assert payload["content"] == "hello"
```

- [ ] **Step 2: Run worker tests and confirm RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest Tests/Local_Ingestion/test_ingest_parse_worker.py -k "progress or import_excludes" -q`

Expected: signature/initializer failures because no callback path exists.

- [ ] **Step 3: Implement initializer, binding, and observable stage reports**

```python
def initialize_ingest_parse_worker(progress_queue=None) -> None:
    silence_ingest_worker_import_noise()
    install_parse_progress_sink(progress_queue)

def run_parse_job(file_path, options, progress_context=None):
    progress_callback = None
    if progress_context is not None:
        generation, job_id = progress_context
        progress_callback = lambda phase, message, percent=None: emit_parse_progress(
            generation, job_id, phase, message, percent
        )
    ...
    payload = parse_local_file_for_ingest(
        file_path, options, progress_callback=progress_callback
    )
```

Inside `parse_local_file_for_ingest`, call a `_report_ingest_progress` helper that catches every callback exception. Report `Inspecting source`, then one truthful type-specific processing message immediately before the branch call/read. Report `Chunking extracted text` only around the shared `_chunk_text_for_ingest` call and `Analyzing extracted text` only around `_run_chat_analysis`. Do not claim internal processor phases that are not observable.

- [ ] **Step 4: Write RED audio/video callback-adapter tests**

Extend the existing fake audio/video processors so their `process_*` method invokes the received `transcription_progress_callback(37.0, "Transcribing segment 3 of 8", {"private": object()})`. Assert the public progress callback receives only `("transcribing", "Transcribing segment 3 of 8", 37.0)` and never the provider data object.

- [ ] **Step 5: Forward existing measurable transcription callbacks**

```python
def transcription_progress(percent, message, _data=None):
    _report_ingest_progress(
        progress_callback,
        "transcribing",
        str(message or "Transcribing audio"),
        percent,
    )
```

Pass that adapter as `transcription_progress_callback` to both `process_audio_files` and `process_videos`. Keep direct Local STT executor routing unchanged; it does not run through this pool path.

- [ ] **Step 6: Preserve lightweight import and real spawn behavior**

Update the isolated import guard to allow only the new stdlib-only progress module while still rejecting `local_file_ingestion`, audio/video processing, transcription, torch, docling, and nltk at module import. Extend the existing real spawn test to pass a queue through `initialize_ingest_parse_worker`, submit `run_parse_job(..., (1, "ingest-job-1"))`, and assert at least one bound event plus the unchanged success payload.

- [ ] **Step 7: Run Task 3 suites and static checks**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Local_Ingestion/test_ingest_parse_progress.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/Local_Ingestion/test_local_file_ingestion.py -q
..\..\.venv\Scripts\python.exe -m ruff check tldw_chatbook/Local_Ingestion/ingest_parse_progress.py tldw_chatbook/Local_Ingestion/ingest_parse_worker.py tldw_chatbook/Local_Ingestion/local_file_ingestion.py Tests/Local_Ingestion/test_ingest_parse_progress.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/Local_Ingestion/test_local_file_ingestion.py
```

Expected: all three suites pass; Ruff is clean.

- [ ] **Step 8: Commit Task 3**

```powershell
git add tldw_chatbook/Local_Ingestion/ingest_parse_worker.py tldw_chatbook/Local_Ingestion/local_file_ingestion.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/Local_Ingestion/test_local_file_ingestion.py
git commit -m "feat: report truthful local parse stages"
```

---

### Task 4: Parse-pool progress resource lifecycle

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Library/test_library_ingest_runner.py`
- Modify: `Tests/UI/test_library_shell.py` (fake-pool signature only)

**Interfaces:**
- Consumes: `initialize_ingest_parse_worker`, `ParseProgressCoalescer`, constants, and events from Tasks 1 and 3.
- Produces: private `_IngestParsePoolResources(pool: Any, progress_queue: Any | None)`.
- Produces: `_start_ingest_parse_progress_drain(generation, progress_queue, stop_event) -> threading.Thread`.
- Extends: `_shutdown_ingest_workers_off_thread(executor, pool, progress_queue, progress_thread)`.

- [ ] **Step 1: Write RED atomic-construction and drain-coalescing tests**

```python
def test_create_pool_returns_progress_resources_and_uses_combined_initializer(monkeypatch):
    captured = {}
    class _Context:
        def Queue(self, maxsize):
            captured["maxsize"] = maxsize
            return _ClosableQueue()
        def Pool(self, **kwargs):
            captured.update(kwargs)
            return _FakeIngestParsePool(auto_run=False)
    monkeypatch.setattr(multiprocessing, "get_context", lambda _name: _Context())
    resources = LibraryIngestQueueMixin._create_ingest_parse_pool(_bare_mixin())
    assert resources.progress_queue is not None
    assert captured["initializer"] is initialize_ingest_parse_worker
    assert captured["initargs"] == (resources.progress_queue,)
```

Add a failure test where `Pool` raises and assert the already-created queue is closed. Add a drain test with two updates for one job and an injected clock, asserting one latest-event batch is marshaled.

- [ ] **Step 2: Run resource tests and confirm RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest Tests/Library/test_library_ingest_runner.py -k "progress_resources or progress_drain" -q`

Expected: failures because pool creation returns only a pool and no drain owner exists.

- [ ] **Step 3: Implement atomic resource construction under the stderr workaround**

```python
@dataclass(frozen=True)
class _IngestParsePoolResources:
    pool: Any
    progress_queue: Any | None
```

Create the bounded queue and pool inside the same valid-stderr context. Use `initialize_ingest_parse_worker` with `(progress_queue,)`. If queue or pool construction fails, close/cancel-join any queue already created before re-raising. Update fake-pool overrides to return `_IngestParsePoolResources(fake_pool, queue.Queue(maxsize=64))` without spawning a process.

- [ ] **Step 4: Implement generation-owned drain startup**

After `_ensure_ingest_parse_pool` assigns the new generation and stop event, store the queue, start one daemon thread, and have it:

```python
while not stop_event.is_set() and not self._ingest_shutdown:
    try:
        event = progress_queue.get(timeout=0.05)
    except queue.Empty:
        event = None
    if event is not None:
        coalescer.accept(event)
    batch = coalescer.take_due(time.monotonic())
    if batch:
        self._marshal_ingest_pool_call(
            self._on_ingest_parse_progress_batch, generation, batch
        )
```

Force-flush nothing during shutdown; terminal results are authoritative and stale pending telemetry should be discarded.

- [ ] **Step 5: Write RED off-thread cleanup and Windows spawn tests**

Assert broken-pool and normal shutdown set the stop event, detach queue/thread references, and run pool terminate/join plus queue close/cancel-join off the caller thread. Add a `@pytest.mark.skipif(sys.platform != "win32", reason="Windows spawn/resource-tracker boundary")` test that uses the production resource factory, receives a real worker event, gets the parse result with a bounded timeout, invokes cleanup, and joins the cleanup thread with a bounded timeout.

- [ ] **Step 6: Implement unified off-thread cleanup**

Pass pool, queue, and progress thread into the existing daemon cleanup helper. Terminate/join the pool first, close/cancel-join the queue, then give the already-stopped daemon drain thread a bounded join. Log unexpected cleanup errors without raising into the app shutdown path. Preserve executor-first ordering.

- [ ] **Step 7: Run Task 4 coordinator subset**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Library/test_library_ingest_runner.py -k "create_pool or progress_drain or shutdown or broken_pool or worker_exit" -q
..\..\.venv\Scripts\python.exe -m ruff check tldw_chatbook/app.py Tests/Library/test_library_ingest_runner.py Tests/UI/test_library_shell.py
```

Expected: focused lifecycle tests pass; the Windows test passes locally and is skipped elsewhere; Ruff is clean on changed code.

- [ ] **Step 8: Commit Task 4**

```powershell
git add tldw_chatbook/app.py Tests/Library/test_library_ingest_runner.py Tests/UI/test_library_shell.py
git commit -m "feat: own ingest progress resources per parse pool"
```

---

### Task 5: Coordinator submission and stale-event fences

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Library/test_library_ingest_runner.py`

**Interfaces:**
- Consumes: `ParseProgressEvent`, transient registry updates, and generation resource state from Tasks 1, 2, and 4.
- Produces: `_on_ingest_parse_progress_batch(generation: int, events: tuple[ParseProgressEvent, ...]) -> None`.
- Changes: pool submission passes `(generation, job_id)` as `run_parse_job`'s third positional argument.

- [ ] **Step 1: Write RED submission-context and accepted-event tests**

```python
@pytest.mark.asyncio
async def test_pool_submission_binds_generation_and_job_and_applies_transient_progress(tmp_path):
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(_make_db(tmp_path), pool_factory=lambda: pool)
    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(_write_text_file(tmp_path, "a.txt", "a")))
        await pilot.pause()
        generation = app._ingest_parse_pool_generation
        assert pool.calls[0]["args"][2] == (generation, job.job_id)
        app._on_ingest_parse_progress_batch(
            generation,
            (ParseProgressEvent(generation, job.job_id, "extracting", "Extracting", 25.0),),
        )
        assert app.library_ingest_jobs.get_job(job.job_id).progress["percent"] == 25.0
```

Attach a store spy and assert no extra upsert for the accepted local tick.

- [ ] **Step 2: Write RED fence matrix tests**

Parameterize wrong generation, job absent from the generation set, non-`PARSING` state, hidden/terminal job, and payload-ready job. For each, apply an event and assert progress is unchanged. Add a race test where `_on_ingest_parse_complete` stores a payload before a late progress batch; the late extraction event must not land.

- [ ] **Step 3: Run focused tests and confirm RED**

Run: `..\..\.venv\Scripts\python.exe -m pytest Tests/Library/test_library_ingest_runner.py -k "submission_binds_generation or parse_progress_batch or late_progress" -q`

Expected: missing handler/context failures.

- [ ] **Step 4: Implement submission context and UI-thread batch application**

```python
pool.apply_async(
    run_parse_job,
    (source_path, options, (generation, job_id)),
    callback=...,
    error_callback=...,
)

def _on_ingest_parse_progress_batch(self, generation, events):
    if self._ingest_shutdown or generation != self._ingest_parse_pool_generation:
        return
    generation_jobs = self._ingest_parse_jobs_by_generation.get(generation)
    if generation_jobs is None:
        return
    for event in events:
        job = self.library_ingest_jobs.get_job(event.job_id)
        if (
            event.generation != generation
            or event.job_id not in generation_jobs
            or event.job_id in self._ingest_parsed_payloads
            or job is None
            or job.state is not IngestJobState.PARSING
        ):
            continue
        self.library_ingest_jobs.update_progress(
            event.job_id,
            progress={"phase": event.phase, "message": event.message, **({"percent": event.percent} if event.percent is not None else {})},
            persist=False,
        )
```

Revalidate through `make_parse_progress_event` or a dedicated parent normalizer before building the dict; never trust a queue object solely because its nominal type is correct.

- [ ] **Step 5: Preserve local-STT control metadata**

When `_on_ingest_local_stt_event` replaces a phase, remove any old `percent` and set a readable `message` for the new phase while retaining `cancel_requested`. Keep `cancel_local_ingest_job` and Force stop action state unchanged.

- [ ] **Step 6: Run the full runner module**

Run: `..\..\.venv\Scripts\python.exe -m pytest Tests/Library/test_library_ingest_runner.py -q`

Expected: all runner tests pass. If the documented Windows Proactor/network-guard socketpair setup conflict occurs, rerun the new and directly affected tests individually with their established `allow_network` marker and record the unrelated harness failure separately.

- [ ] **Step 7: Commit Task 5**

```powershell
git add tldw_chatbook/app.py Tests/Library/test_library_ingest_runner.py
git commit -m "feat: fence local ingest progress events"
```

---

### Task 6: Stable in-place Library progress UI

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_ingest_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_library_ingest_canvas.py`
- Modify: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: `format_ingest_progress_line`, `ingest_progress_action_signature`, and registry progress listeners from Task 2.
- Produces: `_handle_library_ingest_progress_changed(before, after) -> None` on `LibraryScreen`.

- [ ] **Step 1: Write RED canvas composition/copy tests**

```python
@pytest.mark.asyncio
async def test_parsing_row_reserves_progress_line_before_first_worker_tick():
    app = _QueuePanelHost(_state_with_job(state=IngestJobState.PARSING, progress=None))
    async with app.run_test() as pilot:
        widget = pilot.app.query_one("#library-ingest-progress-ingest-job-1", Static)
        assert widget.display is True
        assert str(widget.renderable) == "Preparing import"

@pytest.mark.asyncio
async def test_progress_line_uses_formatter_without_repeating_state():
    app = _QueuePanelHost(_state_with_job(
        state=IngestJobState.PARSING,
        progress={"phase": "extracting", "message": "Extracting page 2 of 5", "percent": 40.0},
    ))
    async with app.run_test() as pilot:
        widget = pilot.app.query_one("#library-ingest-progress-ingest-job-1", Static)
        assert str(widget.renderable) == "40% · Extracting page 2 of 5"
```

Update the old absent-widget expectation: `PARSING` and `WRITING` rows always mount a visible reserved line; queued rows and terminal rows without receipts may remain absent.

- [ ] **Step 2: Implement stable widget composition and muted styling**

Always yield the job-id progress `Static` for `PARSING` and `WRITING`; update its text with `format_ingest_progress_line`. `PARSING` without a payload renders `Preparing import`, so the line is reserved before the first tick. Keep terminal receipt composition. Add a primary-row class that removes its bottom gap whenever a progress line follows, then style the detail without overlap:

```css
.library-ingest-progress {
    width: 100%;
    height: auto;
    color: $ds-text-muted;
    margin: 0 0 1 2;
}

.library-ingest-row-with-progress {
    margin-bottom: 0;
}
```

Regenerate the CSS bundle with `..\..\.venv\Scripts\python.exe tldw_chatbook/css/build_css.py`. The progress rule must not use a negative margin, border, animation, or overlay positioning.

- [ ] **Step 3: Write RED screen identity/fallback tests**

Mount the real Library ingest canvas, focus `#library-ingest-path`, set a non-zero canvas scroll offset, capture the primary row, progress widget, input, and queue-panel identities, then call `registry.update_progress(..., persist=False)`. After `pilot.pause()`, assert every captured widget is the same object, focus/cursor and scroll are unchanged, and only progress text/display changed.

Add a second test changing local-STT progress from `transcribing` to the same phase with `cancel_requested=True`; assert the queue structural path runs and Force stop replaces Cancel.

- [ ] **Step 4: Implement progress listener lifecycle and patcher**

Register both listeners in `on_mount` and remove both in `on_unmount`:

```python
registry.add_listener(self._handle_library_ingest_registry_changed)
registry.add_progress_listener(self._handle_library_ingest_progress_changed)
```

The progress handler compares `ingest_progress_action_signature(before)` with `ingest_progress_action_signature(after)`. Those are the only row-structure fields that a progress-only mutation can change; state, origin, result ids, error details, and terminal actions cannot change through `update_progress`. If the signatures differ, call `_update_library_ingest_dynamic_regions()`. Otherwise update the mounted progress `Static` by job id. Do nothing when the screen is unmounted or another canvas is selected.

Use `getattr` guards when registering/removing progress listeners so existing minimal registry doubles in `test_library_shell.py` remain valid unless the test specifically exercises progress.

- [ ] **Step 5: Run canvas and screen progress tests**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/UI/test_library_ingest_canvas.py -q
..\..\.venv\Scripts\python.exe -m pytest Tests/UI/test_library_shell.py -k "ingest and (progress or context or registry)" -q
..\..\.venv\Scripts\python.exe -m ruff check tldw_chatbook/Widgets/Library/library_ingest_canvas.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_ingest_canvas.py Tests/UI/test_library_shell.py
```

Expected: all canvas tests and focused screen tests pass; row/input/progress identity assertions prove the in-place path; Ruff is clean on changed code.

- [ ] **Step 6: Run CSS and constrained-layout verification**

Run the CSS builder a second time and assert `git diff --exit-code -- tldw_chatbook/css/tldw_cli_modular.tcss` after staging the first generated output. Run progress canvas tests at both the repository's normal Library size and a constrained supported viewport; assert the secondary line remains below the row and does not obscure actions or neighboring entries.

- [ ] **Step 7: Commit Task 6**

```powershell
git add tldw_chatbook/Widgets/Library/library_ingest_canvas.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_library_ingest_canvas.py Tests/UI/test_library_shell.py
git commit -m "feat: update ingest progress rows in place"
```

---

### Task 7: Documentation, integrated verification, and task closeout

**Files:**
- Modify: `Docs/User_Guide/library.md`
- Modify: `backlog/tasks/task-207 - Live-parse-progress-for-ingest-jobs-progress_percent-progress_message.md`

**Interfaces:**
- Consumes: all prior task behavior and verification evidence.
- Produces: user documentation and complete TASK-207 implementation notes.

- [ ] **Step 1: Update the user guide**

Document that Local imports show stage detail, a percentage appears only when the parser knows a real total, indeterminate work intentionally shows text alone, and `Saving to Library` identifies the writer stage. Do not promise resumable percentage state or continuous updates.

- [ ] **Step 2: Run focused integrated tests**

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Local_Ingestion/test_ingest_parse_progress.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/Local_Ingestion/test_local_file_ingestion.py Tests/Library/test_library_ingest_jobs.py Tests/Library/test_library_ingest_state.py Tests/Library/test_server_ingest_reconcile.py Tests/Library/test_library_ingest_runner.py Tests/UI/test_library_ingest_canvas.py -q
```

Expected: all affected non-screen modules pass. Run the focused `test_library_shell.py` progress/registry selection separately because the Windows network guard may conflict with Textual's Proactor socketpair setup.

- [ ] **Step 3: Run static and repository checks**

```powershell
..\..\.venv\Scripts\python.exe -m ruff check tldw_chatbook/Local_Ingestion/ingest_parse_progress.py tldw_chatbook/Local_Ingestion/ingest_parse_worker.py tldw_chatbook/Local_Ingestion/local_file_ingestion.py tldw_chatbook/Library/library_ingest_jobs.py tldw_chatbook/Library/library_ingest_state.py tldw_chatbook/Widgets/Library/library_ingest_canvas.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/app.py Tests/Local_Ingestion/test_ingest_parse_progress.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/Local_Ingestion/test_local_file_ingestion.py Tests/Library/test_library_ingest_jobs.py Tests/Library/test_library_ingest_state.py Tests/Library/test_library_ingest_runner.py Tests/UI/test_library_ingest_canvas.py Tests/UI/test_library_shell.py
..\..\.venv\Scripts\python.exe -m compileall -q tldw_chatbook/Local_Ingestion/ingest_parse_progress.py tldw_chatbook/Local_Ingestion/ingest_parse_worker.py tldw_chatbook/Library/library_ingest_jobs.py tldw_chatbook/Library/library_ingest_state.py
git diff --check
```

Re-run `tldw_chatbook/css/build_css.py` and confirm no generated diff. Run the repository Backlog guard/duplicate-ID check used by current CI before closeout.

- [ ] **Step 4: Perform failure-oriented self-review**

Inspect the complete branch diff for: any blocking queue put, UI-thread join, unbounded message, non-picklable provider data, stale generation path, payload-ready race, local tick persistence, stale percentage under writing, repeated lifecycle prefix, conditional active progress widget, or hand-edited CSS bundle. Add a regression test before fixing any discovered issue.

- [ ] **Step 5: Update TASK-207 closeout**

Check each acceptance criterion only after its evidence passes. Add `## Implementation Notes` summarizing the process contract, truthful-stage limits, transient persistence decision, UI identity behavior, Windows spawn evidence, tests/static checks, ADR-061, and any documented unrelated Windows harness failures. Set status to Done only when every Definition-of-Done item is satisfied.

- [ ] **Step 6: Commit closeout**

```powershell
git add Docs/User_Guide/library.md 'backlog/tasks/task-207 - Live-parse-progress-for-ingest-jobs-progress_percent-progress_message.md'
git commit -m "docs: document live local ingest progress"
```

- [ ] **Step 7: Final verification from committed HEAD**

Re-run the new contract tests, the full registry/state/server-reconcile suites, the complete runner module, the complete ingest-canvas module, focused screen identity tests, Ruff, compileall, CSS regeneration check, `git diff --check`, and `git status --short`. Record exact pass/skip/failure counts in the task notes before requesting review.
