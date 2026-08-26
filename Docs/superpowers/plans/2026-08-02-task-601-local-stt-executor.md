# TASK-601 Generation-Fenced Local STT Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Library Parakeet ONNX and transcribe.cpp batch work into one app-owned spawn process that safely reuses one model, fences stale callbacks, owns artifact leases and decoder descendants, and leaves the existing light parse pool and parent writer intact.

**Architecture:** Add a small `LocalSTTExecutor` controller, one spawn entry module, and one narrow process-tree containment helper. The existing Library registry remains the only queue; eligible audio/video jobs go to the executor, all other parsing stays in the current pool, and successful heavy results rejoin the existing parsed-payload writer. A worker-owned injected transcription callable lets the existing audio/video parse path reuse the resident provider without creating another `TranscriptionService`. Each generation uses a private parent-owned scratch directory so force-stop cleanup happens only after the contained tree is proven dead.

**Tech Stack:** Python 3.11+, `multiprocessing` spawn/Pipe/Event, standard-library threading and OS process primitives, Windows Job Objects through lazy `ctypes`, existing STT coordinator/providers, existing `ModelArtifactService`, Textual Library coordinator, pytest, Ruff.

---

## Preconditions and scope

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-601-local-stt-executor` on `codex/task-601-local-stt-executor`.
- Governing design: `Docs/superpowers/specs/2026-08-02-task-601-local-stt-executor-design.md`.
- ADR required: no new ADR.
- ADR paths: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md` and `backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md`.
- Reason: ADR-025 already fixes the single app-owned heavy-process, resident-identity, lease, generation-fence, process-tree, and retry boundaries; ADR-041 already fixes the direct-local GGUF exception. TASK-601 implements those decisions without changing them.
- Use `superpowers:test-driven-development` for each behavior slice and `superpowers:verification-before-completion` before any completion claim.
- Run only tests and static checks covering files changed by TASK-601. Do not run the full repository suite or wait on unrelated CI.
- Preserve the Windows/Linux native execution gate. This branch may merge after focused macOS evidence and review, but must not claim untested Windows/Linux native proof.
- Do not add another queue, database table, setting, downloader, managed import flow, dictation policy, or provider-removal work.
- Do not route faster-whisper through this executor. Existing semantic `auto` and unsupported-language work continues through its current path.
- Do not delete legacy provider code. TASK-605 owns removal after the release gates pass.

## File map

- Create `tldw_chatbook/STT/executor.py` — dependency-light IPC objects, local-source snapshot helpers, generation/attempt terminal guard, and parent `LocalSTTExecutor`.
- Create `tldw_chatbook/STT/executor_worker.py` — spawn entry point, managed acquisition/local revalidation, one resident provider runtime, and heavy parse invocation.
- Create `tldw_chatbook/STT/executor_process_tree.py` — POSIX session/group and Windows kill-on-close Job Object containment for one executor generation.
- Modify `tldw_chatbook/STT/transcribe_cpp.py` — split current one-shot load/use/close flow into a reusable runtime while preserving `transcribe_file()`.
- Modify `tldw_chatbook/STT/__init__.py` — export only the parent-facing executor API; keep the spawn entry and OS helper private.
- Modify `tldw_chatbook/Local_Ingestion/audio_processing.py` — accept one optional injected transcription callable and fall back to the current `TranscriptionService` behavior.
- Modify `tldw_chatbook/Local_Ingestion/video_processing.py` — pass the same callable to its owned `LocalAudioProcessor`.
- Modify `tldw_chatbook/Local_Ingestion/local_file_ingestion.py` — pass the callable into audio/video processors only; keep the parsed payload contract unchanged.
- Modify `tldw_chatbook/app.py` — lazily own the executor, dispatch eligible heavy jobs separately, fence callbacks, and close it during shutdown.
- Create `Tests/STT/test_local_stt_executor.py` — protocol, controller, identity, generation, cancel, retry, crash, lease, and spawn behavior.
- Create `Tests/STT/test_executor_process_tree.py` — narrow containment unit/contract tests and native POSIX descendant cleanup evidence.
- Create `Tests/STT/executor_test_support.py` — importable spawn targets and deterministic fake resident runtimes used only by executor process tests.
- Modify `Tests/STT/test_transcribe_cpp.py` — reusable runtime and preserved one-shot behavior.
- Modify `Tests/Local_Ingestion/test_ingest_parse_worker.py` — injected runner compatibility and unchanged light worker contract.
- Modify `Tests/Transcription/test_parakeet_onnx_vertical_slice.py` — one service instance reuses the Parakeet model and retains existing result semantics.
- Modify `Tests/Library/test_library_ingest_runner.py` — heavy/light dispatch, stale callback, failure, writer handoff, and shutdown integration.
- Modify `backlog/tasks/task-601 - Add-generation-fenced-local-STT-executor.md` through Backlog CLI — record this plan, then add implementation notes without prematurely marking the platform gate complete.

Before Task 1, record this reviewed plan on TASK-601 with Backlog CLI and commit the plan document plus task-file update. Implementation must then start from a clean worktree.

### Task 1: Define the private IPC and resident-identity contract

**Files:**
- Create: `tldw_chatbook/STT/executor.py`
- Create: `Tests/STT/test_local_stt_executor.py`

- [ ] **Step 1: Write failing protocol, privacy, and terminal-guard tests**

Add tests that require:

- `ModelIdentity` equality to include provider, model, managed root revision, closure fingerprint, precision, device, and local snapshot token;
- `ExecutorRequest`, `ExecutorEvent`, `ExecutorResult`, and `ExecutorFailure` to carry both `generation` and `attempt_id`;
- all protocol objects to round-trip through `pickle`;
- model paths, managed store paths, media paths, and local snapshot tokens to be excluded from reprs;
- failure envelopes to contain only an existing `TranscriptionFailureCode`, bounded recovery actions, sanitized provenance, and an optional typed `DeviceFailureOrigin` — never a raw exception;
- the phase enum to contain only `preparing`, `loading`, `transcribing`, and `post_processing` for worker-originated progress;
- an attempt terminal guard to accept exactly one of result/failure/cancelled and reject duplicates or a mismatched attempt/generation.

Use frozen, slotted dataclasses and `field(repr=False)` for private fields. Keep these transient IPC values out of persisted provenance.

- [ ] **Step 2: Run the focused tests to verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_local_stt_executor.py -k "protocol or identity or redacts or terminal" -v
```

Expected: collection/import fails because `tldw_chatbook.STT.executor` does not exist.

- [ ] **Step 3: Implement only the dependency-light contracts**

Create `executor.py` with:

- frozen, slotted `ModelIdentity`, `LocalSourceSnapshot`, request/event/result/failure dataclasses;
- a small worker-phase enum;
- strict `__post_init__` validation for non-empty generation/attempt/provider/model identities and bounded actions;
- an internal `_AttemptTerminalGuard` that compares generation and attempt before consuming the one terminal slot;
- no provider, ONNX, transcribe.cpp, Textual, artifact acquisition, or ingestion imports at module scope.

Do not build process startup or application routing in this task.

- [ ] **Step 4: Run the focused tests and static checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_local_stt_executor.py -k "protocol or identity or redacts or terminal" -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/STT/executor.py Tests/STT/test_local_stt_executor.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/STT/executor.py Tests/STT/test_local_stt_executor.py
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit the protocol slice**

```bash
git add tldw_chatbook/STT/executor.py Tests/STT/test_local_stt_executor.py
git commit -m "feat(stt): define local executor protocol"
```

### Task 2: Contain one worker generation and all descendants

**Files:**
- Create: `tldw_chatbook/STT/executor_process_tree.py`
- Create: `Tests/STT/test_executor_process_tree.py`
- Modify: `Tests/STT/executor_test_support.py`

- [ ] **Step 1: Write failing POSIX, Windows-contract, and ordering tests**

Add tests that require:

- on POSIX, a spawned worker calls `setsid()`, reports its own process-group identity, and cannot pass its admission wait until the parent explicitly signals it;
- on Windows, a lazy `ctypes` adapter creates a Job Object, sets `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, assigns the worker handle, and signals admission only after assignment succeeds;
- failed assignment never signals admission and terminates the unadmitted worker;
- `terminate_tree()` detaches/marks termination before signalling or killing, then proves the worker has exited with a bounded join;
- failure to prove tree death returns false and leaves the containment object quarantined rather than reusable;
- a macOS/POSIX spawn helper launches a real long-lived descendant, writes its PID through a pipe, and proves forced worker-group cleanup removes both worker and descendant before the parent removes the generation scratch directory;
- the parent does not remove generation scratch when tree death cannot be proven, preventing cleanup from racing a live descendant;
- module import on the opposite platform does not touch unavailable Win32/POSIX symbols.

The test helper must be a normal importable module so `spawn` never depends on a nested pytest function.

- [ ] **Step 2: Run the focused tests to verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_executor_process_tree.py -v
```

Expected: import fails because the narrow containment helper does not exist.

- [ ] **Step 3: Implement the narrow generation-containment helper**

Create `executor_process_tree.py` with only the behavior TASK-601 needs:

- a worker-side `enter_worker_containment()` returning the PID/process-group bootstrap data;
- a parent-side containment object that owns admission and force-stop lifecycle for exactly one generation;
- POSIX `os.killpg` termination with TERM, bounded join, then KILL if needed;
- lazy Windows Job Object creation/assignment/termination/close through `ctypes`, with kill-on-close configured before admission;
- idempotent close and bounded status values;
- no extraction of the large Notes-specific containment module and no general subprocess framework.

The entire worker is the containment unit, so existing FFmpeg descendants inherit it without introducing a second child registry.

- [ ] **Step 4: Run containment tests and static checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_executor_process_tree.py -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/STT/executor_process_tree.py Tests/STT/test_executor_process_tree.py Tests/STT/executor_test_support.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/STT/executor_process_tree.py Tests/STT/test_executor_process_tree.py Tests/STT/executor_test_support.py
git diff --check
```

Expected on macOS: all unit tests and the native POSIX descendant-cleanup test pass. Windows behavior is contract-tested but remains a native release gate.

- [ ] **Step 5: Commit the containment slice**

```bash
git add tldw_chatbook/STT/executor_process_tree.py Tests/STT/test_executor_process_tree.py Tests/STT/executor_test_support.py
git commit -m "feat(stt): contain executor process trees"
```

### Task 3: Implement the generation-fenced executor controller and spawn loop

**Files:**
- Modify: `tldw_chatbook/STT/executor.py`
- Create: `tldw_chatbook/STT/executor_worker.py`
- Modify: `tldw_chatbook/STT/__init__.py`
- Modify: `Tests/STT/test_local_stt_executor.py`
- Modify: `Tests/STT/executor_test_support.py`

- [ ] **Step 1: Write failing controller and real-spawn lifecycle tests**

Add dependency-free controller tests and real-spawn fake-runtime tests for:

- lazy `multiprocessing.get_context("spawn")` process creation and parent admission after containment;
- one active request and no private backlog;
- same-identity requests reusing the same PID/generation after the first request completes;
- every individual identity field change closing the idle worker and creating a new generation;
- the constructor-only completed-job bound recycling the worker after the configured count;
- reader-thread delivery of matching phase/result/failure messages only;
- stale generation, wrong attempt, duplicate terminal, and post-detach messages being ignored before callbacks;
- cooperative `cancel(attempt_id)` setting the generation event only for the matching active attempt;
- clearing the cancellation event while holding the controller lock before a successor dispatch;
- force stop consuming one cancelled terminal, detaching first, terminating off the caller thread, and creating no replacement until a later submit;
- failed force stop putting the executor in an unavailable state and preventing a second generation;
- sentinel/native crash producing one sanitized `ENGINE_CRASHED` failure for only the active attempt and marking only a loading/transcribing identity unhealthy;
- explicit retry clearing that identity once;
- only a typed, qualified device failure triggering one CPU retry with the same attempt in a fresh generation; generic errors and strings do not retry;
- `close()` detaching, stopping the reader, killing an active tree when necessary, and remaining idempotent.

Use deterministic fake worker modes in `executor_test_support.py`: succeed, hold, emit stale, crash in phase, typed device failure, ignore cancel, and fail shutdown.

- [ ] **Step 2: Run the focused tests to verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_local_stt_executor.py -k "controller or spawn or reuse or recycle or stale or cancel or crash or retry or shutdown" -v
```

Expected: tests fail because `LocalSTTExecutor` and the worker loop are not implemented.

- [ ] **Step 3: Implement the controller state machine**

In `executor.py`:

- own one Process, duplex Pipe endpoint, generation Event, admission Event, containment handle, and daemon reader thread;
- serialize `submit`, `cancel`, `force_stop`, callback acceptance, and `close` with one controller lock;
- increment the generation on every new worker;
- create a mode-`0700` generation scratch directory in the parent, pass it as a repr-hidden bootstrap field, and remove it only after normal worker exit or proven process-tree death;
- clear cancel and install the active terminal guard before sending a request;
- detach the generation before tree termination;
- quarantine both the executor and its scratch directory when tree death cannot be proven;
- never call user callbacks while holding the controller lock;
- expose bounded `busy`, `generation`, `resident_identity`, and `unavailable` state needed by the Library coordinator/tests, without exposing model paths;
- accept constructor seams for the worker target, completed-job bound, and bounded shutdown timings only — no user settings.

In `executor_worker.py`:

- keep module scope limited to stdlib plus the protocol and containment helper;
- enter/report containment, wait for admission, then lazily import heavy modules;
- point process-local temporary-file creation at the generation scratch directory before importing or invoking media/provider code;
- process one command at a time, emit only bounded phase/terminal envelopes, and close resident state in `finally`;
- reject a request whose attached generation does not equal the worker generation.

Export `LocalSTTExecutor` and the parent-facing request/identity types from `STT/__init__.py`; do not export worker internals.

- [ ] **Step 4: Run executor tests and static checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_local_stt_executor.py Tests/STT/test_executor_process_tree.py -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/STT/executor.py tldw_chatbook/STT/executor_worker.py tldw_chatbook/STT/executor_process_tree.py tldw_chatbook/STT/__init__.py Tests/STT/test_local_stt_executor.py Tests/STT/test_executor_process_tree.py Tests/STT/executor_test_support.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/STT/executor.py tldw_chatbook/STT/executor_worker.py tldw_chatbook/STT/executor_process_tree.py tldw_chatbook/STT/__init__.py Tests/STT/test_local_stt_executor.py Tests/STT/test_executor_process_tree.py Tests/STT/executor_test_support.py
git diff --check
```

Expected: all commands exit zero and no spawned child remains after pytest exits.

- [ ] **Step 5: Commit the controller slice**

```bash
git add tldw_chatbook/STT/executor.py tldw_chatbook/STT/executor_worker.py tldw_chatbook/STT/executor_process_tree.py tldw_chatbook/STT/__init__.py Tests/STT/test_local_stt_executor.py Tests/STT/test_executor_process_tree.py Tests/STT/executor_test_support.py
git commit -m "feat(stt): add generation-fenced local executor"
```

### Task 4: Reuse provider runtimes and hold artifact/local identities safely

**Files:**
- Modify: `tldw_chatbook/STT/transcribe_cpp.py`
- Modify: `tldw_chatbook/STT/executor.py`
- Modify: `tldw_chatbook/STT/executor_worker.py`
- Modify: `tldw_chatbook/Local_Ingestion/audio_processing.py`
- Modify: `tldw_chatbook/Local_Ingestion/video_processing.py`
- Modify: `tldw_chatbook/Local_Ingestion/local_file_ingestion.py`
- Modify: `Tests/STT/test_transcribe_cpp.py`
- Modify: `Tests/STT/test_local_stt_executor.py`
- Modify: `Tests/Local_Ingestion/test_ingest_parse_worker.py`
- Modify: `Tests/Transcription/test_parakeet_onnx_vertical_slice.py`

- [ ] **Step 1: Write failing injected-runner and reusable-provider tests**

Add tests that prove:

- `LocalAudioProcessor` uses an injected callable when present and never constructs `TranscriptionService` in that case;
- without injection, current `TranscriptionService()` construction and all existing result/error behavior are unchanged;
- `LocalVideoProcessor` forwards the injected callable to its audio processor;
- `parse_local_file_for_ingest` accepts the internal callable for audio/video and returns the same picklable payload shape, while `run_parse_job` remains unchanged for light work;
- a new reusable transcribe.cpp runtime performs admission/load once, transcribes two files, and closes once; the existing `transcribe_file()` still performs load/use/close once and preserves all failure redaction/actions/provenance;
- one worker-owned `TranscriptionService` reuses its Parakeet ONNX cache for two matching jobs;
- a local GGUF or Parakeet required-file snapshot is created in the parent, excluded from repr/log/result, and equality-checked immediately before load and reuse;
- changing any required local file maps to existing `ARTIFACT_INCOMPATIBLE` without loading or silently retrying different bytes;
- a managed request carries only `ArtifactRef`, store root, and expected closure fingerprint; the worker calls `ModelArtifactService.acquire()`, checks the returned fingerprint, uses paths only from the handle, and holds `LeasedArtifactHandle` while idle;
- exclusive deletion remains blocked for both root and dependency while the resident worker is idle, then succeeds after normal close and after forced worker death;
- the parent never owns a duplicate managed lease.

Use the real artifact service/lease primitives and isolated `tmp_path` store for lease process tests. Do not activate the deferred managed GGUF import prototype or build acquisition UI.

- [ ] **Step 2: Run the focused tests to verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_transcribe_cpp.py Tests/STT/test_local_stt_executor.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py -k "reusable or injected or resident or snapshot or managed or lease" -v
```

Expected: tests fail because there is no injected runner, reusable transcribe.cpp runtime, or worker-owned acquisition/runtime path.

- [ ] **Step 3: Add the smallest audio/video injection seam**

- Add an optional callable constructor argument to `LocalAudioProcessor`; `_transcribe_audio` invokes it with the same path/progress/options contract before considering `TranscriptionService`.
- Add the matching optional constructor argument to `LocalVideoProcessor` and pass it to `LocalAudioProcessor`.
- Add one keyword-only internal argument to `parse_local_file_for_ingest` and use it only when constructing audio/video processors.
- Leave `ingest_parse_worker.run_parse_job(file_path, options)` unchanged so light workers cannot receive or construct a heavy callable.
- Do not create a new media payload type or second post-processing stage.

- [ ] **Step 4: Split transcribe.cpp load/use/close without changing its public one-shot behavior**

Refactor `transcribe_cpp.py` around a reusable runtime that:

- reruns existing bounded admission before initial load;
- owns the current adapter/registry/coordinator and exposes `transcribe(...)` plus idempotent `close()`;
- checks the expected local snapshot before each reuse;
- preserves capability equality, result normalization, error codes, recovery actions, provenance, and native exception/path redaction;
- lets `transcribe_file()` remain a thin `load → transcribe → finally close` compatibility wrapper.

No native runtime import may move to module scope.

- [ ] **Step 5: Add worker-owned runtime and lease residency**

In `executor_worker.py`:

- build a Parakeet runtime from one `TranscriptionService` instance or the reusable transcribe.cpp runtime according to the exact resolved provider;
- for managed input, construct `ModelArtifactService` inside the worker, acquire the root, equality-check the expected closure fingerprint, derive model paths from the returned handle, and keep the handle beside the runtime until worker exit;
- for local input, revalidate the private snapshot before load and before same-identity reuse;
- close runtime before leased handle, and both in a `finally` path on command-loop exit;
- call `parse_local_file_for_ingest(..., transcription_runner=resident.transcribe)` and return its existing payload;
- map only bounded stable failures across IPC and keep raw exceptions inside the worker.

Do not add managed downloads or infer artifact identity from an untrusted IPC path.

- [ ] **Step 6: Run provider, ingestion, lease, and static checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/STT/test_transcribe_cpp.py Tests/STT/test_local_stt_executor.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py Tests/Model_Artifacts/test_operation_leases_process.py -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/STT/transcribe_cpp.py tldw_chatbook/STT/executor.py tldw_chatbook/STT/executor_worker.py tldw_chatbook/Local_Ingestion/audio_processing.py tldw_chatbook/Local_Ingestion/video_processing.py tldw_chatbook/Local_Ingestion/local_file_ingestion.py Tests/STT/test_transcribe_cpp.py Tests/STT/test_local_stt_executor.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/STT/transcribe_cpp.py tldw_chatbook/STT/executor.py tldw_chatbook/STT/executor_worker.py tldw_chatbook/Local_Ingestion/audio_processing.py tldw_chatbook/Local_Ingestion/video_processing.py tldw_chatbook/Local_Ingestion/local_file_ingestion.py Tests/STT/test_transcribe_cpp.py Tests/STT/test_local_stt_executor.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py
git diff --check
```

Expected: all focused tests pass; lease tests prove both normal and crash release; the optional native runtimes are not required because their load surfaces are faked.

- [ ] **Step 7: Commit the resident-runtime slice**

```bash
git add tldw_chatbook/STT/transcribe_cpp.py tldw_chatbook/STT/executor.py tldw_chatbook/STT/executor_worker.py tldw_chatbook/Local_Ingestion/audio_processing.py tldw_chatbook/Local_Ingestion/video_processing.py tldw_chatbook/Local_Ingestion/local_file_ingestion.py Tests/STT/test_transcribe_cpp.py Tests/STT/test_local_stt_executor.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py
git commit -m "feat(stt): retain resident batch runtimes"
```

### Task 5: Route eligible Library jobs and preserve the single writer

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Library/test_library_ingest_runner.py`

- [ ] **Step 1: Write failing Library dispatch and fencing tests**

Extend the existing `_IngestRunnerHarness` with a fake executor and tests that require:

- audio/video jobs whose resolved provider is `parakeet-onnx` or `transcribe-cpp` submit to the executor and never call general-pool `apply_async`;
- documents and audio/video work resolved to `faster-whisper` continue through the existing parse pool unchanged;
- the current heavy-lane count remains one, while available general-pool slots continue admitting light jobs;
- an executor success with matching generation/attempt stores the existing payload in `_ingest_parsed_payloads` and wakes the existing parent writer;
- executor progress only updates stable phase state and never fabricates a percentage;
- stale/wrong/duplicate executor results and failures never reach `_ingest_parsed_payloads`, `mark_failed`, or the writer;
- executor `ARTIFACT_INCOMPATIBLE`, `ENGINE_CRASHED`, and `CANCELLED` failures map to existing job failure/cancellation/provenance/recovery behavior without raw detail;
- executor startup failure marks only the triggering heavy job retryable and does not retire the light parse pool;
- executor crash/force stop affects only the active heavy job and leaves light pool object/generation/jobs intact;
- top-up resumes after every heavy terminal result;
- shutdown sets the shared ingest shutdown flag, detaches callbacks, starts executor close off the Textual event loop, and still tears down the light pool through its existing path.

Use `route.provider` as the eligibility decision. Do not infer eligibility from file extension alone and do not send faster-whisper to this executor.

- [ ] **Step 2: Run the focused Library tests to verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_ingest_runner.py -k "executor or heavy or light or stale or shutdown" -v
```

Expected: new executor-routing expectations fail because every claimed job still uses the general parse pool.

- [ ] **Step 3: Integrate one lazily owned executor into `LibraryIngestQueueMixin`**

Modify `app.py` to:

- lazily create one `LocalSTTExecutor` for the app and retain it until shutdown;
- split top-up admission so one eligible heavy request may run while remaining general-pool capacity admits light jobs;
- build the private identity/request from already resolved batch options without logging private model data;
- bind success/failure/progress callbacks to generation and attempt, then marshal accepted callbacks to the Textual thread;
- feed accepted success into `_ingest_parsed_payloads` and `_start_library_ingest_queue_if_idle()` exactly like current parse completion;
- keep the existing parent DB writer and parsed payload contract untouched;
- keep the parse-pool broken-generation handler scoped to only general-pool jobs;
- close the executor during app shutdown without blocking the event loop on an uninterruptible native call.

Do not create a second persistent queue or mirror Library jobs inside the executor.

- [ ] **Step 4: Run Library and cross-seam focused tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_ingest_runner.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/STT/test_local_stt_executor.py Tests/STT/test_executor_process_tree.py Tests/STT/test_transcribe_cpp.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/app.py tldw_chatbook/STT/executor.py tldw_chatbook/STT/executor_worker.py tldw_chatbook/STT/executor_process_tree.py tldw_chatbook/STT/transcribe_cpp.py tldw_chatbook/Local_Ingestion/audio_processing.py tldw_chatbook/Local_Ingestion/video_processing.py tldw_chatbook/Local_Ingestion/local_file_ingestion.py Tests/Library/test_library_ingest_runner.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/STT/test_local_stt_executor.py Tests/STT/test_executor_process_tree.py Tests/STT/test_transcribe_cpp.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/app.py tldw_chatbook/STT/executor.py tldw_chatbook/STT/executor_worker.py tldw_chatbook/STT/executor_process_tree.py tldw_chatbook/STT/transcribe_cpp.py tldw_chatbook/Local_Ingestion/audio_processing.py tldw_chatbook/Local_Ingestion/video_processing.py tldw_chatbook/Local_Ingestion/local_file_ingestion.py Tests/Library/test_library_ingest_runner.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/STT/test_local_stt_executor.py Tests/STT/test_executor_process_tree.py Tests/STT/test_transcribe_cpp.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py
git diff --check
```

Expected: all commands exit zero; documents and faster-whisper still use the existing pool; both targeted local providers use the executor; the writer remains parent-only.

- [ ] **Step 5: Commit the Library integration slice**

```bash
git add tldw_chatbook/app.py Tests/Library/test_library_ingest_runner.py
git commit -m "feat(library): route local STT through executor"
```

### Task 6: Focused macOS evidence, task notes, and review-ready closeout

**Files:**
- Modify: `backlog/tasks/task-601 - Add-generation-fenced-local-STT-executor.md` through Backlog CLI
- Verify: every TASK-601 production and test file listed above

- [ ] **Step 1: Run the complete TASK-601 focused verification set once**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_library_ingest_runner.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/STT/test_local_stt_executor.py Tests/STT/test_executor_process_tree.py Tests/STT/test_transcribe_cpp.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py Tests/Model_Artifacts/test_operation_leases_process.py -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/app.py tldw_chatbook/STT/__init__.py tldw_chatbook/STT/executor.py tldw_chatbook/STT/executor_worker.py tldw_chatbook/STT/executor_process_tree.py tldw_chatbook/STT/transcribe_cpp.py tldw_chatbook/Local_Ingestion/audio_processing.py tldw_chatbook/Local_Ingestion/video_processing.py tldw_chatbook/Local_Ingestion/local_file_ingestion.py Tests/Library/test_library_ingest_runner.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/STT/test_local_stt_executor.py Tests/STT/test_executor_process_tree.py Tests/STT/test_transcribe_cpp.py Tests/STT/executor_test_support.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/app.py tldw_chatbook/STT/__init__.py tldw_chatbook/STT/executor.py tldw_chatbook/STT/executor_worker.py tldw_chatbook/STT/executor_process_tree.py tldw_chatbook/STT/transcribe_cpp.py tldw_chatbook/Local_Ingestion/audio_processing.py tldw_chatbook/Local_Ingestion/video_processing.py tldw_chatbook/Local_Ingestion/local_file_ingestion.py Tests/Library/test_library_ingest_runner.py Tests/Local_Ingestion/test_ingest_parse_worker.py Tests/STT/test_local_stt_executor.py Tests/STT/test_executor_process_tree.py Tests/STT/test_transcribe_cpp.py Tests/STT/executor_test_support.py Tests/Transcription/test_parakeet_onnx_vertical_slice.py
git diff --check origin/dev...HEAD
```

Expected: all TASK-601-related tests and checks pass. Do not expand to the full suite unless a focused failure proves a changed dependency requires one additional named test file.

- [ ] **Step 2: Collect explicit native macOS process evidence**

Run the native spawn/descendant, lease-residency, crash-release, stale-callback, CPU-retry, and shutdown tests by exact names with `-vv`, and retain the command/result summary in the task Implementation Notes. Confirm no executor/FFmpeg test process survives.

Do not claim Windows or Linux native proof. Record those gates as preserved/open.

- [ ] **Step 3: Review the complete branch diff**

```bash
git diff --stat origin/dev...HEAD
git diff --check origin/dev...HEAD
git status --short
```

Review specifically for:

- model/media paths or snapshot tokens in repr/log/error/result payloads;
- callbacks invoked while the controller lock is held;
- a path that starts a new generation before old-tree death is proven;
- scratch cleanup that occurs before normal exit/proven tree death, or cleanup after an unproven stop;
- an executor-side queue or DB write;
- a provider load at import time;
- managed paths trusted from IPC instead of the acquired handle;
- faster-whisper accidentally moved out of the general pool;
- shutdown joins performed on the Textual event loop;
- unrelated edits or tests.

- [ ] **Step 4: Update TASK-601 without overstating the platform gate**

Use Backlog CLI to:

- add concise Implementation Notes covering the controller/worker boundary, provider reuse, lease lifetime, Library routing, process containment, focused verification, and deviations;
- check acceptance criteria whose implementation and macOS evidence are complete;
- explicitly record Windows/Linux native validation as open;
- leave status `In Progress` if the project interprets AC #6/#7 as requiring the unavailable native matrix, rather than marking Done prematurely.

- [ ] **Step 5: Commit closeout documentation**

```bash
git add 'backlog/tasks/task-601 - Add-generation-fenced-local-STT-executor.md'
git commit -m "docs: record task 601 implementation evidence"
```

- [ ] **Step 6: Request code review before PR/merge work**

Use `superpowers:requesting-code-review` against `origin/dev...HEAD`. Address only evidence-backed issues within TASK-601 scope, rerun the smallest affected test set after each fix, then rerun Step 1 once before claiming the branch is review-ready.
