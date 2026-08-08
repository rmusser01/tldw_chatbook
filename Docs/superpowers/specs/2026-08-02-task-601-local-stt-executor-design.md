# TASK-601 Local STT Executor Design

**Date:** 2026-08-02

**Status:** Approved for implementation planning

**Task:** TASK-601 — Add generation-fenced local STT executor

**Governing decisions:** [ADR-025](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md), [ADR-041](../../../backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md)

## Purpose

Move Parakeet ONNX and transcribe.cpp batch inference out of the general
Library parse pool into one app-owned, spawn-isolated process. The executor must
reuse one compatible resident model, hold managed-artifact leases for the full
resident lifetime, contain crashes and decoder subprocesses, and prevent stale
callbacks from reaching the existing parent-side media writer.

This is executor hardening, not another ingestion system. The existing Library
job registry remains the only persistent queue, the existing parsed-payload
contract remains the worker result, and the existing parent transaction remains
the only media writer.

## Scope

TASK-601 will:

- Add one app-owned `LocalSTTExecutor` with exactly one spawn-context worker.
- Route Library audio/video batch jobs for Parakeet ONNX and transcribe.cpp to
  that worker while leaving documents and other light parsing in the current
  parse pool.
- Reuse the resident model only when its complete private identity matches.
- Recycle the worker on identity change, native crash, force stop, qualified
  CPU retry, shutdown, or an internal completed-job bound.
- Retain managed root and dependency leases while their model is resident.
- Fence every request and callback by attempt identity and executor generation.
- Contain the worker and its decoder descendants as one platform process tree.
- Preserve the current parent-side writer, normalized provenance, and explicit
  faster-whisper recovery behavior.

TASK-601 will not:

- Add another queue, scheduler, database table, or user-visible setting.
- Add managed downloads, local import, or artifact-browser behavior.
- Promote Parakeet to the semantic default or remove legacy providers.
- Implement dictation coalescing, priority, or UI backpressure; TASK-603 owns
  those policies. The executor protocol may carry a bounded buffer later.
- Move unrelated standalone `TranscriptionService` callers into the executor.
- Claim native Windows or Linux execution without those hosts.

## Chosen Architecture

### Dedicated process, not a pool

`LocalSTTExecutor` owns one `multiprocessing.get_context("spawn").Process`.
Parent and worker communicate through a small duplex connection. A dedicated
parent reader thread drains worker events so the Textual event loop never
blocks on IPC.

The process also receives a one-shot admission event. It may initialize the
stdlib protocol loop, but it cannot import providers, load models, or launch a
preparation child until the parent has established process-tree containment
and signalled admission. This closes the Windows start-to-Job-assignment race.

A one-worker `multiprocessing.Pool` was rejected because targeted worker
replacement, generation fencing, bidirectional progress, and force-stop
containment are awkward around pool-managed worker lifecycles. Separate
provider processes were rejected because they would permit two resident heavy
models and duplicate the lease/process boundary.

The executor accepts one active request. It does not hold a second backlog;
the existing Library coordinator decides which queued job runs next.

### Small, dependency-free protocol

The protocol consists of frozen, picklable data objects with no native runtime
imports:

- `ModelIdentity`: provider, model, managed root revision when present,
  dependency-closure fingerprint when present, precision, effective execution
  target, and a private local-source snapshot token when needed.
- `ExecutorRequest`: generation, attempt and Library job identities, source or
  bounded buffer description, existing batch/retry context, resolved
  transcription options, and model identity. Model and media paths use
  redacted repr fields.
- `ExecutorEvent`: generation, attempt identity, stable phase, and optional
  bounded progress.
- `ExecutorResult`: generation, attempt identity, and the existing picklable
  parsed-media payload.
- `ExecutorFailure`: generation, attempt identity, stable error code, bounded
  recovery actions, sanitized failure provenance, and no raw exception text.

These are transient IPC contracts, not persisted schemas. They do not add a
separate provenance format.

### Parent controller

The controller owns:

- The monotonically increasing executor generation.
- The current worker process, IPC endpoint, generation-scoped cancellation
  event, and process-tree containment handle.
- The active attempt and one terminal-state guard.
- The current resident identity reported by the worker.
- A session-local unhealthy identity after a relevant native crash.
- An internal completed-job recycle bound with a constructor test seam, not a
  new user setting.
- A controller lock that serializes request, cancel, close, and transport send
  operations.

Before dispatch, the controller attaches the current generation. It accepts a
worker event only when both generation and attempt identity match the active
request. Detaching a generation happens before force termination or replacement,
so a late success from the old process is discarded before it can reach the
writer.

An idle identity change or completed-job recycle uses a bounded graceful
`close` command before termination. An active request is never replaced merely
because a different job is queued; the existing Library queue waits for the
single heavy lane to become available.

The unhealthy circuit is deliberately small: one relevant native crash pauses
automatic dispatch for the same model identity. An explicit user retry clears
that identity once. Unrelated models and providers continue.

`cancel(attempt_id)` sets the generation event only while that exact attempt is
active. The controller clears the event while holding its lock immediately
before sending the next request, preventing a late cancellation for a completed
attempt from cancelling its successor.

### Worker runtime

The spawn entry module keeps module scope limited to the standard library and
protocol imports. Provider, artifact, and ingestion modules load only inside
the worker entry point.

The worker owns one batch runtime at a time. It reuses the existing Parakeet
ONNX and transcribe.cpp implementations rather than introducing another plugin
registry. The existing audio/video parse path receives a worker-owned
transcription runner so it does not construct or load another local STT model.
The worker returns the same payload that the current parse worker returns; it
never opens the media database.

For transcribe.cpp, the current one-shot wrapper will be split around its
existing reusable adapter: the direct wrapper may still load/use/close once,
while the executor worker holds the loaded model and adapter across matching
jobs. Admission, runtime capability probing, coordinator validation, result
normalization, and path redaction remain unchanged.

For Parakeet ONNX, one worker-owned service/runtime instance retains the model
cache only for the current identity. An identity change replaces the process
instead of asking ONNX Runtime or a native provider to unload in place.

Keeping the complete existing audio/video parse call in the heavy worker is an
intentional compatibility choice. It avoids inventing a new partial-media
payload or a second processing stage. Optional post-transcription analysis may
therefore occupy the one heavy lane longer, but behavior and writer ownership
remain unchanged.

## Model Identity and Leases

The canonical resident identity contains:

`(provider, model, root revision, closure fingerprint, precision, execution target, local snapshot token)`

The local snapshot token is absent for immutable managed artifacts. It is
private transient data: it must be excluded from reprs, logs, generic errors,
results, and persisted provenance.

For managed Parakeet bundles, the request carries the managed root
`ArtifactRef` and expected closure fingerprint, not trusted payload paths. The
worker opens `ModelArtifactService` over the configured store, calls its
existing `acquire()` boundary, and equality-checks the returned closure
fingerprint before model load. The resulting shared root/dependency lease set
stays open across idle same-identity reuse and closes only when the resident
runtime closes or the process exits. The parent does not hold a duplicate
lease.

For a legacy/local Parakeet directory, the required model files are snapshotted
before dispatch and revalidated immediately before load or reuse. For a
direct-local GGUF, TASK-597 admission supplies the bounded source identity and
is rerun inside the worker. A mismatch fails safely instead of reusing the
resident model. A source that changed between parent snapshot and worker
admission uses the existing `ARTIFACT_INCOMPATIBLE` failure and is not
automatically retried against different bytes. Unmanaged local files do not
pretend to have managed leases.

## Request Lifecycle

1. The Library coordinator resolves the existing batch route and builds a
   private model identity off the Textual event loop.
2. Document and light-media jobs continue to the general parse pool.
3. Parakeet ONNX and transcribe.cpp audio/video jobs are claimed from the same
   Library queue and submitted to `LocalSTTExecutor`.
4. The controller starts a worker if none exists. If the requested identity
   differs from the resident identity, it closes the old generation and starts
   a new one before dispatch.
5. The worker revalidates local sources or acquires managed lease sets, then
   loads or reuses the model.
6. The worker prepares media, transcribes, normalizes the result and provenance,
   and returns the existing parsed payload.
7. The parent verifies generation and attempt identity, then hands the payload
   to the existing single-writer stage.
8. The writer atomically persists transcript content and provenance and marks
   the Library job complete.

Failure before step 7 produces no media write. Writer failure continues to
roll back through the existing transaction.

## Progress and Terminal States

Stable phases remain:

`queued → preparing → loading → transcribing → post-processing → saving → complete`

The worker reports bounded phase transitions through post-processing. The
parent writer owns saving and complete. TASK-601 does not add high-frequency
percentage transport; a later provider may add real bounded progress without
changing generation fencing. The executor never invents a percentage.

The parent terminal guard allows exactly one of success, failure, or
cancellation for an attempt. Duplicate, stale, detached-generation, or
wrong-attempt envelopes are ignored and recorded without rendering private
payload data.

## Cancellation, Crash Recovery, and CPU Retry

Each worker generation receives one shared cancellation event at process
creation. The controller clears it under the controller lock immediately before
dispatch and sets it only for the matching active attempt. Preparation and
providers check it at their supported boundaries. This avoids adding a listener
thread inside the worker merely to receive a cancel command while native
inference blocks.

Queued cancellation is handled by the existing Library registry. During active
work, cooperative cancellation is attempted first. If an uninterruptible native
call does not return, the UI may invoke force stop. Force stop:

1. Detaches the active generation from the writer path.
2. Records the attempt's single cancelled terminal state.
3. Terminates the contained worker process tree.
4. Joins cleanup off the Textual event loop.
5. Starts a fresh generation only when another request is dispatched.

Step 5 is allowed only after the controller has confirmed the old process tree
is dead. If containment termination or join cannot prove that, the executor
enters `unavailable` state and rejects further heavy dispatch instead of
allowing two worker generations to coexist. Recovery requires app shutdown or
restart; TASK-601 does not add another reset API or recovery control.

The parent monitors the worker sentinel and remembers the latest phase. A crash
during loading or transcribing uses the existing `ENGINE_CRASHED` failure. A
crash while preparing remains an existing sanitized parse-stage failure and
does not mark the model unhealthy. Only the active audio/video attempt fails.
The general parse pool remains alive.

A typed, provider-qualified device failure may retry once on CPU using the same
attempt identity in a fresh generation. The successful result records the
device fallback warning. Generic errors and string matching never trigger this
retry, and no cross-engine fallback is automatic.

## Process-Tree Containment

The entire worker generation is the containment unit.

- On POSIX, the worker enters its own session/process group, reports the
  resulting process-group identity, and waits for the parent admission signal
  before launching preparation subprocesses. Cooperative cleanup signals the
  active decoder group before removing temporary files; force stop terminates
  the worker group.
- On Windows, the parent assigns the spawned worker to a kill-on-close Job
  Object before signalling the worker admission event. Decoder descendants
  then inherit containment. Cooperative cleanup stops the active decoder before
  temporary cleanup; force stop terminates or closes the job and then joins the
  worker.

This is intentionally narrower than a general subprocess framework. It exists
only to guarantee that FFmpeg or another preparation child cannot outlive the
heavy worker generation.

## Application Integration

`TldwCli` owns one `LocalSTTExecutor`, creates it lazily, and closes it during
application shutdown without waiting for an uninterruptible transcription.
The Library coordinator sends only eligible audio/video work to it. Its existing
heavy-lane count remains the admission/backpressure mechanism during migration.

The batch migration covers Parakeet ONNX and transcribe.cpp. Standalone legacy
`TranscriptionService` callers remain process-local for compatibility in this
task. TASK-603 later moves Console dictation through the shared executor and
adds bounded coalescing and priority. TASK-605 later removes obsolete provider
paths after every release gate passes.

## Error and Privacy Rules

- Native exception text never crosses the process boundary.
- Model/artifact paths and filesystem snapshot tokens never appear in reprs,
  logs, generic errors, UI state, result payloads, or persisted provenance.
  Existing media-source-path handling is unchanged by this task.
- Existing stable STT error codes and bounded recovery actions remain
  authoritative.
- Eligible failures may offer explicit **Retry with faster-whisper**; the
  executor never performs that cross-engine retry silently.
- A direct-local artifact change continues to offer **Choose another GGUF…**
  where TASK-604 already allows it.
- Worker and controller logs use attempt, job, generation, provider, stable
  model ID, phase, and error code only.

## Testing Strategy

Only TASK-601-related tests and static checks will run.

### Dependency-free tests

- Protocol objects are picklable and redact private identity data.
- The controller reuses a matching identity and recycles on every identity
  component change and the completed-job bound.
- Generation and attempt fencing rejects stale progress, results, and errors.
- The terminal guard allows exactly one terminal outcome.
- Cooperative cancel sets the shared event; force stop detaches before kill.
- A late cancel cannot affect the next attempt, and a failed force stop cannot
  create a second live generation.
- Only a typed qualified device failure retries once on CPU in a new generation.
- Native crashes mark only the relevant identity unhealthy.

### Spawn-process tests

Fake worker runtimes will prove same-model reuse, provider/identity replacement,
crash release, stale callbacks, cancellation, CPU retry, shutdown, and that the
general parse pool is not affected. Tests use real spawn processes where the
behavior depends on process lifetime rather than mocks.

### Artifact and ingestion integration

- An idle resident managed model retains root and dependency leases.
- Worker exit releases the OS-backed lease set.
- Library audio/video dispatch reaches the executor and the existing parent
  writer; document dispatch remains in the parse pool.
- Parakeet ONNX and transcribe.cpp batch paths reuse one resident runtime and
  preserve normalized results, failures, provenance, and recovery actions.
- Direct-local source changes fail without path leakage.

### Platform containment

Focused platform-contract tests cover POSIX session/group and Windows Job
Object setup, cooperative child cleanup, and force-stop ordering. Native macOS
spawn/descendant evidence will be collected locally. Windows and Linux native
execution remain preserved release gates and will not be claimed while hosts
are unavailable.

## Rollout and Completion Gate

The implementation may merge after focused macOS verification and review while
the unavailable Windows/Linux evidence remains explicitly open, matching the
existing gate policy. TASK-601 must not be represented as cross-platform proven
until the native matrix is complete. TASK-602 may build on the merged executor
without weakening that release gate, and TASK-605 cannot promote defaults or
remove legacy providers until all required platform evidence passes.

## ADR Check

**ADR required:** no new ADR.

**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
and `backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md`.

**Reason:** ADR-025 already establishes the app-owned one-process heavy lane,
resident identity, lease lifetime, generation fencing, process-tree cleanup,
and provider/runtime boundary. ADR-041 already establishes the direct-local
GGUF exception and later executor migration. This design narrows implementation
details without changing those accepted decisions.

## Rejected Alternatives

- **One-worker `multiprocessing.Pool`:** pool lifecycle obscures targeted
  replacement, stale callback rejection, and force-stop process-tree ownership.
- **One process per provider:** permits multiple resident heavy models and
  duplicates leases, cancellation, and crash handling.
- **Another persistent executor queue:** duplicates the Library registry and
  creates synchronization and recovery problems without user value.
- **Moving every transcription caller now:** crosses into TASK-603 dictation and
  TASK-605 compatibility/removal work.
- **Reimplementing managed acquisition:** TASK-595/TASK-596/TASK-598/TASK-1915
  own that lifecycle; the executor consumes resolved identities only.
