# ADR-061: Library Ingest Parse Progress Channel

Status: Accepted
Date: 2026-08-12
Related Task: [TASK-207](../tasks/task-207%20-%20Live-parse-progress-for-ingest-jobs-progress_percent-progress_message.md)
Supersedes: N/A

## Decision

Local Library parse workers report best-effort, stage-scoped progress through a
bounded, non-blocking, generation-fenced process channel; the app validates and
coalesces those events before updating the existing job projection, while
terminal results and backend lifecycle state remain authoritative.

## Context

Library ingest already has the right user-facing lifecycle states (`queued`,
`parsing`, `writing`, and terminal outcomes), a structured `progress` field on
`LibraryIngestJob`, server progress reconciliation, and a secondary queue-row
line. Local process-pool jobs, however, normally jump from `parsing` to
`writing` without explaining lengthy extraction, transcription, chunking, or
analysis work.

Sending progress across the existing `multiprocessing.Pool` boundary is a
runtime-contract decision rather than a cosmetic UI change. This pool uses the
Windows `spawn` context, a real-stderr resource-tracker workaround, generation
fencing, worker-sentinel monitoring, and off-UI-thread termination specifically
to avoid hangs and stale callbacks. A new channel must preserve those
properties. It must also avoid turning optional telemetry into backpressure,
SQLite write amplification, or fabricated progress.

ADR-014 remains authoritative for Library ingest ownership, backend authority,
and recovery. This ADR only defines the transient local parse-progress path and
its UI projection.

## Decision Details

- Create one bounded progress queue with each real parse-pool generation. Create
  the queue and pool under the same spawn context and stderr/resource-tracker
  workaround, and pass the queue to workers through the pool initializer.
- A worker emits a small picklable event containing pool generation, job id,
  phase, message, and optional percentage. Values are reduced to bounded plain
  primitives before IPC, then revalidated in the parent. Queue writes use
  `put_nowait` and silently drop a tick when the queue is full or closing.
- Percentages are finite values inside the documented 0-100 input range and are
  stage-scoped. Invalid or out-of-range values are omitted rather than coerced
  into plausible-looking progress. A parser reports a percentage only when it
  has a real bounded measurement. Phase changes clear the previous phase's
  percentage.
- The app-owned drain thread keeps only the latest event per job and marshals a
  bounded batch to the Textual thread no more than approximately four times per
  second. Its clock and flush callback remain injectable for deterministic
  tests. Progress cadence does not alter parse-result cadence.
- The UI-thread handler accepts an event only when its pool generation is
  current, its job is still assigned to that generation, the job remains
  `PARSING`, and no completed parse payload is already awaiting the writer.
- The parent normalizes phase/message/percentage before updating the registry.
  Messages are single-line, length-bounded display text; unknown fields and
  provider callback data do not reach the UI.
- Live local parse ticks are memory-only. Server polling keeps its existing
  persistence behavior, while lifecycle transitions and terminal receipts
  remain persisted. `mark_writing` replaces parse detail with `Saving to
  Library`, preventing a stale extraction percentage from appearing under the
  writing state.
- Active queue rows always mount a progress-detail widget. Ordinary progress
  changes update that widget in place. A change that also alters row actions
  (for example local STT cancel/force-stop availability) may recompose the queue
  panel through the existing context-preserving path.
- Shutdown sets the progress stop flag before detaching the pool. Worker
  termination, drain-thread cleanup, and queue close/join cancellation happen
  off the Textual thread. A progress callback that races shutdown no-ops.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Recompose the queue panel for every progress tick | It needlessly remounts rows and risks focus, scroll, and click instability during high-frequency work. |
| Pass a callback object directly in each pool task | UI callbacks and most closures are not spawn-picklable, and blocking callbacks would couple parser throughput to the event loop. |
| Use a manager-backed queue or database polling | It adds another process or persistent coordination layer, latency, cleanup, and failure modes for transient telemetry. |
| Make progress events reliable and blocking | A full or closing progress channel could stall the actual ingest, reversing the authority relationship between telemetry and results. |
| Synthesize an overall percentage from lifecycle stages | Stage durations vary radically by file type and options; weighted guesses would present false precision. |
| Replace the process pool with threads | It would discard current CPU concurrency, crash isolation, and explicit termination behavior to simplify a non-authoritative signal. |
| Persist every local progress tick | It adds SQLite churn and does not improve recovery because interrupted active local jobs cannot resume at an extraction percentage. |

## Consequences

Long local imports gain truthful, quiet feedback without adding lifecycle states
or a visually dominant progress bar. A job may skip intermediate ticks under
load, and indeterminate phases may show text without a percentage; both are
intentional. Parse completion, failure, cancellation, pool failure, and shutdown
remain correct even if every progress event is lost.

The parse-pool generation now owns an additional queue and drain thread, so real
Windows spawn and shutdown verification is required. Parser adapters may forward
existing measurable callbacks over time, but they may not invent percentages or
leak arbitrary provider payloads into the shared progress contract.

## Links

- [TASK-207 design](../../Docs/superpowers/specs/2026-08-12-task-207-live-ingest-progress-design.md)
- [ADR-014: Library Ingest Service Authority and Recovery](014-library-ingest-service-authority-and-recovery.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
