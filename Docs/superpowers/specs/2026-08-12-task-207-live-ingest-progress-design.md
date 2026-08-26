# TASK-207: Live Local Ingest Progress Design

## Purpose

The Library ingest queue should explain what a long-running local import is
doing without pretending to know more than the parser knows. Existing lifecycle
states stay unchanged. A quiet secondary line reports the current parse stage
and, only where a real bounded measurement exists, its stage-scoped percentage.

Examples:

- `42% · Extracting page 21 of 50`
- `Transcribing audio`
- `Chunking extracted text`
- `Saving to Library`

The design deliberately avoids animated borders and progress bars. The filename,
state, and row actions remain primary; progress is supporting detail that never
obscures an entry.

## Existing foundation

The current implementation already provides more than TASK-207's 2026-07-12
description assumed:

- `LibraryIngestJob.progress` is a persisted structured payload.
- `LibraryIngestJobRegistry.update_progress` rejects terminal jobs.
- server reconciliation maps `progress_percent` and `progress_message` into that
  payload.
- the ingest queue renders a secondary progress line.
- local STT reports coarse worker phases through its own callback path.

The missing slice is a safe process-worker-to-app channel for ordinary local
parse jobs, truthful progress normalization and formatting, and an in-place UI
update path that does not recompose the queue on every tick.

## Product and interaction design

Progress is an explanation, not another job state. The primary row continues to
read `parsing · filename.pdf` or `writing · filename.pdf`. The secondary line:

- uses sentence-style stage copy without repeating `parsing` or `writing`;
- includes an integer percentage followed by ` · ` only when the event contains
  a trustworthy bounded value;
- stays visible in the same location throughout `PARSING` and `WRITING`;
  `Preparing import` is the honest pre-tick fallback, so the first worker event
  updates a reserved line instead of moving neighboring rows;
- is visually subordinate through a dedicated muted
  `.library-ingest-progress` rule (the current widget has no specific CSS
  treatment, so this task adds one);
- has no animation, border, or color-only meaning;
- is markup-disabled, single-line, and length-bounded.

Phase-only local STT events receive human-readable copy (`Transcribing audio`,
not an empty or machine-token line). When a parse finishes, the transition to
`WRITING` immediately replaces parse progress with `Saving to Library`. Terminal
rows retain their existing outcome receipts.

## Progress semantics

The local in-flight payload uses these concepts:

- `phase`: a controlled machine token such as `inspecting`, `extracting`,
  `transcribing`, `chunking`, `analyzing`, or `writing`;
- `message`: short user-facing text for the current phase;
- `percent`: optional finite stage completion in the inclusive 0-100 range;
- existing local-STT control metadata such as `cancel_requested` remains
  registry-owned and is not accepted from arbitrary parser/provider data.

Percentages are stage-scoped, not overall-job estimates. A new phase without a
known total clears the old percentage. The UI does not interpolate, smooth, or
weight phases. When an underlying parser supplies `(current, total)` or an
existing percentage callback, the adapter may forward it. Otherwise it reports
only a message.

The worker emission boundary first reduces values to bounded, picklable plain
primitives so an arbitrary provider object or oversized string cannot poison the
queue's feeder thread. The app repeats validation rather than trusting IPC: it
removes line breaks and control characters, caps display length, accepts
percentages only when they are finite and inside 0-100, and ignores unknown
event fields. Invalid values are omitted rather than clamped into
plausible-looking progress. Raw provider metadata never becomes display copy.

## Cross-process architecture

Each real parse-pool generation owns one resource bundle containing:

1. a bounded `spawn`-context multiprocessing queue;
2. a daemon parent drain thread;
3. a stop event associated with that generation.

The queue is created before the pool and passed through the pool initializer so
spawned workers inherit a valid endpoint. Queue and pool construction are
treated atomically and run under the existing real-stderr resource-tracker
workaround; partial construction closes already-created resources before the
error escapes. The initializer preserves the current import-noise suppression
and installs the worker-side progress sink.

`run_parse_job` receives a small progress context containing the pool generation
and Library job id. It creates an in-process callback and passes it to
`parse_local_file_for_ingest`; downstream adapters forward existing progress
callbacks through that seam. Emission uses `put_nowait`. `Full`, broken/closing
pipe, and shutdown errors drop the event. Progress failure never changes the
parse result.

The parent drain thread coalesces by job id and schedules at most one latest
event per job per approximately 250 ms interval. Its monotonic clock and flush
callback are injectable so cadence tests do not depend on wall-clock sleeps. It
uses `call_from_thread` to apply a batch on the Textual thread. This thread is
independent of the pool's result-handler thread, so progress cannot block result
delivery.

## Ordering and authority

Separate result and progress channels do not guarantee arrival order. Therefore
the UI-thread handler accepts a progress event only when all of these remain
true:

- the application is not shutting down;
- the event generation equals the current pool generation;
- the job id remains in that generation's in-flight set;
- the registry job remains in `PARSING`;
- `_ingest_parsed_payloads` does not already contain the job.

The last condition closes the payload-ready window: a result may arrive before a
late queued progress event while the job still says `PARSING` waiting for the
single writer. Once a parse result is accepted, no later extraction tick may
overwrite its next-stage receipt.

Terminal registry guards remain defense in depth. A broken pool invalidates the
generation and fails its still-dependent jobs as it does today; queued progress
from that generation is ignored.

## Registry and persistence

The existing `progress` field remains the one UI projection; no schema migration
or new job state is needed. Registry progress updates gain an explicit transient
mode for local live ticks so they notify the UI without writing SQLite several
times per second. Server polling retains its current persistence behavior.

The registry exposes a progress-specific listener carrying before/after job
snapshots. Ordinary lifecycle listeners remain unchanged. This lets the screen
update only the mounted progress widget when row structure is stable, while
still recomposing when progress changes local-STT Cancel or Force stop actions.

`mark_writing` replaces the parse payload with a writing-stage message rather
than preserving a stale percentage. Lifecycle and terminal mutations remain
persisted through their existing paths.

## UI update path

Every `PARSING` and `WRITING` queue row mounts a stable, visible progress
`Static`; `PARSING` falls back to `Preparing import` before its first event.
The progress listener formats and updates that widget in place without replacing
it. This preserves row, form-widget, focus, scroll, and vertical layout identity
during rapid ticks.

If the before/after row projections differ in action availability or another
structural property, the screen uses the existing dynamic-region recompose path
instead. Lifecycle transitions continue through the existing registry listener
and queue recomposition because their state, glyph, actions, and receipts can
legitimately change.

## Shutdown and failure handling

Shutdown follows the existing no-UI-thread-join rule:

1. set `_ingest_shutdown` and the generation's progress stop event;
2. detach pool, local STT executor, queue, and drain-thread references;
3. on the existing daemon cleanup thread, terminate/join workers, let the drain
   thread exit with a bounded wait, then close the queue and cancel queue-thread
   joining where supported.

A drain callback that crossed the shutdown check immediately before teardown is
allowed to reach the still-free UI loop and then no-op. The UI thread never
waits for the pool, queue, or drain thread. Queue EOF/broken-pipe errors during
teardown are expected and quiet.

## Initial instrumentation scope

All local parse branches report useful stage transitions the current seams can
observe truthfully:

- inspect/classify source;
- extract content;
- transcribe audio/video where applicable;
- chunk content when the shared text-tail performs chunking;
- analyze content when the shared text-tail performs analysis;
- complete parse and hand off to writing.

Existing measurable transcription callbacks may supply percentages. Other
extractors report a percentage only if their current API already exposes a real
total. A processor that internally combines extraction, chunking, and analysis
behind one opaque call keeps one honest `Processing …` message rather than
claiming unobservable internal phase changes. This task does not restructure
PDF, document, ebook, or web extractors solely to manufacture page counts,
progress values, or finer stages.

## Verification

Tests must prove:

- normalization accepts valid stage progress, omits indeterminate percentages,
  clears percentage on phase change, and sanitizes malformed messages/values;
- worker emission is non-blocking and a full/closing channel cannot fail a
  parse;
- parent coalescing retains the latest job event and respects its cadence;
- generation, membership, state, terminal, and payload-ready fences reject stale
  events;
- `mark_writing` clears extraction percentage and shows the saving message;
- phase-only local STT events render readable copy and retain action changes;
- ordinary progress updates preserve the mounted row/progress widget, focus, and
  scroll identity;
- lifecycle changes still recompose correctly;
- local live ticks do not persist while server progress and lifecycle outcomes
  retain their existing persistence contracts;
- a Windows-gated real spawned-pool test exercises event delivery and clean
  shutdown, in addition to portable deterministic fake-pool tests.

Focused registry, worker, runner, state, and canvas suites are required. Broader
tests that encounter the repository's known Windows Proactor/network-guard
socketpair conflict must be reported separately rather than mistaken for TASK-207
failures.

## ADR check

ADR required: yes.

ADR path: `backlog/decisions/061-library-ingest-parse-progress-channel.md`.

Reason: the work adds a durable cross-process message contract, resource
lifecycle, backpressure policy, shutdown ordering, and UI projection boundary.
ADR-061 complements ADR-014 without changing ingest authority or recovery.
