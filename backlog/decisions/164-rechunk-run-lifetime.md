# ADR-164: Keep Library Re-chunk work and feedback for the app session

- Status: Accepted
- Date: 2026-09-17
- Task: TASK-32719
- Related: ADR-078, ADR-003, ADR-031, ADR-150
- Allocation: all reachable decision paths and 27 worktrees have maximum 163.

## Context

Re-chunk runs in a thread owned by a replaceable Search/RAG panel. Library rail
navigation or leaving Library removes that panel. The returned panel loses the
running state and receipt; the original worker's UI callbacks still target the
removed panel. Four navigation tests reproduce enabled controls while work is
active and missing completion notices when it finishes while away.

Re-chunk already operates on older-engine items in the local Library, independently
of the Search/RAG query and selected sources. Its admission guard is shared with
Settings backfill. Navigation is not an explicit cancellation request.

## Decision

Give this operation one lazy, ephemeral UI run object per Textual app. The app
owns its thread worker. The run object stores only running state and the latest
completion summary, and publishes changes on Textual's existing Signal mechanism.
Mounted panels subscribe, read current state on entry, and unsubscribe on removal.
Completion refreshes each attached panel's legacy count. All state changes and
publication occur on the UI thread; no widget is retained by the worker.

Keep the existing group, explicit Re-chunk/backfill slot refusal, local scope
service and runtime-policy admission. Publish completion and release the slot in
one UI-thread callback, so a newly admitted run cannot be overwritten by an old
completion. Failure clears running feedback, surfaces the existing error notice,
and permits retry. Failure to schedule a worker releases the slot too.

The summary survives Library canvas and main-destination navigation during this
app session. Starting a new run replaces it. It is not persisted, is not a job
history, and does not promise resumption after process exit. Shutdown retains
the app's existing worker/executor handling and per-item derived-data safety.

## Alternatives considered

- Keep panel ownership and copy a busy flag on return: a new panel still has no
  completion source, and the removed panel's callbacks can fail.
- Retain hidden Search/RAG panels: couples the whole Library navigation layout
  to one operation and does not cover Library screen replacement.
- Add a durable job system or poll shared slots: unnecessary for a session-local
  action; slots describe exclusion, not completion receipts.

## Consequences

The UI gets a small app-session owner without changing the data service or public
service contracts. Tests must cover completion while away, return before completion,
screen replacement, failure/retry and shared exclusion. Actual semantic reindexing
remains conditional and disclosed per ADR-078.
