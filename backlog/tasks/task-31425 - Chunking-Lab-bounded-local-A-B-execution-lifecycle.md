---
id: TASK-31425
title: Chunking Lab - bounded local A-B execution lifecycle
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 23:13'
updated_date: '2026-09-05 01:31'
labels:
  - chunking
  - chunking-lab
dependencies:
  - TASK-31421
  - TASK-31422
  - TASK-31423
  - TASK-31424
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run one or two captured recipes locally without freezing the TUI or confusing old results with the current experiment. Covers spec sections 4 and 6 and AC 3, 7-10, 15-16, 21, 24. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: process execution, cancellation, immutable batches, and backend provenance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run both validates and durably captures the sample, both recipes, runtime identities, and candidate IDs before A starts; edits and catalog changes cannot alter queued B.
- [x] #2 Only one bounded local process runs at a time; cancellation, navigation, restore, and Clear stop current work and pending batch members before a new run starts, and late replies cannot publish into newer epochs.
- [x] #3 2 MiB sample, 10000 chunk, 32 MiB result, and 60 second preview limits have visible outcomes, preserve previous outputs, and never silently clip data; intermediate resource behavior is verified.
- [x] #4 Failed A may be followed by B, cancel stops the queue, and old outputs remain explicitly previous rather than filling a failed batch member; all terminal outcomes retain backend and input provenance.
- [x] #5 Restore and Clear cannot publish replacement authority until the old worker is terminated and reaped and its queue is stopped; failed replacement preserves the original in-memory session and writer retry authority.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: implements the approved bounded local runtime and immutable batch authority. 1. Read Task 5 brief/context and current state/writer/recovery contracts. 2. Write failing process timeout/cancel/limit and coordinator ordering tests. 3. Implement one bounded fresh child and immutable A-before-B lifecycle with manifest commit before launch and honest terminal states. 4. Verify intermediate resource behavior, fresh Textual-compatible first launch, off-loop work, late-result fencing and stop-before-restore/Clear. 5. Run targeted runner/coordinator/recovery/autosave checks plus static checks, self-review, independent review and evidence notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented one fresh stdlib subprocess per immutable preview, bounded framed JSON,
off-loop admission/serialization/supervision, and termination followed by kill/reap.
The coordinator commits the captured A/B manifest before launching either member,
keeps edits out of queued inputs, preserves unsaved outcomes after storage failure,
and quiesces the physical worker before restore, Undo restore, Clear, and close.
Small copied status/invalidation events reuse the borrowed read-only session
snapshot convention, avoiding report copies on every edit. Initial recovery is an
async load that never dispatches execution or assumes authority after a read failure.

Implements ADR-118, including the controller-approved shared 16-operation and 2 MiB
per canonical recipe ceilings. Sample-dependent admission estimates at most 32 MiB
of intermediate working payload, including formatting, overlap/contributor history,
section capture amplification, and normalization replacement temporaries. This is
NOT a process-memory cap. macOS accepted RLIMIT_CPU but rejected RLIMIT_AS. Real
child-lifetime wait4 measurements were: dense tokens, estimate 33,483,328 bytes / RSS
118,259,712 bytes; repeated formatting, estimate 32,468,700 / RSS 480,313,344; repeated
overlap, estimate 30,641,728 / RSS 91,701,248. Seven admitted stress cases were
qualified; Windows is explicitly unqualified and Linux was not run on this host.
No security sandbox, encryption, secure-erasure, network, or vendor-engine change.

Final targeted runner/coordinator/recovery/autosave/preflight/execution selection:
155 passed, 2 existing warnings in 19.64s. Scoped Ruff, formatter, and diff checks
pass. Tests include a fresh mounted Textual first launch with stderr fd -1, a real
SIGTERM-ignoring child reaped before replacement/Clear/close, catalog/source changes
during A, failed A followed by successful B, malformed imports, canceled restore
awaits, failed close Retry, stale replies, and real SQLite round-trip of a real
worker result. Details and exact RED/GREEN commands are recorded in
`.superpowers/sdd/2026-09-04-chunking-lab/task-5-report.md`.

Changed runner/coordinator, narrowly extended Lab preflight, added their targeted
tests, and recorded the proven early-RSS-measurement trap in testing lessons.
Status remains In Progress pending the controller's independent review.

Review fix round 1: quiescence now waits for a dedicated Run completion signal
resolved after Run cleanup, rather than retaining the caller's entire task.
Clear/restore/close finish while a Run caller continues unrelated work; a caller
may also join an already pending close without forming a circular wait. Close
rechecks its completed state after joining another transition. Four regressions
failed before the fix; the covering coordinator file now passes 22 tests with one
known Requests warning in 4.73s. Scoped Ruff/format/diff checks pass. No process,
storage, resource-policy, or public-interface changes; ADR-118 still applies.
