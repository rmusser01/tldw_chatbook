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
- [ ] #1 Run both validates and durably captures the sample, both recipes, runtime identities, and candidate IDs before A starts; edits and catalog changes cannot alter queued B.
- [ ] #2 Only one bounded local process runs at a time; cancellation, navigation, restore, and Clear stop current work and pending batch members before a new run starts, and late replies cannot publish into newer epochs.
- [ ] #3 2 MiB sample, 10000 chunk, 32 MiB result, and 60 second preview limits have visible outcomes, preserve previous outputs, and never silently clip data; intermediate resource behavior is verified.
- [ ] #4 Failed A may be followed by B, cancel stops the queue, and old outputs remain explicitly previous rather than filling a failed batch member; all terminal outcomes retain backend and input provenance.
- [ ] #5 Restore and Clear cannot publish replacement authority until the old worker is terminated and reaped and its queue is stopped; failed replacement preserves the original in-memory session and writer retry authority.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: implements the approved bounded local runtime and immutable batch authority. 1. Read Task 5 brief/context and current state/writer/recovery contracts. 2. Write failing process timeout/cancel/limit and coordinator ordering tests. 3. Implement one bounded fresh child and immutable A-before-B lifecycle with manifest commit before launch and honest terminal states. 4. Verify intermediate resource behavior, fresh Textual-compatible first launch, off-loop work, late-result fencing and stop-before-restore/Clear. 5. Run targeted runner/coordinator/recovery/autosave checks plus static checks, self-review, independent review and evidence notes.
<!-- SECTION:PLAN:END -->
