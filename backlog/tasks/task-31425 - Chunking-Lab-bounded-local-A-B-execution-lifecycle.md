---
id: TASK-31425
title: Chunking Lab - bounded local A-B execution lifecycle
status: To Do
assignee: []
created_date: '2026-09-04 23:13'
labels:
  - chunking
  - chunking-lab
dependencies: [TASK-31421, TASK-31422, TASK-31423, TASK-31424]
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
<!-- AC:END -->
