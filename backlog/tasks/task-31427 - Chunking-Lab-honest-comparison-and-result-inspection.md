---
id: TASK-31427
title: Chunking Lab - honest comparison and result inspection
status: To Do
assignee: []
created_date: '2026-09-04 23:13'
labels:
  - chunking
  - chunking-lab
dependencies: [TASK-31421, TASK-31422]
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users inspect what changed between captured results without implying unsupported alignment, comparable units, or retrieval quality. Covers spec section 7 and AC 3, 9-10, 14-15, 19, 22, 25. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: result interpretation and comparison contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Comparisons require matching sample, backend, engine, and execution versions but allow different methods and tokenizers; mismatches explain why and offer rerunning both.
- [ ] #2 Common character and chunk measurements, explicit method budgets, and named token-count identities avoid incompatible deltas; elapsed time is labeled an observation and no quality score is invented.
- [ ] #3 Configuration diffs show selected immutable result snapshots including ordered operations, captured defaults, classifier and metadata view, runtime differences, and newer-draft staleness.
- [ ] #4 Chunk inspection is bounded and keyboard usable at 10000 chunks; linked source highlights and overlap measurements use verified spans only, with transformed-text inspection or unavailable explanations otherwise.
<!-- AC:END -->
