---
id: TASK-31426
title: Chunking Lab - conflict-safe local template saving
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
Save Lab recipes through the existing canonical service with truthful validation, builtin protection, and atomic conflict detection. Covers spec section 9 and AC 3-4, 7, 16, 18, 20. Reuses the save semantics requested in existing TASK-24404 without adding a Settings form. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: optimistic concurrency and Lab save service contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Lab create and update validate the final full flat body and record fields with parity validation plus the Lab capability gate, preserving advanced data and reporting structured errors without requiring a preview.
- [ ] #2 Updates compare ID, UUID, version, builtin protection, and live state in the same transaction; intervening changes or deletion preserve the draft and offer Reload or Save as new without overwrite.
- [ ] #3 Builtins default to Save as new, reserved auto names are refused, concurrent creates respect live-name uniqueness, and stored-invalid rows remain visible and repairable.
- [ ] #4 Save A persists its pinned recipe and Save B its current valid draft; neither save changes Library content or defaults, and successful changes can trigger ingest-picker refresh.
<!-- AC:END -->
