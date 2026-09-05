---
id: TASK-31422
title: Chunking Lab - lossless draft and candidate state
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 23:11'
updated_date: '2026-09-05 00:00'
labels:
  - chunking
  - chunking-lab
dependencies:
  - TASK-31421
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make controls, JSON, sample edits, and A/B snapshots share one recoverable authoring state without dropping advanced configuration or running stale valid data. Covers spec sections 4-5 and AC 3-5, 8-10, 18, 21, 23, 26. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: durable draft identity and editing authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Invalid raw JSON and incomplete control strings retain exact text and editing authority; switching views cannot discard pending edits or enable Run, Pin, or Save on an older valid document.
- [ ] #2 Known control edits patch only their documented paths in ADR-078 flat bodies; unknown metadata, classifier rules, advanced options, and ordered operations survive Controls/JSON/import/export round trips.
- [ ] #3 Stable candidate identities support editable B, deliberate pin or replacement of A from a current completed B result, correct staleness, and immutable captured run inputs; v1 rejects more than two candidates.
- [ ] #4 Sample and template replacements and pinning are undoable; every recovery-relevant edit increments revision while profile, epoch, and immutable identities prevent cross-session mutation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: implements the approved draft identity and editing authority boundary. 1. Read the Task 2 brief and existing execution values. 2. Write failing lossless authoring, candidate, snapshot and epoch tests. 3. Implement detached serializable state and pure transitions, retaining invalid raw text and last-valid state separately. 4. Run targeted tests, lint and changed-code formatting; self-review and independent review. 5. Record implementation notes and acceptance evidence.
<!-- SECTION:PLAN:END -->
