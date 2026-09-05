---
id: TASK-31424
title: Chunking Lab - recovery export restore and undo
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 23:12'
updated_date: '2026-09-05 00:55'
labels:
  - chunking
  - chunking-lab
dependencies:
  - TASK-31423
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users a usable recovery path when local persistence fails, including safe snapshot restoration and bounded undo of replacement. Covers spec section 8 and AC 13, 16, 24, 26. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: private-data transfer and transactional replacement policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Versioned bounded JSON export from in-memory state preserves exact samples, invalid JSON, pending controls, full result snapshots, and integrity references without requiring a writable recovery database.
- [ ] #2 Restore validates structure, sizes, versions, and reference digests without executing templates, reading embedded paths, or making network calls; malformed input leaves the active session intact.
- [ ] #3 Explicit replacement preserves the displaced checkpoint atomically, changes epoch only after commit, rejects old writers, and waits for canceled work to stop; failure keeps the old session and its retry authority.
- [ ] #4 Undo restore remains available across view-only autosaves until content changes; Clear removes all recovery and undo references and cannot be reversed by late writes.
- [ ] #5 Repeated edits and reruns retain only one application content-action undo plus reachable current, previous and undo-needed snapshots; native editor undo remains separate, view-only changes preserve undo, and exceeding active recovery bounds refuses the edit without losing the prior value.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: implements approved bounded private recovery transfer and atomic replace/undo. 1. Read Task 4 brief/context and current store/writer contracts. 2. Write failing export/import integrity and bounds tests plus atomic replacement/undo cases. 3. Implement structural recovery validation and writer-owned replace/undo with epoch changes only on commit. 4. Bound application content Undo and prune unreachable active snapshots while preserving current/previous evidence. 5. Run targeted recovery/storage/state/autosave tests and static checks, self-review and independent review, then record evidence.
<!-- SECTION:PLAN:END -->
