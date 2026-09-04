---
id: TASK-31424
title: Chunking Lab - recovery export restore and undo
status: To Do
assignee: []
created_date: '2026-09-04 23:12'
labels:
  - chunking
  - chunking-lab
dependencies: [TASK-31423]
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
<!-- AC:END -->
