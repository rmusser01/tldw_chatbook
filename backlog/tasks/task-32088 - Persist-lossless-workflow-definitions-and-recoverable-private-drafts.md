---
id: TASK-32088
title: Persist lossless workflow definitions and recoverable private drafts
status: To Do
assignee: []
created_date: '2026-09-08 20:54'
labels:
  - workflows
  - local-runtime
dependencies:
  - TASK-32077
references:
  - Docs/superpowers/plans/2026-09-08-workflows-local-file-to-note.md
documentation:
  - Docs/superpowers/specs/2026-09-08-workflows-local-first-parity-design.md
  - backlog/decisions/138-portable-workflow-definitions-and-local-execution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users retain portable workflow definitions, immutable revisions, and invalid advanced-editor drafts without data loss. This is the document foundation of the approved local file-to-note milestone, not file exchange or server synchronization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Unknown definition, metadata, step, and configuration fields survive read-edit-save; stable identities and immutable revision ancestry are retained.
- [ ] #2 Raw invalid draft text and its last valid projection survive reopen; stale generations and conflicting revision saves cannot overwrite newer data.
- [ ] #3 Workflow SQLite storage follows the private-owner policy, versioned migrations, and declared backup coverage; tests use real isolated SQLite.
<!-- AC:END -->
