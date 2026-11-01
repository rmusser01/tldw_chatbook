---
id: TASK-32093
title: Build the recoverable three-pane Workflows editor
status: To Do
assignee: []
created_date: '2026-09-08 21:08'
labels:
  - workflows
  - ui
dependencies:
  - TASK-32088
  - TASK-32089
references:
  - Docs/superpowers/plans/2026-09-08-workflows-local-file-to-note.md
documentation:
  - Docs/superpowers/specs/2026-09-08-workflows-local-first-parity-design.md
  - backlog/decisions/138-portable-workflow-definitions-and-local-execution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the Workflows shell with the approved workflow library, step navigator, linear overview and focused continuous collapsible form, preserving unfinished edits across navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can author and save the supported workflow fields with stable step identities, typed references and explicit structural versus execution readiness.
- [ ] #2 Invalid raw JSON, failed flushes, historical inspection and revision switching never silently overwrite or discard another draft.
- [ ] #3 The real Textual screen supports keyboard-only use at wide, ordinary and 60x20 sizes; hidden panes leave focus traversal, validation preserves typing focus, and visible actions work.
<!-- AC:END -->
