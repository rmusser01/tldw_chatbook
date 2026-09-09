---
id: TASK-32095
title: Qualify the local file-to-note workflow milestone
status: To Do
assignee: []
created_date: '2026-09-08 21:12'
labels:
  - workflows
  - ui
  - runtime
dependencies:
  - TASK-32094
references:
  - Docs/superpowers/plans/2026-09-08-workflows-local-file-to-note.md
documentation:
  - Docs/superpowers/specs/2026-09-08-workflows-local-first-parity-design.md
  - backlog/decisions/138-portable-workflow-definitions-and-local-execution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove that the real redesigned Workflows screen completes a local file-to-model-to-human-review-to-note path without tldw_server or external internet, while documenting the supported subset honestly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A real preinstalled local model completes the production UI file-to-note path in an isolated profile with only its selected loopback endpoint permitted; a skipped live test is not milestone completion.
- [ ] #2 Human rejection, restart at a persisted wait, cancellation ownership, uncertain note receipts and historical result provenance have targeted evidence with real SQLite and controlled race tests.
- [ ] #3 Actual Textual screenshots and keyboard walkthroughs at wide, ordinary and 60x20 dimensions verify editing, validation, review and result accessibility.
- [ ] #4 User documentation and evidence distinguish the tested five operation subsets from the full 21-step v1 target, later exchange and paired-server v1 sync; all task acceptance and scoped quality checks pass.
<!-- AC:END -->
