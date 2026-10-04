---
id: TASK-34355
title: Persist device-local response rules and inert recovery
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-04 05:53'
labels: []
dependencies:
  - TASK-34354
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/219-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 2. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Saved Chat rules reopen locally and temporary Chat rules adopt atomically on save.
- [ ] #2 Scoped drafts remain inspectable after invalid or failed learning and binding edits use compare-and-swap.
- [ ] #3 Imported rules are inactive without changing preserved local owners or rollback snapshots.
<!-- AC:END -->
