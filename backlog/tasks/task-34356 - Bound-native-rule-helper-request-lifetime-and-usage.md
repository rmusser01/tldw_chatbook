---
id: TASK-34356
title: Bound native rule helper request lifetime and usage
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-04 05:54'
labels: []
dependencies:
  - TASK-34355
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/219-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 3. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Four app-wide and one per-Chat helper slots remain held through physical provider settlement.
- [ ] #2 Native helpers obey deadlines and zero transport retries without changing existing auxiliary behavior.
- [ ] #3 Late cancelled output is rejected and original-owner usage is charged once.
<!-- AC:END -->
