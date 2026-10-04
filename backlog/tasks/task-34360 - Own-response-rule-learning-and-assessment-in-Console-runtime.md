---
id: TASK-34360
title: Own response rule learning and assessment in Console runtime
status: To Do
assignee:
  - '@codex'
created_date: '2026-10-04 05:55'
labels: []
dependencies:
  - TASK-34359
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/219-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 7. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Eligible settled responses are checked before completion release and next queue drain.
- [ ] #2 Stop, foreground input and source changes fence late activation or correction.
- [ ] #3 App-owned state survives view remount and public editor testing never autoactivates changes.
<!-- AC:END -->
