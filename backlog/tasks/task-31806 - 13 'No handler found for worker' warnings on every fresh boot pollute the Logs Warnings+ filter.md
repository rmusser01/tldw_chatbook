---
id: TASK-31806
title: 13 'No handler found for worker' warnings on every fresh boot pollute the Logs Warnings+ filter
status: To Do
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - logs
  - boot
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). Every fresh boot emits 13 'No handler found for worker ...' WARNING lines, drowning the Warnings+ filter's signal. Either register handlers, downgrade to debug, or fix the worker wiring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A fresh boot emits zero spurious 'No handler found for worker' warnings.
<!-- AC:END -->
