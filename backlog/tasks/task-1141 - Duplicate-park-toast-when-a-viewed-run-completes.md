---
id: TASK-1141
title: 'Duplicate park toast when a viewed run completes'
status: To Do
assignee: []
created_date: '2026-07-27 18:05'
labels: [console, approvals, uat]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT (Docs/superpowers/qa/parallel-agents-uat-2026-07-27, F2): with session B parked (park toast already shown), completing a run in the VIEWED session A re-fires "Agent in B (workspace) needs approval." for B's unchanged round. The once-per-card toast guard does not survive the re-marshal/re-park performed by the viewed-run-completion sync. Repro: park B; run and complete a run in viewed A; second toast fires at A's completion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A parked round toasts exactly once across its lifetime, including viewed-run completions, visits, and re-derives.
- [ ] #2 Regression test reproducing the viewed-completion re-toast path.
<!-- AC:END -->
