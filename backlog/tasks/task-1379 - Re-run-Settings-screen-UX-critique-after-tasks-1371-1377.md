---
id: TASK-1379
title: Re-run Settings screen UX critique after tasks 1371-1377
status: To Do
assignee: []
created_date: '2026-08-06 00:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run the dual-agent Settings critique (static design review + sandboxed headless-TUI UAT) against the post-1371-1377 state to measure score movement from 30/40 and surface any remaining or newly-introduced issues. Persist the snapshot under .impeccable/critique/ and print the trend line (30 -> 29 -> 31 -> 24 -> 24 -> 30 -> ?). Cover the four personas (first-time/power x technical/non-technical) and re-check the remaining Questions to Consider (Scope Inspector purpose, mode strip vs manual sync placement, App Areas in primary nav).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Critique run against HEAD including tasks 1371-1377 fixes,Score and per-heuristic table persisted to .impeccable/critique/ with timestamp,Findings triaged: new issues filed as backlog tasks, no-regression confirmed on previously-fixed paths
<!-- AC:END -->
