---
id: TASK-19301
title: Sweep research-side task-18060 citations to 19300
status: To Do
assignee: []
created_date: '2026-08-21 02:30'
labels:
  - research
  - backlog-hygiene
dependencies:
  - TASK-19300
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-18060 was a duplicate ID (8th collision); the research-durability task
was renumbered to TASK-19300, but ~39 comment/docstring citations of
`task-18060` in `tldw_chatbook/Research_Interop/*`, `Tests/Research/*`, and
`tldw_chatbook/DB/AgentRuns_DB.py`'s research-adjacent lines still cite the
old ID, which now resolves to the Done inspector-rail task. They were left
in place deliberately: an agent is actively working that code on
`feat/durable-research-jobs`, and a sweep in the renumber PR would have
manufactured conflicts. Sweep them once that branch lands.

Caution: only research-side citations move. Inspector-side `task-18060`
citations (Console/UI, review-rail spec references in `AgentRuns_DB.py`
v10-v12 migration comments, `Docs/User_Guide/console/agent-runs-and-tools.md`)
are CORRECT and must not be touched. `task-15452`'s "18041-18060" is a
source line-number range, not a task reference. See the renumbering note in
the TASK-19300 file for the full classification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 No `task-18060` citation remains in Research_Interop, Tests/Research, or the durable-research plan doc; all renamed to task-19300.
- [ ] #2 Inspector-side task-18060 citations are untouched (spot-checked against the classification in TASK-19300's renumbering note).
<!-- AC:END -->
