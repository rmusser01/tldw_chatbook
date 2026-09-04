---
id: TASK-31382
title: Name the asking sub-agent on the ask_user card and its transcript marker
status: To Do
assignee: []
created_date: '2026-09-04 19:28'
labels:
  - console
  - agents
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PRD A11 says the question card names the asking agent. M2 (PR #2379) ships the attribution as a kind only -- the card title says 'A sub-agent has N questions for you' and the marker says 'Questions from a sub-agent' -- because a run carries no display label on dev: CurrentRunActor exposes kind, run_id and parent_run_id, and the transcript's own sub-agent markers are generic ('A sub-agent edited 3 files'). A user running a fleet cannot tell WHICH child is asking, which matters exactly when two children are working in parallel. Needs a run-id to display-label mapping (the fleet coordinator or AgentRuns_DB is the natural owner), threaded into the question payload's asked_by and read by the card title and the marker header.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The question card title names the asking sub-agent by its display label when one exists, falling back to the current generic copy
- [ ] #2 The A14 transcript marker header carries the same label
- [ ] #3 A primary-agent question is unchanged
- [ ] #4 The label lookup never blocks the worker thread or raises into the round
<!-- AC:END -->
