---
id: TASK-41952
title: "A primary agent run on an inactive branch may still get a phantom Trace turn"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- trace
- bug-suspected
priority: low
references:
- tldw_chatbook/UI/Console_Modules/review_selection.py
dependencies:
- TASK-41650
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** Not verified: TASK-41650 drops child calls whose turn is not shown, but a primary run whose assistant message maps it to a turn on an inactive branch (after Edit & resend) probably gets its own phantom turn, and its sub-agents inherit it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test builds an inactive-branch primary run and shows whether a phantom turn appears; if it does, it is dropped (red first)
<!-- AC:END -->
