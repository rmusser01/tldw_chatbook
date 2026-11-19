---
id: TASK-41964
title: "About 80 Console UI nodes still fail behind the raw_source_selection_changed trap: resume active path and approval rounds"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_resume_active_path.py
- Tests/UI
dependencies:
- TASK-41633
- TASK-41637
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** The TASK-38111/41313 config-sandbox trap still hides two groups: 22 nodes of `Tests/UI/test_console_resume_active_path.py` (`RecoveryRequired: raw_source_selection_changed`; one more fails an assertion at ~line 2500), and about 60 non-`bootstrap_profile` approval-round UI nodes seen during TASK-41637.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each file red only behind the trap passes with the `bootstrap_profile` marker
- [ ] #2 Nodes that stay red after the marker are filed or fixed with their causes
<!-- AC:END -->
