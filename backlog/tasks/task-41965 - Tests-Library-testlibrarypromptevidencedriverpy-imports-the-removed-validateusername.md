---
id: TASK-41965
title: "Tests/Library/test_library_prompt_evidence_driver.py imports the removed `validate_username`"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- library
priority: low
references:
- Tests/Library/test_library_prompt_evidence_driver.py
- tldw_chatbook/Utils/input_validation.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** All 4 nodes are red on dev: the driver imports `validate_username` from `Utils/input_validation`, which no longer exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The driver uses the current validator (or the test is retargeted with the reason recorded) and the 4 nodes pass
<!-- AC:END -->
