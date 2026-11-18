---
id: TASK-41653
title: "Tests/ChaChaNotesDB historical bootstrap cases 82..86 are red and run on no CI lane"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- tests
- database
priority: medium
references:
- Tests/ChaChaNotesDB/test_historical_bootstrap.py
- Tests/ChaChaNotesDB/historical_bootstrap.py
dependencies:
- TASK-41310
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41310 A (schema v86→v87) found the historical bootstrap cases for versions 82..86 red: the bootstrapped schemas have tables the historical declarations do not list. No CI lane runs the file, so the failures went unnoticed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The historical bootstrap cases 82..86 pass
- [ ] #2 The file runs on a CI lane
<!-- AC:END -->
