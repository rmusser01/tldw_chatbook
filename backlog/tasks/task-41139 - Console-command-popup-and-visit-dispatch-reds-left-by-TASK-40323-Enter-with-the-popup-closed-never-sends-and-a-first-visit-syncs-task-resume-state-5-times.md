---
id: TASK-41139
title: "Console command popup and visit dispatch reds left by TASK-40323: Enter with the popup closed never sends, and a first visit syncs task-resume state 5 times"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- console
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_console_command_popup.py
- Tests/UI/test_console_visit_dispatch_dedupe.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40323 laned the green nodes of these files and left these reds, which may be product defects: a send that never happens, and redundant work on every Console visit.

**Evidence (TASK-40323's notes, base 26b2e677dd, files under `bootstrap_profile`).**
- `test_console_command_popup.py::test_enter_with_popup_closed_sends_normally`: "Expected mock to have been awaited"; the Send press never reaches `submit_draft`. Red with the stand-in factory too.
- `test_console_visit_dispatch_dedupe.py::test_console_first_visit_refreshes_skills_and_task_resume_once`: "Console synced task-resume state 5x for one visit".
- `test_console_visit_dispatch_dedupe.py::test_console_later_resume_still_refreshes`: "Returning to Console must re-sync task-resume state".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each node is classified as test drift or a product defect, with the commit that moved it named for drift
- [ ] #2 A product defect is fixed with the node red before the fix; drift is re-pinned
- [ ] #3 The nodes run on a PR lane once green
<!-- AC:END -->
