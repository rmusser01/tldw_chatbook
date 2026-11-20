---
id: TASK-42135
title: "Restore Change Review's opt-in and privacy paragraphs to the Console user guide"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- docs
- change-review
priority: medium
references:
- Docs/User_Guide/console/agent-runs-and-tools.md
- tldw_chatbook/Workspaces/change_review_consent.py
- tldw_chatbook/Workspaces/change_retention.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the docs stream (round 18). Squash 6a6242ce3a (2026-09-12) removed the opening paragraphs of "Change review — reviewing a turn's file changes" in `Docs/User_Guide/console/agent-runs-and-tools.md` (the heading is now line 1046); 2e5d256a04 had added them. `git show 6a6242ce3a^:Docs/User_Guide/console/agent-runs-and-tools.md` has the text at lines 685-703. It said:
- Change Review is off for every new workspace until the user enables it in **Settings > Workspaces**; the global `[change_review] enabled` setting is only a capability switch;
- enabling prepares each bound folder in a bounded background queue (`preparing`, `ready` or `failed`, with a retry), and a turn that starts before a folder is ready continues with a warning;
- Change Review keeps shadow Git history, file contents included, in the application's data directory for 30 days by default (`[change_review] retention_days`), and disabling a workspace does not erase it.

No User Guide page states that retention of file contents now. Another session is changing Change Review code (Git LFS filters, TASK-42101, and `Workspaces/change_review_consent.py`), so each sentence must be checked against the code at the time this task is done, not at the filing base. `backlog/docs/lessons-backlog-hygiene.md` ("A 'page has no section for X' finding can be a section a squash deleted") describes the restore procedure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The "Change review" section again states the per-workspace opt-in, the preparation queue and its states, and the shadow Git retention of file contents with its default and its setting, each sentence checked against the code at the time of the change
- [ ] #2 A sentence of the old text that no longer matches the code is corrected rather than restored as it was, and the notes list each correction
- [ ] #3 The notes cite 6a6242ce3a as the source of the restored text
<!-- AC:END -->
