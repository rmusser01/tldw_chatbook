---
id: TASK-3021
title: Home-surface import vocabulary and first-click honesty audit
status: To Do
assignee: []
created_date: '2026-08-07 12:20'
labels:
  - home
  - ux-copy
  - uat-2026-08-06
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-2857 unified the Library's user-facing vocabulary on Import/Export, deliberately leaving
Home-surface strings out of scope. Recorded during that arc (positions at `6672ed276`):

1. `Home/active_work_adapter.py:410` — "Opening Library ingest job details."
2. `app.py:4403` — `HomeControlResult` message "This ingest job can no longer be retried."
   (rendered on the Home screen's Retry action)
3. `Docs/User_Guide/home.md` ("opens Study at flashcards", ~line 75) — Home's Study rows need the
   same first-click-honesty audit task-2854 applied to the Library rail (does the first click land
   on Study, or on a staging surface?). Verify live before rewording.

Scope: bring Home's user-facing strings in line with the Import/Export vocabulary where they name
the same concept, and make Home's Study glosses honest about their first-click destination.
"Chatbook"-as-app-name usages (File Notes panels) are a separate, larger naming decision — out of
scope here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Home-surface strings naming Library import jobs use the Import vocabulary
- [ ] #2 Home's Study row glosses/docs describe the actual first-click destination (verified live)
- [ ] #3 Changed strings inventoried in the task notes; affected user-guide pages updated or re-stamped
<!-- AC:END -->
