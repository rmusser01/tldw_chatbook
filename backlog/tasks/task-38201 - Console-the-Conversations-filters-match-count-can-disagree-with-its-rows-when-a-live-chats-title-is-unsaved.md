---
id: TASK-38201
title: >-
  Console: the Conversations filter's match count can disagree with its rows
  when a live chat's title is unsaved
status: To Do
assignee: []
created_date: '2026-10-30 06:05'
labels:
  - console
  - left-rail
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Impact.** Since TASK-33622.6 the 'Filter visible titles' count comes from the database's title-only search while the rows also include live, in-memory sessions. When a live chat's in-memory title differs from its stored title (a rename not yet persisted), the two can disagree: the stored title can be counted while the builder's title filter hides the row, which would bring back the 'N match. Showing 0 of N' status that task removed.

**Evidence.** Found by code reading in the TASK-33622.6 review, not reproduced: UI/Console_Modules/workspace.py _current_console_browser_rows merges native rows first, so the live session's title wins over the stored one, while the total is the persisted title_only count. Reproducing it needs a live session whose title differs from the stored conversation title at the moment a filter pass runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With a live chat whose in-memory title differs from its stored title, the filter's match count and the rows it shows follow one title (the 'Showing K of N' disclosure is true of the rows on screen)
- [ ] #2 A test that renames a live chat without persisting it, then filters by the old and the new title, asserts the count and the rows agree; it fails on the pre-fix code, or the task records evidence that the state cannot occur and closes without a code change
<!-- AC:END -->
