---
id: TASK-33281
title: 'PERF-22: Settings and Personas - reusable routes and targeted category/list
  updates'
status: To Do
created_date: 2026-09-28 18:03
dependencies:
- TASK-33264
- TASK-33265
labels:
- performance
- settings
- personas
- ui
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Settings and Personas routes are not reusable, so every visit constructs and styles a fresh screen: Settings 356-513 ms of loop time, Personas 0.7-1.0 s. The Settings category switch re-mints both panes (215-900 ms) and triggers a mount-echo Changed storm. Personas and Roleplay lists remount every row on each search, page and sort. The inspector conversation search runs a BEGIN IMMEDIATE per keystroke. Owner decision D4: make both routes reusable. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-22; every issue with file:line is listed under PERF-22 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Owner decision D4 is recorded
- [ ] #2 Warm Settings and Personas visits build no new screen instance
- [ ] #3 A Settings category switch mounts only the changed pane content, with no Changed echo storm
- [ ] #4 Personas/Roleplay lists update rows without remounting the full list, and inspector search is debounced
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-10-23 (Roleplay frame B2a, TASK-33910.3, final review): when this task makes the Roleplay (`personas`) route reusable, `PersonasScreen._end_visit` (B2a's leave hook) does not stop the library search's debounce timer; `on_unmount` does. So a leave during the 0.2 s debounce would start a window worker while Roleplay is suspended, against B2a AC#11's wording. It is dormant today: the route is not reusable, so every leave is an unmount. New with B2a, not fixed by it. B2a's suspend path (Task 7's rows-fence re-stamp, the reload on return) also runs only in tests until then; TASK-33910.34 carries the re-check.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
