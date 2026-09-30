---
id: TASK-33264
title: 'PERF-05: Screen leaks - Settings signal subscription, Personas worker pin,
  Home recompose pin'
status: To Do
created_date: 2026-09-28 18:02
labels:
- performance
- memory
- ui
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every Settings visit retains the whole SettingsScreen, about +71k objects and +10.7 MB per visit. theme_changed_signal is subscribed in on_mount and never dropped, and Textual's Signal keeps subscribers in a WeakKeyDictionary whose bound-method values pin their keys. Departed Personas screens (6 of 10) are pinned by the thread-worker active_worker ContextVar. The reused Home screen whole-screen recomposes on every visit and its query_one cache pins the discarded trees. These leaks inflate every gen-2 GC pause. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-05; every issue with file:line is listed under PERF-05 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After 10 Settings visits and 10 Personas visits, no departed screen instance remains reachable
- [ ] #2 Home visits no longer whole-screen recompose when the targeted triage sync suffices
- [ ] #3 A regression test fails if a screen subscribes to an app-level Signal without unsubscribing on unmount
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
