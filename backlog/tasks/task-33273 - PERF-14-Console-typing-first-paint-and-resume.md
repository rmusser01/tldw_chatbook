---
id: TASK-33273
title: 'PERF-14: Console typing, first paint and resume'
status: To Do
created_date: 2026-09-28 18:02
dependencies:
- TASK-33265
labels:
- performance
- console
- ui
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each typing pause fires a 250-415 ms loop stall: the debounced spend refresh rebuilds the settings summary and inspector state outside a derivation scope. Every keystroke still forces a whole-screen layout through three removable triggers (TASK-21120). Console character search recomposes twice per keystroke with 5 SQLite round trips and no debounce. First paint constructs AgentRunsDB and the agent bridge on the loop, runs launch-wake discovery on the loop, and builds all Canvas machinery unused. Mount and resume await LocalSkillsService.get_context in a coroutine worker. After _ui_ready there are 5-12 s of 30-75% loop load. Warm resume restyles all 533 nodes. TASK-24452's re-mint claim is stale: Console is a reused route. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-14; every issue with file:line is listed under PERF-14 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typing pauses cause no loop stall above 50 ms on the typing probe
- [ ] #2 A keystroke does not trigger a whole-screen layout
- [ ] #3 Character search is debounced and recomposes at most once per settled query
- [ ] #4 First Console paint constructs no agent-runtime DB, launch-wake connection or Canvas service on the event loop
- [ ] #5 TASK-24452 is updated to reflect route reuse
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
