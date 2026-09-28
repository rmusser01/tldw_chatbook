---
id: TASK-33272
title: 'PERF-13: Console idle, tick and streaming render'
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
At idle the Console wakes about 45 times per second. The 4 Hz credential/readiness poll, duplicated on 5 surfaces, rebuilds readiness every tick. The 0.2 s tick re-runs context_control_inputs (durable snapshots, branch-memory SQL, a recovery config scope) and character-context fingerprint SQL. Message-row class sync removes and re-adds managed classes every sync, restyling the whole Markdown subtree each streaming tick. left_rail.sync_model_recovery forces a screen relayout per tick. The conversations tray rebuilds for tooltip-only age deltas, the cost chip re-captures turn config at 5 Hz, and transcript actions re-parse replies with a fresh MarkdownIt 2-3 times per pass. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-13; every issue with file:line is listed under PERF-13 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Credential/readiness updates are event-driven; no surface polls readiness at 4 Hz while nothing changes
- [ ] #2 Tick-level derivations are memoized on transcript/config revision and do no DB work when nothing changed
- [ ] #3 A streaming tick does not restyle unchanged message rows (restyle census)
- [ ] #4 Idle Console wakeups fall below 10 per second on the idle probe
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
