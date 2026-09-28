---
id: TASK-33260
title: 'PERF-01: Perf guards that count admissions, helper spawns and pre-import payload'
status: To Do
created_date: 2026-09-28 18:01
labels:
- performance
- ci
- testing
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The two biggest regressions since the 09-04 perf review shipped with every perf guard green. Per-call ADR-126 storage admission and a private-SQLite helper spawn per connection went unseen. The keystroke census reported 0 work per key while each key ran 27-69 guarded load_settings calls. The Console mount profiler is broken by route reuse. The screen pre-import payload ratchet is red (554/500 modules) but perf-guard.yml never runs it. About 14 startup guards are masked by RecoveryRequired raised at module-scope APP_CONFIG. Every later PERF task needs a guard that sees its unit. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-01; every issue with file:line is listed under PERF-01 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The keystroke census and a new idle/visit census count config admissions, storage admissions, private-SQLite helper spawns and open() calls, and fail on today's dev behaviour
- [ ] #2 The screen pre-import payload guard runs in perf-guard.yml; it is either paid down or re-pinned at the measured value with a follow-up task named in the pin
- [ ] #3 run_console_mount_profile.py produces a profile against the reusable Console route
- [ ] #4 Startup/footer guards previously masked by RecoveryRequired('raw_source_selection_changed') run and report real results
- [ ] #5 The stale CSS-source meta-test matches the current source count
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
