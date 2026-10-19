---
id: TASK-35598
title: Console response-rules phase-event test times out in the PR gate about half
  the time
status: To Do
created_date: 2026-10-19 18:04
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_console_response_rules.py::test_rule_phase_events_project_only_the_mounted_current_session has timed out on 4 of ~7 PR-gate runs observed across three different PR heads on 2026-10-18/19 (variants [rebuild] twice and [maintenance] once; each time green on rerun). Seen on PR #3140's runs (39152595713, 39160862810, 39275849597, 39287157642) — never related to that PR's diff (zero overlap). It was added to the PR-gate census recently (TASK-35444). Under runner congestion it hits the private-profile timeout machinery; suspect a hang in phase-event projection during maintenance/rebuild phases that only manifests under load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Repro or hang-site identified under load,Test passes reliably in the PR gate or is quarantined with a tracked reason
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
