---
id: TASK-40724
title: Bound self-hosted regression lanes without dropping coverage
status: In Progress
created_date: 2026-11-13 19:39
assignee:
- '@codex'
updated_date: 2026-11-13 19:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first sustained local gate reaches the Console P0 and Dreams job time limits before completing their existing tests. Keep the required coverage while allowing the measured workloads to finish within bounded jobs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console P0 executes each existing sandboxed and admission-sensitive target exactly once across required matrix jobs.
- [x] #2 Console P0 and Dreams retain per-test deadlines and required verdicts with bounded 35-minute job budgets below the controller deadline.
- [ ] #3 Focused workflow contracts pass and a new normal PR gate verifies the revised execution under the local worker pool.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Rebalance existing required test invocations inside the accepted disposable runner boundary.

1. Preserve cancelled gate43487945840 evidence and compare executed steps with the previous hosted run.
2. Split Console P0's two unchanged pytest invocations using a fail-fast-disabled matrix; bound it and Dreams at35minutes, preserving test deadlines and required aggregation.
3. Update and run focused workflow contracts, independently review, then start the normal PR gate after the coordinated idle benchmark completes. Record full-run timing separately from failed attempts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Split Console P0's unchanged sandboxed and admission-sensitive pytest commands into two matrix profiles with fail-fast disabled. Each profile and Dreams/Guardian now uses35minutes, below the controller45-minute deadline. Required aggregation, runner routing, command arguments, per-test deadlines and isolation are unchanged. Updated exact workflow contracts, profile-selection/count checks, ADR-103 current capacity statements, local-runner README and the incident lesson. Existing ADR-359 applies; no new boundary.

Evidence: gate43487945840 cancelled after37m49s following two annotated20-minute job timeouts; not a successful full timing. Its final run/jobs/summary files are retained in /tmp/tldw-ci-third-attempt-final-*.json. New profile contract failed before the workflow edit. Focused run passed833tests under --noconftest; its sole harness-dependent storage-lease test passed separately with normal conftest. Ruff and small-file format checks passed; parsed comparison confirms every job shell command and runner expression unchanged. Independent review found no blocking issues; stale combined-order documentation corrected. AC3 remains pending the new normal PR gate after the coordinated benchmark releases the host.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
