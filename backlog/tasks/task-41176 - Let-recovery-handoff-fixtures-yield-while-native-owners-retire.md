---
id: TASK-41176
title: Let recovery handoff fixtures yield while native owners retire
status: In Progress
created_date: 2026-11-15 23:07
assignee:
- '@codex'
parent_task_id: TASK-41117
updated_date: 2026-11-15 23:16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The shared startup handoff fixture blocks its own event loop while waiting for native owner retirement. Investigate the one-second drain failure observed in hosted dev job 132963900278, preserve the existing deadline and authority checks, and qualify the corrected fixture on the local CI migration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Event-loop retirement can complete while the fixture waits, retaining its one-second deadline, same-task authority and all later handoff assertions.
- [x] #2 A controlled real pending owner reproduces the old failure and passes after the fix; an owner that never retires still fails at the original bound.
- [ ] #3 Focused affected callers, static checks and independent review pass; actual local dev CI evidence is recorded before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only correction preserving the existing runtime and isolation boundary.
1. Trace the reported child assertion and reproduce it with a real pending lease that needs an event-loop turn.
2. Match the production monitor waiting strategy while preserving the original one-second bound and same-task authority.
3. Verify the positive and never-retiring negative controls, affected shared fixture callers, lint and independent review.
4. Record actual local dev-push evidence before completion; do not close the separate later-snapshot issue in TASK-41117.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reproduced the hosted dev child failure with a real acquired storage lease whose close was queued on the same loop. The original synchronous drain failed at 1.012s with the lease still held and produced the same secondary TTS cancellation noise. Added a retiring outcome to the existing native handoff test; it failed before the fix. The shared script now follows production monitor_app: immediate drain probes, yielding on the original task, and current-thread cache retirement. The original one-second budget remains enforced, including after a successful probe; all later authority, capture, cancellation and readmission assertions remain. No production code changed.

All 14 targeted startup/monitor, real run-log, accepted sync/MCP, Skills/Chatbooks and local embedding callers passed in 59.14s. The never-retiring real lease negative control still failed at 1.018s. Original flaky run-log node also passed before the change in 4.10s, so quiet success alone was not evidence. Ruff and formatting pass for the changed fixture. Evidence: /tmp/tldw-ci-drain-regression-red.log, /tmp/tldw-ci-drain-negative.log, /tmp/tldw-ci-backup-fixture-focused.log. Added the incident to lessons-testing-evidence.md. ADR not required: test-only correction preserves existing boundaries. Independent review and actual local dev CI remain pending; TASK-41117 retains its separate later-snapshot and loaded-repeat criteria.
Independent requesting-code-review review found no actionable issues in the rebased head plus final diff; same-task authority, strict one-second bound and shared substitution anchors were verified. The remaining optional RAG backfill caller also passed (1 test,5.11s), bringing affected caller coverage to15 passing tests. Actual local dev CI remains the open acceptance item.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
