---
id: TASK-37654
title: Console composer restyle acceptance test syncs the action row twice intermittently
status: To Do
created_date: 2026-10-30 08:10
labels:
- console
- flaky-test
priority: medium
references:
- backlog/tasks/task-33620.15.8 - Console-after-acceptance-the-composers-class-sync-and-the-durable-turn-publication-hold-run-starts-longest-UI-loop-callback.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_composer_state_restyle.py::test_the_acceptance_projection_clears_now_and_restyles_send_after fails intermittently on dev with 'The projection should sync the action row exactly once, after its step, from the cleared draft: ['', '']'. It failed in PR #3286's UI Fast Lane (6) on 2026-10-30 and in 2 of 3 local runs on dev ff815f8559, so it is not caused by that PR. A flaky required-lane test evicts queued PRs, so either the projection double-syncs in a real race or the test's timing assumption is wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the double action-row sync is identified as a product race or a test timing assumption, with evidence.
- [ ] #2 The test passes 20 consecutive local runs on dev and does not weaken the exactly-once assertion unless the product contract changes.
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
