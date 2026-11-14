---
id: TASK-39803
title: Preserve the async contract of the Console readiness test observer
status: Done
created_date: 2026-11-09 05:26
labels:
- ci
- console
- testing
priority: high
references:
- Tests/UI/test_console_send_ready_attachment.py
updated_date: 2026-11-09 21:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The mounted cold-readiness convergence test sometimes gave Textual a synchronous observer where the Console submits an asynchronous worker callback. CI could then fail before exercising the intended readiness behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce the hosted Textual rejection with the original observer and show that the repaired observer runs through the same worker while preserving immediate publication observation and argument forwarding.
- [x] #2 The unchanged UI2 selector roster passes locally within its existing limits, preserving all original convergence assertions and runtime behavior.
- [x] #3 Independent review and the published repair commit’s own hosted gates verify the fixture correction and the original UI2 roster.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This repairs an observer in a test fixture and leaves the production worker, ownership, and storage boundaries unchanged.

1. Retain the original hosted UI2 log and reproduce the callable rejection with Textual's worker and the exact observer from the fixture.
2. Preserve the observer's immediate caller detection and coroutine return, and expose that coroutine contract to Textual. Preserve the original assertions, selectors, and runtime limits.
3. Verify the same worker route locally, run the original UI2 roster, and bind the exact test-only change to independently reviewed source. Reuse the original UAT only through verified product and UAT input identity.
4. Retain the successor's own hosted gates and close this task with TASK-39703 in the narrow closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Textual’s async worker accepts the observer through `inspect.markcoroutinefunction`. The observer still records the caller when `_refresh` requests publication and returns the real coroutine with the same arguments. The controlled old worker rejected the original function, and the corrected function delegated successfully. Every original test assertion and the observer body remained intact.

All 158 original UI2 cases passed in the 66-selector local shard with the 180-second case limit. That run preceded the separate credential-poll correction. This task remains In Progress until the combined commit passes independent review and its own hosted gates. No ADR is required for the fixture change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
