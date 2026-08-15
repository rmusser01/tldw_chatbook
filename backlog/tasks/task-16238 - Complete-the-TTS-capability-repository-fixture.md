---
id: TASK-16238
title: Complete the TTS capability repository fixture
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 09:56'
updated_date: '2026-08-14 09:57'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore TTS profile-capability tests by keeping their fail-loud repository collaborator structurally aligned with the current profile service contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The capability fixture satisfies the complete profile repository protocol.
- [x] #2 Unused persistence methods remain fail-loud.
- [x] #3 The focused capability module and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Identify the protocol members missing from the shared availability fixture.
2. Add fail-loud implementations for the unused clone-reference persistence seams.
3. Run the three regressions, full capability module, and scoped static checks.
4. Record verification and close the task.

ADR required: no
ADR path: N/A
Reason: This is a test-double conformance repair for an existing TTS repository contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the shared availability repository double with fail-loud create_profile_with_reference and get_reference seams required by the current runtime-checkable profile repository protocol. The full capability module passed 43 tests; Ruff check/format and git diff --check passed. ADR required: no (test-double conformance only).
<!-- SECTION:NOTES:END -->
