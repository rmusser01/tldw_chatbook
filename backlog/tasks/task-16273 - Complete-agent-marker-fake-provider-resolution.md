---
id: TASK-16273
title: Complete agent-marker fake provider resolution
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 20:56'
updated_date: '2026-08-14 21:01'
labels:
  - testing
  - console
  - agents
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep agent-marker E2E tests on the real bridge by supplying the complete provider-resolution fields the streaming adapter consumes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The scripted gateway supplies a model and both marker lifecycle scenarios complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the failed runs and inspect their persisted error outcome.
2. Add the required model field to the test-only resolution fake.
3. Run focused mutation evidence, the integration module, original checkpoint, and static verification.

ADR required: no
ADR path: N/A
Reason: test-double completion for an existing provider boundary; no production change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the required model identity to the integration test's scripted provider resolution, matching neighboring Console E2E fakes and allowing the real streaming adapter to run.
- Preserved RED evidence: both lifecycle scenarios produced empty failed assistants; the persisted run recorded `SimpleNamespace` missing `model`. Removing the field reproduces the provider-boundary failure.
- Verified the four affected nodes (4 passed), the owning modules (103 passed), and the original 25-file checkpoint (524 passed). Ruff lint and `git diff --check` pass; the file's pre-existing formatter drift is unchanged from `HEAD`.
- ADR required: no. This completes a test double for an existing provider contract.
<!-- SECTION:NOTES:END -->
