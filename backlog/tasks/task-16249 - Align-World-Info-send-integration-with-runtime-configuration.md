---
id: TASK-16249
title: Align World Info send integration with runtime configuration
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 12:38'
updated_date: '2026-08-14 12:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep World Info provider-branch integration coverage isolated from the live agent/provider runtime by using the supported pre-construction Console configuration seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both World Info send tests use the supported runtime configuration before mounting.
- [x] #2 No test attempts network egress and the capturing gateway receives the provider payload.
- [x] #3 The full World Info integration module and checkpoint chunk 52 pass.
- [x] #4 Static and task hygiene checks are complete.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a test-harness correction using an existing configuration boundary.

1. Preserve the provider-capture and network-tripwire failures as RED evidence.
2. Configure agent runtime off before the Console controller is built and remove the ineffective private mutation.
3. Run the exact failing nodes, full module, and checkpoint chunk 52.
4. Run scoped static checks, self-review, document evidence, and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Configured `console.agent_runtime` off before mounting both World Info provider-branch harnesses, matching the already-correct dictionary integration tests and ensuring controller construction selects the provider path.
- Removed the ineffective post-construction `_agent_runtime_enabled` writes; the capturing gateway now receives the model-bound payload and the global network tripwire records no socket attempt.
- RED evidence was two capture failures plus two teardown egress failures to the unstubbed local provider. The full World Info module passed 3 tests, and exact checkpoint chunk 52 passed 264 tests in 4m20s.
- Scoped Ruff lint and `git diff --check` passed. The file retains only its exact HEAD formatter drift; the changed lines introduce no new format delta.
- ADR check: no ADR was required because this changes only test setup at an existing public configuration boundary.
<!-- SECTION:NOTES:END -->
