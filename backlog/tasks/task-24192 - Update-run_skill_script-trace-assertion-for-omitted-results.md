---
id: TASK-24192
title: Update run_skill_script trace assertion for omitted results
status: Done
assignee: []
created_date: '2026-08-29 03:58'
updated_date: '2026-08-29 04:50'
labels: []
dependencies: []
references:
  - >-
    backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the full-suite gate after the Trace v2 privacy contract intentionally omitted run_skill_script payloads from durable agent-step rows while older runtime-tool assertions continued to require plaintext refusal and script output in those fields.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unwired run_skill_script path is still classified as a blocked tool outcome.
- [x] #2 The durable agent-step result remains omitted instead of persisting sensitive script output.
- [x] #3 The model-visible tool result still contains the permission refusal needed for loop continuation.
- [x] #4 Focused agent runtime and Notes verification pass.
- [x] #5 Successful sub-agent script dispatch remains observable to the model while its durable payload is classified as successful and omitted.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md. Reason: this test-only correction aligns an older assertion with the existing Trace v2 privacy contract and changes no production behavior or architecture. 1. Preserve the reproducible failing assertion as RED evidence. 2. Replace the stale plaintext durable-row assertion with consumer-visible refusal plus structured blocked/omitted assertions. 3. Run the focused runtime-tool module and Notes verification gates. 4. Review the diff and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the two stale run_skill_script service-level assertions to verify consumer-visible tool results separately from privacy-safe durable AgentStep projection. The unwired path now asserts a model-visible permission refusal plus blocked/omitted structured state; the successful sub-agent path asserts model-visible stdout plus success/omitted structured state. No production code or storage contract changed. Verification: the runtime-tool module passed 10 tests; the focused Notes regression batch passed 16 tests; Ruff lint/format and git diff checks passed. A fresh full-suite first-failure probe advanced through 1,765 passes and 8 skips before stopping at the separate test_real_closure_recovers_full_content_beyond_both_caps failure, proving the original blockers no longer stop the gate. The expanded Notes matrix was stopped after 1,025 passes and unrelated Settings/legacy Library hub-media baseline failures. ADR: existing ADR-080; no new ADR required. No reusable lesson was added because this is direct enforcement of the already documented privacy-owner testing lesson.
<!-- SECTION:NOTES:END -->
