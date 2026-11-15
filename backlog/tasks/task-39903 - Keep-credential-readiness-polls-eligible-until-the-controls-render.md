---
id: TASK-39903
title: Keep credential readiness polls eligible until the controls render
status: Done
created_date: 2026-11-09 05:42
labels:
- ci
- console
- readiness
priority: high
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/UI/test_console_credential_poll_gate.py
- Tests/UI/test_console_switcher_local_probe.py
updated_date: 2026-11-09 21:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console credential poll can acknowledge a readiness change while its control bar defers the render. Later polls then skip the same evidence, so the header and send hint can stay stale after config contention.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A real credential poll leaves the previous snapshot and gate in place when a changed control render defers, then renders the same evidence on a later tick and returns to skipping idle ticks.
- [x] #2 The original held-config UI18 assertion and whole unchanged UI18 roster pass without raising runtime limits, with fresh original software UAT.
- [x] #3 Independent review and the published repair’s own hosted gates verify the credential poll fix.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Restore the documented existing poll acknowledgment contract without changing config admission, ownership, durable state, or worker budgets.

1. Preserve the hosted held-lock failure. Pin a changed readiness snapshot whose control render declines once, and verify the old poll acknowledges it without rendering.
2. Move the existing acknowledgment behind the successful control render or the unchanged-snapshot branch. Keep all existing locks, retries, source checks, and idle gates.
3. Run the focused poll contract and the unchanged UI18 shard with its original limits. Run the original browser and mounted software UAT for the production change.
4. Bind the local evidence and independent review to the published commit. Require that commit's own hosted gates before closing this task with the other in-scope tasks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The credential poll keeps the previous acknowledgment when the changed control bar returns False. The next normal tick can retry the same evidence version. An unchanged snapshot or a successful render still closes the idle gate. The regression showed the old poll changed the gate from evidence version 0 to 1 while keeping the previous snapshot. The four focused poll checks passed after the fix.

All 204 original UI18 cases passed with the 180-second case limit, including the held-config refusal and subsequent render. The official diagnostic inventory passed, along with all 15 original browser cases and the original mounted journey. This task remains In Progress until the combined commit passes independent review and its own hosted gates. No ADR is required because the fix restores the existing poll contract.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
