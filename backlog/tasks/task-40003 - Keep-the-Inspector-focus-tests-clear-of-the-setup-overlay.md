---
id: TASK-40003
title: Keep the Inspector focus tests clear of the setup overlay
status: In Progress
created_date: 2026-11-09 07:22
labels:
- ci
- console
- testing
priority: high
references:
- Tests/UI/test_console_edge_focus_restyle_scope.py
- Tests/UI/test_console_native_chat_flow.py
updated_date: 2026-11-09 07:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Inspector focus tests hide the setup overlay while the Console still requires setup. A later real refresh can put the overlay back in front of the rail before Textual delivers the original click.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce the original click failure with a late real refresh and verify the compositor targets the blocking setup card.
- [x] #2 Both original focus tests retain their real clicks and style assertions and pass with the existing send-ready fixture and original limits.
- [ ] #3 Independent review and the published repair’s own hosted gates verify the fixture correction.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: The failure is in the mounted test setup. It does not require changing production readiness, compositor behavior, or the focus contract.

1. Retain the exact hosted click failure and the original local controlled late-refresh reproduction through the compositor.
2. Use the existing send-ready Console fixture in the two focus tests so production keeps the setup card hidden. Keep both original clicks, test assertions, dimensions, and time limits.
3. Repeat the controlled late refresh with the repaired fixture and run both original tests. Verify production and the original UAT input scope are unchanged.
4. Bind independent review to the exact repair commit and require its own hosted gates before closing this task with the other in-scope records.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
A controlled late refresh restored the genuine blocking setup modal before the original click was delivered. At the same click point, the original fixture hit the modal and returned False. With the standard send-ready fixture, the production refresh kept the modal hidden and the click hit the Inspector open control. The hosted log recorded the failed click without its hit target.

Both original focus tests passed locally with the 180-second case cap. Their clicks, viewport and assertions are unchanged. The test now uses actual ready setup state, and the production and workflow inputs remain byte-identical. The original browser and mounted UAT carry through the unchanged runtime and unrelated test-only delta. This task remains In Progress until independent review and the repair commit’s own hosted gates pass. No ADR is required for the test setup change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
