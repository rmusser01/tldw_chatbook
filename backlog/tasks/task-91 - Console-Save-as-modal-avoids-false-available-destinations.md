---
id: TASK-91
title: Console Save as modal avoids false available destinations
status: Done
assignee:
- '@codex'
labels:
- console
- ux
- uat
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure selected-message Save as... does not present unavailable destinations as actionable, and renders any explicitly wired destination as a real control instead of static text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Save as modal marks unwired destinations as WIP with recovery copy.
- [x] #2 Available Save as destinations render as focusable, clickable controls.
- [x] #3 Default selected-message Save as destinations do not imply Chatbook export is wired when it is not.
- [x] #4 Focused regression tests cover destination state and modal activation behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the Console selected-message Save as honesty fix. `ConsoleMessageActionService` now treats Save as destination availability as opt-in, so default selected-message actions no longer claim Chatbook export is wired. `ConsoleSaveAsModal` now renders explicitly available destinations as real buttons and renders unavailable destinations as WIP rows with visible recovery copy. Added focused service and mounted modal regressions, then verified the Console message-action suites and captured an actual browser screenshot at `/private/tmp/console-save-as-fixed-modal.png`.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: This is a focused UI honesty/actionability bug fix. It does not change storage, sync, runtime boundaries, service contracts, security policy, or long-lived application architecture.

1. Add failing regressions proving default Save as destinations do not mark Chatbook as available when selected-message export is not wired.
2. Add a modal regression proving explicitly available destinations render as focusable/clickable controls and WIP destinations remain non-actionable with recovery copy.
3. Update the Console message action service to make Save as destination availability opt-in.
4. Update the Save as modal to render available destinations as buttons and unavailable destinations as explicit WIP rows.
5. Run focused Console message-action tests, diff checks, and a CDP screenshot pass against the running textual-web Console.

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Selected-message Save as no longer presents unavailable destinations as active. The modal is now explicit: no wired destinations by default, with Chatbook, Note, Media, and Prompt shown as WIP until their export paths are implemented.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
