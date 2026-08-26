---
id: TASK-16255
title: Preserve prompt-row focus after discard
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 15:01'
updated_date: '2026-08-14 15:01'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep keyboard focus on the current Prompt row when Discard exits the compatibility editor, despite the asynchronous browse refresh completing afterward.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Discard browse refresh carries the current Prompt row as its focus identity.
- [x] #2 Late browse completion cannot replace the returned row focus with the Sort control.
- [x] #3 Prompt compatibility and broader prompt-canvas tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a focused UI race fix within the existing Library prompt focus contract.

1. Preserve the deterministic RED showing Discard returns to the list but focus settles on Sort or nowhere.
2. Capture the current prompt-row identity before editor reset and pass it through the browse refresh.
3. Assert the exact browse focus handoff and final live row focus.
4. Run the focused node, prompt-canvas module, lint, formatting characterization, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Captured the selected Prompt id before resetting editor state and carried its row identity through the asynchronous browse refresh. The compatibility regression now asserts that exact handoff and the live row regains focus after the refresh. The focused regression and both prompt failures observed under the loaded module run pass together; Ruff and diff hygiene pass. Both touched files are already formatter-red at HEAD, so no unrelated formatting churn was introduced.
<!-- SECTION:NOTES:END -->
