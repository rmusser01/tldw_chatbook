---
id: TASK-31383
title: Make failed Console replies offer Retry instead of Continue
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 19:29'
updated_date: '2026-09-04 19:44'
labels:
  - console
  - messages
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent failed provider responses from presenting Continue as the recovery action, so users can retry the original turn without the empty composer guidance implying that new message content is required.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Failed assistant messages expose Retry and do not expose Continue
- [x] #2 Completed assistant messages retain Continue
- [x] #3 Retry still reuses the failed response path and no composer content is required
- [x] #4 Focused action-service and mounted Console regressions pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md
Reason: This narrows when an existing message action is applicable and preserves the established Continue, Retry, storage, and provider/runtime contracts.

1. Add a focused action-service regression asserting failed assistant rows expose Retry without Continue while completed rows retain Continue.
2. Run the regression before production edits and confirm it fails on the unexpected Continue action.
3. Remove Continue from the failed-assistant action list at the shared action-catalog seam.
4. Run the focused action-service and mounted Console retry/continue regressions.
5. Run targeted Ruff and diff checks, self-review the change, and complete TASK-31383 with evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the Console message-action catalog so failed assistant replies expose Retry without Continue, while completed, stopped, and failed user-authored messages retain their existing Continue behavior. Added a dispatch-boundary guard so stale Continue events cannot start a new assistant turn from a failed reply. Expanded service and mounted Console coverage, including retry recovery without composer content. Verification: focused action-service and mounted retry tests passed; the mounted completed-Continue regression passed separately; targeted Ruff and git diff checks passed. Independent review found and verified the dispatch guard and assistant-only scope corrections. ADR check: no new ADR required; this preserves the contracts in backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md. No new general lesson was needed because the mounted-button interaction follows the existing live-verification lesson.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Failed assistant replies now offer Retry instead of Continue, and the action service rejects stale Continue requests for failed assistant messages without changing stopped or user-message behavior.
<!-- SECTION:FINAL_SUMMARY:END -->
