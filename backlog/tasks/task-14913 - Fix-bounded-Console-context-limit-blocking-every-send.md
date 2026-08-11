---
id: TASK-14913
title: Fix bounded Console context limit blocking every send
status: Done
assignee: []
created_date: '2026-08-11 03:50'
updated_date: '2026-08-11 04:03'
labels: []
dependencies: []
references:
  - backlog/decisions/052-console-conversation-memory-and-compaction-policy.md
modified_files:
  - tldw_chatbook/Chat/console_chat_controller.py
  - Tests/Chat/test_console_context_compaction.py
  - backlog/docs/lessons-testing-evidence.md
  - backlog/tasks/task-14914 - Add-deterministic-visual-transcript-compaction.md
  - Docs/superpowers/qa/console-context-memory-uat-2026-08-10/README.md
  - Docs/superpowers/qa/console-context-memory-uat-2026-08-10/uat_context_memory.py
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore Console message sending when a user has configured a valid bounded conversation context limit. The current send path incorrectly reports that compaction cannot run safely even when the saved per-conversation limit should make the request admissible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A valid custom conversation-token limit saved from Console model settings is applied to the next request for both new and restored conversations
- [x] #2 Messages send normally when mandatory context and the active request fit within the effective bounded capacity
- [x] #3 A genuinely unsatisfiable request remains blocked with recovery guidance based on the actual limiting segment
- [x] #4 Regression tests cover the settings-to-policy-to-prepared-request path and prevent the false compaction-unrecoverable state
- [x] #5 Focused controller, policy, provider-gateway, and Console UI tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the saved custom limit from the Console modal through session state, persistence, policy resolution, and prepared-request admission. 2. Add a failing end-to-end regression for a bounded conversation that currently cannot send. 3. Fix the narrow ownership or capacity calculation defect without weakening mandatory-context safety. 4. Verify new and restored conversations across relevant compaction modes. 5. Run focused tests, review the diff, and record implementation evidence. ADR required: no. ADR path: backlog/decisions/052-console-conversation-memory-and-compaction-policy.md. Reason: this is a compatibility bug in the existing accepted policy, not a new boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Separated compaction availability from provider send admission. UNKNOWN_WINDOW and NON_COMPACTABLE now pass through when the exact immutable prepared request has no known overflow; known overflow remains blocked, and the recovery copy now leads with the request-fit failure, identifies mandatory material or the absence of older complete turns, and explains why summarizing cannot help. Added regressions for the default inherited automatic budget on an unknown model, bounded custom compaction on an unknown model, bounded active memory with no older eligible unit, and genuine mandatory-material overflow. Filed TASK-14914 for deterministic visual-transcript compaction. Completed an isolated full-app new-user UAT and senior UX/HCI review with a persistent nine-finding register, wide/narrow captures, keyboard focus evidence, invalid-save validation, global Settings inspection, and a deterministic fake-provider send. Addressed the PR's only substantive review comment by documenting the intent of all four new async regressions. Verification: 42 compaction tests passed; the 221-test policy/lifecycle/preparation/gateway/UI matrix passed; 172 controller tests passed; the final 61-test compaction/context-controls/settings matrix passed; the full-app bounded-send UAT passed; scoped Ruff checks and git diff whitespace checks passed. Two repository-baseline E721 findings outside this diff were confirmed to predate the branch and were not changed. ADR required: no; ADR-052 already defines this separation and the fix restores that contract.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console sends are no longer blocked merely because compaction is unavailable: unverified-but-not-known-overflow requests proceed, bounded custom policy is honored, and proven overflows still stop with precise recovery guidance.
<!-- SECTION:FINAL_SUMMARY:END -->
