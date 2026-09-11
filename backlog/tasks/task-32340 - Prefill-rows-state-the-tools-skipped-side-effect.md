---
id: TASK-32340
title: >-
  Prefill rows state the tools-skipped side effect
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review D4. While a prefill is armed, tool calling (including MCP) is silently skipped for that send; the docs disclose it but the Inspector's 'Prefill (next send only)' / 'Prefill (pinned)' rows do not. Append the consequence to the armed row so the state is self-describing.

Filed from the 2026-09-10 Console rail UX review (review item D4).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add armed_prefill_row_value helper + suffix constant with tests. 2. Use it for both armed prefill rows in the conversation inspector rows builder. 3. Update the user-guide prefill section. 4. Re-run prefill suite.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Armed prefill rows indicate that tools will be skipped for the send
- [x] #2 Unarmed/absent prefill rendering is unchanged
- [x] #3 Row status class semantics unchanged
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** The row LABEL ("Prefill (next send only)" / "Prefill
(pinned)") is a stable key consumed by inspector row-id ownership and the
send-authority summary lookups, so the consequence rides on the VALUE: a
new `armed_prefill_row_value()` helper in `Chat/console_prefill.py`
(`describe_prefill_preview` + " — tools skipped this send") is used for
both armed rows in `_selected_console_conversation_inspector_rows`
(chat_screen.py). Unarmed/absent rendering unchanged; row status class
semantics unchanged (labels untouched).

**ADR check.** Not required — copy + pure helper, no boundary moved.

**Modified.** `tldw_chatbook/Chat/console_prefill.py`,
`tldw_chatbook/UI/Screens/chat_screen.py`,
`Tests/Chat/test_console_prefill.py` (+2 tests),
`Docs/User_Guide/console/context-and-rag.md`. Verified:
`pytest Tests/Chat/test_console_prefill.py` — 21 passed.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED (code): labels at chat_screen.py:14265-14292; prefill genuinely blocks agent/tools dispatch (console_chat_controller.py:17170-17178) and nothing surfaces it in the rows.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
