---
id: TASK-695
title: Approval card can truncate a tool call's target path out of view
status: Done
assignee: []
created_date: '2026-07-26 06:45'
labels:
  - ui
  - agents
  - security
dependencies:
  - TASK-545
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ChatApprovalCard` caps a tool call's whole JSON argument summary at 80 characters (`Widgets/Chat_Widgets/chat_approval_card.py:71`). Because `json.dumps` preserves the model's key order, a `write_file` call that emits `content` before `file_path` shows the user a truncated blob of file content with the destination path scrolled off the end.

That makes the approval prompt ask "may I write this?" without showing *where*. The decision the card exists to collect cannot be made correctly from what it displays.

This is pre-existing widget behavior, but TASK-545 P2 is what makes it reachable: `write_file` is the first gated built-in whose arguments are routinely larger than the cap, and it is precisely the tool where the target matters most. Blast radius is bounded by the sandbox root, so this is a clarity defect rather than an escape.

Fix should ensure security-relevant arguments (paths, destinations) survive truncation — e.g. summarize per-argument with its own budget, or hoist known path-like keys to the front — rather than raising the global cap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] A `write_file` approval row shows the target path regardless of the model's argument key order and regardless of `content` length
- [x] Long argument values are still bounded so one call cannot dominate the card
- [x] A test drives an approval row for a call whose `content` precedes `file_path` and exceeds the cap, asserting the path is visible in the rendered summary
- [x] MCP approval rows keep their existing summary behavior, or the change is applied deliberately to both with the shared behavior tested
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two changes in `_summarize_arguments`, neither of which raises the global cap (raising it only moves the cliff):

1. **Destination keys are hoisted.** `_is_destination_key` matches whole TOKENS, not substrings -- `profile` contains "file" and `urinal` contains "uri", and a false positive pushes the real destination later in a budget-limited line, which is the defect this exists to fix. camelCase is split, so `filePath` matches.
2. **Every value gets its own budget.** Destinations render at the full per-value limit; what they leave is split evenly among the remaining arguments, with a floor so a value never renders as pure ellipsis. Without the split, the SECOND bulk argument still starves everything after it -- a fixed per-value cap just moves the cliff along by one key.

Redaction runs BEFORE reordering and clipping, so neither can expose a secret (pinned by a test).

AC#4: this is the shared summariser, so MCP and built-in rows change together -- deliberate, and the whole point is that it is the one place a target can be hidden.

**Both halves are mutation-verified, and the first attempt to verify hoisting failed to.** Removing the hoisting left all 25 tests green, because with a handful of arguments the per-value budget alone keeps the destination on screen wherever it sits. Hoisting only becomes load-bearing when the destination is last among MANY arguments -- the shared budget shrinks, the total still reaches the line cap, and the tail is clipped. That case now has its own test, and removing the hoisting fails only that one.
<!-- SECTION:NOTES:END -->
