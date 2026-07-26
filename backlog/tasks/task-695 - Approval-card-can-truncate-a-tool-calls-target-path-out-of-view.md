---
id: TASK-695
title: Approval card can truncate a tool call's target path out of view
status: To Do
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
- [ ] A `write_file` approval row shows the target path regardless of the model's argument key order and regardless of `content` length
- [ ] Long argument values are still bounded so one call cannot dominate the card
- [ ] A test drives an approval row for a call whose `content` precedes `file_path` and exceeds the cap, asserting the path is visible in the rendered summary
- [ ] MCP approval rows keep their existing summary behavior, or the change is applied deliberately to both with the shared behavior tested
<!-- AC:END -->
