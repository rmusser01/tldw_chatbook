---
id: TASK-42351
title: "Record each tool call's before and after file content for Change Review"
status: To Do
assignee: []
created_date: 2026-11-22 16:37
updated_date: 2026-11-22 16:37
labels:
- follow-up
- console
- review
- tools
priority: low
references:
- tldw_chatbook/Workspaces/change_turn_tracker.py
- tldw_chatbook/Workspaces/change_tracking.py
- tldw_chatbook/Workspaces/change_retention.py
- tldw_chatbook/Workspaces/change_bounds.py
- Docs/Architecture/console-file-authority.md
- Docs/superpowers/specs/2026-08-02-agent-change-review-design.md
dependencies:
- TASK-38942
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Proposed while round 20 corrected TASK-38942 (fu/r20-backlog, 2026-11-22). Read at 59cc99e61b.

**Why.** Change Review records one Begin and one End snapshot per root per turn (`Workspaces/change_turn_tracker.py`, "Per-turn B/E snapshot"; each `TurnChangeRecord` holds `baseline_sha`, `end_sha`, change counts and any `tracking_error`). It can show a file's net change over a turn, but not what each tool call did to it. TASK-38942 therefore steps through a turn's tool calls beside the net diff, and labels each call's arguments as requested input, not verified file state. It names per-call capture as out of scope until it has its own task and ADR. A user who wants to see which call broke a file, when one file was written several times in a turn or a `shell_exec` wrote files it does not name, needs the file's content before and after each call.

Capturing content per call changes what is stored, how long it is kept, and what a later reader may see under current workspace and exclusion rules. That is why an ADR comes first.

`Workspaces/native_git.py` and `Workspaces/change_review_consent.py` are in another session's open work (tldw-chatbook-30). Coordinate before editing either.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An accepted ADR decides what is captured per call (including for `shell_exec`), the storage and retention bounds, behaviour when capture fails or is truncated, and how current workspace and exclusion authority applies to historical content
- [ ] #2 For each file-changing tool call in a turn, Change Review can show that file's recorded content before and after the call, or an explicit missing, truncated, binary or capture-failed state
- [ ] #3 Capture never changes what a tool call does or returns, and stays within the ADR's storage bounds on a turn with many writes to one large file
- [ ] #4 Tests cover several writes to one file in one turn, a `shell_exec` that writes files it does not name, a capture failure, and revoked access to a captured path
<!-- AC:END -->
