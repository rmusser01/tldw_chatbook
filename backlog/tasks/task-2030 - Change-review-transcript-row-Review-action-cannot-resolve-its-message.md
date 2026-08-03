---
id: TASK-2030
title: 'Change review: transcript ✎ row''s Review action cannot resolve its message'
status: To Do
assignee: []
created_date: '2026-08-03 00:45'
labels:
  - change-review
  - console
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-1980 live UAT (the headline finding). The ✎ change-summary row
says "review with `v`", but on the live app BOTH the `v` binding and the
row's own Review action button toast "Console message action target no
longer exists" — every time, including seconds after the run on the very
turn that produced the row.

Root cause: the transcript renders TOOL marker rows through the
resume/recompute pipeline (`console_agent_bridge.py` `_console_historical_blocks`
area, ~line 2300), which synthesizes fresh `ConsoleChatMessage` objects that
are never placed in the store. Their auto-generated ids end up in the action
button id, and `handle_console_message_action` (`chat_screen.py` ~18079)
resolves `store.get_message(message_id)` BEFORE dispatching `review-changes`
— KeyError → toast. The run id the action actually needs is already ON the
row (`change_review_run_id`); the store lookup is only how the handler reads
it back.

The inspector route (`Run` panel → Review changes) works and is how the UAT
proceeded. 24 change-tracking tests pass because they drive the handler with
store message ids — the fixture-invented-shape trap, fifth occurrence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selecting a ✎ row and pressing `v` (or clicking its Review action) opens the Review screen on that turn, on the live app, including after a session switch or resume
- [ ] #2 A test drives the real transcript render path (synthesized marker rows, not store ids) and pins the fix
- [ ] #3 The failure toast still appears for genuinely-deleted targets (no blanket bypass of the store lookup for other actions)
<!-- AC:END -->
