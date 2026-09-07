---
id: TASK-2030
title: 'Change review: transcript ✎ row''s Review action cannot resolve its message'
status: Done
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
- [x] #1 Selecting a ✎ row and pressing `v` (or clicking its Review action) opens the Review screen on that turn, on the live app, including after a session switch or resume
- [x] #2 A test drives the real transcript render path (synthesized marker rows, not store ids) and pins the fix
- [x] #3 The failure toast still appears for genuinely-deleted targets (no blanket bypass of the store lookup for other actions)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing test that drives the REAL render path: transcript built from the historical/recompute pipeline (synthesized ✎ marker rows not in the store), invoke the row's review-changes action, assert the Review screen opens on the right run — watch it fail with the "target no longer exists" shape
2. Fix: carry the run id in the review-changes dispatch at render time (it is display data already on the row) so the handler's review-changes branch never needs the store lookup; every other action keeps the store resolution and its failure toast
3. Sabotage-verify, run the change-tracking/transcript/screen suites
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause was sharper than filed: `ConsoleChatStore._message_or_raise`
resolves from the session's TREE NODES, and display-only TOOL markers are
deliberately never tree nodes (its own comment says so) — so
`store.get_message(<marker id>)` ALWAYS raises, live-appended or resumed.
No resume needed; every ✎ row action died at the handler's pre-dispatch
lookup.

Fix: `ConsoleTranscript.display_message(id)` (public display-model lookup
over the rendered rows) + the `review-changes` branch in
`handle_console_message_action` hoisted ABOVE the store lookup, resolving
the run id via `_console_change_review_run_id` (display model first, store
fallback for tree-node rows, toast on neither). Every other action keeps
the store resolution and failure toast byte-identical.

Test drives the REAL chain the live UAT exercised: bridge run → store rows
→ `_sync_native_console_transcript` → rendered ✎ row → `select_message` →
`action_invoke_selected_action("review-changes")` → button dispatch →
handler → pushed ChangeReviewScreen with the right turn. Watched it fail
with the exact live shape first. AC#3 pinned: unknown target still toasts,
no screen. Both directions sabotage-verified (None-branch removed; display
lookup blinded — each caught).

Files: `Widgets/Console/console_transcript.py`, `UI/Screens/chat_screen.py`,
`Tests/Chat/test_change_turn_tracking.py`. 361 tests green across
change-review + transcript + message-action suites.
<!-- SECTION:NOTES:END -->
