---
id: TASK-15660
title: Re-attach a survivor's full usage when its last child finishes (fleet F3)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 21:30'
updated_date: '2026-08-13 15:23'
labels:
  - console
  - agents
  - cost
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 3a-1 made a surviving sub-agent's post-turn token spend OBSERVABLE (the Console cost chip's `Sub-agents: N tok (not priced)` line) rather than attributed. The real fix needs a "last child done" signal the bridge does not emit today; PR 3a-2 builds exactly that signal for auto-wake, so this task consumes it rather than building a second one. Re-attach is already known to be idempotent (`_attach_stream_usage` recomputes from all payloads and `set_message_usage` REPLACES), and that idempotence is pinned by a test, so the path is safe to reuse.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A sub-agent that finishes after its turn has its usage folded into the originating assistant message's own usage row, not only into the cost chip
- [x] #2 Re-attaching twice produces the same stored total (the existing idempotence guard still passes)
- [x] #3 A conversation export includes a survivor's spend once the re-attach has run
- [x] #4 The chip's unattributed line falls to zero for a run whose children have all been re-attached
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the gap RED against unmodified production: full path (real bridge, real gated fleet child, real controller) — survivor bills after the turn's one attach; message row stays at the turn total; export excludes it; teardown variant.
2. Register a bridge-lifetime `FleetDrained` consumer ("usage-reattach") from the controller, next to bridge attachment (constructor + `update_agent_runtime`).
3. Record, per originating assistant message, the turn's (signals, resolution, partial) at watch time — only while the bridge says the conversation still owes a drain (`has_unsettled_children`, new bridge seam reading the drain-paired counter).
4. On drain: hop from the child's thread to the loop captured at watch time (`call_soon_threadsafe`), recompute-all + REPLACE via the existing `_attach_stream_usage`, re-baseline the session watch so `unattributed_fleet_tokens` reads zero, pop the source.
5. Ride `usage_json` on `export_conversation_to_json`'s per-message payload.
6. Mutation-test every new test; run the PR 3a-2 gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped on `feat/fleet-autowake` (PR 3a-2 Task 3). All four ACs verified by
`Tests/Chat/test_fleet_usage_reattach.py` (10 tests, red-first, 12/12
mutations killed — one mutation initially SURVIVED and forced a test
strengthening: a drain whose source is missing but whose session has a
watch from ANOTHER turn must attach nothing, or it becomes wrong-turn
attribution).

- AC#1: `test_survivor_spend_folds_into_the_message_row_when_the_last_child_settles`
  — full production path (real `ConsoleAgentBridge`, gated fleet child,
  `_finalize_agent_reply`); 120 tok at attach, survivor bills 45 more,
  drain folds the row to 165. The fold's store write is pinned to run on
  the loop the turn-end attach used, never the child's thread.
- AC#2: `test_re_attaching_twice_yields_the_same_stored_total` (end to
  end; the 6b pin `test_re_attaching_the_same_signals_is_idempotent`
  stays green untouched).
- AC#3: `test_a_conversation_export_includes_survivor_spend_after_the_fold`
  — `export_conversation_to_json` now carries per-message `usage` parsed
  from `messages.usage_json`; the folded 165 rides the export.
- AC#4: same full-path test asserts `unattributed_fleet_tokens == 0`
  after the fold (the watch's attached-count is re-baselined).

Files: `tldw_chatbook/Chat/console_chat_controller.py` (source map, loop
capture, consumer + hop, watch signature), `tldw_chatbook/Chat/
console_agent_bridge.py` (`has_unsettled_children`),
`tldw_chatbook/Character_Chat/Character_Chat_Lib.py` (export),
`Docs/User_Guide/console/agent-runs-and-tools.md` (honest limits),
`Tests/Chat/test_fleet_usage_reattach.py` (new).

Design note: there is deliberately NO mount-time reconcile half. The plan
brief assumed a survivor's tokens persist in `agent_runs` rows; verified
FALSE (the table has no usage column; `fleet.finish(total_tokens=...)` is
in-memory only; run-log manifests cover the primary's turn only). The
live path covers everything for the life of the PROCESS — the app loop
and the ChaChaNotes DB both outlive the Console screen, proven by
`test_the_fold_lands_durably_when_the_child_settles_after_console_teardown`
(fold read back through a fresh DB handle after `controller.shutdown()`).
Spend a survivor bills after its turn and before app exit that never
drains is durable nowhere; a mount reconcile could not recover it either.
Remaining honest limit recorded in the User Guide.
<!-- SECTION:NOTES:END -->
