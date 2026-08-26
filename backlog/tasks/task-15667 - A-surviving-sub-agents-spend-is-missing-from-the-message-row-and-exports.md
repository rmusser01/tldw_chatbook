---
id: TASK-15667
title: A surviving sub-agent's spend is missing from the message row and exports
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
PR 3a-1 delivered audit F3 as OBSERVABLE, not fixed: a survivor's post-turn token spend reaches the Console cost chip's tooltip (`Sub-agents: N tok (not priced)`) and the chip's own token total, and nothing else. It is absent from the assistant message's stored usage row and from conversation exports, and it is remembered only for the lifetime of the controller instance - close the Console screen and it is gone. This is the partiality that remains after F3; the full re-attach is tracked separately and depends on a signal PR 3a-2 builds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A conversation export accounts for sub-agent spend, or states explicitly that it excludes it
- [x] #2 The assistant message row's usage reflects the sub-agents that ran underneath that turn
- [x] #3 The figure survives closing and reopening the Console screen
- [x] #4 The User Guide's honest-limits paragraph is updated when the limit no longer holds
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Satisfied by TASK-15660's fix (PR 3a-2 Task 3, same branch): the
`FleetDrained` fan-out's "usage-reattach" consumer folds a survivor's
spend back onto the originating assistant message when the conversation's
last fleet child settles. This task re-verifies its own four ACs against
that fix rather than shipping a second mechanism.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed as satisfied-by-15660's-fix — not a duplicate closure on faith;
each AC was re-verified by execution on `feat/fleet-autowake`
(`Tests/Chat/test_fleet_usage_reattach.py`):

- AC#1 (export accounts for sub-agent spend, or states it excludes it):
  BOTH halves now hold. `export_conversation_to_json` carries per-message
  `usage` from `messages.usage_json` (test:
  `test_a_conversation_export_includes_survivor_spend_after_the_fold`);
  the plain-TEXT export still has no token figures, and the User Guide's
  honest-limits bullet now states that explicitly.
- AC#2 (message row reflects the sub-agents that ran underneath):
  `test_survivor_spend_folds_into_the_message_row_when_the_last_child_settles`
  — the row's stored total includes post-turn survivor billing after the
  fold; `error`/`cancelled` children's partial spend included
  (`test_an_earlier_turns_survivor_folds_onto_its_own_message_after_a_later_turn`
  uses a cancelled, run-id-less child).
- AC#3 (figure survives closing and reopening Console): the fold writes
  `usage_json` through the persistence adapter's version-neutral column
  write, which outlives the screen —
  `test_the_fold_lands_durably_when_the_child_settles_after_console_teardown`
  runs the exact `on_unmount` sequence (`busy_fleet_session_count` +
  `shutdown()`), lets the child settle AFTER teardown, and reads the
  folded figure back through a FRESH DB handle. A remount loads it from
  the DB. The one case that still loses the remainder is quitting the
  APP before the last child settles — that spend is durable nowhere
  (verified: `agent_runs` has no usage column), and no mount-time
  reconcile could recover it; stated in the User Guide.
- AC#4 (User Guide honest-limits updated): the chip-only bullet in
  `Docs/User_Guide/console/agent-runs-and-tools.md` is rewritten — fold
  on last-child-done, durable row, JSON export inclusion, and the two
  remaining honest limits (app-exit remainder; text export).
<!-- SECTION:NOTES:END -->
