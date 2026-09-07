---
id: TASK-1972
title: 'Change review: per-turn transcript summary row + inspector action'
status: Done
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - console
  - change-review
  - ux
dependencies:
  - TASK-1971
  - TASK-1973
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Display surfaces for a turn's changes: a transcript row in the TOOL-marker display-only family (`✎ Edited 3 files  +92 −468 — review with `v``, markup=False), a `review` selected-row action, and a 'Review changes' row in the run inspector's actionable group. Deliberately NO destructive control here: Undo-all lives on the Review screen behind a confirm — a one-keystroke destructive action in the transcript would repeat the approval-card mistake TASK-1845 fixed. Honesty degradations surface here too: 'change tracking failed (reason)', 'N nested repositories not tracked'.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A turn with changes renders the summary row with real counts; a turn without renders nothing
- [x] #2 The row survives subsequent messages and session switch/resume (TOOL-marker anchoring rules)
- [x] #3 `v` on the selected row and the inspector action both open the Review screen for THAT turn
- [x] #4 tracking_error and nested-repo warnings render on the row/inspector, in monochrome-legible text
- [x] #5 No control in the transcript mutates files
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `ConsoleChatMessage.change_review_run_id` (session-only field, like `tool_output_full`) + store passthrough — the row must carry WHICH turn it reviews, or `v` can only guess.
2. Shared `format_change_summary_marker(records)` in console_agent_bridge (the format_agent_step_marker discipline: live and resume render byte-identical); live emit in run_reply's finally after end_turn (counts row, or ⚠ tracking-failed row); resume emit in `resume_marker_messages` from change_snapshots rows.
3. `review-changes` action: offered by ConsoleMessageActionService only when the message carries a run id; transcript BINDING `v`; chat_screen dispatch opens ChangeReviewScreen via a new bridge seam `change_review_provider(conversation_id)` and selects THAT turn.
4. Inspector actionable row "Review changes" (the honest replacement for the 1843-removed dead action): enabled only when the conversation has snapshot rows, with a reason otherwise.
5. TDD: bridge-level marker emission (counts / none / ⚠), marker survival past the next message (TOOL anchoring), resume parity, action-offering rules, screen-open wiring; sabotage first-try passes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`ConsoleChatMessage.change_review_run_id` (session-only, resume re-derives it); shared `format_change_summary_marker`/`format_change_tracking_failure_marker` used by BOTH the live emit (run_reply's finally, after rows are stored) and resume (`resume_marker_messages` reads change_snapshots) — byte-identical per the marker discipline; `review-changes` action offered only on rows carrying a run id; transcript `v` binding; chat_screen dispatch + `_open_change_review`; inspector "Review changes" actionable row (the honest replacement for 1843's dead action) enabled on tracker presence — deliberately NOT on a per-tick DB query; the SCREEN owns the empty state.

**The opener wiring test caught two real production races** (and earlier, an invented method name — three defects in one small opener, none visible by reading):
1. post-push `call_after_refresh(select_turn)` fired before the pushed screen composed → NoMatches; fixed as constructor state (`initial_run_id`) applied by the screen itself;
2. the screen's own `on_mount` queried children that don't exist yet when pushed onto a LIVE app — the standalone 1973 harness had passed by timing luck; initialization now defers via `call_after_refresh`, the ChatApprovalCard pattern.

Also: bridge tests that run `run_reply` from an async test MUST `asyncio.to_thread` it (its internal loop collides with pytest-asyncio's) — as production does.

Three sabotages, each failing its tests: live emit removed (3), resume emit removed (parity), action over-offered (offering). AC#4's nested-repo warnings land with TASK-1976 (that detection does not exist yet); tracking_error warnings are in.
<!-- SECTION:NOTES:END -->
