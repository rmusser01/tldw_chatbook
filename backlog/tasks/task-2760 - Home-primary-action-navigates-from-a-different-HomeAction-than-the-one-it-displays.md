---
id: TASK-2760
title: 'Home primary action navigates from a different HomeAction than the one it displays'
status: Done
assignee: ['@t2-home']
created_date: '2026-08-06'
updated_date: '2026-10-11'
labels: [home, bug, ui, honesty]
dependencies: []
---

## Description (the why)

The Home canvas labels its primary action button — and renders the
`Next: <label> — <reason>` callout — from `canvas.next_action`
(`Widgets/Home/home_canvas.py:110-127`), which, when a failed item is
selected on the top suggestion's own route, is the **H3-suppressed**
recomputation (`Home/dashboard_state.py:1260-1278`, excludes
`review_failed_work` and, with nothing running, `resume_active_work`).

But the press handler navigates from a **different object**:
`_activate_home_primary_action` (`UI/Screens/home_screen.py:739-748`) posts
`NavigateToScreen(dashboard.next_action.target_route, …)` where
`dashboard = summarize_home_dashboard(state)` calls
`choose_next_best_action(state)` with **no exclusions**
(`dashboard_state.py:771`).

Live repro (guide-G5 verification, dev @ 84e4b33f0, 2026-08-06): with one
failed Library ingest job auto-selected, the button and callout both read
"Start a conversation — Console is ready for a task." and clicking the
button opens **Library**. On a quiet profile the same button opens Console.
The user-facing promise and the routing disagree exactly when the H3
suppression fires.

Documented as a Quirk in `Docs/User_Guide/home.md`.

## Acceptance Criteria (the what)

- [x] The primary-action button always navigates to the route of the SAME
      `HomeAction` whose label it displays.
- [x] The screen context passed on press is built from that same action.
- [x] A test pins label/destination agreement for the suppressed case (failed
      item selected on the top action's route).

## Implementation Plan

1. Premise-verify at origin/dev 9f8533686c: canvas labels from
   `triage.canvas.next_action` (H3-suppressed recomputation) while
   `_activate_home_primary_action` routed from
   `_current_dashboard.next_action` (unsuppressed) — confirmed live in code.
2. Red-first: add a pinned test (failed Library ingest selected on its own
   route, nothing running, ready Console) asserting the button reads
   "Start a conversation" AND the press routes to `chat`; observed RED
   (`library != chat`) pre-fix.
3. Keep the canvas's own `next_action` on the screen
   (`_current_canvas_next_action`, set wherever `_current_canvas_controls`
   is set) and dispatch the primary press from it, with the dashboard action
   as a defensive fallback only; build the press context from that same
   action.
4. Re-pin the neighbor test that encoded the old disagreeing behavior, and
   add a count-only companion keeping the unsuppressed
   review-failed-work -> subscriptions/runs-context path covered.

## Implementation Notes

- Approach: single-source the dispatched action. `HomeScreen` already kept
  the selection-scoped canvas CONTROLS (`_current_canvas_controls`, T152)
  for exactly this class of display/dispatch disagreement; the primary
  action now follows the same pattern via `_current_canvas_next_action`
  (set in `compose_content` and `_sync_home_triage` from
  `triage.canvas.next_action`). `_activate_home_primary_action` dispatches
  route AND nav context from that action; `_current_dashboard.next_action`
  remains only as a defensive fallback.
- Tests: `test_home_primary_action_routes_from_the_action_it_displays`
  (RED pre-fix: navigated `library` while displaying "Start a conversation";
  GREEN post-fix: `chat`, empty context). The pre-existing
  `test_home_failed_watchlist_primary_action_opens_watchlist_runs_context`
  had pinned the disagreeing behavior (button "Start in Console" navigating
  to subscriptions); it now pins label/destination agreement, and a new
  count-only companion
  (`test_home_failed_count_primary_action_opens_watchlist_runs_context`,
  `active_detail_route="subscriptions"`) keeps the unsuppressed
  review-failed-work suggestion's subscriptions + runs-section context
  covered.
- Files: `tldw_chatbook/UI/Screens/home_screen.py`,
  `Tests/UI/test_home_screen.py`.
- Verified: full `Tests/UI/test_home_screen.py` +
  `Tests/UI/test_home_triage_rail.py` green (79 passed) at the fix commit.

ADR required: no
ADR path: N/A
Reason: local dispatch-coherence fix inside one screen; no schema, sync,
provider-boundary, or cross-module contract change.
