---
id: TASK-2760
title: 'Home primary action navigates from a different HomeAction than the one it displays'
status: To Do
assignee: []
created_date: '2026-08-06'
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

- [ ] The primary-action button always navigates to the route of the SAME
      `HomeAction` whose label it displays.
- [ ] The screen context passed on press is built from that same action.
- [ ] A test pins label/destination agreement for the suppressed case (failed
      item selected on the top action's route).
