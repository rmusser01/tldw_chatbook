---
id: TASK-37000
title: >-
  Console: a first send rebuilds the collapsed Inspector's run section and lays
  out hidden widgets
status: Done
assignee:
  - '@claude'
created_date: '2026-10-26 12:30'
updated_date: '2026-10-27 23:14'
labels:
  - console
  - performance
dependencies:
  - TASK-36012
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while classifying the send-transition requesters for TASK-36012 (headless, 160x45, collapsed Inspector, 2026-10-26). A chat's first send persists the conversation, which turns the Inspector's dictionary and world-book rows from "No active chat" into "No ... in play" and enables their Attach actions. `ConsoleRunInspector.sync_state` treats that as a structural change and recomposes, re-mounting the run section, the More control and every bounded section and viewport. The Inspector rail is collapsed (`display: none`), so none of it is visible, yet each mount and the rail cascade that then hides the new children ask Textual for a whole-screen layout: on TASK-36012's branch the first send's start still made about 190 layout requests over 14 full reflows, against 92 over 9 for a later send in the same chat.

The same is true, more mildly, of every send: the run inspector's live-work row and the selected-turn rows update their auto-height copy inside the hidden rail, and each update lays out the whole screen. Those widgets' heights can genuinely change, so the request is not wrong in itself; it is wrong only while nothing of the rail is drawn.

Fix direction to evaluate (not prescribed): let the collapsed Inspector defer its rebuilds and auto-height copy until it opens (opening already resizes the rail and reconciles it), keeping its state current for readers and tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Measured first, paired against the base build: layout requests and full reflows from collapsed-Inspector widgets during a first send and a later send
- [x] #2 While the Inspector is collapsed, a send causes no layout request from widgets inside it, and opening it afterwards shows the same rows, actions and geometry as a rail that was open throughout
- [x] #3 A deterministic test pins both halves and fails on the base build
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure first (headless, 160x45, collapsed Inspector): count layout/recompose requests and full reflows from widgets inside #console-right-rail during a first send and a later send on origin/dev; A/B the rail opened after a collapsed send against one open throughout.
2. Tests first: one test counting in-rail layout requests across a first and a later send (positive control with the rail open), one A/B witness (rows, actions, classes, geometry) of an Inspector opened after the send vs one open throughout. Both must fail on origin/dev for the right reason.
3. Fix: while the rail's collapse cascade hides them, ConsoleRunInspector and ConsoleInspectorSection keep their state current, re-write row copy without a layout, and hold any rebuild for the rail's opening; the cascade's opening runs the held rebuild. Display writes into the hidden rail (the Agents section) record into the cascade's marker instead of bypassing it.
4. Mutation-check the fix against the tests; re-measure headless; live paired spot-check with the pf2probe kit (base vs branch).
5. Related suites by test id vs origin/dev, ratchets, preflight, rebase onto newest origin/dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
While the Inspector rail is collapsed, a send no longer re-mounts or lays out anything inside it. The collapsed Inspector keeps its state current, re-writes row copy without a layout, and holds any rebuild until the rail opens. The rail's opening runs the held rebuild before its first paint.

Approach
- The collapse cascade (`ChatScreen._sync_console_rail_descendant_visibility`) already marks every hidden descendant with `_console_rail_prior_display`. `hidden_by_console_rail(widget)` in `console_inspector_section.py` reads that mark, so no new state was added to the screen.
- `ConsoleRunInspector.sync_state` and `ConsoleInspectorSection.sync_state` behave as follows while hidden. `state`, `rows` and `summary` update at once. An in-place change re-writes its Statics with `layout=False`. A structural change sets `_rail_rebuild_deferred` and returns. When the cascade opens the rail, it calls `rebuild_if_deferred()` on each restored child, so the rebuild lands with the opening's own layout. While a rebuild is held, every later sync skips the in-place path.
- `set_display_under_console_rail` routes display writes inside the hidden rail (the Agents section's show/hide, and a row's secondary line) into the cascade's record. A direct write there was undone by the opening: on dev, a rail opened after a collapsed first send has no Agents section ("Run history" / "View all runs"), and the same rail kept open throughout has one.
- The readiness card's MCP row is written by the sync tick after a chat's first agent run publishes its MCP catalog. The live check found that write; inside the collapsed rail it now passes `layout=False`.

Measured (AC #1)
- Headless, 160x45: on origin/dev 3be493413a, a first send's start window made 56 in-rail layout requests (the run inspector's and Agents section's recomposes). Every send made 2 more (the "Live work" row). On origin/dev 0923ec0ca2 the count is 62, now that the test's first send also publishes an MCP catalog. The branch makes 0.
- Live, real app with Anthropic claude-haiku-4-5 through the review proxy, pf2probe, 160x45, Inspector collapsed, B F F B, 4 sends per launch. The run logs (tools/evidence/bd5/P37/runs/ and the files named below) were kept with the verifying session's local tooling, not in this repository.
  - r4: base origin/dev 0923ec0ca2 against the final head. First send in-rail requests: base 107 and 59, fix 0 and 0. Later sends: base 2 each, fix 0. Base had 4 and 5 reflows per first-send window that included rail requesters; the fix had 0. The MCP-row write about 2-4 s after the first send appears in both base launches and in neither fix launch (mcp-row-timing.txt).
  - r3 (load 42-57) is consistent: every launch lost send 0 to a late render, so send 1 was the first send. In-rail requests: base 107 and 59, fix 0 and 0.
  - r1 and r2 (pre-rebase head, base 3be493413a) are also consistent, except for the MCP row, which this branch then fixed.
- The total full-reflow count per send window is not measurably lower: the rail's requests coalesce into reflows the send makes anyway. At load 55-70, reflow time is noise. What the fix removes is the run inspector's and Agents section's recompose and mount work, and every layout request from the hidden rail.
- Visual (visual_final/): the base rail opened after a send has no Agents section. On the final head, the rail opened late matches the rail open first, row for row.

Tests (AC #3)
- `Tests/UI/test_console_collapsed_inspector_send_layout.py` has two mounted Console cases. The first counts in-rail layout requests over a first and a later send; its positive control has the rail open, and its first send now publishes an MCP catalog. The second is an A/B witness (rows, copy, disabled state, classes, geometry, every painted frame of the opening) of a late-opened rail against one open throughout.
- `Tests/UI/test_console_inspector_rail_deferral.py` has four fast widget cases. They drive each path (held rebuild, in-place patch) through the production cascade.
- All six fail on origin/dev 0923ec0ca2 for the reasons above. The deferral cases name the hidden recomposes, the "Live work" row and the section's secondary line. The send case reports 62 layouts. The parity case is missing the Agents section at index 22 of the witness. The MCP case also failed on the pre-fix branch head, naming `Static#console-live-work-source-mcp`.
- Four existing pending-facts tests read a collapsed inspector's rows through its opening projection. One journey test opens the Inspector before pressing Review approval.

Trade-offs
- A child's mark does not cover a widget mounted into the rail after the collapse. Such a widget is still treated as visible. It is correct, but it can still lay out (see TASK-37000.1 for other display writes that bypass the cascade).
- The first draft's eleven settings-row writes (ConsoleSettingsSummary) happen before Send, on base and branch alike. They were filed as TASK-37000.2.
- ChatScreen's size row was raised 24319 -> 24324 under the owner rule, with a dated comment, and to 24325 by the review's marker-constant tidy-up (one import line). The UI PR census floor rose by this task's 3 tail entries (dev's 965 -> 968 after the rebase onto 773cc9729c).

Files: Widgets/Console/console_inspector_section.py, Widgets/Console/console_run_inspector.py, UI/Screens/chat_screen.py, the two new test files, four adjusted pending-facts tests, Tests/Architecture/test_screen_size_ratchet.py, scripts/check_ui_pr_gate_census.py + ui_pr_gate_census.txt, Docs/User_Guide/console/agent-runs-and-tools.md, backlog/docs/lessons-testing-evidence.md (two entries), TASK-37000.1 and TASK-37000.2.

Review follow-ups (2026-10-27, review verdict ship, no blocking findings)
- Focus: the review held that a responsive collapse leaves focus on a hidden Inspector control, which the held rebuild would then drop. A headless probe on base and branch refuted it: Textual's Hide event blurs the control at the collapse's first layout, and focus ends in the same place on both. The comment now says so, and test_a_control_focused_at_the_collapse_leaves_no_focus_to_carry pins it, for a change made later and in the same tick as the collapse.
- The cascade and the run inspector name the marker through CONSOLE_RAIL_PRIOR_DISPLAY (ChatScreen row 24324 -> 24325, dated comment).
- The pending-facts helper reads mounted rows unless a rebuild is actually held, and then projects with the inspector's own ownership, not STRICT.
- New full-app case test_an_inspector_opened_after_chat_switches_matches_one_open_throughout: after a collapsed first send, a new tab, a temporary tab and the saved chat again, each met collapsed, match a rail open throughout. It leaves the outer fold hint out, because a rail opened late after a switch drops that hint on dev as well (TASK-37000.5). It is not in the UI PR census: it took 170 s here at host load about 70, too slow for that lane.
- Filed: TASK-37000.1 widened to the Context rail's agent-section display writes; TASK-37000.3 (one rail module for the cascade and helpers); TASK-37000.4 (a larger paired sample of first-send full reflows; r4 had two launches per arm); TASK-37000.5 (fold hint after a switch).

Merge evidence (2026-10-27, head after the review follow-ups, base origin/dev 773cc9729c, no rebase needed)
- New tests, plain command: test_console_inspector_rail_deferral 6/6; test_console_collapsed_inspector_send_layout 3/3 (113 s at host load about 70).
- On a detached origin/dev worktree: the deferral file fails 6/6 (the four layout cases; the focus pin only because dev holds no rebuild). The chat-switch case fails at 'back to the saved chat': the late-opened rail lacks the Agents section. A first dev run failed earlier at 'new tab' on the staged-context tray's height (4 rows against 3); the branch matched there in every run that reached that step.
- Mutation: with the cascade's rebuild_if_deferred call removed, the chat-switch case fails at 'new tab' (the run inspector keeps its held rows).
- Neighbours on the branch: pending_interrupt_projection 15/15, both cold-readiness files 1/1, first_persistence 1/1. By test id against dev, no branch-only reds: inspector_section 3 failed on both, run_inspector 79 setup errors on both, right_rail 34 failed on both, run_tick_layout_requests 14/14 on both.
- Ratchets: test_screen_size_ratchet 4/4 (ChatScreen 24325). test_module_size_ratchet: the same 8 rows fail on dev, none for a touched file. ui-ready census: 1032 against 1031 on dev and branch alike, identical output. Boot CSS budget passes; no CSS changed.
- ./scripts/preflight.sh exit 0 (UI census 968, floor 968).
- No new live round: this session's code changes rename a marker to a constant of the same value and correct a comment, so runtime behaviour is unchanged from the live-verified head.
<!-- SECTION:NOTES:END -->
