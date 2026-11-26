---
id: TASK-41959
title: "Console Inspector rail order and workbench contract drift: staged-context tray position and five workbench nodes"
status: Done
assignee: []
created_date: '2026-11-19 07:30'
updated_date: '2026-11-26 02:00'
labels:
- follow-up
- tests
- console
- ui
priority: low
references:
- Tests/UI/test_console_workbench_contract.py
- Tests/UI/test_console_session_settings.py
dependencies:
- TASK-41633
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** The sub-agents section (`#console-agent-section-subagents`) now sits where the staged-context tray was pinned (`test_console_inspector_hosts_staged_context_above_source_readiness`, `test_console_left_rail_keeps_session_and_moves_staged_context_out`), and three more workbench-contract nodes are red on dev (TASK-41633 notes F3/F10).

**Failing nodes (dev d0486978dd, TASK-41633 runs).**
- `Tests/UI/test_console_workbench_contract.py::test_composer_menu_opens_current_draft_directly_in_recommended_review[size1]`
- `Tests/UI/test_console_workbench_contract.py::test_console_approvals_chip_activation_focuses_the_decision_not_submit`
- `Tests/UI/test_console_workbench_contract.py::test_console_blocked_inspector_explains_impact_and_next_action`
- `Tests/UI/test_console_workbench_contract.py::test_console_inspector_prioritizes_actionable_status_before_secondary_groups`
- `Tests/UI/test_console_workbench_contract.py::test_console_left_rail_keeps_session_and_moves_staged_context_out`
- `Tests/UI/test_console_session_settings.py::test_console_inspector_hosts_staged_context_above_source_readiness`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The intended Inspector order is confirmed (product) and the tests pin it
- [x] #2 The workbench-contract nodes pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (owner ruled the order 2026-11-25; one yield moves, the rest is test drift)

1. Owner decision (2026-11-25): the staged-context tray goes back directly after Environment/Tasks, above the run inspector and source readiness; the sub-agents section (`#console-agent-section-subagents`) moves after the tray's own group (tray, Library search, Scope row) and ahead of the run inspector. Move the one `yield` in `right_rail.py`; keep the section's id, collapse default, hide-when-empty and sync path untouched.
2. Re-pin the fleet's own order tests (`test_console_right_rail.py` boundary inventory and direct-children pin, `test_console_live_work_handoffs.py`) to the new order; the two tray tests already pin it. Red-first: the tray tests fail on dev; negative control by reverting the move.
3. Blocked inspector test: trace why the inspector reads empty. Approvals-chip and actionable-order tests: trace the focus target and the disabled Review approval. Product regression -> fix product red-first; intended change -> update the test and cite the commit.
4. Update `Docs/User_Guide/console/context-and-rag.md` where it lists the Inspector order.
5. Run both named files whole (`-n 4`) plus every other changed test file; preflight, ruff F, ruff gate, size ratchets.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Order (product change).** Owner ruling 2026-11-25: the staged-context tray belongs directly under Environment/Tasks, above the run inspector and its source readiness; the sub-agents section is status and goes after it. `right_rail.py` now yields `#console-agent-section-subagents` after the Scope row and before `#console-run-inspector`. Body order: Environment, Tasks, Sources tray, Library search, Scope, Agents, run inspector, live work. The section's id, `open=False`, hide-when-empty and its sync path are unchanged. It went after the whole tray group, not straight after the tray, so TASK-24611's "Library search directly beneath the tray" and the Scope row's place beside the tray still hold.

**Cause.** 7e904737c7 (TASK-31450, #2411, 2026-09-05), its task-10 step, moved the fleet section from the left rail into the Inspector "directly below Tasks", ahead of the tray. The same squash updated the two tray tests for its task-9 step (`children[2] is tray`) but not for task-10, so both were red from that merge. The config-sandbox trap hid them until TASK-41633 (2026-11-18). The fleet's own pins (`test_console_right_rail.py` boundary inventory and direct-children order, `test_console_live_work_handoffs.py::test_console_live_work_card_swap_keeps_tray_on_top_and_cards_at_bottom`) now pin the new order.

**Stale tests (no product change; each traced to the commit that changed the behaviour on purpose).**
- `test_console_blocked_inspector_explains_impact_and_next_action` read `''` because the Inspector was closed. d5e1a26ab3 (TASK-23197, #2233) stops the 118-128 column auto-open while the Context rail is open. With that guard patched out, the unchanged test passed (probe). With the rail opened, the inspector still says "Blocked impact: Send is blocked…" and "Next action: Set up provider". The test now opens the rail first, as TASK-41308 did for its ready sibling.
- `test_console_inspector_prioritizes_actionable_status_before_secondary_groups`: Review approval stayed disabled. The old test failed 3 of 3 in my runs (alone and whole-file); the reviewer saw dev's original pass 1 of 3 serially. So it was a race, not a steady red. The test never staged an approval. Since 55f86c9538 (#2953) a mounted Console counts the controller's pending rounds plus a legacy tool card, and `app.console_pending_approval_count` is read only when no controller exists yet. That only happens in the startup race. The test now stages a real pending card. Four things were needed for that:
  - wait until `console_pending_startup_work()` is empty on two polls, with `_wait_for_console_quiet` shared from `test_console_right_rail.py`. In the review, a card staged earlier was wiped by the runtime's attach (`_set_console_pending_decision` writes `pending_approval=None`), and the first rewrite failed 2 of 3 loaded serial runs;
  - open the rail, because TASK-37000 holds a closed rail's rebuild. With the card staged and the rail closed, the count read 1 and the button stayed disabled;
  - wait for the card to survive the opening;
  - call `_sync_console_rail_and_controls()` explicitly. Staging refreshes only the task cards. Without that call, Review approval never enabled within 10 s: 2 of 2 probe runs on head, and the reviewer saw the same on dev.
- `test_console_approvals_chip_activation_focuses_the_decision_not_submit`: a2fcbe61bc (#3036) moved the keyboard landing from the row's decision Select (now behind More options) to the neutral `approval-request-summary`. TASK-40097 re-pinned the card-level test the same way. Run alone, focus landed there. In the whole-file `-n 4` run it was still on the status line after the fixed 0.1 s pauses, so the test now stages its card through the same quiet-then-stage helper. It also waits (10 s cap) for focus to leave the line, and it still refuses `approval-submit`.

**Evidence** (dev 2959b48314, scratch-profile wrapper, load 17-56).
- Red first: `test_console_left_rail_keeps_session_and_moves_staged_context_out` and `test_console_inspector_hosts_staged_context_above_source_readiness` failed on dev (`ConsoleInspectorSection(id='console-agent-section-subagents') is ConsoleStagedContextTray`).
- Negative control: with dev's `right_rail.py` restored over the fix, 4 of the 5 order nodes failed: both tray tests, the right-rail census and the handoffs swap test. The fifth, the static inventory tuple, is a pure literal. The fix was then restored byte-for-byte.
- Whole files at `-n 4`, base -> head:
  - `test_console_workbench_contract.py`: 4 failed/72 passed -> 76/76. `test_composer_menu_opens_current_draft_directly_in_recommended_review[size1]`, also named in this task, passed on base and on head in every run here. TASK-41633 saw it fail once in two runs (an action outside the 100x30 scroll). Nothing was changed for it.
  - `test_console_right_rail.py`: 37/37 -> 37/37.
  - `test_console_live_work_handoffs.py`: 6 failed/74 passed -> 5 failed/75 passed. All 5 are shared with dev (workflows/artifacts destinations, staged launch across screen recreation, WC action button). The set moves between runs: `..._keeps_console_launch_disabled_without_active_run` and `..._wc_live_work_action_button_routes_run_details` each failed in some head runs and passed in others, so they are flaky reds shared with dev, not differences.
  - `test_console_session_settings.py`: 9 failed + 8 errors -> 8 failed + 8 errors. The tray test went green. Every remaining node is listed in TASK-41962 (F6 reds and F11 teardown errors).
- Fleet neighbours, run whole on head; every red is shared with base:
  - `test_console_collapsed_inspector_send_layout.py`: 3/3 passed.
  - `test_console_inspector_navigation.py`: 39 passed, 1 failed. `test_responsive_hide_and_reveal_hands_off_focus_without_losing_offset` also fails on dev's `right_rail.py`.
  - `test_console_fleet_panel.py` and `test_console_environment_wiring.py` have no `bootstrap_profile` marker, so every node hit the config-sandbox trap. With a throwaway plugin that adds the marker:
    - fleet_panel: 11/12 passed. The one red writes an evidence SVG to a missing `.superpowers/` folder.
    - environment_wiring: 47/49 passed. `test_row_activation_toggles_expansion_and_rerenders` and `test_unbound_landing_replaces_the_previous_repos_paint` also fail with dev's `right_rail.py`.
- Static checks: `ruff --select F` clean; `ruff_gate.py` OK; size ratchets 238 passed; preflight passed.

**Review round (2026-11-25, local review of #3777 at 44a791f679).**
- *Fleet auto-open revealed nothing (Important 1, product fix).* The fleet's one-shot section expand runs while the rail is still hidden, so the section's `scroll_on_expand` reveal scrolled nothing. Below the tray that loss is visible. Red first on 44a791f679: the auto-opened rail at 150x40 with a 6-row tray put the Agents header at y=42, with the body ending at y=31. Fix, in `Console_Modules/agent.py`, because `chat_screen.py` is at its size budget:
  - `open_fleet_section_for_rows` keeps the section when it opens it behind a closed rail.
  - `reveal_fleet_after_rail_auto_open` runs at the screen's one rail-becomes-visible seam and scrolls the section into view with a minimal scroll once it has a region.
  - A rail the user opened (`preferred_right_open`) keeps its scroll, and the owed reveal is dropped.
  - Round 2 below changed the details.
  - Why not `top=True`: probed at 180x48 with 12 tray rows, the rail's first layouts shrank the rows above by 3 (scrollbar width and tray cap), which pushed a top-pinned header out of the body.
  - `chat_screen.py` changes by -1 line net.
- Tests in `test_console_right_rail.py`: `test_fleet_auto_open_scrolls_the_agents_section_into_view` at 150x40 with 6 tray rows and at 180x48 with 12, and `test_manual_inspector_open_keeps_its_scroll_with_fleet_rows` (140x40, where the auto-open is declined, opened by Alt+I's action).
- Mutations, each red:
  - no seam call: both sizes fail;
  - no `preferred_right_open` guard: the manual open jumped `scroll_y` to 21;
  - no region retry: both sizes fail, because the first refresh precedes the rail's layout.
- *Actionable-order race (Important 2).* Fixed as described above. Stability under load (Tests/Architecture serially plus whole-file runs as load):
  - actionable-order alone at `-n 4`: 10/10 at load 14-27, then 10/10 more at load 23-29;
  - approvals-chip alone at `-n 4`: 10/10 at load 23-29;
  - whole workbench file at `-n 4`: 76/76 twice (load about 25-38).
- *Final whole-file runs* at `-n 4`, with Tests/Architecture running serially beside them:
  - workbench 76/76;
  - right_rail 40/40;
  - agent_rail 51/51;
  - navigation 39/40, the shared responsive-reveal red;
  - handoffs 7 failed / 73 passed. All seven are the shared workflows/artifacts/staged-launch/WC reds plus `test_artifacts_destination_reopens_console_saved_chatbook_with_citation_metadata`, which passed 3 of 3 alone and in every other run.
- *Run with a throwaway plugin that adds the `bootstrap_profile` marker:*
  - environment_wiring 47/49, the same two reds as dev;
  - agent_controller 3/7, the same four reds on dev's product files;
  - fleet_panel: the SVG red plus 1-2 teardown errors ("Native app workers remain uncertain; config retirement refused", the TASK-41962 F11 kind). Dev's product files showed it too. The two head nodes, run alone as a pair, were clean in 5 of 7 runs.
- *Static checks (review round):* `ruff --select F` clean; `ruff_gate.py` OK; size ratchets 238 passed. The first placement in `chat_screen.py` hit its budget (+53 over 24,595), so the code moved to `agent.py`. Tests/Architecture run serially: 1507 passed, 5 skipped. Preflight passed.
- *Minors.*
  - `agent-runs-and-tools.md` no longer says "near the top".
  - `context-and-rag.md` says the self-opened rail scrolls to Agents.
  - The lesson now separates the closed-rail cause from the actionable test's count-source, startup-race and re-projection causes, and its recipe includes the startup-quiet wait.
  - The design spec carries a dated note pointing here.
  - These notes now correct the handoffs flake, the `[size1]` node and the "red in every run" claim.

**Review round 2 (2026-11-25, delta 44a791f679..5ec1696d01).**
- *I1, importer cap.* `_wait_for_console_quiet` and `_wait_for_right_rail_condition` moved into the non-test helper `Tests/UI/console_rail_section_helpers.py`. `test_console_right_rail.py` and the workbench contract both import them from there. `scripts/ci/tier_policy._importers` now gives `test_console_right_rail.py` 6 transitive test importers (cap 8): environment_wiring, footer_shutdown, inspector_focus_carriers, inspector_navigation, inspector_rail_minors, library_activity. With the earlier import it was 9.
- *M1, a landing pushed the fleet back down (product; the every-landing replay below was replaced in round 3).*
  - What happened: opening the rail starts the Environment fetch. Its landings arrive in tiers (unbound or local first, the PR and its checks later), and a rich one grew the sections above and pushed the fleet out. A one-shot replay on the first landing would not have covered the later PR tier, so `replay_fleet_reveal` instead redoes the minimal reveal on every landing (`_land_console_environment` hands it the Environment and Tasks sections).
  - It stops for good once the section closes, the user opens the rail themselves, the busy window ends, or the body's offset differs from the one the last reveal left (the user scrolled). The reveal also re-checks the offset when it runs, so a scroll made between a landing and its deferred reveal is kept.
  - Timing: the reveal is queued with `call_next` behind each landed section's rebuild (and behind the section's own held rebuild on opening), then `call_after_refresh`. A plain `call_after_refresh` ran against the old layout: in a probe the Environment section was still 4 rows when the scroll ran, then grew to 9.
  - Red first: `test_fleet_reveal_survives_the_first_environment_landing[150x40/6, 180x48/12]` landed a rich snapshot (branch, 3 changed files, PR with 3 checks, branch task). The section ended at y=29-30 (body 15-31) and y=35 (body 15-39). Control: `test_environment_landing_keeps_a_scroll_the_user_made`.
- *M2, any automatic opening reveals.* Kept as intended. The `chat_screen.py` seam comment, the `agent.py` docstrings and `context-and-rag.md` now say "when the Inspector opens itself while sub-agents are running". `test_band_open_after_context_closes_reveals_running_sub_agents` pins the 124-column band case: the Context rail is closed by preference, the band opens the Inspector with `preferred_right_open=False`, and the fleet is shown.
- *M3, no fixed pause.* The manual-open test spies on `scroll_visible` and asserts it is never called. It waits for the Agents section to have a region, then ten rounds of `pilot.pause()` plus `wait_for_refresh()`. `wait_for_refresh` alone did not run the queued rebuilds, and the mutated user-scroll guard then passed (see the new lesson).
- *M4.* `_sync_console_fleet_surface_flags`'s rows-absent branch calls `drop_fleet_reveal()`. That is now the only guard for ended rows: the reveal's rows-present check was removed as redundant. `test_reveal_is_dropped_when_the_sub_agents_finish_first`: at 140x40 the rows end, the bridge keeps a run-history page so the section stays shown, then an automatic opening happens, and `scroll_visible` is never called.
- *Nits.*
  - Zero DOM: the controller acts only on the sections the screen hands it, and the module docstring says so.
  - `open_fleet_section_for_rows` is typed `ConsoleInspectorSection`.
  - Lesson wording now says "a reviewer" and "the implementer".
  - `agent-runs-and-tools.md` is rewrapped.
  - `_approval_card_displayed` uses `_is_displayed`.
- *Mutations* (the reveal tests at `-n 4`). Each was red in both matrix runs unless noted:
  - no landing hook: the 2 landing nodes fail;
  - no user-scroll check: the control fails;
  - no wait behind the landed rebuilds: the 2 landing nodes fail;
  - no seam reveal: 6 nodes fail;
  - no `preferred_right_open` guard: the manual test fails;
  - no drop: the rows-ended test fails;
  - no region retry: the band test failed in 1 of 2 matrix runs. The retry stays as a bounded safety net (20 refreshes).
- *Stability:* the 9 reveal tests passed 6 of 6 runs at load 30-59 and 4 more under the final load (below). One earlier pass of a since-changed version failed 2 nodes; the cause was the harness's own "unbound" landing arriving out of order. The tests now give the Environment controller a root (`/w`) with stubbed dispatches, so the only landing is the test's.
- *Final runs on this code* (whole files at `-n 4`, load 28-75):
  - right_rail 45/45;
  - workbench 76/76;
  - agent_rail 51/51;
  - navigation 39/40, the shared responsive-reveal red;
  - handoffs 5 failed / 75 passed, all in the shared workflows/artifacts/staged-launch set.
- The 9 reveal tests passed 4 more times at load 25-33.
- Static checks: `ruff --select F` clean; `ruff_gate.py` OK; size ratchets 238 passed. Tests/Architecture serially: 1507 passed, 5 skipped. Preflight passed.
- `chat_screen.py` is at exactly its 24,595-line budget.

**Review round 3 (2026-11-25, delta 5ec1696d01..79355fe61f): bounded replay.**
- *Shape* (coordinator's call):
  - The auto-open reveal runs once, at the rail-becomes-visible seam.
  - Replays are bounded by time, not by tier. Telling the opening's network-tier landing apart would have meant exposing the tier from `environment.py`, and a missing `gh` never lands one. So: Environment landings within `FLEET_REVEAL_REPLAY_SECONDS` (5 s) of the reveal may redo it, at most `FLEET_REVEAL_REPLAY_LIMIT` (2) times (the opening's local tier, then its PR-and-checks tier). (Round 4 corrected two parts of this: the window was too short for `gh`, and a real poll can land inside it.)
  - Any user gesture in the rail stands it down for good: `_InspectorOuterBody.user_gestures` counts focus moving into the body, a mouse press and any section collapse toggle except the fleet's own one-shot open; `_handle_console_environment_row` calls `cancel_fleet_reveal()`; a moved offset counts as a user scroll.
- *I1:* the review saw a row expand moving the focused row out of the body. It is covered by the cancel, the gesture count and the bound. Tests: `test_a_gesture_in_the_rail_stops_the_reveal_for_good[row, row-in-flight, environment-collapse, agents-collapse, focus, mouse-down]`.
- *I2:* the replay carries the stored offset into the reveal, so a scroll inside the rebuild window is kept. Test: `test_environment_scroll_inside_the_rebuild_window_is_kept`.
- *M1 (stale in-flight reveal):* a token, bumped by every cancel and every new reveal, plus a check that the fleet rows are still present. Both are checked when the replay starts and when the reveal runs. A stale re-armed replay fails the token check, so `note_offset` needs no guard of its own. Test: `test_rows_ending_as_the_rail_opens_leave_nothing_armed`. With the rows-present check, the screen's `drop_fleet_reveal()` call went away.
- *Removed as redundant, so there is nothing left to mutate:*
  - the replay's `not section.open`: collapsing Agents is a `CollapseToggled` gesture, pinned by `[agents-collapse]`;
  - `note_offset()` before the scroll: Textual's `scroll_to_region` calls `on_complete` whether or not it scrolls;
  - the in-`_fleet_reveal_current` self-cancel: a reveal that may not scroll arms no replay.
- *Nits:*
  - The screen passes the rail body into `open_fleet_section_for_rows`.
  - The `waiting` countdown is gone: the replay chains `environment.call_next(tasks.call_next, ...)`.
  - The hidden test is one condition (`node.display is False`).
  - The lesson pointers and title are fixed.
  - `context-and-rag.md` describes the once-plus-early-landing behaviour in plain words.
- *Mutations*, all 20 reveal tests at `-n 4`; each named mutation fails at least one test:
  - M01 replay `preferred_right_open` -> the hand-reopen test;
  - M02 reveal-time check `if False:` -> the rebuild-window, rows-ended and row-in-flight tests;
  - M03 `hidden` -> `test_a_section_opened_in_view_owes_no_later_reveal`;
  - M04 the collapse-gesture count -> both collapse cases;
  - M05 the row cancel -> both row cases;
  - M06 the deadline and M07 the limit -> the bound test;
  - M08 rows-present -> the rows-ended and rows-ending tests;
  - M09 the gestures clause -> the collapse, focus and mouse-down cases;
  - M10 the offset clause -> the user-scroll and rebuild-window tests;
  - M11 the token -> row-in-flight;
  - M12 the seam `preferred_right_open` -> the manual-open test;
  - M13 the landing hook and M14 the wait behind rebuilds -> both landing nodes;
  - M15 the seam reveal -> 14 nodes;
  - M17 the focus count -> `[focus]`;
  - M18 the mouse count -> `[mouse-down]`.
  - M16 (the region retry) is still green in both runs this round. It failed the band test once in round 2's matrix and stays as a bounded load-only safety net.
  - The first round-3 matrix used a `-k` that left the rebuild-window test out. It and the tests' new landing sentinel (`_settle_landing`) were added before the final matrix.
- *Stability:*
  - the 20 reveal tests passed 10 of 10 runs at load 13-21, and 10 of 10 more with whole-file runs as generated load (load 18-23);
  - at load 22-85 an earlier pass went 8 of 10: two runs timed out the 180x48 auto-open wait at the default 5 s, with the section already in view when the wait gave up. The reveal waits now go through `_wait_reveal` (15 s).
- *Final whole files* at `-n 4`:
  - right_rail 56/56 (twice);
  - workbench 76/76. In one other run `test_composer_menu_opens_current_draft_directly_in_recommended_review[size1]` failed at load 52; that is TASK-41633's known flake, an action outside the 100x30 scroll.
  - agent_rail 51/51;
  - inspector_navigation 39/40, the shared responsive-reveal red.
- *Static checks:* `ruff --select F` clean; `ruff_gate.py` OK; size ratchets 238 passed; Tests/Architecture serially 1507 passed, 5 skipped; preflight passed.
- `chat_screen.py` stays at exactly its 24,595-line budget: the cancel line and the body argument are paid for by dropping the drop call and tightening the seam comment.

**Review round 4 (2026-11-25, delta 79355fe61f..2bbb2aa2b3).**
- *I1, no-op landings used up the budget:*
  - What went wrong: unchanged landings (the mount-time poll, focus regain, a TTL re-land) each spent a replay, so the PR tier found none left.
  - The fix: a reveal now checks whether the section is already inside the body and, if so, re-arms without scrolling or spending. Only a replay that scrolls spends one (`_FleetReveal.spends`; the opening's reveal does not), at most `FLEET_REVEAL_REPLAY_LIMIT` (2).
  - The window is 15 s, cross-referenced in the constant's comment to the ~12 s measured for `gh` in `environment.py`.
  - Red first on 2bbb2aa2b3: `test_landings_that_change_nothing_spend_no_replay` (two no-op landings, then the rich one).
  - *M3 wording:* the comment, the bound test (now `test_replays_are_bounded_and_end_with_the_window`), these notes and `context-and-rag.md` say "a landing that changes nothing never scrolls; nothing replays after the window".
- *I2(a):* the region retry runs whenever a landing arrives while the rail is hidden. The comment now says so, and `test_a_landing_while_hidden_is_revealed_once_laid_out` pins it.
- *I2(b):* `test_rows_and_the_rail_opening_in_one_tick` pins the fleet's own-open exclusion (rows and the rail opening with no pause between).
- *Found while pinning M1:* with 20 staged sources, the opening shrank the Sources list to its cap after the reveal ran, and the Agents section rose above the top (y=1 and y=-39, body 15-31). The reveal now also waits, within its 20-refresh bound, while `_InspectorOuterBody.layout_settling()` reports a re-fit pending (round 5 narrowed this to the rail's outer reconcile flag). Red first on 2bbb2aa2b3: `test_fleet_auto_open_scrolls_the_agents_section_into_view[150x40/20]`.
- *M1, inner scroll areas:* at the reveal, `_InspectorOuterBody.inner_scroll_state()` snapshots every `ScrollableContainer` in the body, and `inner_scrolled_since()` treats a moved offset as a gesture. (Round 5: a replaced widget is not skipped. It keeps its frozen last offset, since Textual 8.2.8 never clears `is_mounted`.)
  - Can a rebuild change an inner offset? Only from a non-zero offset (clamping); from 0 it cannot. A surviving area only leaves 0 by being scrolled, which is a gesture.
  - Tests: `test_a_scroll_inside_a_rail_scroll_area_stops_the_reveal` (the capped Sources list scrolled by 1) and `test_a_landing_rebuild_is_not_an_inner_scroll` (no false stand-down). Both use 20 staged sources, so the list scrolls by 20.
- *M2:* every reveal test takes the `frozen_reveal_clock` fixture; only the bound test moves the clock.
- *M4:* `arm` is one-shot per reveal, and `scroll_visible(immediate=True)` lets the first `on_complete` read the offset the scroll left. `test_one_reveal_arms_the_replay_once` counts exactly one arm per reveal; the review probe counted 6.
- *N1:* the replay's `rail_state.right_open` check is restored. `test_a_landing_after_a_manual_close_schedules_nothing` checks that nothing is scheduled.
- *N2:* the bound test alternates rich and lean landings and asserts exactly `FLEET_REVEAL_REPLAY_LIMIT` calls, with the third push-down left unreplayed.
- *Simplified:* I removed the run-time limit check; the replay entry already refuses an empty budget.
- *Gesture tests strengthened:* after I1, a landing that leaves the fleet in view calls nothing, so "never called" no longer detected a deleted focus or mouse guard (both mutations stayed green in the first round-4 matrix). The gesture test now also asserts that no replay is armed (`_fleet_replay is None`), and the non-row cases land the rich snapshot only after the gesture. The trap is in a new lesson, together with the cap re-fit.
- *Mutations, final matrix:* all 27 reveal tests at `-n 4`; every one of the 26 mutations fails at least one test, with none left unpinned.
  - Unchanged from round 3: M01-M15, M17 and M18.
  - M04b, the fleet-open exclusion -> the same-tick test.
  - M16, the region retry -> the hidden-landing and same-tick tests.
  - M19, the inner-scroll clause -> the scroll-area test.
  - M20, arm once -> the arms-once test.
  - M21, `immediate` -> 8 nodes.
  - M22, the opening spends a replay -> the bound test.
  - M23, in view costs nothing -> the no-op test.
  - M25, `right_open` -> the manual-close test.
  - M26, the settle wait -> the 20-row auto-open and both inner-scroll tests.
- *Red first on 2bbb2aa2b3 product code:* 6 of the new tests fail — 20-row auto-open, manual close, arms once, no-op landings, landing rebuild, scroll area.
- *Final runs:*
  - the 27 reveal tests passed 8 of 8 (load 12-25);
  - right_rail 64/64;
  - workbench 76/76;
  - agent_rail 51/51;
  - inspector_navigation 39/40, the shared responsive-reveal red.
  - `chat_screen.py` unchanged this round (24,595).

**Review round 5 (2026-11-26, delta 2bbb2aa2b3..c086cf52b6).**
- *I-1, a no-op landing spent a replay when Agents is taller than the body:* the in-view check now crops the section to the body's size first, `window.contains_region(section.region.crop_size(window.size))`, which is what `scroll_visible` can show. The test helper `_fleet_section_inside_rail_body` uses the same rule. Red first on c086cf52b6: `test_landings_that_change_nothing_spend_no_replay[150x30]`.
- *m-1, the settle wait ran out while paints were held:*
  - The fix: a wait attempt made while `_paints_held(app)` (the one helper reading Textual 8.2.8's private `App._batch_count`) does not count against the 20. That holds up to `FLEET_REVEAL_PAINT_HOLD_CAP_SECONDS = 2.0`, which is twice `hold_paints`' 1 s self-release.
  - The 190x40/20 case still ended in view on c086 in this environment. What it did fail was "no scroll while still settling", which the auto-open test now asserts for every case. With c086's controller the [190x40/20] case scrolled with the re-fit pending (`[True]`).
  - `test_a_paint_hold_that_never_ends_cannot_stall_the_reveal` forces a hold that never ends and a layout that never settles, and needs the cap to finish.
- *m-2:* I mutated each clause of `layout_settling` alone, across 32 reveal tests: outer-only passed and sections-only passed. I kept the outer flag; it stays scheduled while any bounded section reconciles. Returning False fails the 150x40/20 and 190x40/20 auto-opens and both inner-scroll tests.
- *m-3:*
  - `inner_scrolled_since` compares every snapshotted area, with no `is_mounted` filter. A replaced area keeps its frozen last offset, which reads as unmoved, or as moved if the user scrolled it before the rebuild. The docstring and class comment say so.
  - `_fleet_reveal_current` loses its dead `section.is_mounted` check.
  - There is a new lesson: `is_mounted` stays True after removal in Textual 8.2.8.
- *m-4:* `test_a_landing_rebuild_is_not_an_inner_scroll` restages the tray. It waits until the capped list is a new widget, then lands, and it fails under the fresh-compare mutation.
- *m-5:* `context-and-rag.md` now says "Refreshes that leave Agents in view never scroll or count toward that limit."
- *m-6:* `arm` checks the token. `test_a_cancel_beats_the_openings_late_on_complete` holds the reveal's `on_complete`, cancels, then fires it, and it fails on c086cf52b6.
- *Nits:*
  - n-1: the bound-test docstring and message describe the real sequence: rich #0 and lean #1 scroll, rich #2 is not replayed.
  - n-2: the 15 s comment cites the local tier, `_GH_TIMEOUT_SECONDS` (5.0) and the ~12 s measurement, and says not to shrink it to 5 s.
  - n-3: `arm` no longer re-snapshots the inner areas. The reveal's own scroll moves only the body, because the section is the body's direct child.
  - n-4: the rail injects `layout_settling=self._outer_layout_settling` like `on_geometry_changed`; no more `getattr` on the parent.
  - n-5: `inner` is typed.
  - n-6: the post-wait `if not section.region: return` is pinned by `test_a_reveal_whose_rail_closed_again_shows_nothing`, where the rail closes in the same tick.
  - n-7: the scroll-area test runs at 160x70 with 40 sources, so the capped list is on screen. It posts a real `MouseScrollDown` to it.
- *Mutations, round 5:* 35 mutations, each run across all 32 reveal tests at `-n 4`; every one fails at least one test.
  - M01-M26 as before; M23b (an in-view landing spends) fails both no-op cases.
  - M27 (outer flag returns False) -> 5 tests.
  - M28 (no crop) -> `[150x30]`.
  - M29 (held attempts count) -> `[190x40/20]`.
  - M30 (no hold cap) -> the never-ending-hold test.
  - M31 (no arm token) -> the late `on_complete` test.
  - M32 (no post-wait region return) -> the closed-again test.
  - M33 (fresh compare) -> the same-tick and rebuild tests.
  - M34 (empty opening snapshot) -> the scroll-area test.
- *Red first on c086cf52b6:* `[150x30]`, the late-`on_complete` test, `[190x40/20]` (with c086's controller) and the never-ending-hold test fail.
- *Final runs:*
  - the 32 reveal tests passed 5 of 5 at load 24-29;
  - right_rail 69/69;
  - workbench 76/76;
  - agent_rail 51/51;
  - inspector_navigation 39/40, the shared responsive-reveal red.
  - ruff F clean; ruff gate OK; size ratchets 238 passed; `chat_screen.py` unchanged (24,595).

**Review round 6 (2026-11-26, coordinator review of 6861880954).**
- *Held attempts spun the loop.* Under a real `batch_update` hold, `Screen._on_idle` runs after-refresh callbacks at once, so `call_after_refresh(run, attempts)` with an unchanged count re-ran as fast as the loop could go. A held attempt now re-checks on a 0.02 s `set_timer`; counted attempts still use `call_after_refresh`.
  - Measured with `_paints_held` wrapped at 190x40/20, with no forced hold: 23-25 wait runs, 15 of them held (before), against 13 wait runs, 2 held (after).
  - With a real 0.5 s `_begin_batch()` hold: 2,549 held runs before, 14 after.
  - `test_a_paint_hold_is_waited_out_on_a_timer_not_a_spin` bounds the held runs at 60 and checks the reveal still lands. It fails on 6861880954 with 2,454.
- *Nits.* Cut the test-history sentence from `_outer_layout_settling`'s docstring. Earlier rounds recorded that each clause alone kept every test green and that dropping the outer flag failed 5. `hold_cap` now carries a comment saying it reads real time on purpose.

**CI note.** `scripts/ci/tier_policy.py` selects the full lanes for this diff, not the light tier. `test_console_workbench_contract.py` has more than `IMPORTER_CAP` (8) transitive test importers.

**Docs.** `Docs/User_Guide/console/context-and-rag.md` lists the new Inspector order in both places that gave the old one.

Files: `tldw_chatbook/UI/Console_Modules/right_rail.py`, `tldw_chatbook/UI/Console_Modules/agent.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `Docs/User_Guide/console/agent-runs-and-tools.md`, `Docs/superpowers/specs/2026-09-04-console-inspector-environment-redesign-design.md`, `Tests/UI/console_rail_section_helpers.py`, `Tests/UI/test_console_workbench_contract.py`, `Tests/UI/test_console_right_rail.py`, `Tests/UI/test_console_live_work_handoffs.py`, `Docs/User_Guide/console/context-and-rag.md`, `backlog/docs/lessons-testing-evidence.md` (closed-Inspector trap at 118-128 columns).
<!-- SECTION:NOTES:END -->
