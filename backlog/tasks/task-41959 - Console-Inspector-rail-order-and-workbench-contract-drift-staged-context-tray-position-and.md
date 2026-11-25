---
id: TASK-41959
title: "Console Inspector rail order and workbench contract drift: staged-context tray position and five workbench nodes"
status: Done
assignee: []
created_date: '2026-11-19 07:30'
updated_date: '2026-11-25 15:40'
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
  - `open_fleet_section_for_rows` records a pending reveal when it opens the section behind a closed rail.
  - `reveal_fleet_after_rail_auto_open` runs at the screen's one rail-becomes-visible seam. It retries after refresh until the section has a region, then scrolls it into view with a minimal scroll.
  - A rail the user opened (`preferred_right_open`) keeps its scroll, and so does one whose busy window ended. In both cases the pending reveal is dropped.
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

**CI note.** `scripts/ci/tier_policy.py` selects the full lanes for this diff, not the light tier. `test_console_workbench_contract.py` has more than `IMPORTER_CAP` (8) transitive test importers.

**Docs.** `Docs/User_Guide/console/context-and-rag.md` lists the new Inspector order in both places that gave the old one.

Files: `tldw_chatbook/UI/Console_Modules/right_rail.py`, `tldw_chatbook/UI/Console_Modules/agent.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `Docs/User_Guide/console/agent-runs-and-tools.md`, `Docs/superpowers/specs/2026-09-04-console-inspector-environment-redesign-design.md`, `Tests/UI/test_console_workbench_contract.py`, `Tests/UI/test_console_right_rail.py`, `Tests/UI/test_console_live_work_handoffs.py`, `Docs/User_Guide/console/context-and-rag.md`, `backlog/docs/lessons-testing-evidence.md` (closed-Inspector trap at 118-128 columns).
<!-- SECTION:NOTES:END -->
