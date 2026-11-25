---
id: TASK-41959
title: "Console Inspector rail order and workbench contract drift: staged-context tray position and five workbench nodes"
status: Done
assignee: []
created_date: '2026-11-19 07:30'
updated_date: '2026-11-25 14:20'
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
- `test_console_inspector_prioritizes_actionable_status_before_secondary_groups` (red in every local run, alone and whole-file, despite the hand-off saying it passed): Review approval stayed disabled. Since 55f86c9538 (#2953) a mounted Console counts the controller's pending rounds plus a legacy tool card. `app.console_pending_approval_count` is only the no-controller fallback. The test now stages a real pending card. It also opens the rail, because TASK-37000 holds a closed rail's rebuild: with the card staged and the rail closed, the count read 1 and the button stayed disabled for 5 s (probe).
- `test_console_approvals_chip_activation_focuses_the_decision_not_submit`: a2fcbe61bc (#3036) moved the keyboard landing from the row's decision Select (now behind More options) to the neutral `approval-request-summary`. TASK-40097 re-pinned the card-level test the same way. Run alone, focus landed there. In the whole-file `-n 4` run it was still on the status line after the fixed 0.1 s pauses, so the test now waits (10 s cap) for the card to display and for focus to leave the line. It still refuses `approval-submit`.

**Evidence** (dev 2959b48314, scratch-profile wrapper, load 17-56).
- Red first: `test_console_left_rail_keeps_session_and_moves_staged_context_out` and `test_console_inspector_hosts_staged_context_above_source_readiness` failed on dev (`ConsoleInspectorSection(id='console-agent-section-subagents') is ConsoleStagedContextTray`).
- Negative control: with dev's `right_rail.py` restored over the fix, 4 of the 5 order nodes failed: both tray tests, the right-rail census and the handoffs swap test. The fifth, the static inventory tuple, is a pure literal. The fix was then restored byte-for-byte.
- Whole files at `-n 4`, base -> head:
  - `test_console_workbench_contract.py`: 4 failed/72 passed -> 76/76.
  - `test_console_right_rail.py`: 37/37 -> 37/37.
  - `test_console_live_work_handoffs.py`: 6 failed/74 passed -> 5 failed/75 passed. All 5 are base reds (workflows/artifacts destinations, staged launch across screen recreation, WC action button). `..._keeps_console_launch_disabled_without_active_run` failed on base only.
  - `test_console_session_settings.py`: 9 failed + 8 errors -> 8 failed + 8 errors. The tray test went green. Every remaining node is listed in TASK-41962 (F6 reds and F11 teardown errors).
- Fleet neighbours, run whole on head; every red is shared with base:
  - `test_console_collapsed_inspector_send_layout.py`: 3/3 passed.
  - `test_console_inspector_navigation.py`: 39 passed, 1 failed. `test_responsive_hide_and_reveal_hands_off_focus_without_losing_offset` also fails on dev's `right_rail.py`.
  - `test_console_fleet_panel.py` and `test_console_environment_wiring.py` have no `bootstrap_profile` marker, so every node hit the config-sandbox trap. With a throwaway plugin that adds the marker:
    - fleet_panel: 11/12 passed. The one red writes an evidence SVG to a missing `.superpowers/` folder.
    - environment_wiring: 47/49 passed. `test_row_activation_toggles_expansion_and_rerenders` and `test_unbound_landing_replaces_the_previous_repos_paint` also fail with dev's `right_rail.py`.
- Static checks: `ruff --select F` clean; `ruff_gate.py` OK; size ratchets 238 passed; preflight passed.

**CI note.** `scripts/ci/tier_policy.py` selects the full lanes for this diff, not the light tier. `test_console_workbench_contract.py` has more than `IMPORTER_CAP` (8) transitive test importers.

**Docs.** `Docs/User_Guide/console/context-and-rag.md` lists the new Inspector order in both places that gave the old one.

Files: `tldw_chatbook/UI/Console_Modules/right_rail.py`, `Tests/UI/test_console_workbench_contract.py`, `Tests/UI/test_console_right_rail.py`, `Tests/UI/test_console_live_work_handoffs.py`, `Docs/User_Guide/console/context-and-rag.md`, `backlog/docs/lessons-testing-evidence.md` (closed-Inspector trap at 118-128 columns).
<!-- SECTION:NOTES:END -->
