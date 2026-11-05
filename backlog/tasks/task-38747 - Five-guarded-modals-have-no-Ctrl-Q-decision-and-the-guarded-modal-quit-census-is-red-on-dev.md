---
id: TASK-38747
title: "Five guarded modals have no Ctrl+Q decision and the guarded-modal quit census is red on dev"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 06:34
updated_date: '2026-11-05 22:20'
labels:
- follow-up
- architecture
- tests
- quit-safety
priority: high
references:
- Tests/Architecture/test_guarded_modal_quit_census.py
- tldw_chatbook/Widgets/quit_while_working.py
- tldw_chatbook/Widgets/confirmation_dialog.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

`Tests/Architecture/test_guarded_modal_quit_census.py::test_every_guarded_modal_close_has_a_ctrl_q_decision` (:532) is red on dev. Run serially through the scratch-profile wrapper, it fails at :535. It names five modals whose close is guarded, but which have no Ctrl+Q decision in `GUARDED_CLOSE_DECISIONS` (:63):
- `Widgets/Console/console_goal_setup_modal.py`: `ConsoleGoalSetupModal.action_cancel` (:255);
- `Widgets/Console/console_hooks_review_modal.py`: `ConsoleHooksReviewModal._perform_safe_cancel` (:521);
- `Widgets/Console/response_rules_modal.py`: `ResponseRulesModal._perform_safe_cancel` (:689);
- `Widgets/Persona_Widgets/portrait_composition_review.py`: `PortraitCompositionReview.action_close` (:349);
- `Widgets/Persona_Widgets/portrait_region_editor.py`: `PortraitRegionEditor.action_cancel` (:459).

None of the five defines `confirm_quit`. Ctrl+Q is a priority binding (TASK-33622.10), so it quits straight past each modal's own guard. An operation in flight, or an unsaved edit, is lost without the confirmation that Escape or Cancel gives in the same modal.

**Which lane runs `Tests/Architecture`.** No PR lane runs the directory whole. `derived-artifacts.yml` names 17 of its 78 files individually, and this census is not one of them. Only the scheduled `nightly-deep.yml` and `test.yml` (on a push to main, or on dispatch) collect it, with the whole `Tests/` tree. That is how it went red unnoticed. The census file was last changed on 2026-10-04 (060919ae72); the five modal files were added between 2026-09-28 and 2026-10-24.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the five modals has a Ctrl+Q decision: either a `confirm_quit` that refuses while an operation runs (`refuse_quit_while_working`) or confirms before discarding edits (`confirm_quit_discarding_edits`), with a CONFIRM_QUIT row; or an exemption row that says why quitting there loses nothing
- [x] #2 For each modal given a `confirm_quit`, a test presses Ctrl+Q while the guarded state holds and shows that the quit is refused or confirmed; each test is red before the change
- [x] #3 `test_guarded_modal_quit_census.py` passes serially
- [x] #4 The census runs on a PR lane, so the next modal that guards its close without a Ctrl+Q decision turns that PR red; the placement and its measured time are recorded in the notes
<!-- AC:END -->

## Implementation Plan

1. Read each of the five close guards and decide: a `confirm_quit` where Escape refuses or asks (goal setup while its launch runs, region editor while it saves, response rules while a test or save runs and before unsaved editor text is discarded); an exemption where the census sees a path that never holds anything (hooks review's `_submit_result` keeps the review open only for a ready result, the composition review is read-only).
2. Add the hooks with the lazy imports of the existing precedents, and the rows to `GUARDED_CLOSE_DECISIONS`.
3. Real-app Ctrl+Q tests in `Tests/UI/test_app_quit_in_flight_modals.py`, one per hooked modal, each shown red before its hook; stand-in rows in the lane-run `Tests/UI/test_modal_quit_in_flight_hooks.py`.
4. Run the census serially; append it to the fast PR contract lane in `derived-artifacts.yml` and to `FAST_LANE_TARGETS`; record the measured time.

## Implementation Notes

**Decisions, one per census row.**
- `ConsoleGoalSetupModal.action_cancel`: Escape and Cancel are refused while the launch runs (`_busy`: the review `configure` call or the `start` call). New `confirm_quit` refuses through `refuse_quit_while_working` ("The goal is still being set up."). When idle, Escape drops the setup without asking, so the quit goes ahead. CONFIRM_QUIT row.
- `PortraitRegionEditor.action_cancel`: Escape and Cancel are refused while the regions save (`_saving`). New `confirm_quit` refuses ("The portrait regions are still being saved."). When idle, Cancel drops unsaved regions without asking. CONFIRM_QUIT row.
- `ResponseRulesModal._perform_safe_cancel`: while a test or save runs (`_busy`), Escape stops a test and stays; with unsaved editor text, Escape asks "Keep editing?". New `confirm_quit` refuses while `_busy`, and otherwise asks through `confirm_quit_discarding_edits` when `_has_unsaved()` ("Your edits to this response rule are not saved."). CONFIRM_QUIT row.
- `ConsoleHooksReviewModal._perform_safe_cancel`: exemption. The scan reads `_submit_result` as a path that stays open, but only a `ready` result stays open; the cancel always reaches `dismiss_safe_once`, busy or not, so Escape always closes.
- `PortraitCompositionReview.action_close`: exemption. It is a read-only review of saved frames ("without side effects"); `retire()` closes it whenever it is mounted, and the unmounted branch has nothing to close.

The hooks import their helpers lazily, as the existing ones do, so the quit-anyway fixture's patch of `confirmation_dialog.confirm_quit_discarding_edits` reaches them.

**Tests.**
- `Tests/UI/test_app_quit_in_flight_modals.py` (real `TldwCli`, real Ctrl+Q): `test_ctrl_q_waits_for_a_goal_that_is_starting`, `test_ctrl_q_waits_for_portrait_regions_being_saved`, `test_ctrl_q_waits_for_a_rule_test_then_asks_before_discarding_edits`. Each holds the operation, presses Ctrl+Q and sees "still working" with no quit. It then lets the operation finish, and Ctrl+Q quits. The rule test then also gets "Discard changes and quit?"; Keep editing keeps the text, and the confirm quits. Red before the hooks, all three on the `cleanups == []` assertion ("Ctrl+Q quit straight past ..."). The region editor awaits its save inside the Save button's handler, so the modal's message queue is busy for the whole save, and Pilot's press and pause (which wait for that queue) time out. That test sends Ctrl+Q through `App._press_keys` and polls with plain sleeps until the save ends. It also releases the held save in a `finally`, because a held save kept the app from shutting down after a failure (its first red run hung to the 300 s timeout).
- `Tests/UI/test_modal_quit_in_flight_hooks.py` (a UI-census file, so on a PR lane): the goal setup and region editor join `_IN_FLIGHT` (still working, quit anyway on a repeat, idle quits). `test_response_rules_ctrl_q_waits_for_a_test_and_asks_before_dropping_edits` covers the rule modal's three answers. Red under a temporary edit that made each hook's guard always pass: 5 failed (the 4 `_IN_FLIGHT` cases and the rule test), 36 passed. Green: 41 passed.

**Census.** `test_guarded_modal_quit_census.py`: 1 failed (the five rows) at base, 6 passed serially after.

**Lane (AC#4).** The census is appended to the sandboxed fast PR contract step (`pr-fast-lane`, "Run fast PR contract", serial, `--timeout=180`), after TASK-38702's census riders, and to `FAST_LANE_TARGETS`. Measured with that step's flags (`pytest -p no:xdist -p no:cov Tests/Architecture/test_guarded_modal_quit_census.py --timeout=180 --tb=short`, through the scratch wrapper, load average 31-33): 3 of 3 passed, 5.06 s, 3.43 s and 3.70 s in pytest (6.6-8.8 s wall). The scan itself takes 2.7-4.0 s. The other quit-flow censuses (`test_flush_screens_have_quit_hooks.py`, `test_quit_flow_prompt_choke_point.py`) are already in that step. `Tests/CI/test_ci_queue_pressure_contract.py` and `test_derived_artifacts_workflow.py`: 173 passed.

**Covering runs** (`-n 4`, through the scratch wrapper, load average 8-40). `test_app_quit_in_flight_modals.py` 16 passed; `test_modal_quit_in_flight_hooks.py` 41 passed; serially, the census, `test_quit_flow_prompt_choke_point.py` and `test_flush_screens_have_quit_hooks.py` 15 passed. Each file run on its own: goal setup 14, goal command 5, goal navigation 5, `test_response_rules_modal.py` 25, response-rules end to end 5, composition review 11, modal wide tier 131, close under the quit question 12, discard guards 29, hooks review 14 passed. Two results are red on dev too, with the same command and the three modal files reverted to the base:
- `test_console_response_rules.py::test_active_run_copy_takes_priority_over_an_earlier_rule_verdict` fails (`'Waiting for reply…' == 'Streaming response.'`).
- `test_portrait_region_editor.py` fails `test_dialog_validation_is_coalesced_owned_and_drained[...]` and then hangs, in 2 of 3 runs at the base and 2 of 3 after; the third run passes 25 of 25.

Neither file is on a PR lane.

**Docs.** `Docs/User_Guide/index.md`'s Ctrl+Q row names starting a goal, saving portrait regions, testing or saving a response rule, and a response rule's unsaved edits.

**Lesson.** `backlog/docs/lessons-testing-evidence.md`: a modal that awaits its work in a message handler blocks Pilot and app shutdown.

**Files.** `tldw_chatbook/Widgets/Console/console_goal_setup_modal.py`, `tldw_chatbook/Widgets/Console/response_rules_modal.py`, `tldw_chatbook/Widgets/Persona_Widgets/portrait_region_editor.py`, `Tests/Architecture/test_guarded_modal_quit_census.py`, `Tests/UI/test_app_quit_in_flight_modals.py`, `Tests/UI/test_modal_quit_in_flight_hooks.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Docs/User_Guide/index.md`.
