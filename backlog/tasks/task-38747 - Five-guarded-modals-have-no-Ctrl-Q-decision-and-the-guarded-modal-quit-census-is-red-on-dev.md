---
id: TASK-38747
title: "Five guarded modals have no Ctrl+Q decision and the guarded-modal quit census is red on dev"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 06:34
updated_date: '2026-11-06 01:56'
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
- `ResponseRulesModal._perform_safe_cancel`: while a test, save or pin change (Disable/Enable, Exclude, Delete local pin) runs (`_busy`), Escape stops a test and stays; with unsaved editor text, Escape asks "Keep editing?". New `confirm_quit` refuses while `_busy` ("The response rule change is still being tested or saved."), and otherwise asks through `confirm_quit_discarding_edits` when `_has_unsaved()` ("Your edits to this response rule are not saved."). CONFIRM_QUIT row.
- `ConsoleHooksReviewModal._perform_safe_cancel`: exemption. The scan reads `_submit_result` as a path that stays open, but only a `ready` result stays open; the cancel always reaches `dismiss_safe_once`, busy or not, so Escape always closes.
- `PortraitCompositionReview.action_close`: exemption. It is a read-only review of a draft's candidate frames (`draft.composition`, held in memory). The draft stays with the Personas screen, and the review holds no edit or write. `retire()` closes it whenever it is mounted, and the unmounted branch has nothing to close.

The hooks import their helpers lazily, as the existing ones do, so the quit-anyway fixture's patch of `confirmation_dialog.confirm_quit_discarding_edits` reaches them.

**Tests.**
- `Tests/UI/test_app_quit_in_flight_modals.py` (real `TldwCli`, real Ctrl+Q): `test_ctrl_q_waits_for_a_goal_that_is_starting`, `test_ctrl_q_waits_for_portrait_regions_being_saved`, `test_ctrl_q_waits_for_a_rule_test_then_asks_before_discarding_edits`. Each holds the operation, presses Ctrl+Q and sees "still working" with no quit. It then lets the operation finish, and Ctrl+Q quits. The rule test then also gets "Discard changes and quit?"; Keep editing keeps the text, and the confirm quits. Red before the hooks, all three on the `cleanups == []` assertion ("Ctrl+Q quit straight past ..."). The region editor awaits its save inside the Save button's handler, so the modal's message queue is busy for the whole save, and Pilot's press and pause (which wait for that queue) time out. That test sends Ctrl+Q through `App._press_keys` and polls with plain sleeps until the save ends. It also releases the held save in a `finally`, because a held save kept the app from shutting down after a failure (its first red run hung to the 300 s timeout).
- `Tests/UI/test_modal_quit_in_flight_hooks.py` (a UI-census file, so on a PR lane): the goal setup and region editor join `_IN_FLIGHT` (still working, quit anyway on a repeat, idle quits). `test_response_rules_ctrl_q_waits_for_a_test_and_asks_before_dropping_edits` covers the rule modal's three answers. Red under a temporary edit that made each hook's guard always pass: 5 failed (the 4 `_IN_FLIGHT` cases and the rule test), 36 passed. Green: 41 passed.

**Census.** `test_guarded_modal_quit_census.py`: 1 failed (the five rows) at base, 6 passed serially after.

**Lane (AC#4).** The census is appended at the end of the "Run Dreams and Guardian boundary contracts" step (`dreams-boundary` in `derived-artifacts.yml`; serial, `--timeout=180 --tb=short`). It is pinned in `DREAMS_BOUNDARY_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`) and in that step's argv pin in `Tests/CI/test_response_rules_critical_gate.py`. It is not in `pr-fast-lane`: review r8g-rev1 read 54 runs of that job (2026-11-04/05), median 32.0 and max 34.8 min of its 35, with 8 cancelled at the cap. `dreams-boundary` runs median 6.7 and max 7.8 of its 20, as with TASK-36532. That step uses the same install line and flags as the fast lane, so the census runs in its sandbox unchanged. Measured with the step's flags (`pytest -p no:xdist -p no:cov Tests/Architecture/test_guarded_modal_quit_census.py Tests/Chat/test_console_rail_state_prune.py --timeout=180 --tb=short`, through the scratch wrapper, load average 15-16): 3 of 3 passed, 15 tests each, 4.55 s, 3.81 s and 3.07 s in pytest. The census scan itself took 2.58 s, 2.50 s and 1.78 s. The whole step, run once locally with both riders in one process: 238 passed in 141 s. `Tests/CI` (`-n 4`): 1211 passed, 1 skipped. (First placed on `pr-fast-lane`, where it measured 3.43-5.06 s; moved on review.)

**Covering runs** (`-n 4`, through the scratch wrapper, load average 8-40). `test_app_quit_in_flight_modals.py` 16 passed; `test_modal_quit_in_flight_hooks.py` 41 passed; serially, the census, `test_quit_flow_prompt_choke_point.py` and `test_flush_screens_have_quit_hooks.py` 15 passed. Each file run on its own: goal setup 14, goal command 5, goal navigation 5, `test_response_rules_modal.py` 25, response-rules end to end 5, composition review 11, modal wide tier 131, close under the quit question 12, discard guards 29, hooks review 14 passed. Two results are red on dev too, with the same command and the three modal files reverted to the base:
- `test_console_response_rules.py::test_active_run_copy_takes_priority_over_an_earlier_rule_verdict` fails (`'Waiting for reply…' == 'Streaming response.'`).
- `test_portrait_region_editor.py` fails `test_dialog_validation_is_coalesced_owned_and_drained[...]` and then hangs, in 2 of 3 runs at the base and 2 of 3 after; the third run passes 25 of 25.

Neither file is on a PR lane.

**Docs.** `Docs/User_Guide/index.md`'s Ctrl+Q row names starting a goal, saving portrait regions, testing, saving or changing a response rule, and a response rule's unsaved edits. The portrait regions save also joins the row's list of dialogs that close themselves when done, and the matching list in `Widgets/quit_while_working.py`'s docstring. The editor's save closes through `dismiss_safe_once_when_on_top`, so a close made under the quit question finishes after Wait. No test covers that path for this editor; `test_close_under_quit_question.py` pins the helper.

**Review of #3418 (Qodo): each operation gets its own first refusal.** `refuse_quit_while_working` remembered the refused activity for the modal's lifetime. So after one refused Ctrl+Q during a response-rule operation, the first Ctrl+Q during a later operation asked "Quit anyway" instead of refusing. New `forget_quit_refusal(screen)` in `Widgets/quit_while_working.py` clears it, and does nothing if nothing is remembered. It is called where each operation ends:
- `ResponseRulesModal._action`'s `finally` (test, save, enable/disable, exclude, delete);
- `ConsoleGoalSetupModal.launch`'s `finally`;
- `PortraitRegionEditor._pressed`'s save `finally`;
- `ConsolePromptsModal.on_worker_state_changed`, when the mix-in save worker finishes (TASK-38748's code, edited here).

The module's "Known limits" paragraph now says which modals forget it and lists the 13 other call sites, which still remember it. `test_the_next_operations_first_ctrl_q_refuses_again` (4 cases, in the lane-run `test_modal_quit_in_flight_hooks.py`) runs each modal's own operation code on a stand-in: refuse, finish, start again, refuse again without asking. Red with the helper made a no-op (4 failed). Red with each call site removed alone: only that modal's case failed.

**Lesson.** `backlog/docs/lessons-testing-evidence.md`: a modal that awaits its work in a message handler blocks Pilot and app shutdown.

**Files.** `tldw_chatbook/Widgets/Console/console_goal_setup_modal.py`, `tldw_chatbook/Widgets/Console/response_rules_modal.py`, `tldw_chatbook/Widgets/Persona_Widgets/portrait_region_editor.py`, `Tests/Architecture/test_guarded_modal_quit_census.py`, `Tests/UI/test_app_quit_in_flight_modals.py`, `Tests/UI/test_modal_quit_in_flight_hooks.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_response_rules_critical_gate.py`, `tldw_chatbook/Widgets/quit_while_working.py` (docstring), `Docs/User_Guide/index.md`.
