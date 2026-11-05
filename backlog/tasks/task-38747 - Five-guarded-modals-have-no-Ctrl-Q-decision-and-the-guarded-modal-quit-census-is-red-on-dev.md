---
id: TASK-38747
title: "Five guarded modals have no Ctrl+Q decision and the guarded-modal quit census is red on dev"
status: To Do
assignee: []
created_date: 2026-11-04 06:34
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
- [ ] #1 Each of the five modals has a Ctrl+Q decision: either a `confirm_quit` that refuses while an operation runs (`refuse_quit_while_working`) or confirms before discarding edits (`confirm_quit_discarding_edits`), with a CONFIRM_QUIT row; or an exemption row that says why quitting there loses nothing
- [ ] #2 For each modal given a `confirm_quit`, a test presses Ctrl+Q while the guarded state holds and shows that the quit is refused or confirmed; each test is red before the change
- [ ] #3 `test_guarded_modal_quit_census.py` passes serially
- [ ] #4 The census runs on a PR lane, so the next modal that guards its close without a Ctrl+Q decision turns that PR red; the placement and its measured time are recorded in the notes
<!-- AC:END -->
