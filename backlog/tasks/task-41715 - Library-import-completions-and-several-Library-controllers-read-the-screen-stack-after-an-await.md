---
id: TASK-41715
title: "Library import completions and several Library controllers read the screen stack after an await"
status: Done
assignee: ['@claude']
created_date: 2026-11-18 06:57
updated_date: 2026-11-19 16:39
labels:
- follow-up
- ui
- library
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41702's review (round 16). TASK-41702 guarded Home's and Library's footer reads of `app.screen` once a screen is detached at shutdown. The same `self.is_mounted and self.app.screen ...` shape is in five Library import completions (library_screen.py ~26270, 26299, 26324, 27827, 28024) and in controller sites (library_artifacts_controller.py ~206/992, library_artifacts_share_controller.py ~89, library_artifacts_navigation.py ~99, library_navigation_controller.py ~144, library_ingest_controller.py ~2221, library_note_autosave.py ~226, library_prompts_controller.py ~1535/3033). Each can raise `ScreenStackError` if it completes after the stack empties. Some tests reach these through SimpleNamespace fakes (`Tests/Skills/test_skills_import.py`, `Tests/UI/test_library_canvas_scoped_sync.py`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 None of these sites raises once its screen is detached after the exit fence, and each still applies its result while its screen is attached and current
- [x] #2 A test steps at least one completion past the fence and is red before the change
- [x] #3 The fakes the change touches are updated
<!-- AC:END -->

## Implementation Plan

1. At each listed site, read `is_attached` before `app.screen` (the TASK-41702 shape: `is_mounted and is_attached and app.screen is ...`). Once the exit fence is set every node reads detached, so the stack is never read; while the screen is attached and current the result still applies.
2. Leave sites that already cannot raise (an `except Exception` around the read, or an `is_attached` check ahead of it) and say so in the notes.
3. Add `is_attached=True` to the SimpleNamespace fakes that reach a changed site.
4. Add a test that steps an import completion past the fence with the stack emptied, red before the change.

## Implementation Notes

Each listed site now reads `is_attached` before `app.screen`, the TASK-41702 shape (`is_mounted and is_attached and app.screen is ...`). Shutdown sets the exit fence before Textual empties the stack, and every node reads detached from then on, so the stack is never read past the fence. While the screen is attached and current the condition is unchanged and the result still applies.

- **Sites changed.** `library_screen.py`: `_library_skills_import_row_visible`, `_present_library_skills_import_snapshot`, `_present_library_skills_import_choice_if_needed`, `_apply_library_prompts_import_status`, `_run_library_prompts_import` (the five import completions). Controllers: `library_artifacts_controller.py` (`presentable`, `_scripts_closed`), `library_artifacts_share_controller.py` (`_active`), `library_artifacts_navigation.py` (`resume`), `library_navigation_controller.py` (`present_pending_repair`), `library_note_autosave.py` (`resume_library_note_autosave`), `library_prompts_controller.py` (`_report_library_prompt_action_result`).
- **Sites left alone, already safe.** `library_ingest_controller.py` (`_library_ingest_submit_may_fire`) reads the stack inside `except Exception: return False`. `library_prompts_controller.py`'s membership focus restore checks `not opener.is_attached` ahead of its stack read.
- **Size pins (review r17ls-rev1).** The first commit squeezed each check onto the `is_mounted` line to stay inside the ratchets. Each check now has its own line, like the rest of each chain, and the five ratcheted modules' pins were raised by the measured growth, with dated comments naming this task: `library_screen.py` 32823 -> 32828 (`Tests/Architecture/test_screen_size_ratchet.py`), `library_artifacts_controller.py` 1063 -> 1065, `library_artifacts_share_controller.py` 339 -> 340, `library_navigation_controller.py` 202 -> 203 and `library_prompts_controller.py` 5068 -> 5069 (`Tests/Architecture/test_library_modules_size_ratchet.py`). `library_artifacts_navigation.py` (215 -> 216) and `library_note_autosave.py` (278 -> 279) are not ratcheted.
- **Fakes (AC#3).** `is_attached=True` on the SimpleNamespace screens in `Tests/Skills/test_skills_import.py` (2), `Tests/UI/test_library_canvas_scoped_sync.py` (2), `Tests/UI/test_library_artifact_navigation.py`, `Tests/UI/test_library_character_repair.py`, and on `_ResumableScreen` in `Tests/Library/test_library_note_autosave_max_wait.py`. Without them 56 nodes failed on `AttributeError: ... no attribute 'is_attached'`.
- **Test.** `Tests/UI/test_screen_navigation.py::test_library_completions_after_the_exit_fence_skip_the_empty_stack`, after TASK-41702's test and with its window (set `_exit`, clear the stack, restore). A prompts import (one JSON prompt) runs with its lookup and save stubbed for its own task only; the save enters the window and returns without yielding, so the completion reaches its stack check inside the window. It must apply nothing (`import_path` unchanged). Then eleven readers are called inside the window: the prompts status, the skills import row check, snapshot and choice, the Notes autosave resume, the prompt action result, the artifacts share `_active`, the artifacts controller's `presentable` and `_scripts_closed` (a controller built for the test, with an Artifacts row selected), `LibraryArtifactsNavigation.resume` (`_eligible` set) and `present_pending_repair` (a pending context set). That state is set only for the window and restored after it.
- **Red/green.** With the seven source files at base: the test failed on its assertion, all seven readers raising `ScreenStackError('No screens on stack')`, the prompts import completion first. With the change: passed (serially, load 44-49). After review r17ls-rev1 added the five readers, the same base run listed all twelve (the completion and eleven readers) raising `ScreenStackError`; with the guards it passed (serially, load 36-38).
- **Covering runs.** `Tests/Skills/test_skills_import.py`, `Tests/UI/test_library_canvas_scoped_sync.py`, `Tests/Library/test_library_note_autosave_max_wait.py`, `Tests/UI/test_library_artifact_navigation.py`, `Tests/UI/test_library_artifacts_sharing.py`, `Tests/UI/test_library_character_repair.py` at `-n 4`: head 27 failed / 108 passed, base 29 failed / 106 passed. Every failure at both is the local `RecoveryRequired: raw_source_selection_changed` trap, except `test_notes_per_click_updates_keep_screen_and_canvas_identity` (a `NoMatches` at load 62 at head; 3 of 3 passed serially) and three `test_library_character_repair` nodes that failed at base only.
- **Lane.** A tail entry in `scripts/ui_pr_gate_census.txt` next to TASK-41702's node (floor 2279): 10.0 s call serially with the lane's flags at load 63-65.

**Files.** `tldw_chatbook/UI/Screens/library_screen.py`; `tldw_chatbook/UI/Library_Modules/{library_artifacts_controller,library_artifacts_share_controller,library_artifacts_navigation,library_navigation_controller,library_note_autosave,library_prompts_controller}.py`; `Tests/UI/test_screen_navigation.py`; the five fake-bearing test files above; `Tests/Architecture/test_screen_size_ratchet.py`; `Tests/Architecture/test_library_modules_size_ratchet.py`; `scripts/ui_pr_gate_census.txt`; `scripts/check_ui_pr_gate_census.py`.
