---
id: TASK-1612
title: Bench authoring copy polish batch
status: Done
assignee: []
created_date: '2026-07-31 15:10'
updated_date: '2026-08-01 02:26'
labels:
  - evals
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Copy findings from the task-1482 whole-branch review, none blocking: (1) a rename collision renders the DB's "Task name already exists" — wrong vocabulary ("Task" not "bench") and, worse, after deleting a bench its name stays reserved (UNIQUE has no deleted_at exemption) so the message can appear with no visible bench of that name — explain the trap ("a deleted bench may still hold this name"); the pinned test only asserts "already exists" so copy can change freely. (2) The zero-target blocked reason still reads "This bench has no targets yet" right after the user STAGES one — nothing says Save is the arming step; append "…and Save". (3) llama_targets() silently uses list_models' default limit=100 unlike the documented _LIST_LIMIT reads — align for consistency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rename-collision copy speaks bench vocabulary and explains the deleted-name reservation
- [x] #2 The zero-target blocked reason names Save as the arming step
- [x] #3 llama_targets() uses the documented list limit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. (a) In bench_editor.py's _on_save_pressed except clause, special-case ConflictError separately from ValueError/RuntimeError and render bench-vocabulary copy explaining the deleted-name-reservation trap; update/relax the existing substring test if needed and confirm it still passes, keep it as substring "already exists" per task note (copy can change freely) unless a more precise pin is warranted.
2. (b) Append the Save-is-the-arming-step clause to evals_screen.py's zero-target tooltip; update test_evals_screen.py's exact-match assertion.
3. (c) Pass limit=_LIST_LIMIT into EvalsViewModel.llama_targets()'s list_models() call and note in its docstring why (matches other read methods in the module).
4. Run the targeted test files, add/adjust assertions, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
(a) bench_editor.py's _on_save_pressed now catches ConflictError in its own except clause (split out of the former except (ValueError, ConflictError, RuntimeError) tuple) and renders bench-vocabulary copy naming the trap: 'A bench named "<name>" already exists -- choose a different name. (Deleting a bench does not free its name: a deleted bench may still be holding it.)' The DB's raw "Task name already exists" is never surfaced to the UI now. Pinned the existing rename-collision test (test_renaming_to_a_taken_name_renders_the_conflict_callout) to the exact new copy instead of the prior substring check.
(b) evals_screen.py's zero-target primary-action tooltip now reads "...add one in the bench editor and Save." naming Save as the arming step. Updated test_primary_action_state_stays_disabled_for_a_target_less_bench's exact-match assertion.
(c) EvalsViewModel.llama_targets() (evals_state.py) now passes limit=_LIST_LIMIT into list_models(), matching every other read method on the class; docstring/inline comment explain why. Added a new red-before-green unit test (test_llama_targets_uses_the_documented_list_limit_not_list_models_default) that creates 120 llama_cpp models and asserts none are dropped -- verified it fails against the pre-fix call (only 100 returned) and passes after.
Ran Tests/UI/test_evals_bench_editor.py + Tests/UI/test_evals_screen.py together: 119 passed.
<!-- SECTION:NOTES:END -->
