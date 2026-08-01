---
id: TASK-1614
title: update_task description hygiene parity
status: Done
assignee: []
created_date: '2026-07-31 15:10'
updated_date: '2026-08-01 02:33'
labels:
  - evals
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-1482 (Task 3) fixed the name-hygiene asymmetry between create_task and update_task via a shared helper, but the DESCRIPTION parameter still differs: create_task filters control characters, update_task does not. Same parity fix, same shared-helper approach.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 update_task applies the same description cleaning as create_task, via a shared helper
- [x] #2 A test pins a control-character description round-trip on both paths
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep all callers of EvalsDB.update_task that pass description= to check whether newly stripping control characters would break any caller's expectations (storage.py save_bench: UI-form-driven, fine; local_evaluations_service.py/evaluation_scope_service.py: pass-through, but their own tests use FakeEvalsDB/fakes, not the real EvalsDB, so no risk).
2. Add a `_clean_task_description` helper next to the existing `_clean_task_name` helper (same file-level location, same "shared by create_task and update_task" framing), matching create_task's existing inline control-char-only filter behavior exactly (no strip, no blank rejection -- description has no NOT NULL constraint).
3. Replace create_task's inline filter block with a call to the new helper (no behavior change -- pin with a create_task-path test first if none exists).
4. Apply the same helper inside update_task's `if description is not None` branch (this is the actual parity fix).
5. Add a control-character description round-trip test for both create_task and update_task in Tests/Evals/test_evals_db.py; red-before-green the update_task one (temporarily revert to raw passthrough, confirm it fails, restore).
6. One mutation check on the shared helper: neutralize its filter body, confirm BOTH new tests catch it, restore.
7. Run Tests/Evals/test_evals_db.py + Tests/Evals/test_eval_properties.py + Tests/Evals/word_bench/test_storage_authoring.py + the Evaluations_Interop suites, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added _clean_task_description (Evals_DB.py, next to _clean_task_name) filtering control characters only -- no strip, no blank rejection, since eval_tasks.description carries no NOT NULL constraint and a falsy value must round-trip unchanged (including an explicit "clear the description" update). create_task's pre-existing inline filter block was replaced with a call to the helper (behavior-preserving, covered by the existing hypothesis roundtrip property test plus a new pinned unit test). update_task's `if description is not None` branch now calls the same helper -- this is the actual parity fix, since it previously passed description straight through with zero cleaning.
Grepped every update_task(description=...) caller: storage.py's save_bench (UI-form-driven, fine to newly filter) and the Evaluations_Interop pass-through path (local_evaluations_service.py / evaluation_scope_service.py) -- the latter's own tests use FakeEvalsDB/FakeLocalEvaluationService fakes, never the real EvalsDB, so stripping control characters there carries no test-breakage risk; ran both interop suites to confirm (34 passed).
Added test_create_task_strips_control_characters_from_description and test_update_task_strips_control_characters_from_description (Tests/Evals/test_evals_db.py), both asserting the identical control-character round-trip. Red-before-green: temporarily reverted update_task's call to a raw passthrough -- the update test failed as expected (create_task's own pre-existing behavior stayed covered by the existing property test). Mutation check on the shared helper (the batch's most substantive change): neutralized its filter body to a no-op -- BOTH new tests failed, confirming each pins real behavior rather than tautology. Restored after each check.
Ran Tests/Evals/test_evals_db.py + test_eval_properties.py + word_bench/test_storage_authoring.py (73 passed) and both Evaluations_Interop suites (34 passed).
<!-- SECTION:NOTES:END -->
