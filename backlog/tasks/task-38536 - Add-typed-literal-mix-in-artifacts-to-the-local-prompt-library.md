---
id: TASK-38536
title: Add typed literal mix-in artifacts to the local prompt library
status: Done
created_date: 2026-11-01 19:30
dependencies:
- TASK-38535
labels:
- prompts
- console
references:
- backlog/decisions/286-console-prompt-mix-ins.md
documentation:
- Docs/superpowers/specs/2026-11-01-prompt-mix-ins-design.md
- Docs/superpowers/plans/2026-11-01-prompt-mix-ins-implementation.md
- backlog/decisions/286-console-prompt-mix-ins.md
updated_date: 2026-11-01 20:22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give reusable response modifiers a distinct local library identity while preserving existing Prompt and Recipe behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Strict schema-v3 text_mixin definitions preserve literal body bytes and Before/After placement, reject invalid or empty bodies, and cannot be applied as normal Prompts, System replacements or Recipes.
- [x] #2 Local CRUD, brief/detail/search filters, optimistic updates and retained versions round-trip mix-ins through existing services and advertise explicit capabilities.
- [x] #3 An executed file-backed migration extends the artifact-type constraint without changing existing prompt identities, keywords, FTS, collections, retained history or sync records; rollback and targeted compatibility tests pass.
- [x] #4 Existing Prompt v4/v5 backups validate and migrate safely to v6 through the restricted recovery authorizer, and v6 backups validate without data loss.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing strict literal decoding and local persistence regressions. 2. Implement typed schema-v3 dispatch, local capabilities, filtered CRUD and snapshot reads with safe Prompt eligibility. 3. Add a transactional file-backed migration and verify retained data and rollback. 4. Run targeted compatibility checks, scoped lint and self-review; update architecture notes. ADR required: yes; ADR path: backlog/decisions/286-console-prompt-mix-ins.md; Reason: existing approved ADR governs the storage and source contract.
5. Preserve Prompt backup recovery across schema v4/v5 to v6: update DB/recovery_core_schema.py, DB/recovery_core.py and Backup_Recovery/sqlite_validation.py; add focused restore and migration-authorizer coverage.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented strict literal v3 definitions, canonical local source writes and optimistic updates, exact UUID snapshots and type-filtered browse/count/search. Rebuilt Prompt v5 to v6 from executed SQL with identity, collection, FTS, history and sync preservation; scoped backup recovery retains exact v4/v5 catalogs and migrates staged copies under its authorizer. ADR: backlog/decisions/286-console-prompt-mix-ins.md. See .superpowers/sdd/2026-11-01-prompt-mix-ins-implementation/task-1-report.md for commands and results.

Known exact-base failures reproduced at 76171c2750a0: Tests/DB/test_migration_file_sources.py::TestNoDecorativeMigrationFiles::test_every_migration_file_is_load_bearing; Tests/Prompt_Management/test_prompt_batch_models.py::test_prompt_management_lazy_public_exports_preserve_class_identities; Tests/UI/test_console_prompt_picker.py::{test_console_picker_captures_ordinary_segment_draft_at_opening,test_console_picker_refuses_when_draft_changes_after_opening,test_prompt_picker_css_blocks_pinned_in_source_and_bundle}; Tests/UI/test_console_prompts_controller.py::{test_improvement_target_is_pinned_once_and_shared_across_callbacks,test_improvement_activation_refuses_a_moved_system_prompt,test_improvement_snapshot_refuses_an_unpinned_provider_target}; Tests/UI/test_console_prompts_modal.py::{test_no_matches_retry_and_source_unavailable_are_explicit,test_source_switch_clears_foreign_rows_before_unavailable_result,test_real_recipe_fill_with_additional_context_reaches_block_review,test_mapped_context_blocks_recipe_save_until_deleted}. The guard reports the same seven pre-existing Dreams/Guardian/Collections files; the new Prompt SQL is recognized and independently executed and rollback-tested. AC 3 remains open for independent review of this inherited guard failure.
Review round 1: shared local mix-in write preparation now fences explicit expected_version=None to the merged record version; real interleaved-write tests cover both adapters. Focused tests: 232 passed. Independent review accepted AC 3 migration evidence; the global migration-file guard retains the same seven inherited BASE findings listed above. Status remains In Progress pending scoped re-review.
Independent task review identified an explicit-None version-fence race and duplicated preparation; both were fixed in 79751133d760a5ebf3deb59e53163b117a55b1ab. Scoped re-review approved both findings, with no new breakage or deferred observations. Original implementation 7f511816fc had 472 targeted passes; focused fix coverage had 232 passes (overlapping tests, not an aggregate count). The new Prompt migration is recognized by the migration-file guard; its same seven unrelated BASE findings remain documented. Runtime authority, queue/retry capture and portable boundaries are owned by the subsequent existing plan tasks. No full suite, push or merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
