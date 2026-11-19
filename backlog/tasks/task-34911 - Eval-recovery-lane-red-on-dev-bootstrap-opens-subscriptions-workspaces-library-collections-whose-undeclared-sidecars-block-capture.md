---
id: TASK-34911
title: 'Eval recovery lane red on dev: bootstrap opens subscriptions/workspaces/library_collections
  whose undeclared sidecars block capture'
status: Done
created_date: 2026-10-12 22:02
labels:
- bug
priority: high
updated_date: 2026-10-13 17:49
assignee:
- '@zcode'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Backup_Recovery/test_eval_retained_definitions.py::test_complete_rebackup_retains_eval_after_source_and_candidate_removal fails deterministically on pristine origin/dev (verified twice in a clean worktree at 38654061ca, sys.path-isolated) and in the 'Eval recovery regressions' required CI lanes (macos-15/ubuntu-24.04/windows-2022 all red on PR #3083 heads; green on PRs that predate the recent goals/personal-context merges). The full-app child exits 1 because capture classifies tldw_chatbook_subscriptions.db-shm/-wal, tldw_chatbook_library_collections.db-shm/-wal and tldw_chatbook_workspaces.db-shm/-wal as ('unknown','unsupported') - sidecars of stores that are opened at bootstrap by the recently merged goals/personal-context/Library work but are NOT declared participants of the test profile, so inventory._sqlite_sidecars (which only derives sqlite.transient for declared schema-policy adapters) never claims them. Every required Eval recovery lane will stay red for every PR until this is fixed on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three stores' sidecars are either declared participants or classified as transient sidecars at capture, per the Backup_Recovery governance ruling recorded in an ADR or decision note if scope expands,The eval recovery lanes (macos/ubuntu/windows) are green on dev again,The child classification report in test_complete_rebackup_retains_eval_after_source_and_candidate_removal contains zero unknown/unsupported entries
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
RE-SCOPED after first fix attempt: the unknown-sidecar classification was real but secondary. Primary blocker (verified on pristine origin/dev worktree, twice): the headless full-app child's capture preview is incomplete - preview.complete False with ~83-89 stores classified 'unsupported' including config, db.chachanotes.primary, chat.dictionaries (identical on dev without the perf PR). Suspected headless encryption-unlock/capture contract regression from the recent goals/personal-context merges. Owned by Backup_Recovery + the feature workstreams. The perf PR carries the still-correct partial fixes: recovery catalogs for the task-34399 indexes, and sidecar claiming for declared-but-unselected stores.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed on PR #3083 (commit f0299c94ce). Two dev-side defects: (1) darwin25 provenance xattr made file_inventory._metadata_supported refuse every raw-file owner (config.toml, chat_dicts, generated_images) - flistxattr now enumerates and only non-provenance attributes refuse; (2) sqlite_validation authorize() lacked db.evals admissions for the task-34399 migration statements (CREATE INDEX idx_eval_results_run_created, INSERT sqlite_master from the column rebuild, REINDEX). Plus the catalog stamps and sidecar claiming already on the branch. Eval file 12/12; 23-failure lifetime set verified identical with changes stashed; full suite 6575 passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
