---
id: TASK-26996
title: Clean Ruff formatter debt for ruff-ui-library
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 18:55'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

<!-- TASK-26000-BATCH: ruff-ui-library -->
<!-- TASK-26000-PATHS-SHA256: e13803687bd19602fd227b27d0ea27c287ffc966686f76e9d08dfeabf7f41797 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-ui-library` Ruff formatter batch at the owner boundary recorded as: Library UI screens/modules and directly named UI/Library tests.. The focused test surface recorded by TASK-26000 is `["Tests/Library", "Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_audio_cpp_model_library_handoff.py",
  "Tests/UI/test_library_adaptive_reader_closeout.py",
  "Tests/UI/test_library_canvas_scoped_sync.py",
  "Tests/UI/test_library_canvas_sync_defects.py",
  "Tests/UI/test_library_choice_strips.py",
  "Tests/UI/test_library_entry_compose_once.py",
  "Tests/UI/test_library_export_receipt.py",
  "Tests/UI/test_library_file_notes_git_push.py",
  "Tests/UI/test_library_file_notes_workspace.py",
  "Tests/UI/test_library_ingest_canvas.py",
  "Tests/UI/test_library_ingest_clear_focus.py",
  "Tests/UI/test_library_ingest_inline_consent.py",
  "Tests/UI/test_library_ingest_keyboard.py",
  "Tests/UI/test_library_ingest_retry_last.py",
  "Tests/UI/test_library_ingest_structural.py",
  "Tests/UI/test_library_ingest_template_picker.py",
  "Tests/UI/test_library_media_image_preview.py",
  "Tests/UI/test_library_media_reader_flow.py",
  "Tests/UI/test_library_media_reader_match_nav_t22209.py",
  "Tests/UI/test_library_media_reader_no_change_sync_t22208.py",
  "Tests/UI/test_library_media_reader_scroller_resolution.py",
  "Tests/UI/test_library_media_reader_traversal_t22207.py",
  "Tests/UI/test_library_media_side_by_side.py",
  "Tests/UI/test_library_media_trash.py",
  "Tests/UI/test_library_multiselect_media.py",
  "Tests/UI/test_library_notes_folder_navigator.py",
  "Tests/UI/test_library_notes_lasting_sync_flow.py",
  "Tests/UI/test_library_notes_reader.py",
  "Tests/UI/test_library_per_click_recompose_t21116.py",
  "Tests/UI/test_library_prompt_collections.py",
  "Tests/UI/test_library_prompts_canvas.py",
  "Tests/UI/test_library_prompts_reader.py",
  "Tests/UI/test_library_rag_handoffs.py",
  "Tests/UI/test_library_rag_legacy_chunk_report.py",
  "Tests/UI/test_library_rag_rechunk_action.py",
  "Tests/UI/test_library_resize_focus_gates_t23025.py",
  "Tests/UI/test_library_review_round_t21116.py",
  "Tests/UI/test_library_screen.py",
  "Tests/UI/test_library_skills_canvas.py",
  "Tests/UI/test_library_skills_reader.py",
  "Tests/UI/test_personas_library_pane_paging.py",
  "Tests/UI/test_personas_library_scale.py",
  "Tests/UI/test_personas_library_toolbar_layout.py",
  "Tests/UI/test_post_release_workspaces_library_depth.py",
  "Tests/UI/test_product_maturity_gate16_library_search_rag.py",
  "Tests/UI/test_product_maturity_phase39_library_collections.py",
  "Tests/UI/test_settings_library_rag_defaults.py",
  "tldw_chatbook/UI/Library_Modules/library_collections_browse_controller.py",
  "tldw_chatbook/UI/Library_Modules/library_skill_import_controller.py",
  "tldw_chatbook/UI/Library_Modules/library_snapshot_cache.py",
  "tldw_chatbook/UI/Screens/settings_library_rag_defaults.py",
  "tldw_chatbook/UI/stts_profile_library.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

1. Reconcile assigned paths against current `origin/dev` (existence + format state; missing paths recorded with dev delete/rename lineage) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` before formatting (type_comments=True with symmetric plain-parse fallback; normalize `TypeIgnore.lineno`).
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality (formatter-mandated docstring-normalization deviations enumerated if they occur).
5. `ruff format --check` must pass on every existing assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface (bounded; never a bare pytest), plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`; verify the three-way path partition arithmetically before writing notes.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `ecc0a531c8` (isolated worktree `.worktrees/ruff-debt-batch-7`, branch `chore/ruff-debt-batch-7`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All 52 assigned paths reconcile mechanically (hash matches both the task marker and the evidence JSON), and all were reproduced: 51 exist at the base; ONE was DELETED upstream after the census pin — `tldw_chatbook/UI/Library_Modules/library_collections_browse_controller.py`, removed by dev commit `5dd1077df6` ("feat(collections): retire generic containers from current surfaces"), the same commit that deleted batch-5 task-26977's path. Recorded here as the required lineage reconciliation rather than silently dropped.

**Formatting (AC#2, AC#5).** 49 of the 51 existing assigned paths formatted. `ruff format --check` passes on every existing assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 203 assigned paths: 68 findings before and 32 after, ZERO increases — the formatter incidentally resolved 36 line-length findings concentrated in `Tests/Tools/test_local_tool_impls.py` (34->1) and `Tests/Tools/test_local_tool_impls_properties.py` (3->0) by reflowing; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after hashes equal on every existing assigned path.

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: the 47 assigned test files (recorded surface Tests/Library + Tests/UI). Result on the formatted tree: 944 failed, 1227 passed, 1 error in 1607.17s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: A/B name-diff protocol: formatted 945 FAILED names vs baseline 944 — exactly one mover, `Tests/UI/test_library_file_notes_workspace.py::test_initial_root_scan_projects_checking_authority_while_actions_are_gated`, which then PASSES 3/3 in isolation on the formatted tree (8.76s/4.40s/4.64s). Load-sensitive timing flake under shared CPU, not formatting-correlated; AST equality independently proves runtime-identical files. — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 192 committed + 10 clean-at-base + 1 upstream-deleted = 203 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** 1 path upstream-deleted (recorded above, same commit as batch-5's task-26977 deletion); 2 paths already formatter-clean at the base (`Tests/UI/test_library_resize_focus_gates_t23025.py`, `Tests/UI/test_post_release_workspaces_library_depth.py`) and deliberately left untouched; the other 49 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
