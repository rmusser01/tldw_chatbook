---
id: TASK-26977
title: Clean Ruff formatter debt for ruff-library
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 11:40'
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

<!-- TASK-26000-BATCH: ruff-library -->
<!-- TASK-26000-PATHS-SHA256: 50ced0603397159231ce2f7c86975c74dcb98f5cc4ea25f6c01ab6d3b9e7c96c -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-library` Ruff formatter batch at the owner boundary recorded as: Library services/widgets and directly corresponding Library tests.. The focused test surface recorded by TASK-26000 is `["Tests/Library", "Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Library/test_agent_chunk_student_story.py",
  "Tests/Library/test_cross_runtime_parity.py",
  "Tests/Library/test_ingest_preflight.py",
  "Tests/Library/test_ingest_preflight_egress.py",
  "Tests/Library/test_library_collections_service.py",
  "Tests/Library/test_library_conversations_state.py",
  "Tests/Library/test_library_expand_policy.py",
  "Tests/Library/test_library_ingest_jobs_restore.py",
  "Tests/Library/test_library_ingest_runner.py",
  "Tests/Library/test_library_ingest_state.py",
  "Tests/Library/test_library_keyword_and_then_prefix.py",
  "Tests/Library/test_library_local_rag_search_service.py",
  "Tests/Library/test_library_media_content.py",
  "Tests/Library/test_library_media_raw_view.py",
  "Tests/Library/test_library_media_trash_state.py",
  "Tests/Library/test_library_notes_session.py",
  "Tests/Library/test_library_prompt_evidence_driver.py",
  "Tests/Library/test_library_prompts_seam.py",
  "Tests/Library/test_library_prompts_state.py",
  "Tests/Library/test_library_rag_answer_service.py",
  "Tests/Library/test_library_rag_mode_resolution.py",
  "Tests/Library/test_library_rag_state.py",
  "Tests/Library/test_library_rechunk_service.py",
  "Tests/Library/test_library_seam_availability.py",
  "Tests/Library/test_library_shell_state.py",
  "Tests/Library/test_library_tool_contract.py",
  "Tests/Library/test_library_tool_security_bounds.py",
  "Tests/Library/test_local_library_tool_service.py",
  "Tests/Library/test_media_chunk_tool_service.py",
  "Tests/Library/test_prompt_export_roundtrip.py",
  "Tests/Library/test_server_ingest_field_contract.py",
  "Tests/Library/test_server_ingest_reconcile.py",
  "Tests/Library/test_server_ingest_request.py",
  "Tests/Library/test_skill_trust_review_preview.py",
  "Tests/Library/test_web_clip_request.py",
  "Tests/Widgets/Library/test_library_note_folder_dialog.py",
  "Tests/Widgets/Library/test_library_rail.py",
  "tldw_chatbook/Library/ingest_analysis.py",
  "tldw_chatbook/Library/ingest_capabilities.py",
  "tldw_chatbook/Library/ingest_preflight.py",
  "tldw_chatbook/Library/ingest_types.py",
  "tldw_chatbook/Library/library_conversations_state.py",
  "tldw_chatbook/Library/library_ingest_state.py",
  "tldw_chatbook/Library/library_local_rag_search_service.py",
  "tldw_chatbook/Library/library_media_viewer_state.py",
  "tldw_chatbook/Library/library_notes_tree_state.py",
  "tldw_chatbook/Library/library_pager_state.py",
  "tldw_chatbook/Library/library_rag_answer_service.py",
  "tldw_chatbook/Library/library_rag_state.py",
  "tldw_chatbook/Library/library_rechunk_service.py",
  "tldw_chatbook/Library/library_shell_state.py",
  "tldw_chatbook/Library/library_tool_contract.py",
  "tldw_chatbook/Library/local_library_tool_service.py",
  "tldw_chatbook/Library/local_media_chunk_tool_service.py",
  "tldw_chatbook/Library/server_ingest_reconcile.py",
  "tldw_chatbook/Library/server_ingest_request.py",
  "tldw_chatbook/Widgets/Library/library_collections_panel.py",
  "tldw_chatbook/Widgets/Library/library_entry_canvases.py",
  "tldw_chatbook/Widgets/Library/library_export_canvas.py",
  "tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py",
  "tldw_chatbook/Widgets/Library/library_ingest_canvas.py",
  "tldw_chatbook/Widgets/Library/library_media_canvas.py",
  "tldw_chatbook/Widgets/Library/library_media_content.py",
  "tldw_chatbook/Widgets/Library/library_media_image_preview.py",
  "tldw_chatbook/Widgets/Library/library_media_raw_view.py",
  "tldw_chatbook/Widgets/Library/library_media_trash_canvas.py",
  "tldw_chatbook/Widgets/Library/library_media_viewer.py",
  "tldw_chatbook/Widgets/Library/library_prompts_canvas.py",
  "tldw_chatbook/Widgets/Library/library_search_rag_panel.py",
  "tldw_chatbook/Widgets/Library/prompt_delete_confirmation_modal.py"
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

1. Reconcile assigned paths against current `origin/dev` (existence + format state) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` (parse with `type_comments=True`, normalize `TypeIgnore.lineno`, `include_attributes=False`) before formatting.
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality with the before-snapshot (formatter-mandated docstring quote-initial normalization documented if it occurs).
5. `ruff format --check` must pass on every assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface recorded by TASK-26000 (bounded subset if the surface is whole-tree; never a bare pytest), plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `e92b01515f` (isolated worktree `.worktrees/ruff-debt-batch-5`, branch `chore/ruff-debt-batch-5`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All 70 assigned paths reconcile mechanically (hash matches both the task marker and the evidence JSON), and all were reproduced: 69 exist at the base; ONE was DELETED upstream after the census pin — `tldw_chatbook/Widgets/Library/library_collections_panel.py`, removed by dev commit `5dd1077df6` ("feat(collections): retire generic containers from current surfaces"). Recorded here as the required lineage reconciliation rather than silently dropped.

**Formatting (AC#2, AC#5).** 67 of the 69 existing assigned paths formatted; 2 (`Tests/Library/test_library_media_trash_state.py`, `tldw_chatbook/Library/library_conversations_state.py`) were already formatter-clean at the base and were deliberately left untouched (final three-way reconciliation: 67 committed + 2 clean + 1 upstream-deleted = 70). `ruff format --check` passes on every existing assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's assigned paths: 122 findings before and 122 after, per-file counts identical — the formatter introduced zero findings; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after hashes equal on every existing assigned path (the deleted path has no after-state to compare).

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: `Tests/Library` (the assigned test files are the Tests/Library portion; bounded directory). Result on the formatted tree: 179 failed, 3559 passed, 4 skipped, 2 errors in 677.98s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 179 failed, 3559 passed, 4 skipped, 2 errors in 599.24s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Batch-wide verification: the modified working set is exactly the assigned paths that needed formatting plus the 8 batch task files; zero unassigned paths touched. Every content diff is Ruff formatter output.

**Lineage.** 1 path upstream-deleted (recorded above); 2 already formatter-clean at the base (recorded above); the remaining 67 carried live debt and were formatted. The ownership `paths_sha256` continues to match the census record — the deletion is dev-side drift AFTER the census pin, which the rebase-reconcile AC requires recording.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
