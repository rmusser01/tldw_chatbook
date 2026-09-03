---
id: TASK-26996
title: Clean Ruff formatter debt for ruff-ui-library
status: To Do
assignee: []
created_date: '2026-08-31 18:31'
updated_date: '2026-08-31 18:31'
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
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
