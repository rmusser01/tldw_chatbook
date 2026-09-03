---
id: TASK-26977
title: Clean Ruff formatter debt for ruff-library
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
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
