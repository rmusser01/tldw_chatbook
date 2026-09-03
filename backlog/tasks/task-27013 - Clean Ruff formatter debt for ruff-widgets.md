---
id: TASK-27013
title: Clean Ruff formatter debt for ruff-widgets
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

<!-- TASK-26000-BATCH: ruff-widgets -->
<!-- TASK-26000-PATHS-SHA256: 363308e7911957a39dc222b8a801bc80733f5824dc6701522c0ec88d5946139c -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-widgets` Ruff formatter batch at the owner boundary recorded as: Shared non-Console widgets and direct widget tests.. The focused test surface recorded by TASK-26000 is `["Tests/Widgets"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Widgets/test_chapter_editor_widget_inplace_edit_refresh.py",
  "Tests/Widgets/test_console_video_card.py",
  "Tests/Widgets/test_console_video_card_rows.py",
  "Tests/Widgets/test_inline_loader_timer.py",
  "Tests/Widgets/test_library_collections_panel.py",
  "Tests/Widgets/test_model_search_picker.py",
  "Tests/Widgets/test_password_dialog_encryption_warning.py",
  "Tests/Widgets/test_pausable_progress.py",
  "Tests/Widgets/test_prune_safe_select.py",
  "Tests/Widgets/test_reactive_default_aliasing.py",
  "Tests/Widgets/test_tool_diff_widgets.py",
  "Tests/Widgets/test_watchlists_operation_card.py",
  "tldw_chatbook/Widgets/AppFooterStatus.py",
  "tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py",
  "tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py",
  "tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py",
  "tldw_chatbook/Widgets/Chat_Widgets/watchlists_operation_card.py",
  "tldw_chatbook/Widgets/Persona_Widgets/character_tts_portability_dialogs.py",
  "tldw_chatbook/Widgets/Persona_Widgets/conversation_attach_picker.py",
  "tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py",
  "tldw_chatbook/Widgets/Persona_Widgets/personas_character_dictionaries.py",
  "tldw_chatbook/Widgets/Persona_Widgets/personas_character_world_books.py",
  "tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py",
  "tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py",
  "tldw_chatbook/Widgets/Persona_Widgets/personas_policy_rules_editor.py",
  "tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py",
  "tldw_chatbook/Widgets/Persona_Widgets/world_book_picker.py",
  "tldw_chatbook/Widgets/detailed_progress.py",
  "tldw_chatbook/Widgets/enhanced_file_picker.py",
  "tldw_chatbook/Widgets/model_search_picker.py",
  "tldw_chatbook/Widgets/project_skills_import_modal.py",
  "tldw_chatbook/Widgets/settings_agents_panel.py",
  "tldw_chatbook/Widgets/settings_image_gen_panel.py",
  "tldw_chatbook/Widgets/settings_internal_prompts_editor_modal.py",
  "tldw_chatbook/Widgets/settings_splash_screen_viewer.py",
  "tldw_chatbook/Widgets/settings_theme_editor.py",
  "tldw_chatbook/Widgets/settings_video_gen_panel.py",
  "tldw_chatbook/Widgets/splash_screen.py",
  "tldw_chatbook/Widgets/workspace_create_modal.py"
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
