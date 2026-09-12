---
id: TASK-26966
title: Clean Ruff formatter debt for ruff-console-modals
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

<!-- TASK-26000-BATCH: ruff-console-modals -->
<!-- TASK-26000-PATHS-SHA256: 17470c695b11e131cb9d6ca0d1e450908c7555e0251bb349282d10cf8ebf3022 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-console-modals` Ruff formatter batch at the owner boundary recorded as: Console modal, dialog, picker, and menu surfaces.. The focused test surface recorded by TASK-26000 is `["Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_console_capture_policy_dialog.py",
  "Tests/UI/test_console_conversation_action_menu.py",
  "Tests/UI/test_console_exchange_export_dialog.py",
  "Tests/UI/test_console_feedback_comment_modal.py",
  "Tests/UI/test_console_library_access_modal.py",
  "Tests/UI/test_console_modal_dismissal.py",
  "Tests/UI/test_console_rag_settings_modal.py",
  "Tests/UI/test_console_review_notes_modal.py",
  "Tests/UI/test_console_scope_picker_modal.py",
  "Tests/UI/test_console_side_chat_modal.py",
  "tldw_chatbook/Widgets/Console/console_capture_policy_dialog.py",
  "tldw_chatbook/Widgets/Console/console_character_picker_modal.py",
  "tldw_chatbook/Widgets/Console/console_citation_sources_modal.py",
  "tldw_chatbook/Widgets/Console/console_conversation_action_menu.py",
  "tldw_chatbook/Widgets/Console/console_exchange_export_dialog.py",
  "tldw_chatbook/Widgets/Console/console_generate_image_modal.py",
  "tldw_chatbook/Widgets/Console/console_rag_settings_modal.py",
  "tldw_chatbook/Widgets/Console/console_review_notes_modal.py",
  "tldw_chatbook/Widgets/Console/console_save_as_modal.py",
  "tldw_chatbook/Widgets/Console/console_save_markdown_modal.py",
  "tldw_chatbook/Widgets/Console/console_scope_picker_modal.py",
  "tldw_chatbook/Widgets/Console/console_session_switcher_modal.py",
  "tldw_chatbook/Widgets/Console/console_settings_modal.py",
  "tldw_chatbook/Widgets/Console/console_setup_modal.py",
  "tldw_chatbook/Widgets/Console/console_side_chat_modal.py",
  "tldw_chatbook/Widgets/Console/console_style_picker_modal.py",
  "tldw_chatbook/Widgets/Console/console_video_capacity_modal.py"
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
