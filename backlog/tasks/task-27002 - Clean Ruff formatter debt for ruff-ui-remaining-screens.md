---
id: TASK-27002
title: Clean Ruff formatter debt for ruff-ui-remaining-screens
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

<!-- TASK-26000-BATCH: ruff-ui-remaining-screens -->
<!-- TASK-26000-PATHS-SHA256: 51da045a69140ab99a01d217f142f1e64471417a311ba46c7c91357f4bc6846b -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-ui-remaining-screens` Ruff formatter batch at the owner boundary recorded as: Remaining non-Console screens and narrowly corresponding UI tests.. The focused test surface recorded by TASK-26000 is `["Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_app_instance_warning.py",
  "Tests/UI/test_artifacts_screen_reports.py",
  "Tests/UI/test_background_signal_bounds.py",
  "Tests/UI/test_ccp_handlers.py",
  "Tests/UI/test_change_review_commit_ui.py",
  "Tests/UI/test_change_review_git_provider.py",
  "Tests/UI/test_change_review_screen.py",
  "Tests/UI/test_chat_screen_sidebar_state_debounce.py",
  "Tests/UI/test_chat_screen_worker_groups.py",
  "Tests/UI/test_chat_task_cards_sync.py",
  "Tests/UI/test_code_repo_copy_paste_window.py",
  "Tests/UI/test_conversation_attach_picker.py",
  "Tests/UI/test_home_screen.py",
  "Tests/UI/test_lab_frame.py",
  "Tests/UI/test_logs_filter_persist_debounce.py",
  "Tests/UI/test_nav_overflow_tick_gating.py",
  "Tests/UI/test_probe_headless_wake_p1_continuity.py",
  "Tests/UI/test_probe_headless_wake_p2_p3_p4.py",
  "Tests/UI/test_probe_launch_wake.py",
  "Tests/UI/test_product_maturity_phase1_first_run.py",
  "Tests/UI/test_product_maturity_phase6_recovery_docs.py",
  "Tests/UI/test_reminder_form.py",
  "Tests/UI/test_screen_preimport.py",
  "Tests/UI/test_screen_preimport_pacing.py",
  "Tests/UI/test_serve_main_args.py",
  "Tests/UI/test_study_flashcards_screen.py",
  "Tests/UI/test_trace_export_ui.py",
  "Tests/UI/test_trajectory_timeline_integration.py",
  "tldw_chatbook/UI/ChatbookCreationWindow.py",
  "tldw_chatbook/UI/Chatbooks_Window_Improved.py",
  "tldw_chatbook/UI/Logs_Window.py",
  "tldw_chatbook/UI/Screens/artifacts_screen.py",
  "tldw_chatbook/UI/Screens/chat_screen_state.py",
  "tldw_chatbook/UI/Screens/home_screen.py",
  "tldw_chatbook/UI/Screens/image_gen_demo_screen.py",
  "tldw_chatbook/UI/Screens/lab_frame.py",
  "tldw_chatbook/UI/Screens/logs_screen.py",
  "tldw_chatbook/UI/Screens/stats_screen.py",
  "tldw_chatbook/UI/Screens/trajectory_screen.py",
  "tldw_chatbook/UI/Widgets/table_click_select.py",
  "tldw_chatbook/UI/image_gen_command_provider.py"
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
