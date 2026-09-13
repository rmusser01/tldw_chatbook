---
id: TASK-26962
title: Clean Ruff formatter debt for ruff-console-foundation-ui
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

<!-- TASK-26000-BATCH: ruff-console-foundation-ui -->
<!-- TASK-26000-PATHS-SHA256: 719613ae3ecb0cd9962153e5728c80800357d22bb9f689c4fa62304d5f29559f -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-console-foundation-ui` Ruff formatter batch at the owner boundary recorded as: Console UI foundations outside narrower semantic surfaces.. The focused test surface recorded by TASK-26000 is `["Tests/Chat", "Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_console_agent_steering_bar.py",
  "Tests/UI/test_console_approval_first_open_render.py",
  "Tests/UI/test_console_browser_search_echo.py",
  "Tests/UI/test_console_context_controls.py",
  "Tests/UI/test_console_internals_decomposition.py",
  "Tests/UI/test_console_mcp_approval.py",
  "Tests/UI/test_console_moved_seam_guard.py",
  "Tests/UI/test_console_parked_payload_rekey.py",
  "Tests/UI/test_console_regenerate_feedback.py",
  "Tests/UI/test_console_roleplay_resume_navigation.py",
  "Tests/UI/test_console_scope_row.py",
  "Tests/UI/test_console_setup_card_fit.py",
  "Tests/UI/test_console_setup_lock_polish.py",
  "Tests/UI/test_console_shell_regions.py",
  "Tests/UI/test_console_skill_install_confirm.py",
  "Tests/UI/test_console_staged_context.py",
  "Tests/UI/test_console_stream_scrollback.py",
  "Tests/UI/test_console_temporary_capture_admission.py",
  "Tests/UI/test_console_watchlists_mounted_uat.py",
  "Tests/UI/test_console_workbench_contract.py",
  "tldw_chatbook/UI/Console_Modules/raw_cli.py",
  "tldw_chatbook/Widgets/Console/console_agent_steering_bar.py",
  "tldw_chatbook/Widgets/Console/console_assistant_turn.py",
  "tldw_chatbook/Widgets/Console/console_auto_speak_consent.py",
  "tldw_chatbook/Widgets/Console/console_command_popup.py",
  "tldw_chatbook/Widgets/Console/console_retrieval_scope_row.py"
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
