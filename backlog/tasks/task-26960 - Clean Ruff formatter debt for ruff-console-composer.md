---
id: TASK-26960
title: Clean Ruff formatter debt for ruff-console-composer
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

<!-- TASK-26000-BATCH: ruff-console-composer -->
<!-- TASK-26000-PATHS-SHA256: cc2c9397df7205d41a82c45372d58218d8e9047257ca0533093be6cc57db5aa0 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-console-composer` Ruff formatter batch at the owner boundary recorded as: Console prompt composition, input, dictation, and queue surfaces.. The focused test surface recorded by TASK-26000 is `["Tests/Chat", "Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_console_command_composer.py",
  "Tests/UI/test_console_composer_caret_nav.py",
  "Tests/UI/test_console_composer_collapse.py",
  "Tests/UI/test_console_composer_history.py",
  "Tests/UI/test_console_composer_overflow.py",
  "Tests/UI/test_console_dictation_firstrun.py",
  "Tests/UI/test_console_prompt_queue.py",
  "Tests/UI/test_console_prompt_queue_modal.py",
  "Tests/UI/test_console_prompts_controller.py",
  "Tests/UI/test_console_prompts_modal.py",
  "tldw_chatbook/UI/Console_Modules/dictation.py",
  "tldw_chatbook/UI/Console_Modules/prompts.py",
  "tldw_chatbook/Widgets/Console/console_prompt_queue_modal.py",
  "tldw_chatbook/Widgets/Console/console_prompts_modal.py"
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
