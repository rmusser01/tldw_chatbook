---
id: TASK-26948
title: Clean Ruff formatter debt for ruff-chat-console-interaction
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

<!-- TASK-26000-BATCH: ruff-chat-console-interaction -->
<!-- TASK-26000-PATHS-SHA256: fda2c56e5364efd07fc80019191a0eaa8641a97e3a20fadbe4458c9d5b011de8 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chat-console-interaction` Ruff formatter batch at the owner boundary recorded as: Console send/edit/rewind/roleplay/session transaction services.. The focused test surface recorded by TASK-26000 is `["Tests/Chat"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Chat/test_console_edit_resend.py",
  "Tests/Chat/test_console_first_send_atomicity.py",
  "Tests/Chat/test_console_regenerate_branching.py",
  "Tests/Chat/test_console_rewind_modal.py",
  "Tests/Chat/test_console_rewind_summarize.py",
  "Tests/Chat/test_console_roleplay_identity.py",
  "Tests/Chat/test_console_roleplay_metadata.py",
  "Tests/Chat/test_console_send_gate_queue_race.py",
  "Tests/Chat/test_console_session_settings.py",
  "Tests/Chat/test_console_stop_reliability.py",
  "Tests/Chat/test_console_switcher_state.py",
  "Tests/Chat/test_console_transaction_contribution.py",
  "tldw_chatbook/Chat/console_image_edit_operations.py",
  "tldw_chatbook/Chat/console_roleplay_identity.py",
  "tldw_chatbook/Chat/console_roleplay_metadata.py",
  "tldw_chatbook/Chat/console_transaction_contribution.py"
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
