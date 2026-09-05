---
id: TASK-26949
title: Clean Ruff formatter debt for ruff-chat-console-library
status: In Progress
assignee:
  - codex
created_date: '2026-08-31 18:31'
updated_date: '2026-09-05 16:33'
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

<!-- TASK-26000-BATCH: ruff-chat-console-library -->
<!-- TASK-26000-PATHS-SHA256: cd664f7c1688da479c38f4c47b8c5b2c997aab2320c9ceef1e2a73bd12476b90 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chat-console-library` Ruff formatter batch at the owner boundary recorded as: Console library activity, policy, and destination services.. The focused test surface recorded by TASK-26000 is `["Tests/Chat", "Tests/Library"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Chat/test_console_chat_store_library_policy.py",
  "Tests/Chat/test_console_library_activity_buffer.py",
  "Tests/Chat/test_console_library_destination.py",
  "Tests/Chat/test_console_library_policy_coordinator.py",
  "Tests/Chat/test_console_library_runtime_policy.py",
  "tldw_chatbook/Chat/console_library_activity_buffer.py",
  "tldw_chatbook/Chat/console_library_destination.py",
  "tldw_chatbook/Chat/console_library_policy.py",
  "tldw_chatbook/Chat/console_library_policy_repository.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same nearest logical owner and significant-token position, using a uniquely fail-closed same-line `except` clause for an `ExceptHandler` header and otherwise the nearest containing AST statement; exclude only AST-neutral parentheses proven by shadow parse/dump comparison, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

1. Reconcile the nine-path batch manifest and its canonical digest against the exact current `origin/dev` base, including upstream lineage since TASK-26000's authority cut.
2. Capture Python 3.12.11 structural evidence and an exact focused-test baseline before making any source edit.
3. Run Ruff 0.15.22 formatting once with all nine assigned paths supplied explicitly, then require structural, comment/directive, scope, lint, format, focused-test, deterministic-replay, diagnostic, and governance parity.
4. Obtain independent task and whole-branch reviews, close the task with exact evidence, rebase onto the latest `dev`, and repeat every base-sensitive gate before integration.

ADR required: no
ADR path: N/A
Reason: This is mechanical formatter cleanup under TASK-26000's existing contract and changes no architecture, schema, storage, security, dependency, or long-lived UX boundary.
