---
id: TASK-26945
title: Clean Ruff formatter debt for ruff-chat-console-context
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-31 18:31'
updated_date: '2026-09-03 04:14'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
  - Docs/superpowers/plans/2026-09-03-task-26945-ruff-chat-console-context.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chat-console-context` Ruff formatter batch at the owner boundary recorded as: Console context, memory, prepared-request, and RAG state services.. The focused test surface recorded by TASK-26000 is `["Tests/Chat"]`.
<!-- SECTION:DESCRIPTION:END -->

<!-- TASK-26000-BATCH: ruff-chat-console-context -->
<!-- TASK-26000-PATHS-SHA256: 11a85c1a0bd495783743a97a47f2b5f0da359158fcdd22b86532a22c229797bd -->
<!-- TASK-26000-FINAL: false -->

## Assigned Paths

```json
[
  "Tests/Chat/test_console_context_compaction.py",
  "Tests/Chat/test_console_context_policy.py",
  "Tests/Chat/test_console_context_policy_cas.py",
  "Tests/Chat/test_console_manual_memory_planning.py",
  "Tests/Chat/test_console_memory_selection.py",
  "Tests/Chat/test_console_prefill.py",
  "Tests/Chat/test_console_prepared_request.py",
  "Tests/Chat/test_console_world_info_application.py",
  "tldw_chatbook/Chat/console_context_compaction.py",
  "tldw_chatbook/Chat/console_context_policy.py",
  "tldw_chatbook/Chat/console_context_repository.py",
  "tldw_chatbook/Chat/console_prefill.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] #2 Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] #3 Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] #4 Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] #5 Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] #6 Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] #7 `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] #8 The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile all twelve TASK-26000 assigned paths against current origin/dev, record drift and the separate TASK-30040 test repair, and capture Python 3.12.11 AST/comment/directive evidence.
2. Run Ruff 0.15.22 format with all twelve paths supplied explicitly, reject any unassigned Python diff, and require the structural comparison to match.
3. Run Ruff lint/format checks, the eight assigned Console context test modules, backlog task-ID uniqueness, and git diff --check.
4. Commit only formatter-owned Python changes, request independent review, then record exact evidence and close TASK-26945 in a task-only commit.

ADR required: no
ADR path: N/A
Reason: Mechanical formatter cleanup under TASK-26000 introduces no architectural, persistence, security, dependency, or long-lived UX decision.

Detailed plan: Docs/superpowers/plans/2026-09-03-task-26945-ruff-chat-console-context.md
<!-- SECTION:PLAN:END -->
