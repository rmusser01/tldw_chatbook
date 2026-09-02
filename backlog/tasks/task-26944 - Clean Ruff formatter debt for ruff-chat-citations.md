---
id: TASK-26944
title: Clean Ruff formatter debt for ruff-chat-citations
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-31 18:31'
updated_date: '2026-09-02 19:56'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/specs/2026-09-02-task-26944-ruff-chat-citations-design.md
  - Docs/superpowers/plans/2026-09-02-task-26944-ruff-chat-citations.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chat-citations` Ruff formatter batch at the owner boundary recorded as: Chat citation construction and trace helpers with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/Chat"]`.
<!-- SECTION:DESCRIPTION:END -->

<!-- TASK-26000-BATCH: ruff-chat-citations -->
<!-- TASK-26000-PATHS-SHA256: 6149803c4606eb131c95d8713504d1213a98aac1b4fc15f4fb5aea4fb9a73129 -->
<!-- TASK-26000-FINAL: false -->

## Assigned Paths

```json
[
  "Tests/Chat/test_citation_service_factory.py",
  "Tests/Chat/test_citation_trace_builder.py",
  "tldw_chatbook/Chat/citation_trace_builder.py"
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

ADR required: no

ADR path: N/A

Reason: this is a formatter-only application of the existing TASK-26000 cleanup
contract and introduces no architectural boundary or durable policy.

1. Fetch and rebase onto current `origin/dev`, reconcile every assigned path, and
   capture the pre-format AST/comment evidence with the pinned toolchain.
2. Run Ruff 0.15.22 on only the reconciled owned paths and require identical
   semantic/comment evidence.
3. Run the scoped Ruff, focused-test, governance, and diff checks; self-review the
   layout-only change.
4. Record exact evidence, check all acceptance criteria, set the task to `Done`,
   and commit the closeout.
