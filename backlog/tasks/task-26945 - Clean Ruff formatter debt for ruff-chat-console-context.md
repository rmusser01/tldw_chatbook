---
id: TASK-26945
title: Clean Ruff formatter debt for ruff-chat-console-context
status: Done
assignee:
  - '@codex'
created_date: '2026-08-31 18:31'
updated_date: '2026-09-03 04:25'
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
- [x] #1 After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] #2 Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] #3 Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] #4 Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] #5 Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] #6 Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] #7 `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] #8 The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Applied Ruff 0.15.22 to the exact twelve-path ruff-chat-console-context allowlist. Ruff reformatted eleven files and left Tests/Chat/test_console_memory_selection.py unchanged because the separately reviewed TASK-30040 prerequisite had already repaired and formatted that assigned path. The formatter commit contains only the expected eleven Python paths; Ruff replay from each parent blob reproduced every resulting file byte-for-byte.

Drift reconciliation: work began from fetched origin/dev de52e8f1e736b54e0cc2b11342e7f60662d106a2, then origin/dev advanced and the branch was rebased onto e76915ccf38a8557f203977f2ebeecc1645afeca before structural capture or batch formatting. Relative to TASK-26000 pin e555df102c950c29beed5e7119f433d35eee1f3c, four assigned paths had upstream modifications and were retained: Tests/Chat/test_console_context_compaction.py and tldw_chatbook/Chat/console_context_compaction.py at 7685a16d21998aa73ec16ab813b6c007703caa2b; Tests/Chat/test_console_prepared_request.py at bf852436ec41c6e0f19a5428324c22696da3c0f1; tldw_chatbook/Chat/console_context_repository.py at a157d87cfaddd884f5a4d5d26ac0d20238f44be6. No assigned path was renamed or deleted. The canonical sorted path digest remained 11a85c1a0bd495783743a97a47f2b5f0da359158fcdd22b86532a22c229797bd.

Structural evidence: Python 3.12.11 parsed all twelve files before and after formatting with type_comments=True; only TypeIgnore.lineno was normalized, and AST dumps matched. Ordered comments, inline directive anchors plus significant-token positions, standalone Ruff directive adjacency, and fmt-range enclosed-node intervals matched. One pre-existing tied deepest-node case was resolved by deterministically anchoring the rightmost deepest AST node completed before the inline directive; capture and compare then covered all twelve paths. Ruff check reported All checks passed; Ruff format --check reported 12 files already formatted; git diff --check passed. Independent review found no Critical, Important, or Minor issue and confirmed exact scope, parent-blob Ruff replay, structural equality, and no behavior, security, or performance change.

Focused-test rationale: the batch owns Console context, memory selection, policy, prefill, prepared-request, and world-info behavior, so the eight assigned modules exercise every changed production owner and its adjacent regression surface without an unrelated full-suite sweep. Exact command: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_context_compaction.py Tests/Chat/test_console_context_policy.py Tests/Chat/test_console_context_policy_cas.py Tests/Chat/test_console_manual_memory_planning.py Tests/Chat/test_console_memory_selection.py Tests/Chat/test_console_prefill.py Tests/Chat/test_console_prepared_request.py Tests/Chat/test_console_world_info_application.py. Before formatting, after TASK-30040 and the final rebase, it reported 280 passed, 1 warning in 19.45s; after formatting it reported 280 passed, 1 warning in 20.19s; the final closeout replay reported 280 passed, 1 warning in 21.06s. The warning is the environment RequestsDependencyWarning; successful runs also emitted unrelated pre-existing pytest temporary-directory cleanup warnings after the summary. Exact governance command /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/CI/test_backlog_task_id_uniqueness.py reported 3 passed, 1 warning in 1.02s and 3 passed, 1 warning in 1.05s on final closeout replay. No full suite was run under repository policy.

Modified Python files: Tests/Chat/test_console_context_compaction.py; Tests/Chat/test_console_context_policy.py; Tests/Chat/test_console_context_policy_cas.py; Tests/Chat/test_console_manual_memory_planning.py; Tests/Chat/test_console_prefill.py; Tests/Chat/test_console_prepared_request.py; Tests/Chat/test_console_world_info_application.py; tldw_chatbook/Chat/console_context_compaction.py; tldw_chatbook/Chat/console_context_policy.py; tldw_chatbook/Chat/console_context_repository.py; tldw_chatbook/Chat/console_prefill.py. Added Docs/superpowers/plans/2026-09-03-task-26945-ruff-chat-console-context.md and updated this task record.

ADR required: no. This is mechanical formatter cleanup under TASK-26000 and introduces no architectural, persistence, security, dependency, or long-lived UX decision.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Formatted the exact twelve-path Console context batch with Ruff 0.15.22, preserving Python semantics and directive/comment structure, with 280 focused tests and all scoped governance checks passing.
<!-- SECTION:FINAL_SUMMARY:END -->
