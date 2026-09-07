---
id: TASK-513
title: Repair stale QuestionAnswerRunner patch targets
status: Done
assignee: []
created_date: '2026-07-24 18:14'
updated_date: '2026-07-24 18:16'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore Evals integration tests after QuestionAnswerRunner moved to eval_runner by patching the class at its actual definition without changing evaluation runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All Evals tests patch QuestionAnswerRunner at its current eval_runner definition
- [x] #2 Concurrent, error-handling, performance, and real integration fixtures retain their existing mocked-LLM behavior
- [x] #3 No production Evals code changes are made
- [x] #4 Focused stale-target tests and the full Evals suite pass
- [x] #5 Task documentation includes the ADR decision, base comparison, verification, and implementation notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the exact failure on both the feature branch and merge base, and inventory every stale specialized_runners.QuestionAnswerRunner patch target.
2. Mechanically redirect those test patches to eval_runner.QuestionAnswerRunner, where the class is defined and instantiated.
3. Run the affected integration test files and full Evals suite.
4. Run Ruff format/check and git diff --check; independently review before completion.

ADR required: no
ADR path: N/A
Reason: This is a test-only correction to stale mock import paths after an existing class move; it changes no runtime boundary, dependency, storage, or architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Repaired all stale QuestionAnswerRunner LLM patch targets in the Evals integration tests without changing production behavior.

Approach and base comparison:
- On both merge base ba6b45cdf4dd548796e072f5933cdcf44c8c0344 and the feature branch, the first affected test failed while resolving tldw_chatbook.Evals.specialized_runners.QuestionAnswerRunner._call_llm because specialized_runners no longer exports that class.
- Confirmed QuestionAnswerRunner is defined in tldw_chatbook/Evals/eval_runner.py and instantiated there by the runner factory.
- Mechanically redirected all eight stale patch strings across the three integration files to tldw_chatbook.Evals.eval_runner.QuestionAnswerRunner._call_llm; side effects, return values, concurrency, error recovery, and performance fixture behavior were otherwise unchanged.

Verification:
- Affected integration files: 40 passed, 2 skipped.
- Full Tests/Evals suite: 279 passed, 13 skipped.
- Ruff format check: 3 files already formatted.
- Ruff check: all checks passed.
- Repository-wide Tests/Evals search: zero stale specialized_runners.QuestionAnswerRunner targets.
- git diff --check: clean for owned files.
- Self-review: the diff contains only eight module-path substitutions plus this task record; no production Evals files changed.

ADR required: no
ADR path: N/A
Reason: Test-only mock maintenance preserves the existing evaluation runtime boundary.

Files modified:
- Tests/Evals/test_eval_integration.py
- Tests/Evals/test_integration.py
- Tests/Evals/test_eval_integration_real.py
- backlog/tasks/task-513 - Repair-stale-QuestionAnswerRunner-patch-targets.md
<!-- SECTION:NOTES:END -->
