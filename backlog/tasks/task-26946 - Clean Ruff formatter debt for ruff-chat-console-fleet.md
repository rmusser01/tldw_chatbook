---
id: TASK-26946
title: Clean Ruff formatter debt for ruff-chat-console-fleet
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-31 18:31'
updated_date: '2026-09-03 21:30'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
  - Docs/superpowers/plans/2026-09-03-task-26946-ruff-chat-console-fleet.md
priority: medium
---

<!-- TASK-26000-BATCH: ruff-chat-console-fleet -->
<!-- TASK-26000-PATHS-SHA256: 42889241af1499332c7ee76af9beac4087f9a7c552364f3dae3229ecdd9b1295 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chat-console-fleet` Ruff formatter batch at the owner boundary recorded as: Console fleet, wake, headless, and run-lifetime services.. The focused test surface recorded by TASK-26000 is `["Tests/Chat"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Chat/test_console_fleet_wake.py",
  "Tests/Chat/test_console_fleet_wake_safety.py",
  "Tests/Chat/test_console_fleet_wake_staleness.py",
  "Tests/Chat/test_console_fleet_wake_view_mark.py",
  "Tests/Chat/test_console_headless_wake_invariants.py",
  "Tests/Chat/test_console_run_state_per_session.py",
  "Tests/Chat/test_console_runtime_lifetime.py",
  "tldw_chatbook/Chat/console_fleet_attention.py",
  "tldw_chatbook/Chat/console_fleet_wake.py",
  "tldw_chatbook/Chat/console_launch_wake.py"
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile all ten TASK-26000 assigned paths against current origin/dev, record upstream lineage and the untouched focused-test baseline, and capture Python 3.12.11 AST/comment/directive evidence.
2. Run Ruff 0.15.22 format with all ten paths supplied explicitly, reject any unassigned Python diff, and require the structural comparison to match.
3. Run Ruff lint/format checks, the seven assigned Console fleet test modules, backlog task-ID uniqueness, and git diff --check; require the post-format focused result to introduce no failure beyond the untouched origin/dev baseline.
4. Commit only formatter-owned Python changes, request independent review, then record exact evidence and close TASK-26946 in a task-only commit.

ADR required: no
ADR path: N/A
Reason: Mechanical formatter cleanup under TASK-26000 introduces no architectural, persistence, security, dependency, or long-lived UX decision.

Detailed plan: Docs/superpowers/plans/2026-09-03-task-26946-ruff-chat-console-fleet.md
<!-- SECTION:PLAN:END -->
