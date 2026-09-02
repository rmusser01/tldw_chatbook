---
id: TASK-26988
title: Clean Ruff formatter debt for ruff-scheduling-notifications
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

<!-- TASK-26000-BATCH: ruff-scheduling-notifications -->
<!-- TASK-26000-PATHS-SHA256: d9351e3fbcb7ea44fb3ecacc65041d557d90c3e8769ca543128d9c579ac73d15 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-scheduling-notifications` Ruff formatter batch at the owner boundary recorded as: Scheduling and notification services with direct scheduling tests.. The focused test surface recorded by TASK-26000 is `["Tests/Scheduling"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Scheduling/test_briefing_handler.py",
  "Tests/Scheduling/test_briefing_projection.py",
  "Tests/Scheduling/test_config_flags.py",
  "Tests/Scheduling/test_handler_timeout.py",
  "Tests/Scheduling/test_missed_fire.py",
  "Tests/Scheduling/test_scheduled_tasks_db.py",
  "Tests/Scheduling/test_scheduled_watchlist_runs.py",
  "Tests/Scheduling/test_scheduler_observability.py",
  "Tests/Scheduling/test_scheduling_service.py",
  "Tests/Scheduling/test_server_client.py",
  "Tests/Scheduling/test_sync_engine.py",
  "Tests/Scheduling/test_watchlist_check_handler.py",
  "tldw_chatbook/Notifications/client_notifications_db.py",
  "tldw_chatbook/Notifications/event_state_repository.py",
  "tldw_chatbook/Scheduling/db/migrations/v1_to_v2.py",
  "tldw_chatbook/Scheduling/db/migrations/v2_to_v3.py",
  "tldw_chatbook/Scheduling/db/scheduled_tasks_db.py",
  "tldw_chatbook/Scheduling/scheduler/loop.py",
  "tldw_chatbook/Scheduling/services/briefing_projection.py",
  "tldw_chatbook/Scheduling/services/scheduling_service.py",
  "tldw_chatbook/Scheduling/services/sync_engine.py"
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
