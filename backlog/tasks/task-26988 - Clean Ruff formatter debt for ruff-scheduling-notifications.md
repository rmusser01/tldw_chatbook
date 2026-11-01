---
id: TASK-26988
title: Clean Ruff formatter debt for ruff-scheduling-notifications
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 14:05'
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
- [x] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

1. Reconcile assigned paths against current `origin/dev` (existence + format state; missing paths recorded with their dev delete/rename lineage) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` before formatting (type_comments=True with symmetric plain-parse fallback where prose `# type:` comments are unparseable; normalize `TypeIgnore.lineno`).
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality (formatter-mandated docstring-normalization deviations enumerated if they occur).
5. `ruff format --check` must pass on every existing assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface (bounded; never a bare pytest), plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`; verify the three-way path partition arithmetically before writing notes.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `f80d3e0090` (isolated worktree `.worktrees/ruff-debt-batch-6`, branch `chore/ruff-debt-batch-6`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All assigned paths exist unchanged at the base — no deletes or renames to reconcile. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 21 of 21 assigned paths formatted. `ruff format --check` passes on every assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 233 assigned paths: 90 findings before and 44 after, ZERO increases — the formatter incidentally resolved 46 line-length findings in `Tests/Performance/run_console_three_turn_profile.py` (8), `Tests/Performance/test_console_three_turn_profile.py` (15), `Tests/RAG/simplified/test_collection_fingerprint.py` (16), and `Tests/RAG/test_active_config_resolution.py` (7) by reflowing; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every assigned path (the batch-5 symmetric plain-parse fallback was not needed on any file this batch).

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: `Tests/Scheduling` (recorded surface, bounded directory). Result on the formatted tree: 95 failed, 997 passed in 832.14s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 95 failed, 997 passed in 747.94s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 226 committed + 7 clean-at-base + 0 missing = 233 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** No already-formatted paths; all 21 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
