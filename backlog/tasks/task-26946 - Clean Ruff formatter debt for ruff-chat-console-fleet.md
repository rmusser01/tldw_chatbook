---
id: TASK-26946
title: Clean Ruff formatter debt for ruff-chat-console-fleet
status: Done
assignee:
  - '@codex'
created_date: '2026-08-31 18:31'
updated_date: '2026-09-03 22:00'
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

## Implementation Notes

Implemented and verified the mechanical Ruff formatter cleanup in commit `2be4dd91cae7bb6905e9b8945c9dff2695178c17` (parent `e4652f9d379639bd39c13e3b2d005269da0e16d6`). Drift reconciliation against the TASK-26000 pin `e555df102c950c29beed5e7119f433d35eee1f3c` preserved the unchanged assigned-path digest `42889241af1499332c7ee76af9beac4087f9a7c552364f3dae3229ecdd9b1295`; there were no renames or deletions. The retained upstream-modified paths and last commits were: `Tests/Chat/test_console_fleet_wake.py` (`cf081bf725b252b4c7ef6c6fc50854791f8d3f82`), `Tests/Chat/test_console_runtime_lifetime.py` (`a5eabe7a872d1ce40bad93c660bd2692e30c6f2c`), `tldw_chatbook/Chat/console_fleet_attention.py` (`b3860842c0f74cb0c6a5a1d37e7d997cf30c9aa9`), and `tldw_chatbook/Chat/console_launch_wake.py` (`cf081bf725b252b4c7ef6c6fc50854791f8d3f82`).

Ruff 0.15.22 and Python 3.12.11 formatted exactly these ten paths: `Tests/Chat/test_console_fleet_wake.py`, `Tests/Chat/test_console_fleet_wake_safety.py`, `Tests/Chat/test_console_fleet_wake_staleness.py`, `Tests/Chat/test_console_fleet_wake_view_mark.py`, `Tests/Chat/test_console_headless_wake_invariants.py`, `Tests/Chat/test_console_run_state_per_session.py`, `Tests/Chat/test_console_runtime_lifetime.py`, `tldw_chatbook/Chat/console_fleet_attention.py`, `tldw_chatbook/Chat/console_fleet_wake.py`, and `tldw_chatbook/Chat/console_launch_wake.py`. Parent-blob Ruff reproduction passed for all ten paths; no unassigned Python path changed. The structural guard parsed with `ast.parse(..., type_comments=True)`, normalized only `TypeIgnore.lineno`, and confirmed AST/comment/directive/fmt-range equality. `ruff check` and `ruff format --check` passed on all ten paths, and `git diff --check` passed.

The seven assigned modules directly exercise fleet wake scheduling/safety/staleness/view marks, headless wake behavior, per-session run state, and runtime lifetime, so no unrelated full-suite sweep was run under repository policy. The exact pre-format command was `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_fleet_wake.py Tests/Chat/test_console_fleet_wake_safety.py Tests/Chat/test_console_fleet_wake_staleness.py Tests/Chat/test_console_fleet_wake_view_mark.py Tests/Chat/test_console_headless_wake_invariants.py Tests/Chat/test_console_run_state_per_session.py Tests/Chat/test_console_runtime_lifetime.py` (exit 1: 87 tests, 51 passed, 36 failures). The exact post-format command `LOGURU_LEVEL=ERROR /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q --tb=line --disable-warnings --junitxml=/tmp/task26946_post.xml Tests/Chat/test_console_fleet_wake.py Tests/Chat/test_console_fleet_wake_safety.py Tests/Chat/test_console_fleet_wake_staleness.py Tests/Chat/test_console_fleet_wake_view_mark.py Tests/Chat/test_console_headless_wake_invariants.py Tests/Chat/test_console_run_state_per_session.py Tests/Chat/test_console_runtime_lifetime.py` exited 1 with 87 tests, 51 passed, 36 failed, 0 errors, and 0 skipped; failure keys were identical (`added=[]`, `removed=[]`). Of the preserved failures, 35 wake/headless failures share the unassigned `tldw_chatbook/Chat/console_chat_controller.py:8169` defect: `AGENT_WAKE` intentionally has no preparation, so `preparation.capture_mode` is invalid. The separate `Tests/Chat/test_console_run_state_per_session.py::test_stop_active_run_cancels_only_viewed_sessions_task` persistence failure ends in unassigned `tldw_chatbook/Chat/console_chat_store.py:15409`; TASK-26946 fixes neither.

Governance verification: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/CI/test_backlog_task_id_uniqueness.py` returned 3 passed, 1 warning in 1.31s. No full suite was run. The tracked branch diff contains exactly the ten assigned Python files, `Docs/superpowers/plans/2026-09-03-task-26946-ruff-chat-console-fleet.md`, and this task record; SDD reports are ignored evidence inputs, not deliverables. ADR required: no; ADR path: N/A; reason unchanged from the plan.

## Final Summary

TASK-26946 is complete: all ten assigned files are Ruff-formatted with structural parity and clean Ruff/governance checks. Focused tests preserve the pre-existing 36-failure baseline; no production behavior was changed.
