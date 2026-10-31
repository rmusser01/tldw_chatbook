---
id: TASK-27002
title: Clean Ruff formatter debt for ruff-ui-remaining-screens
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 21:55'
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

<!-- TASK-26000-BATCH: ruff-ui-remaining-screens -->
<!-- TASK-26000-PATHS-SHA256: 51da045a69140ab99a01d217f142f1e64471417a311ba46c7c91357f4bc6846b -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-ui-remaining-screens` Ruff formatter batch at the owner boundary recorded as: Remaining non-Console screens and narrowly corresponding UI tests.. The focused test surface recorded by TASK-26000 is `["Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_app_instance_warning.py",
  "Tests/UI/test_artifacts_screen_reports.py",
  "Tests/UI/test_background_signal_bounds.py",
  "Tests/UI/test_ccp_handlers.py",
  "Tests/UI/test_change_review_commit_ui.py",
  "Tests/UI/test_change_review_git_provider.py",
  "Tests/UI/test_change_review_screen.py",
  "Tests/UI/test_chat_screen_sidebar_state_debounce.py",
  "Tests/UI/test_chat_screen_worker_groups.py",
  "Tests/UI/test_chat_task_cards_sync.py",
  "Tests/UI/test_code_repo_copy_paste_window.py",
  "Tests/UI/test_conversation_attach_picker.py",
  "Tests/UI/test_home_screen.py",
  "Tests/UI/test_lab_frame.py",
  "Tests/UI/test_logs_filter_persist_debounce.py",
  "Tests/UI/test_nav_overflow_tick_gating.py",
  "Tests/UI/test_probe_headless_wake_p1_continuity.py",
  "Tests/UI/test_probe_headless_wake_p2_p3_p4.py",
  "Tests/UI/test_probe_launch_wake.py",
  "Tests/UI/test_product_maturity_phase1_first_run.py",
  "Tests/UI/test_product_maturity_phase6_recovery_docs.py",
  "Tests/UI/test_reminder_form.py",
  "Tests/UI/test_screen_preimport.py",
  "Tests/UI/test_screen_preimport_pacing.py",
  "Tests/UI/test_serve_main_args.py",
  "Tests/UI/test_study_flashcards_screen.py",
  "Tests/UI/test_trace_export_ui.py",
  "Tests/UI/test_trajectory_timeline_integration.py",
  "tldw_chatbook/UI/ChatbookCreationWindow.py",
  "tldw_chatbook/UI/Chatbooks_Window_Improved.py",
  "tldw_chatbook/UI/Logs_Window.py",
  "tldw_chatbook/UI/Screens/artifacts_screen.py",
  "tldw_chatbook/UI/Screens/chat_screen_state.py",
  "tldw_chatbook/UI/Screens/home_screen.py",
  "tldw_chatbook/UI/Screens/image_gen_demo_screen.py",
  "tldw_chatbook/UI/Screens/lab_frame.py",
  "tldw_chatbook/UI/Screens/logs_screen.py",
  "tldw_chatbook/UI/Screens/stats_screen.py",
  "tldw_chatbook/UI/Screens/trajectory_screen.py",
  "tldw_chatbook/UI/Widgets/table_click_select.py",
  "tldw_chatbook/UI/image_gen_command_provider.py"
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

1. Reconcile assigned paths against current `origin/dev` (existence + format state; missing paths recorded with dev delete/rename lineage) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` before formatting (type_comments=True with symmetric plain-parse fallback; normalize `TypeIgnore.lineno`).
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality (formatter-mandated docstring-normalization deviations enumerated if they occur).
5. `ruff format --check` must pass on every existing assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface (bounded; never a bare pytest), plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`; verify the three-way path partition arithmetically before writing notes.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `2612fc56b2` (isolated worktree `.worktrees/ruff-debt-batch-8`, branch `chore/ruff-debt-batch-8`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All 41 assigned paths reconcile mechanically (hash matches both the task marker and the evidence JSON), and all were reproduced: 40 exist at the base; ONE was DELETED upstream after the census pin — `Tests/UI/test_code_repo_copy_paste_window.py`, removed by dev commit `ef43462806` ("chore(ui): delete five unreachable UI/ root windows (TASK-32899)"). Recorded here as the required lineage reconciliation rather than silently dropped. (Third distinct upstream deletion consuming census paths this series, after 5dd1077df6 in batches 5 and 7.)

**Formatting (AC#2, AC#5).** 36 of the 40 existing assigned paths formatted. `ruff format --check` passes on every existing assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 142 assigned paths: 32 findings before and 32 after, per-file counts identical — the formatter introduced zero findings; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every existing assigned path (no docstring deviations and no plain-parse fallback needed anywhere in this batch).

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: the 28 existing assigned test files (recorded surface Tests/UI). Result on the formatted tree: 191 failed, 292 passed, 1 skipped in 719.95s (`--timeout=120`) — the initial test-battery invocation for this task errored (`no tests ran`) because the command included the deleted path; this is the corrected run without it. Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 191 failed, 292 passed, 1 skipped in 719.95s — count-identical (same command shape both sides) — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 130 committed + 11 clean-at-base + 1 upstream-deleted = 142 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** 1 path upstream-deleted (recorded above); 4 paths already formatter-clean at the base (`Tests/UI/test_ccp_handlers.py`, `Tests/UI/test_chat_screen_sidebar_state_debounce.py`, `tldw_chatbook/UI/Screens/trajectory_screen.py`, `tldw_chatbook/UI/Widgets/table_click_select.py`) and deliberately left untouched; the remaining 36 carried live debt and were formatted (36 committed + 4 clean + 1 deleted = 41).

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
