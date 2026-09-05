---
id: TASK-26948
title: Clean Ruff formatter debt for ruff-chat-console-interaction
status: Done
assignee:
  - '@codex'
created_date: '2026-08-31 18:31'
updated_date: '2026-09-05 15:24'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
  - Docs/superpowers/plans/2026-09-05-task-26948-ruff-chat-console-interaction.md
priority: medium
---

<!-- TASK-26000-BATCH: ruff-chat-console-interaction -->
<!-- TASK-26000-PATHS-SHA256: fda2c56e5364efd07fc80019191a0eaa8641a97e3a20fadbe4458c9d5b011de8 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chat-console-interaction` Ruff formatter batch at the owner boundary recorded as: Console send/edit/rewind/roleplay/session transaction services.. The focused test surface recorded by TASK-26000 is `["Tests/Chat"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Chat/test_console_edit_resend.py",
  "Tests/Chat/test_console_first_send_atomicity.py",
  "Tests/Chat/test_console_regenerate_branching.py",
  "Tests/Chat/test_console_rewind_modal.py",
  "Tests/Chat/test_console_rewind_summarize.py",
  "Tests/Chat/test_console_roleplay_identity.py",
  "Tests/Chat/test_console_roleplay_metadata.py",
  "Tests/Chat/test_console_send_gate_queue_race.py",
  "Tests/Chat/test_console_session_settings.py",
  "Tests/Chat/test_console_stop_reliability.py",
  "Tests/Chat/test_console_switcher_state.py",
  "Tests/Chat/test_console_transaction_contribution.py",
  "tldw_chatbook/Chat/console_image_edit_operations.py",
  "tldw_chatbook/Chat/console_roleplay_identity.py",
  "tldw_chatbook/Chat/console_roleplay_metadata.py",
  "tldw_chatbook/Chat/console_transaction_contribution.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and normalized significant-token position within the nearest logical owner (the same-line `except` clause for an `ExceptHandler` header, otherwise the nearest containing AST statement), excluding only parenthesis pairs independently proven AST-neutral by shadow parse/dump equality; preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile all 16 TASK-26000 assigned paths against current `origin/dev`, retain the one upstream-clean path in the ownership proof, capture the untouched 12-module focused baseline, and record Python 3.12.11 structural evidence.
2. Run Ruff 0.15.22 only on the 16 explicit assigned paths, require AST/comment/directive/fmt-range parity, and commit only the assigned Python files Ruff changes.
3. Require clean Ruff lint/format checks, exact focused-test failure-key parity, immutable-base byte replay, persistent-diagnostic verification, backlog-ID uniqueness, and `git diff --check`.
4. Obtain task-scoped and whole-branch independent reviews, record the exact evidence, check every acceptance criterion, and close TASK-26948 without broadening into unrelated behavior fixes.

ADR required: no
ADR path: N/A
Reason: Mechanical formatter cleanup under TASK-26000 introduces no architectural, persistence, security, dependency, or long-lived UX decision.

Detailed plan: Docs/superpowers/plans/2026-09-05-task-26948-ruff-chat-console-interaction.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

Applied Ruff 0.15.22 to the exact 16-path batch under authority cut `e555df102c950c29beed5e7119f433d35eee1f3c`, with batch digest `fda2c56e5364efd07fc80019191a0eaa8641a97e3a20fadbe4458c9d5b011de8`. The whole-branch base was `4e904f54db74497950eb31594fb37c8cd48568f3`; source blobs for deterministic replay came from the plan/setup revision `6bc95d1cef001f76de65ba7e42a8723ebecab07b`; and the formatter result was committed in Task 1 commit `799e6501e4b77d0b0464912962c6177575a9f66d`. Reconciliation found upstream drift on exactly `Tests/Chat/test_console_session_settings.py` and `Tests/Chat/test_console_switcher_state.py`; the latter was already formatted and remained in every proof. Ruff changed 15 paths and left one unchanged. No unassigned Python path or handwritten behavior changed.

The Python 3.12.11 structural guard (SHA-256 `3fac070e94fe91cd152f956b19093c457c48787ea5449b54945b2305386b7471`) matched all 16 paths, including AST, ordered comments, directive anchors/positions, standalone directive adjacency, and formatter ranges. Ruff `check` passed, and `ruff format --check` reported all 16 files already formatted. Immutable-base replay formatted source blobs from plan/setup revision `6bc95d1cef001f76de65ba7e42a8723ebecab07b` and matched all 16 worktree files byte-for-byte in formatter result commit `799e6501e4b77d0b0464912962c6177575a9f66d`.

The exact 12-module focused pytest command ran before and after formatting (only the JUnit output path differed):

```text
LOGURU_LEVEL=ERROR /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q --tb=line --disable-warnings --junitxml=/tmp/task26948_before.xml Tests/Chat/test_console_edit_resend.py Tests/Chat/test_console_first_send_atomicity.py Tests/Chat/test_console_regenerate_branching.py Tests/Chat/test_console_rewind_modal.py Tests/Chat/test_console_rewind_summarize.py Tests/Chat/test_console_roleplay_identity.py Tests/Chat/test_console_roleplay_metadata.py Tests/Chat/test_console_send_gate_queue_race.py Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_stop_reliability.py Tests/Chat/test_console_switcher_state.py Tests/Chat/test_console_transaction_contribution.py
LOGURU_LEVEL=ERROR /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q --tb=line --disable-warnings --junitxml=/tmp/task26948_after.xml Tests/Chat/test_console_edit_resend.py Tests/Chat/test_console_first_send_atomicity.py Tests/Chat/test_console_regenerate_branching.py Tests/Chat/test_console_rewind_modal.py Tests/Chat/test_console_rewind_summarize.py Tests/Chat/test_console_roleplay_identity.py Tests/Chat/test_console_roleplay_metadata.py Tests/Chat/test_console_send_gate_queue_race.py Tests/Chat/test_console_session_settings.py Tests/Chat/test_console_stop_reliability.py Tests/Chat/test_console_switcher_state.py Tests/Chat/test_console_transaction_contribution.py
```

Both runs collected 452 tests, with 5 failures, 0 errors, 0 skips, and exit code 1; normalized failure keys matched exactly. The inherited failures were `Tests/Chat/test_console_edit_resend.py::test_edit_and_resend_forks_user_sibling_and_streams_reply`, `Tests/Chat/test_console_regenerate_branching.py::test_regenerate_mid_conversation_failure_restores_selected_anchor_not_former_tail`, `Tests/Chat/test_console_regenerate_branching.py::test_regenerate_persists_new_sibling_when_store_has_persistence`, `Tests/Chat/test_console_regenerate_branching.py::test_regenerate_stream_failure_retains_failed_sibling_and_restores_anchor`, and `Tests/Chat/test_console_session_settings.py::test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh`. No full suite ran per repository instruction and lack of opt-in.

Exact governance checks: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/CI/test_backlog_task_id_uniqueness.py` passed (3 passed, with two environment/dependency warnings); `git diff --check` exited 0; `python3.12 scripts/check_persistent_diagnostic_inventory.py --diff` exited 0 with no persistent-diagnostic drift. ADR required: no; ADR path: N/A. This is mechanical TASK-26000 formatter cleanup. Independent Task 1 review was clean after the report correction loop.
