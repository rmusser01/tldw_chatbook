# TASK-26951 general Chat formatter cleanup

Spec: `Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md` and TASK-26951's owner-approved test-only exception.

ADR required: no
ADR path: N/A
Reason: mechanical formatting under the existing TASK-26000 contract, with three explicitly approved test-only lint corrections.

## Global Constraints

Only the 19 Python paths in TASK-26951 may change. Use Python 3.12.11 and Ruff 0.15.22. Preserve AST structure (including type comments), ordered comments, directive anchors, and fmt intervals, except the three exact owner-approved test corrections. No hand-written production behavior changes. No full suite: use all 15 assigned test modules with identical before/after bounded commands. Record inherited failures honestly, not as passing tests. Do not absorb other failures or paths. Root owns task documentation and integration.

## Task 1: Apply and verify the assigned batch

Read TASK-26951 for the complete 19-path list and precise approved exceptions. Capture original structural evidence using `/tmp/task26947_format_guard.py` and run all 15 assigned test modules before editing, using the main checkout's `.venv/bin/python`, `-q --timeout=10 --timeout-method=signal --timeout-disable-debugger-detection --show-capture=no --tb=short`, and `/tmp/task26951_before.xml`. Preserve output and exact commands in the report. Report baseline failures; do not fix unrelated behavior.

Apply only the three approved test corrections using apply_patch; prove that exact AST delta against original files, then capture a corrected pre-format baseline. Run Ruff format on all assigned paths, prove structural parity against the corrected baseline and comment preservation against original, and verify deterministic Ruff byte replay from corrected base blobs. Run Ruff check and format check on all assigned paths and repeat the exact focused suite to `/tmp/task26951_after.xml`; compare every test identity/outcome, not just counts. Commit only assigned Python changes after self-review. Return report with command evidence, tool versions, changed paths, hashes, and concerns.

## Task 2: Review and integrate

Root reconciles authority cut `e555df102c950c29beed5e7119f433d35eee1f3c` with integration base `c0fa6639a1fd294bf2bfbdc043c0dcb70782a689`, verifies governance and diagnostic inventory, and records complete evidence in TASK-26951. Independent task review and final branch review precede PR publication. Rebase if needed, reverify affected gates, request Qodo review, wait for CI, then merge and remove only this clean merged worktree and its branch.
