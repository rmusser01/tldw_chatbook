# TASK-3070.14 Console Decomposition Closeout Plan

> **For Codex:** Execute this plan in the isolated `codex/task-3070-14-console-closeout` worktree. Do not run a local full test suite.

**Goal:** Lock the completed Console decomposition into the one-way size ratchet at the exact final rebased `ChatScreen` counts and close the coordinated backlog parent with reproducible evidence.

**Architecture:** Keep the frozen Wave 6 delivery history unchanged, add an explicit final-closeout count oracle for the live tree, and make the canonical ratchet the single live authority. Synchronize architecture tests that intentionally inspect that authority; do not change production code or introduce a second ratchet.

**Tech Stack:** Python 3.11+, pytest, Ruff, AST source inspection, Backlog.md CLI.

ADR required: no

ADR path: `backlog/decisions/068-console-text-selection-and-annotations.md`

Reason: The closeout applies already approved controller boundaries and preserves ADR-068 ownership. It changes architecture evidence and task records only.

---

## Task 1: Establish the exact live-tree closeout oracle

**Files:**

- Modify: `Tests/Architecture/test_console_wave6_closeout_inventory.py`
- Test: `Tests/Architecture/test_console_wave6_closeout_inventory.py`

1. Add `FINAL_CLOSEOUT_COUNTS` for the measured, final-rebased physical line and direct AST method-definition counts. The initial `16_968 / 562` measurement must be refreshed if dev advances before merge; after PR #2125 and subsequent Console work through PR #2143 landed, the final live value became `17_025 / 565`. This deliberately differs from TASK-3070.13's 532 unique method names and preserves the rule that the landed ratchet exactly matches the live base.
2. Preserve `IMMUTABLE_BUDGETS` as the historical Wave 6 ceiling used by the deficit arithmetic.
3. Require the live source count, `FINAL_CLOSEOUT_COUNTS`, and the canonical live budget to be exactly equal, while retaining the frozen-revision arithmetic against `IMMUTABLE_BUDGETS`.
4. Run the exact closeout evidence test and confirm it fails because the live ratchet still contains 17,727/593.

## Task 2: Lower the one-way ratchet and synchronize its consumers

**Files:**

- Modify: `Tests/Architecture/test_screen_size_ratchet.py`
- Modify: `Tests/Architecture/test_console_review_selection_controller_boundary.py`
- Test: the three modified architecture modules

1. Lower `_BUDGETS` to the exact final-rebased counts without changing the measured production file.
2. Document the Wave 6 closeout measurement and preserve the rule that budgets never increase.
3. Update the review/selection boundary test so its live-ratchet assertion matches the canonical closeout value; keep its frozen task-base projection unchanged.
4. Preserve `_TASK_22507_4_CHAT_SCREEN_BASE` as a separate historical task-local non-worsening guard, not an alternative live decomposition budget.
5. Run the modified architecture modules and confirm green.

## Task 3: Verify the approved focused surface

**Files:**

- Verify only; no production files are expected to change.

1. Run the exact bounded union recorded by TASK-3070.12 and TASK-3070.13: `test_console_realtime_wiring.py`, `test_console_realtime_controller.py`, `test_console_controller_wiring.py`, the mic-button realtime routing node, `test_console_realtime_loop.py`, `test_realtime_mic_tap.py`, `test_realtime_protocol.py`, `test_console_review_selection_controller.py`, `test_change_review_opener_roots.py`, `test_console_annotation_markers.py`, `test_console_selection_end_to_end.py`, `test_console_turn_undo_all.py`, and `test_trajectory_live.py`.
2. Run `test_console_wave6_inventory.py`, `test_console_wave6_closeout_inventory.py`, `test_console_realtime_controller_boundary.py`, `test_console_review_selection_controller_boundary.py`, and `test_screen_size_ratchet.py` as the complete relevant architecture set.
3. Run targeted Ruff check and format verification on the modified Python test files.
4. Run the repository diagnostic inventory, backlog-ID, privacy, live-base drift, and diff gates applicable to this closeout.
5. Assert the branch changes no path under `tldw_chatbook/`, and run `git diff --check`.
6. Do not run a local full suite; required GitHub Actions provide broad integration evidence.

## Task 4: Close repository records

**Files:**

- Modify: `Docs/superpowers/specs/2026-08-23-console-decomposition-wave6-closeout-amendment.md`
- Modify: `backlog/tasks/task-3070.14 - Close-amended-Console-decomposition-ratchet.md`
- Modify: `backlog/tasks/task-3070 - chat_screen-size-ratchet-red-on-dev-after-console-decomposition-wave-3.md`

1. Audit TASK-3070.1 through TASK-3070.13 and sibling TASK-21201 on `origin/dev` for Done status, checked acceptance criteria, implementation plan, implementation notes, ADR/lesson hygiene, focused evidence, and review evidence. TASK-3070.14 remains In Progress until its own required CI and review complete.
2. Record the final exact counts and ratchet closure in the approved amendment.
3. Add interim TASK-3070.14 implementation notes after local evidence, but do not check AC #2 or mark it Done before required CI and review complete.
4. Keep TASK-3070 In Progress until every child satisfies the repository DoD and TASK-3070.14 required CI/review are green.
5. Re-run backlog-ID and diff gates after generated backlog manifests settle.

## Task 5: Prepare review delivery

1. Re-fetch `origin/dev`; if it moved, rebase, re-measure, and repeat the relevant gates before recording final counts.
2. Commit the closeout changes, push the branch, and open a PR with the exact verification evidence and explicit statement that no local full suite was run.
3. Wait for required CI and review feedback, re-check live-base drift, then check TASK-3070.14 and parent acceptance criteria and mark both Done in a final records commit.
4. Re-run backlog-ID and diff gates, push the final records commit, wait for its required CI, and merge only when review, current-base, and required-check conditions remain satisfied.
