---
id: TASK-32568
title: Restore Console recovery actions after failed attempts
status: In Progress
created_date: 2026-09-18 18:54
references:
- https://github.com/rmusser01/tldw_chatbook/issues/2708
- https://github.com/rmusser01/tldw_chatbook/pull/2709
documentation:
- backlog/decisions/079-console-library-conversation-authority.md
- backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md
updated_date: 2026-09-18 19:09
modified_files:
- tldw_chatbook/UI/Console_Modules/dispatch_recovery.py
- tldw_chatbook/UI/Console_Modules/prompt_queue.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/Chat/console_display_state.py
- tldw_chatbook/Widgets/Console/console_composer_bar.py
- Tests/UI/test_console_dispatch_recovery_repeated_actions.py
- Docs/User_Guide/console/chat-basics.md
- backlog/docs/lessons-console-wiring.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix GitHub issue #2708: failed response recovery can leave Retry and Discard inert and mislabel the composer blocker as an active run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #5 Retry and Discard remain actionable after a refused or failed recovery even when the displayed model state is unchanged.
- [x] #6 Pending recovery actions still reject duplicate intents; cancellation and exceptions allow subsequent recovery after completion.
- [x] #7 The composer identifies unresolved response recovery instead of telling the user to wait for an active run, and clears the reason after settlement.
- [ ] #8 All new mounted regressions pass; targeted existing tests introduce no failures relative to the recorded main baseline; touched code has no new lint or Bandit findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Bounded design approved in conversation. 1. Add failing mounted recovery and composer-copy regressions. 2. Reconcile the widget latch from model state and guarantee recovery completion repaint; pass recovery-specific blocker copy through the composer. 3. Run targeted tests, lint, Bandit, review the diff, document the incident, and commit.
ADR required: no new ADR.
ADR paths: backlog/decisions/079-console-library-conversation-authority.md and backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md.
Reason: routine bug fix preserving store-owned recovery, explicit actions, atomic settlement and app-runtime custody.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Worktree: /private/tmp/tldw-chatbook-issue-2708, branch codex/fix-2708-recovery-actions, base origin/main b0dadf1941. Governing ADR-079 defines store-owned recovery, explicit retry, atomic discard and no automatic replay; ADR-094 preserves app-runtime ownership. No new ADR required. Regression coverage exercises production recovery dispatch and real SQLite failure/settlement.
RED: new regressions initially produced 5 expected failures (inert second click/refusal/exception/cancellation and incorrect composer copy), with duplicate-click test passing. GREEN: all 8 final mounted regressions pass, including a real controller claim cancelled before provider entry, real SQLite discard failure, subsequent discard, draft-side repaint, and clear-on-settlement. Broader targeted selection: 172 passed, 3 failed; all 3 failures were reproduced in untouched origin/main baseline (20 passed, 3 failed): test_mounted_recovery_is_literal_actionable_and_owns_send_with_empty_queue; test_enter_hotkey_still_sends_when_send_is_enabled; test_enter_hotkey_queues_draft_behind_accepted_run. Baseline criterion clarified to require no added failures; no tests disabled or modified. Broader run also reported requests dependency compatibility warning and session file-descriptor growth warning. New test file and recovery widget pass Ruff lint/format; full touched-scope Ruff findings compared with HEAD yield no new findings after normalizing shifted source line references. Bandit scanned all five touched production files: 10 baseline findings, 10 current findings, zero new findings. Diff whitespace check passes. Independent read-only review found no blocking issues and requested stronger cancellation coverage, now included. Documentation and incident lesson updated; ADR-079/094 boundaries unchanged.
Full-file Ruff format check also reports existing formatting debt in four touched legacy files. The new test and recovery widget are format-clean; touched snippets are checked against formatting changes to avoid unrelated whole-file rewrites.
Final verification: 8 tests passed in Tests/UI/test_console_dispatch_recovery_repeated_actions.py, using the existing Chatbook .venv. Broader selection additionally included Tests/Chat/test_console_dispatch_recovery.py, Tests/UI/test_console_dispatch_recovery.py, fix_round1.py, fix_round2.py, test_console_send_disabled_state.py, test_console_prompt_queue.py and Tests/Chat/test_console_display_state.py (172 passed, baseline-only 3 failures). HEAD/current formatter comparison confirmed zero new formatting edits in all five production files. All six changed Python files parse. Security report: /tmp/bandit_chatbook_2708.json; baseline: /tmp/bandit_chatbook_2708_baseline.json. Full test log: /tmp/chatbook-2708-verification.txt. No external issue comments or publishing performed.
PR preparation 2026-09-18: user requested base dev. Rebased the unpublished branch cleanly onto origin/dev e89f28d751bc8a5b4f4545b8894b87437252c657 (405 commits ahead of the original main base). Revalidation on dev of the new regression file, UI recovery tests and Chat recovery tests: 68 passed, 8 failed. Four new mounted tests fail before reaching the changed behavior while constructing _ready_host: Backup_Recovery.bootstrap.RecoveryRequired(raw_source_selection_changed). Four existing controller tests fail (accepted retry; terminal-delete recovery success/failure; cancellation settlement timeout). These dev failures have not been independently classified as baseline-only. Previous eight-pass regression result and broader 172-pass/3-known-baseline-failure result apply to the original main-based commit. Opening a draft PR against dev with these limits clearly disclosed; final validation criterion reopened. Log: /tmp/chatbook-2708-dev-pr-tests.txt.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation is available in draft PR #2709 against dev. The recovery latch and composer-copy fixes passed all eight new mounted regressions on the original main base, with no new lint/format/Bandit findings relative to that base. Rebased cleanly onto dev; current targeted verification is 68 passed and 8 failed (four mounted setup failures with raw_source_selection_changed, plus four existing controller-test failures). Validation remains open and the task is In Progress. PR documents these limits and the pending human-written Change summary merge gate.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
