---
id: TASK-34601
title: Cut Console Send latency by measured trace-fix-measure iterations
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-08 11:20'
updated_date: '2026-10-08 11:20'
labels:
  - performance
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console Send on native Windows takes ~6 s (warm) to ~7 s (cold) from the Send action to the provider adapter entry, excluding provider time, against the agreed targets of under 100 ms to first rendered feedback and under one second of application overhead (ADR-222). Earlier work fixed many correctness issues but never demonstrated a whole-Send latency gain. This task runs the loop the owner asked for: trace the current Send, fix the largest measured removable cause, re-measure the same scenario under comparable conditions, repeat. Each iteration ships separately with clean before/after evidence, so every gain is attributable to one change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The native Send probes run at the current source again (harness compatible with the received-intent preparation keyword)
- [ ] #2 Each shipped iteration has interleaved, observer-free native before/after receipts of the same scenario with exact revisions, and the gain is attributed to that one change
- [ ] #3 Windows admission metadata observations cost less per node with unchanged refusal verdicts, descriptor projections and fresh TokenOwner semantics where they matter
- [ ] #4 A fresh ChaChaNotes connection never leaves its journal_mode statement active, so retained frames cannot refuse a Send's commit
- [ ] #5 A Send attempt reads hook permissions once for its preparation consumers while effect gates and the final pre-dispatch check stay fresh
- [ ] #6 The live-turn transcript poll publishes progress without re-running the full config-locked reconciliation on every tick, and full reconciliation still runs on direct requests, periodically and at the end
- [ ] #7 Storage admission avoids repeated native re-walks inside one acquisition and between unchanged warm acquisitions on Windows, with full re-observation on any notified change, error or backstop expiry (ADR-126 amendment)
- [ ] #8 Warm and cold Send-to-provider-entry and first-feedback times are reported against the targets with the remaining unexplained delay
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Repair the measurement harness; establish observer-free baselines at integration HEAD 36c6fb431a (sequential native runs only, guarded against overlapping runs).
2. Attribute the critical path with an out-of-repo all-thread stack sampler aligned to the probe's stage clock (never retaining frames).
3. Iteration 1: cut Windows native primitive cost (security descriptor read, conditional TokenOwner read, single node-tree build and component validation, qualification parse memo).
4. Harden the ChaChaNotes journal_mode statement (found while attributing).
5. Owner-approved levers, each implemented in its own worktree with adversarial review, then integrated and measured one at a time: hook snapshot per attempt, narrowed live poll, single path fence per acquisition, change-notified Windows evidence reuse.
6. Record receipts, ADR amendments, ledger rows and lessons.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Trace → fix → re-measure, one attributable change at a time, on native Windows with the full app (private profile, file-backed DBs, only the provider adapter stubbed). Latency claims come only from observer-free, guarded (no overlapping native run), interleaved A/B runs of `Tests/Performance/test_console_send_wall_clock.py`; attribution came from an out-of-repo all-thread stack sampler aligned to the probe's stage clock.

- **Harness:** both Send probes wrapped `accept_received_intent(intent)` and crashed after `6cad32ed67` added `_configuration_preparation`; they now forward keyword arguments.
- **Iteration 1 (`Utils/windows_files.py`, `Backup_Recovery/qualification.py`):** `security()` uses `NtQuerySecurityObject` (GetSecurityInfo silently re-read the parent's descriptor for every app-created directory); TokenOwner is read only when it can change the projection; the admission snapshot builds its node tree and validates components once; the qualification JSON parse is memoized by exact text. 40-node snapshot 10.2–10.9 → 4.8–5.1 ms; warm Send 6.0 → 4.5 s.
- **PRAGMA hardening (`DB/ChaChaNotes_DB.py`):** the journal_mode statement is fetched. A retained cursor (any frame-retaining tool — here the first sampler) made every later commit fail with "SQL statements in progress".
- **L1 hook read per attempt** (`Agents/hook_permissions.py`, `Agents/run_hooks.py`, `Chat/console_*`): one full hook-permission read serves the pre-commit preparation consumers; effect gates and the final pre-dispatch admission stay fresh; mismatched context falls back to a fresh read. Warm −0.76 s, cold −1.35 s.
- **L2 narrowed live poll** (`UI/Console_Modules/poll_cadence.py`, `UI/Screens/chat_screen.py`): light publication ticks; full reconciliation on direct requests, ≤2 s and at the stop tick. No Send-latency change; fewer UI stalls. Rebased onto #3023, whose own Preparing-poll narrowing (`_sync_console_poll_display_ui`) now serves every poll-driven full pass; only complete (not Preparing-narrowed) passes reset the 2 s cadence.
- **L3 one path fence per acquisition** and **L4 change-notified Windows evidence** (`Backup_Recovery/storage_admission.py`, `Utils/windows_files.py`): see the ADR-126 TASK-34601 amendment. Native opens per warm Send 37.9k → ~20k; the native pause probe's open budget passes again.
- Docs: ADR-126 amendments (iteration 1; L3/L4), ADR-222 note (L1), User Guide cadence note (L2), ledger rows, lessons (sampler frame retention, `monkeypatch.undo`, by-id watch handles, shared `git stash`).

Pre-existing failures observed identically on unmodified HEAD (not caused here): 10 in `test_participant_lifetimes.py`, 4 hook tests (`test_console_run_hooks_regressions` ×2, `test_hook_permissions` unsafe-store and v2-revoke), `test_console_presentation_cadence::test_cancelled_actual_count_read_drains_on_its_receiver_and_retries`, flaky `test_admission::test_known_incompatible_is_refused_until_os_lifetime_ends`; symlink tests need a privilege this host lacks.
<!-- SECTION:NOTES:END -->
