---
id: TASK-33269
title: 'PERF-10: Legacy trace maintenance - park when complete, incremental trace
  GC'
status: Done
created_date: 2026-09-28 18:02
dependencies:
- TASK-33268
labels:
- performance
- console
- database
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
assignee:
- '@claude'
updated_date: 2026-09-29 02:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
New evidence for TASK-31501. The ~1 Hz forever legacy trace-maintenance tick now costs 6-9 ms of CPU per tick, and on an unconnected executor thread it spawns a private-SQLite helper every tick (79 ms wall, about 45 ms child CPU). Trace GC re-marks the whole reachable ledger inside BEGIN IMMEDIATE every 60 s after any send. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-10; every issue with file:line is listed under PERF-10 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Once legacy normalization is logically complete, the maintenance loop performs no periodic DB work until a new exchange is written
- [x] #2 The completion check does not take a write transaction
- [x] #3 Incremental trace GC (work proportional to new data, not ledger size) is split out to TASK-33461; this task does not change the GC pass
- [x] #4 The idle probe shows zero helper spawns and zero write transactions from trace maintenance
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Legacy trace maintenance now **parks** once a pass reports normalization logically complete, instead of calling `run_batch` once a second forever. Each of those calls was a `BEGIN IMMEDIATE` plus an admission and, through `run_owned_db_call`, a fresh connection and private-SQLite helper spawn.

**Wake-up.** The parked loop (`console_runtime.py`) wakes on either:
- the work signal: `chat_persistence_service.signal_trace_maintenance_work()`, a `threading.Event` set by `append_message_exchanges`, the only writer of exchange rows;
- a due physical GC pass, still every `TRACE_PHYSICAL_MAINTENANCE_INTERVAL_SECONDS`.

It polls the event every `LEGACY_TRACE_MAINTENANCE_PARK_POLL_SECONDS` (1 s) in memory only.

**Read-only pre-check.** `LegacyTraceMaintenance._complete_without_pending_work()` does a read-only `transaction()` check before `run_batch` opens its immediate transaction, so a wake with nothing new takes no write lock.

**Scope change.** Incremental trace GC (original AC #3) is not done here; it is split to TASK-33461.

**Evidence**
- `Tests/Chat/test_console_trace_maintenance_parking.py`:
  - parks: 1-2 `run_batch` calls in 0.3 s, where it used to be one per tick;
  - wakes on the exchange signal;
  - the append raises the signal.
- `test_complete_check_without_new_rows_takes_no_write_transaction` in `Tests/Chat/test_console_trace_legacy_migration.py`.
- **Idle probe** (AC #4): real app on a scratch profile with Console open, 15 s idle after the first pass, counting audit events.
  - Base (c174e30f6b): 13 `run_batch` calls, 15 `sqlite3.connect`, 15 `subprocess.Popen`.
  - Branch: **0** `run_batch` calls; 2-7 connects, and a caller trace attributes every one to other subsystems (Workspaces registry, Console character context), none to trace maintenance.
- **Regression:** 48 test files touching trace maintenance, exchanges, persistence and Console runtime. The branch fails 169 of 1,391. The same 169 IDs, rerun on base c174e30f6b, all fail too: the RecoveryRequired class tracked as TASK-33370, plus drift tracked as TASK-33371. No new failures.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
