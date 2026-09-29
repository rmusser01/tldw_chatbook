---
id: TASK-33262
title: 'PERF-03: Logging pipeline defeats level filtering and redacts every record
  three times'
status: Done
created_date: 2026-09-28 18:02
labels:
- performance
- logging
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
assignee:
- '@claude'
updated_date: 2026-09-29 04:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Logging_Config.py forwards every loguru record to stdlib at level TRACE, so loguru's early level check never fires. Every logger.debug costs about 7-8 us instead of 0.15 us, opt(lazy=True) guards (TASK-275) are defeated, and a dropped opt(exception=True).debug formats a full traceback. Every INFO+ record is redacted three times (shouldRollover, emit, Logs buffer), about 340 us, and flushed synchronously on the emitting thread, the event loop included. The DB layer and importer log INFO per row. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-03; every issue with file:line is listed under PERF-03 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A dropped debug record costs under 1 us (measured) and opt(lazy=True) callables are not evaluated when DEBUG is off
- [x] #2 Each INFO+ record is redacted exactly once, with redaction regression tests unchanged or extended
- [x] #3 File and Logs-buffer handlers do not perform file I/O on the event-loop thread
- [x] #4 Per-row DB/importer INFO logs are demoted, and app-owned worker transitions no longer emit a WARNING each
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Level gate: install the loguru->stdlib forwarder at the effective stdlib threshold (min of root and tldw_chatbook levels; TRACE kept when DEBUG) instead of TRACE; add sync_loguru_forward_level() and call it wherever those levels change (configure_application_logging, TldwCli.__init__).
2. Single-pass redaction: redact message+exception+stack once per record (cached on the record) and have RedactingFileFormatter compose each sink's prefix around it; the Logs-buffer handler (moved to Logging_Config as LogsBufferHandler) reuses the same formatter, so shouldRollover, the file emit and the buffer share one redaction.
3. Off-thread sinks: one daemon log-writer thread; the private file handler and the Logs buffer defer emit to it (handlers stay on the root logger, so existing lookups keep working); the buffer hands stored lines back to the UI loop with call_soon_threadsafe; close()/maintenance pause drain first; stop at unmount and atexit (before logging.shutdown).
4. Volume: demote per-row ChaChaNotes CRUD and chatbook-importer per-item INFO to DEBUG; unhandled app-owned worker transitions log DEBUG, WARNING only for ERROR; make @timeit coroutine-aware and log its summary at DEBUG.
5. Tests first for each step; isolated micro-benchmark before/after; run logging/redaction/worker suites + preflight; compare failures to base c174e30f6b.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Single-pass, level-gated, off-thread logging pipeline. Centralised in Logging_Config.py; app.py, base_handler.py and metrics_logger.py carry small call-site changes, and ChaChaNotes_DB.py and chatbook_importer.py only change log levels.

Level gate (AC #1)
- configure_application_logging installs the loguru->stdlib forwarder at _loguru_forward_level(): the lower of the root and tldw_chatbook effective levels, or 0 when that is DEBUG or lower (TRACE is forwarded as DEBUG). It no longer uses TRACE. sync_loguru_forward_level() re-levels the forwarder by adding the new sink before removing the old one, and never brings back a forwarder that someone else removed.
- The level only changes at run time in two places: configure_application_logging (root level, then lowered to the most verbose handler) and TldwCli.__init__'s "Disable debug logging for performance" block. That block used to change stdlib levels only and so had no effect on loguru. Both now call the sync. There is no CLI log-level flag, and the Settings UI only writes config that the next boot reads.

Single redaction (AC #2)
- RedactingFileFormatter redacts each record's body once: message, exception text and stack, i.e. everything a caller controls. The result is cached on the record (_redacted_body), and each sink puts its own prefix around it (timestamp, level, logger name, line number, all set by code). RotatingFileHandler.shouldRollover, the file emit and the Logs buffer now share one sanitizer pass; there used to be three.
- The Logs-buffer handler moved from a class defined inside app.py to Logging_Config.LogsBufferHandler, which uses the same formatter. Its " - " layout is unchanged; Logs_Window._styled_line parses it.
- The truncation cap now applies to the body rather than the whole line. The new test walks a secret across the cap to prove nothing leaks there. No existing redaction or sanitizer test was edited.

Off-thread sinks (AC #3)
- A shared daemon thread, tldw-log-writer, does the work. PrivateRotatingFileHandler and LogsBufferHandler hand records to it once routed. Each record is frozen into one snapshot, with its message rendered at the call site, and all routed sinks share that snapshot and so its one redaction.
- LogsBufferHandler formats on the writer and returns the finished line to the app loop with call_soon_threadsafe. Only the loop touches the deques and widgets. This also fixes the old pattern where worker threads wrote to widgets directly.
- Handlers stay on the root logger, so these keep working unchanged: runtime_maintenance's file-handler lookup, the on_unmount cleanup, the reconfigure detection, and the backup roundtrip test.
- close() and _maintenance_close_admission() drain queued records first. The drain in the pause runs before taking the handler lock, because the writer needs that lock.
- on_unmount and an atexit hook stop the writer. The hook is registered after logging's own, so it runs before logging.shutdown.
- When the writer is stopped, or no app loop is known (tests, early boot), handlers fall back to writing directly.
- The writer never prints: handleError goes to sys.stderr, which Textual captures while the TUI is running.

Why not a stdlib QueueHandler/QueueListener
- It would move the file handler off root. Four lookups would then need changing: runtime_maintenance, on_unmount, the reconfigure detection, and root-level lowering. It would also break Tests/Backup_Recovery/test_selected_local_options_roundtrip.py, which asserts the handler is on root.
- Redacting once, off the emitting thread, still needs custom prepare() on both the QueueHandler and the QueueListener.
- QueueListener cannot drain without stopping, and the maintenance pause and close() need exactly that.
- So it gives no smaller change. The custom writer is about 55 lines (submit, run, drain, stop), plus a 30-line mixin.

Volume (AC #4)
- ChaChaNotes_DB: 40 per-row CRUD INFO calls demoted to DEBUG (add/update/soft-delete/restore of conversations, messages, variants, notes, character cards, generic items, links). list_all_active_conversations is left alone because it is a read, not a row write.
- chatbook_importer: 23 per-item INFO calls inside the import loops demoted to DEBUG. The single summary INFO per import stays.
- Workers: WorkerHandlerRegistry.handle_event now reports unhandled app-owned workers at DEBUG, and at WARNING only when the state is ERROR. StateChanged does not bubble, so every event reaching this hook comes from an app-owned worker. app.py's per-transition WARNING is gone, and its hook debug line is formatted lazily.
- @timeit is now coroutine-aware: it times the awaited body instead of coroutine creation. Its summary logs at DEBUG and is formatted lazily.
- Tests/App/test_worker_failure_event.py::test_unknown_worker_group_still_warns_unhandled now drives an ERROR transition. That is the intended behaviour change.

Measurements
Isolated micro-benchmark: scratch HOME/XDG/TLDW_CONFIG_PATH, Py 3.12.11, loguru 0.7.3, the shipped sink layout (forwarder, root INFO, rotating file with maxBytes > 0, Logs buffer), 2,000 INFO records, median of 5 runs for the dropped-call rows.

| Metric | Before (c174e30f6b) | After |
|---|---|---|
| Dropped `logger.debug` | 7.7–10.3 µs | 0.12–0.13 µs |
| Dropped `opt(lazy=True).debug` | 9.9–10.5 µs, callable evaluated | 0.43–0.44 µs, not evaluated |
| Dropped `opt(exception=True).debug` | 35.5–39.6 µs | 0.44–0.48 µs |
| INFO, 140-char prose line, on the emitting thread | 349.8 µs | 17.5 µs |
| INFO, 140-char prose line, writer throughput | 349.8 µs (all on the emitting thread) | 97.5 µs |
| INFO, 140-char key=value-dense line, on the emitting thread | 479.8 µs | 15.4 µs |
| INFO, 140-char key=value-dense line, writer throughput | 479.8 µs (all on the emitting thread) | 192.3 µs |
| Sanitizer passes per record | 3.0 | 1.0 |

Both trees wrote all 2,050 records to the file and to the buffer. The in-suite fresh-process probe asserts less than 2 µs per dropped debug.

Tests
- New: Tests/test_logging_pipeline_single_pass.py, 17 tests (level gate, single redaction, exception and cap-straddle redaction, writer thread, argument rendering at the call site, drain on close, stop, and the maintenance pause, the Logs-buffer loop hand-back, every-sink masking of a forwarded loguru secret, sandboxed production wiring, DB and importer demotion, worker warning, @timeit).
- At base the full file fails to import (the new API does not exist yet). A base-compatible subset fails on the behaviour itself: 2 redactions per record, per-row INFO present, no ERROR warning, @timeit not a coroutine function.

Verification against base c174e30f6b (throwaway detached worktree; app pre-imported at collection per lessons-testing-evidence, one pytest per top-level Tests/ directory because a directory conftest loads at startup)
- All 226 files that reference Logging_Config, log_sanitizer, redact or on_worker_state_changed, plus Tests/App/test_worker_failure_event.py. Tests/ProductionApp was excluded because its conftest rebinds config at import.
  - Base: 10,389 passed, 2,209 failed, 69 errors.
  - This branch: 10,404 passed, 2,211 failed, 69 errors.
- The base and branch failure sets are identical except for three ids. Each passes or fails the same way on both trees when rerun alone without contention:
  - Tests/UI/test_trace_responsive.py passed 38/38, three times on each tree.
  - Tests/Agents/test_run_log_service_wiring.py showed the same 5 failures on both trees, twice each.
  - Every other base failure is ADR-126 RecoveryRequired or a pre-existing assertion. Examples of the latter: the Client_Media_DB brace-style list, the persona-workspace interpolation check, and the remaining-sentinel clipboard matrix.
- Core logging files (private files, Logs buffer, share-path privacy, persistent-log, diagnostic boundary and sentinel matrices, logging maintenance, crash forensics, stdlib format style, worker failure events, console send diagnostics):
  - Base: 35 failed / 93 passed.
  - This branch: 35 failed / 110 passed. The failed set is identical; the extra passes are the 17 new tests.
- Tests/App/test_worker_failure_event.py, run under private_profile_test in scratch copies: 13/13 pass with the change. At base, only the original (unwrapped) unknown-worker test fails, because it asserts a WARNING on a SUCCESS transition, which this change intentionally drops.
- Architecture tests (derived_artifact_checkers, diagnostic_path_privacy, persistent_diagnostic_inventory, security_logger_write_surface), with -p no:xdist:
  - 2 failed / 257 passed / 1 skipped on both trees.
  - The 2 inventory tests fail with the same messages on both.
- ./scripts/preflight.sh: all derived-artifact checks passed.

Inventory
- Docs/security/production-diagnostic-inventory.json was regenerated with --write after reading every --statements row. The drift was expected: re-levelled DB and importer lines, two new registry lines, the timeit and app-hook rewrites, and a new sink row for sync_loguru_forward_level's forwarder add.

Files
- tldw_chatbook/Logging_Config.py
- tldw_chatbook/app.py
- tldw_chatbook/Event_Handlers/worker_handlers/base_handler.py
- tldw_chatbook/Metrics/metrics_logger.py
- tldw_chatbook/DB/ChaChaNotes_DB.py
- tldw_chatbook/Chatbooks/chatbook_importer.py
- Tests/test_logging_pipeline_single_pass.py
- Tests/App/test_worker_failure_event.py
- Docs/security/production-diagnostic-inventory.json
- backlog/docs/lessons-testing-evidence.md (lesson on a bash-3.2 empty-list runner)
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
