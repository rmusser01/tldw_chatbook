---
id: TASK-246
title: Restore lazy DB debug logging (eager param stringify on every query)
status: Done
assignee: []
created_date: '2026-07-16 14:30'
updated_date: '2026-07-17 00:20'
labels: [performance, db]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ChaChaNotes_DB.execute_query's isEnabledFor guard is commented out, so the debug f-string — including str(params) over raw image BLOBs — is built on EVERY query regardless of log level: measured 14.3ms per 3MB image-message INSERT, on the send-completion persist path. Same pattern in Prompts_DB.py:433, Client_Media_DB_v2.py:626 (full ingested document text), Sync_Client.py:667/674. NOTE (review-verified): `logger` in these modules is LOGURU (ChaChaNotes_DB.py:49) — it has no isEnabledFor(), so restoring the commented guard verbatim would AttributeError; use loguru's lazy form (logger.opt(lazy=True).debug with callables) or a min-level check via loguru's mechanism instead. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P0 A1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No query/param stringification occurs when debug logging is disabled, across all four DB modules
- [x] #2 Param values are truncated before stringification when debug IS enabled
- [x] #3 A regression test proves an image-bearing insert does not stringify its BLOB at default log level
<!-- AC:END -->

## Implementation Notes

Added `DB/sql_logging.py` with a single shared `preview_params()` helper:
bytes-like values become `<N bytes>` (never repr()'d/stringified), long
strings truncate at 80 chars, and the whole rendered preview is capped at
200 chars regardless of param count. Used at all four call sites via
loguru's `logger.opt(lazy=True).debug(msg, *lambda_args)` so the preview
string is only built when a sink actually admits DEBUG (loguru has no
`isEnabledFor`, so the previously-commented `if logger.isEnabledFor(...)`
guard was NOT restored — that would AttributeError on a loguru logger).

Sites fixed:
- `DB/ChaChaNotes_DB.py:~2635` (`logger` = loguru)
- `DB/Prompts_DB.py:~433` (`logging` here is `from loguru import logger as
  logging` — also loguru)
- `DB/Client_Media_DB_v2.py:~626` (this call site's `logging` was the
  *stdlib* module, unconfigured and already silent by default, but still
  paid the eager-stringify cost every query; switched to the module's
  already-imported loguru `logger` for consistency and to fix the cost)
- `DB/Sync_Client.py:667,674` (MediaKeywords link/unlink debug logs; params
  here are small ints, but converted for consistency with the shared
  helper/pattern)

New test `Tests/DB/test_sql_debug_logging.py`: a `bytes` subclass with a
`__repr__`/`__str__` call-counter, inserted as a query param through
`CharactersRAGDB.execute_query`. RED before the fix (2 failures — the
counter was > 0 both at default level and with a DEBUG sink attached);
GREEN after (11 passed) — counter stays 0 in both cases, and a sanity test
confirms the log line still fires when DEBUG is enabled (so "never emits"
isn't passing by accident). Plus unit tests of `preview_params()`'s shapes
(None, tuple, bytes/bytearray/memoryview, long-string truncation, dict,
whole-preview cap, unrepr-able values don't raise).
