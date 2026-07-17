---
id: TASK-246
title: Restore lazy DB debug logging (eager param stringify on every query)
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
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
- [ ] #1 No query/param stringification occurs when debug logging is disabled, across all four DB modules
- [ ] #2 Param values are truncated before stringification when debug IS enabled
- [ ] #3 A regression test proves an image-bearing insert does not stringify its BLOB at default log level
<!-- AC:END -->
