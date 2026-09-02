---
id: TASK-25816
title: Conversation database failure gives recovery advice that cannot work
status: Done
assignee: []
created_date: '2026-08-31 05:07'
updated_date: '2026-08-31 13:39'
labels:
  - console
  - ux-review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the conversation database cannot be opened, Console tells the user to restart and check the app log. Restarting cannot fix an on-disk fault, and the underlying exception is never written to the log, so the instruction leads nowhere. In the observed case a single corrupt index made the whole product unusable while table data and foreign keys were intact and one REINDEX restored it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The underlying database exception is written to the app log before the user is told to consult it
- [ ] #2 Console distinguishes a repairable integrity fault from an unrecoverable one and says which it is
- [ ] #3 A repairable fault offers an in-app repair action rather than only a restart suggestion
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ROOT CAUSE WAS ARCHITECTURAL, not a wording problem. Console said 'check the app log for the database error'; PersistentDiagnosticFilter admits ONLY records marked metadata-only by persist_event(), so every ordinary logger.error -- including the ChaChaNotes schema failure and config's lazy-init failure -- is STRUCTURALLY EXCLUDED from that file. The instruction pointed at a file that by design could not contain the fault. Verified live: a corrupt index left Console unusable and that session's log held 15 INFO lines and no trace of it.

The filter is a deliberate privacy boundary (exception text carries paths and secrets), so weakening it was not an option. Fix works WITH it: ChaChaNotes_DB's schema-failure handler now emits a metadata-only 'database_open_failed' event carrying schema, error_type and a repairable flag -- no message text. Copy corrected to drop 'Restart Chatbook' (a restart cannot repair an on-disk fault) and to point at Logs (F8), which is now true. Kept the 'database could not be opened' fragment that Tests/UI/test_console_degraded_database_send.py pins.

TRAP HIT AND FIXED: my first version ran PRAGMA quick_check via get_connection() inside the failed __init__ to classify the fault; re-entering the connection there DEADLOCKED test_unopenable_database_still_sends_a_temporary_conversation. Classification now comes from the exception alone. Never issue a query from inside a failed database __init__.

Baseline confirmed unchanged: 9 pre-existing failures in Tests/ChaChaNotesDB/ and 1 in test_console_degraded_database_send.py, all present on clean dev.
<!-- SECTION:NOTES:END -->
