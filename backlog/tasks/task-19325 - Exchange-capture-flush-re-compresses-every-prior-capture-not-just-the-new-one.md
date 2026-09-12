---
id: TASK-19325
title: 'Exchange-capture flush re-compresses every prior capture, not just the new one'
status: Done
assignee: []
created_date: '2026-08-21 06:03'
updated_date: '2026-08-21 17:31'
labels:
  - console
  - exchange-capture
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console Conversation Inspector's exchange-capture persistence (task-18300 and subtasks) flushes captured request/response payloads to the message_exchanges table as zlib-compressed JSON blobs (capture_to_blob in console_exchange_capture.py). Each flush point re-serializes and re-compresses the FULL set of captures being written, including ones already compressed and written on an earlier flush of the same message (e.g. a multi-call agent turn that flushes more than once, or a stop-path late attach after an earlier flush). This is pure repeated CPU work with no correctness impact -- found during the final whole-branch review of task-18300 (M10) and explicitly deferred rather than fixed at merge time, since the number of flushes per message is small in practice and no user-visible slowdown was observed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Identify every flush call site for message_exchanges captures and confirm which ones re-process already-flushed captures on a later flush of the same message.
- [x] #2 Either skip re-compressing captures already persisted on an earlier flush, or document why the current behavior is acceptable and close the task without a code change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed as part of Qodo PR #1883 review remediation (finding 4): ConsoleChatStore._exchange_blob_cache memoizes capture_to_blob output keyed by (message.id, run_tag, seq, status), invalidated naturally on a stopped->complete status transition and pruned on every flush plus on message/session removal. See tldw_chatbook/Chat/console_chat_store.py's _persist_exchanges_only and Tests/Chat/test_console_chat_store_exchanges.py (commit 4261f6d32).
<!-- SECTION:NOTES:END -->
