---
id: TASK-30015
title: Abort a wedged provider read so a stalled sync-provider worker thread is torn down
status: To Do
assignee: []
created_date: '2026-09-02'
labels:
  - agents
  - reliability
  - mcp
dependencies:
  - TASK-26003
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-26003's content-stall watchdog bounds the RUN (it raises StreamStallError
at ~90s of no content and frees the turn), but it does not tear down the
underlying work in the generic sync-provider path. For hosted providers,
chat_api_call runs on an asyncio.to_thread worker that pumps
next(normalized_response) and feeds a queue; in the keep-alive-only stall the
worker is blocked INSIDE a single next() that never returns. On stall,
_stream_generic_chat's finally sets stop_event (only polled between chunks, so
never seen), calls close() on the streaming GENERATOR (raises "generator already
executing" cross-thread, swallowed), and awaits the worker with timeout=0 (does
not wait). Net: the worker thread and provider socket stay live until the
provider/proxy drops the connection. Under repeated stalls this leaks default
ThreadPoolExecutor slots and connections, which can eventually stall every
asyncio.to_thread in the app. This is a pre-existing limitation of the
sync-provider-in-a-thread bridge that 26003 exposes rather than introduces.
Found by the 26003 adversarial review (finding I1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On a stall/stop, a provider worker blocked in a read is actually aborted (the socket is closed), not left until the connection drops
- [ ] #2 The sync bridge holds the closeable HTTP response/socket and closes THAT on stop, rather than calling close() on the executing generator
- [ ] #3 Repeated stalls do not accumulate live worker threads or provider connections - verified by a test that stalls N times and asserts no thread/connection growth
- [ ] #4 stdio/local paths and the normal streaming path are unaffected
<!-- AC:END -->
