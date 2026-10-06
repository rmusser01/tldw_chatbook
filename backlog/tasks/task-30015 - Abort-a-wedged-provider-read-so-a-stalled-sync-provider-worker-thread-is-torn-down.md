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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Live evidence from TASK-34100.5 (review round 2, V2-F8, 2026-10-04).** The
same leak reaches self-hosted servers on the first-token path. TASK-34100.5
gave a self-hosted first token a 300 s window and raised the local handler's
HTTP read timeout to that window plus 30 s, so the watchdog decides. When the
window ran out, the run ended at ~300 s, but the request was not aborted: the
CPU-only llama-server kept processing the prompt until the read timeout
closed the connection 30 s later. Log
`setup-wizard-ux-qa/evidence/g5-v2-slow/llama-server-9412.log`: task 111
"new prompt" at 7.21.16, the UI timeout at ~300 s (02-slow-t312), and the
server's "cancel task, id_task = 111" at 12.51.30 (+330 s). Task 1045 shows the
same. The User Guide (console.md, "The first reply from a large local model
takes minutes") now says the server may keep working for up to 30 s. Closing
the held response on `StreamStallError` (AC#2 here) would end that too.
<!-- SECTION:NOTES:END -->
