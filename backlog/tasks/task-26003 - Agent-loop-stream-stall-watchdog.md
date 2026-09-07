---
id: TASK-26003
title: 'Agent loop: stream stall watchdog'
status: Done
assignee: []
created_date: '2026-08-31 15:43'
updated_date: '2026-09-03 01:11'
labels:
  - agents
  - reliability
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A provider emitting keep-alive bytes without content holds a run until the wall budget expires. Verified on origin/dev: the only bound on a silent stream is the httpx read timeout at Chat/console_provider_gateway.py:150, and a named grep for stall, idle_timeout, no_activity, watchdog and first_token_timeout across Chat/console_provider_gateway.py and Chat/console_agent_bridge.py returns only install* false positives. Because bytes keep arriving, the read timeout never fires. Hermes tracks last-content time and trips a cross-turn circuit breaker after repeated stale kills.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A stream producing no new content for a configurable interval is terminated even while transport-level bytes continue to arrive
- [x] #2 The distinction between no bytes and no content is explicit: keep-alive and heartbeat frames do not reset the content clock
- [x] #3 A stall termination is reported distinctly from a network error and from a user cancel
- [x] #4 Repeated stalls against the same provider within a session surface a warning rather than silently retrying forever
- [x] #5 Normal slow generation (long thinking, large responses) does not trip the watchdog - verified with a slow but productive stream
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build a pure content-idle watchdog primitive (fully unit-testable).
2. Wire it at the single stream-consumption chokepoint in the bridge.
3. Report a stall distinctly and track repeats per provider per session.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Key realization from tracing the streaming path: for hosted providers the raw
SSE bytes (keep-alives included) are consumed INSIDE chat_api_call's provider
handler on a worker thread; the gateway only sees decoded content items via
next(normalized_response). So a byte-level watchdog would mean editing every
provider handler (not lazy, high-risk). But that same fact means a CONTENT-idle
watchdog at the single consumption chokepoint is sufficient and correct:
keep-alives never reach the consumer, so only real items reset the clock (AC#2
holds for free), and it works uniformly for both the llamacpp and generic paths
without touching any provider byte loop.

- New Chat/stream_stall_watchdog.py: watch_content_stalls(source, timeout,
  provider=...) wraps an async item stream with asyncio.wait_for on each anext;
  a window with no item -> StreamStallError (AC#1), source aclose()d so the
  worker/HTTP stream is cancelled. Non-positive timeout disables it. CancelledError
  propagates unchanged, so a user cancel is never reported as a stall (AC#3).
  StreamStallError is a plain RuntimeError, distinct from CancelledError and
  httpx errors (AC#3). StallTracker + a session-keyed registry
  (record_session_stall/reset_session_stalls) count per-provider stalls and warn
  at a threshold (AC#4); a productive turn prunes the session entry.
- console_agent_bridge._StreamingModelAdapter.chat_call: the one
  `async for chunk in self._gateway.stream_chat(...)` is wrapped inline with
  watch_content_stalls (no loop-body reindent); the timeout comes from
  [chat_defaults] stream_stall_timeout_seconds (default 90s, read per turn). At
  the future.result() boundary a StreamStallError is caught, recorded per
  session+provider, logged distinctly (repeat -> "surfacing rather than retrying
  silently"), and re-raised so the turn fails as a stall; a productive turn
  resets the streak.
- config.py: [chat_defaults] stream_stall_timeout_seconds documented (commented
  default 90).

AC#5 (slow-but-productive doesn't trip) is covered because any content/thinking/
tool-call item resets the clock; only a contentless window trips.

Tests: Tests/Chat/test_stream_stall_watchdog.py (13: pass-through, stall, disable,
cancel-vs-stall, aclose-on-stall, tracker/registry) + one bridge integration test
(a first-call-stalling gateway surfaces a stall through the real chat_call path).
287 bridge+watchdog tests green; no production behavior change on the normal path.
<!-- SECTION:NOTES:END -->
