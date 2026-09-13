---
id: TASK-19324
title: llama.cpp stream-to-complete fallback makes a second uncaptured HTTP request
status: Done
assignee: []
created_date: '2026-08-21 06:03'
labels:
  - console
  - exchange-capture
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console Conversation Inspector's exchange-capture feature (task-18300 and its subtasks) captures every provider call so a user can see exactly what was sent and received on a past turn. The llama.cpp gateway branch in console_provider_gateway.py has a documented fallback path where a streaming request that fails mid-stream retries as a non-streaming complete_llamacpp_chat call. That second HTTP request is made outside the capture seam -- the retry's request/response never gets its own ExchangeCapture, so a turn that actually made two HTTP calls to the local server shows only one in the inspector, understating what was actually sent. Found during the final whole-branch review of task-18300 (M2), explicitly deferred rather than fixed at merge time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The llama.cpp stream-to-complete fallback path's second HTTP request is captured with its own ExchangeCapture (or the existing capture is amended to reflect both calls), so the Exchange tab shows every HTTP request a turn actually made.
- [x] #2 A test exercises the fallback path (streaming request fails mid-stream, retried as non-streaming) and asserts the retry's request/response appears in exchange_captures().
<!-- AC:END -->

## Implementation Notes

`stream_llamacpp_chat` gained an optional `on_fallback_retry` hook, invoked
only when the non-streaming retry actually runs. `stream_chat` supplies a
callback that opens a SECOND call-scoped exchange off the aggregate, so the
retry lands as its own Inspector row rather than being folded into the
streaming call it replaced. The capture carries the retry's real wire payload
(`stream=false`) and a `retry_of` marker explaining why it exists.

Chose AC #1's primary clause (its own ExchangeCapture) over the "amend the
existing capture" alternative: the user-facing promise is that the Exchange
tab shows every HTTP request a turn made, and one row carrying two requests
does not show that.

The hook is best-effort and swallows its own failures -- capture must never
break a send (task-18300's contract), and this path is already the degraded
one.

Test drives the REAL `stream_llamacpp_chat`, faking only the HTTP client (a
stream that opens and yields nothing) and the retry, so the fallback genuinely
fires rather than being simulated. Red-proofed: removing the wiring drops the
capture count from 2 to 1.

Files: `tldw_chatbook/Chat/console_provider_gateway.py`,
`Tests/Chat/test_console_provider_gateway.py`.
