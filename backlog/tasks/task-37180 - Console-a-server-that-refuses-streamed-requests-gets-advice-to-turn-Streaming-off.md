---
id: TASK-37180
title: >-
  Console: a server that refuses streamed requests gets advice to turn Streaming
  off
status: Done
assignee:
  - '@claude'
created_date: '2026-10-27 18:50'
updated_date: '2026-10-29 15:58'
labels:
  - console
  - streaming
  - copy
  - p3
dependencies: []
references:
  - >-
    backlog/tasks/task-33620.10 -
    Console-Cloud-replies-show-Generating…-for-the-whole-run-then-appear-all-at-once-streaming-ships-off.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since TASK-33620.10 every Console provider streams by default, and `[chat_defaults] streaming = true` now outranks a provider's own `[api_settings.<provider>] streaming = false`. An OpenAI-compatible server that does not support streaming therefore fails the first reply. Live on 2026-10-27, a local server that answers a streamed request with HTTP 400 produced this System row: "Agent run failed: Provider error from Custom OpenAI-compatible: bad request. Status: 400. Custom OpenAI-compatible says: “…” Selected model: qwen2.5-0.5b-instruct. The provider rejected this request. Confirm the model is still available, or choose another model from the model picker." The server's own message is quoted, but the advice points at the model. The fix that works, turning Streaming off for that model or provider, is not mentioned (the User Guide's chat-basics "Turning streaming off" describes it). llama.cpp is not affected: its gateway retries once without streaming.

Evidence: the screen capture of TASK-33620.10's live run on 2026-10-27; the System row above is quoted from it. The capture was kept with that session's working files, not in the repo.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When a streamed Console request that is not retried without streaming (a cloud provider; TASK-37421 retries self-hosted and Custom servers) is rejected before any text arrives with a refusal status (400, 406, 415, 422 or 501) that is not a model-unavailable, context-length or tool-definition failure, the failure's recovery advice names turning Streaming off for this chat, model or provider, alongside the existing model advice. Auth (401/403), payment (402), not-found (404), timeout (408), size (413) and rate-limit (429) failures keep their own advice.
- [x] #2 The advice does not appear for the same error on a request that was not streamed.
- [x] #3 A test drives a streamed request rejected with HTTP 400 and asserts the advice; it fails before the change.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Share one refused-stream classification with TASK-37421's retry (`Chat/refused_stream.py`).
2. In the generic path, compute it where the original exception is still in hand (the cause chain carries the HTTP response a context-length check reads) and pass it to the failure copy.
3. Tests through the real bridge and adapters: a streamed 400 from a cloud provider names Streaming; the same 400 on a non-streamed request does not.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed in TASK-33620.10's branch with TASK-37421 (owner decision 2026-10-29: self-hosted and Custom servers are retried once without streaming, and the copy points at the Streaming setting). A provider that is retried never shows this advice for the refused stream; a retry that also fails shows the non-streamed failure, which does not name Streaming (it was not the cause). So the advice is for providers that are not retried: cloud providers.

`_stream_generic_chat` computes `refused_stream` in its worker (streamed, nothing delivered yet, `refused_stream.is_refused_stream(exc)` on the original adapter exception) and `_provider_error_copy_with_model_recovery(refused_stream=True)` appends: "If this server cannot stream replies, turn Streaming off and send again: in Chat settings for this chat or this model, or in config.toml for this provider." after the existing model advice. A tool-definition rejection keeps its own copy and gets no Streaming advice. AC#1 amended to the statuses implemented (400, 406, 415, 422, 501; not model-unavailable or context-length; auth, payment, not-found, timeout, size and rate-limit failures keep their own advice): a 404's "could not find this model or endpoint" is the right fix, and adding Streaming there would mislead.

Tests (red on 7e2167e0a7 for the missing advice): `Tests/Chat/test_console_refused_stream_retry.py::test_cloud_refused_stream_is_not_retried_and_names_streaming[anthropic|zenmux]` (real `chat_with_anthropic` and the hosted engine against a loopback server, one request) and `Tests/Chat/test_console_provider_gateway.py::test_cloud_refused_stream_is_not_retried_by_the_gateway`; AC#2: `test_cloud_refusal_of_a_request_that_was_not_streamed_does_not_name_streaming` (green on both, a guard). Live 2026-10-29: Anthropic pointed at a local front that refuses `"stream": true` showed the advice in the System row after one request (see TASK-37421's notes). Known gap, filed as TASK-37700: OpenAI's streaming adapter turns a refused stream into an SSE error, so its failure has no status or provider message and cannot carry this advice.
<!-- SECTION:NOTES:END -->
