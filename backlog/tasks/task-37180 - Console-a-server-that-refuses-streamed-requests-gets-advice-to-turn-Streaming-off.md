---
id: TASK-37180
title: >-
  Console: a server that refuses streamed requests gets advice to turn
  Streaming off
status: To Do
assignee: []
created_date: '2026-10-27 18:50'
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
- [ ] #1 When a streamed Console request is rejected by the provider before any text arrives, and the rejection is a client error (4xx) that is not an auth, rate-limit or context-length failure, the failure's recovery advice names turning Streaming off for this chat, model or provider, alongside the existing model advice.
- [ ] #2 The advice does not appear for the same error on a request that was not streamed.
- [ ] #3 A test drives a streamed request rejected with HTTP 400 and asserts the advice; it fails before the change.
<!-- AC:END -->
