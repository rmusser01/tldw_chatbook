---
id: TASK-37181
title: >-
  Console: llama.cpp replies from the automatic non-streaming retry persist no
  usage
status: To Do
assignee: []
created_date: '2026-10-27 18:50'
updated_date: '2026-10-29 15:50'
labels:
  - console
  - streaming
  - cost
  - p3
dependencies: []
references:
  - >-
    backlog/tasks/task-33620.10 -
    Console-Cloud-replies-show-Generating…-for-the-whole-run-then-appear-all-at-once-streaming-ships-off.md
  - TASK-19324
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a llama.cpp server refuses a streamed request (or streams no text), the Console gateway retries once without streaming and the reply arrives. Live on 2026-10-27 (TASK-33620.10 verification), a reply delivered by that retry against a real llama-server (qwen2.5-0.5b-instruct) behind a front that refuses `"stream": true` was persisted with an empty `messages.usage_json`, while the same server's non-streamed reply through the Custom OpenAI-compatible provider persisted provider usage (310 input, 3 output tokens). The retry passes `_automatic_signals` only when `prepared` and `call_signals` are both present (console_provider_gateway.py, the llama.cpp fallback branch), which may be why. Whether a plain streamed llama.cpp reply records usage was not checked in that session.

Evidence: TASK-33620.10's live run on 2026-10-27 (screen captures, the refusing front's request log and the persisted message rows), kept with that session's working files, not in the repo.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A llama.cpp reply delivered by the automatic non-streaming retry persists the usage the server reported for the retry, or the gap is shown to be by design and recorded where the cost chip explains unknown usage.
- [ ] #2 A test drives the retry through the real gateway with a server response that carries usage and asserts what is persisted; it fails before the change if a change is made.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Measured 2026-10-29 (TASK-37421): the gap is not the retry alone. Through the direct llama.cpp path (`stream_chat`'s llama.cpp branch, a request with no native tools), a plain streamed reply, a non-streamed reply and a reply from the automatic retry all record no usage on the Console signals, aggregate or call view, even when every server response carries `usage`: `stream_llamacpp_chat` and `complete_llamacpp_chat` record usage only on `_automatic_signals`, which `stream_chat` passes for automatic work only. A llama.cpp request that goes through the generic path (native tools) now gets TASK-37421's retry, which records the retry's usage once.
<!-- SECTION:NOTES:END -->
