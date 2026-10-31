---
id: TASK-37700
title: >-
  Console: a streamed OpenAI request refused with a 4xx loses its status and is
  sent three times
status: To Do
assignee: []
created_date: '2026-10-29 15:48'
updated_date: '2026-10-29 17:10'
labels:
  - console
  - streaming
  - openai
  - p1
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since TASK-33620.10 OpenAI streams by default in Console. `chat_with_openai` posts a streamed request inside its stream generator, and when the server answers with a client error it catches the `requests` HTTPError and yields an SSE error event ("Stream connection error: 400 Client Error ...") instead of raising. The gateway turns that event into a bare `RuntimeError("Provider stream error.")`, so the failure reaches the agent runtime as a status-less provider error (502), which it treats as transient and retries twice.

Measured 2026-10-29 (TASK-37421, loopback OpenAI-compatible server that answers `"stream": true` with HTTP 400 and the message "Streaming is not supported by this server."): the server saw three streamed requests, and the System row read "Agent run failed: provider returned HTTP 502 (Provider error from OpenAI: unexpected provider error)." The provider's own sentence and the 400 were lost, so neither the model advice nor TASK-37180's Streaming advice can appear. The same server reached through Anthropic, OpenRouter, Groq, DeepSeek, Mistral AI and two hosted-engine presets (ZenMux, Vercel) failed once, with "Status: 400" and the server's message. Any OpenAI 4xx on a streamed Console send (a rejected parameter, a model the key cannot use) takes the same path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A streamed OpenAI request refused with a client error before any text fails once, with the provider's status and its own message, as the non-streamed path does
- [ ] #2 When the refusal is a refused stream (TASK-37180's classification), the failure copy names the Streaming setting
- [ ] #3 A test drives a refused streamed request through the real chat_with_openai against a loopback server and asserts one request and the copy; it fails before the change
- [ ] #4 The User Guide's chat-basics Streaming section and the CHANGELOG no longer carve OpenAI out of the refused-stream advice
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Priority raised 2026-10-29 (TASK-37421 review): TASK-33620.10 makes OpenAI, the shipped default provider, stream by default, so every streamed OpenAI 4xx (a rejected parameter, a model the key cannot stream) now reaches Console as a status-less 502 that is sent three times with no usable copy; with streaming off by default those users saw the status and OpenAI's message once. Land before the next release. Re-measured on the review-fix head: the loopback server saw [True, True, True] and the System row read "Agent run failed: provider returned HTTP 502 (Provider error from OpenAI: unexpected provider error)." Until this lands, the User Guide (console/chat-basics, Turning streaming off) and the CHANGELOG Unreleased entry say OpenAI is the exception; AC#4 removes that.
<!-- SECTION:NOTES:END -->
