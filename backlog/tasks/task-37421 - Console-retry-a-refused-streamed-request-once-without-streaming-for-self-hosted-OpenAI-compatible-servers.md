---
id: TASK-37421
title: >-
  Console: retry a refused streamed request once without streaming for
  self-hosted OpenAI-compatible servers
status: To Do
assignee: []
created_date: '2026-10-29 07:34'
labels:
  - console
  - streaming
  - local-models
  - p3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since TASK-33620.10 every Console provider streams by default, and [chat_defaults] streaming = true outranks a provider's own [api_settings.<provider>] streaming = false. A self-hosted OpenAI-compatible server (Custom, Custom_2, a custom-ep entry) that does not support streaming used to work out of the box because those templates shipped streaming = false; it now fails the first reply with HTTP 400 until the user turns Streaming off for that model or provider (live 2026-10-29, TASK-33620.10 evidence tools/evidence/bd5/U19/rebased/09-custom-refused-stream.txt). The llama.cpp path already retries once without streaming when a streamed request is refused or yields no text (console_provider_gateway.py, the LLAMA_FALLBACK route, with its own trace capture and dispatch authorization). This task decides whether the same one-shot retry should cover self-hosted OpenAI-compatible providers, where a retry costs nothing, and implements it if so. Cloud providers stay out of scope: a 400 there is usually a real request error and a silent second paid call is not wanted. TASK-37180 (advice copy on the failure) is the alternative if the answer is no.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decision (retry automatically for self-hosted OpenAI-compatible providers, or rely on the advice in TASK-37180) is recorded in this task
- [ ] #2 If retry is chosen: a streamed request to a self-hosted OpenAI-compatible provider that is refused before any text arrives is retried once without streaming, the reply completes, and the retry gets its own trace capture and usage like the llama.cpp fallback
- [ ] #3 If retry is chosen: cloud providers are never retried this way, and a test drives a refused stream for each case through the real gateway; the tests fail before the change
<!-- AC:END -->
