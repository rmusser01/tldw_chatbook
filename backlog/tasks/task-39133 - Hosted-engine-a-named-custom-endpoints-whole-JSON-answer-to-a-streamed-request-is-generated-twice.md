---
id: TASK-39133
title: >-
  Hosted engine: a named custom endpoint's whole JSON answer to a streamed
  request is generated twice
status: To Do
assignee: []
created_date: '2026-11-07 02:48'
labels:
  - console
  - streaming
  - hosted-engine
  - p3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A server that ignores "stream": true and answers a streamed request with a whole JSON chat completion is read in one request for the Custom slots and vLLM (PR3304 normalizes the answer). A named custom endpoint (custom-ep:<slug>) runs through the hosted engine, which cannot read that body as a stream; since TASK-37762 Console asks such a server once more without streaming, so the answer is generated twice and the first answer is thrown away. Measured live 2026-11-06 with a real llama-server behind a front that returns the whole JSON answer to a streamed request: the first answer (4 completion tokens) was discarded and the retry generated another (2 tokens). The first call's usage is recorded as unknown, so a paid endpoint may bill both and an automatic budget can refuse the retry. The User Guide's chat-basics Streaming section now warns about this.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A named custom endpoint whose server answers a streamed request with a whole JSON chat completion shows that answer from the one request, with its usage recorded once
- [ ] #2 A body that is neither SSE nor a valid chat completion still fails, or is retried, as it does today
- [ ] #3 The User Guide's Streaming section no longer says such an answer is generated twice
- [ ] #4 A loopback test drives custom-ep against a server that ignores stream and asserts one request and the answer's usage; it fails before the change
<!-- AC:END -->
