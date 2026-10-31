---
id: TASK-37761
title: >-
  Console: a self-hosted server that ignores stream shows its raw JSON answer as
  the reply
status: To Do
assignee: []
created_date: '2026-10-29 17:10'
labels:
  - console
  - streaming
  - local-models
  - p2
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since TASK-33620.10 every Console provider streams by default. Some OpenAI-compatible servers do not support streaming and simply ignore `"stream": true`, answering with one whole `application/json` chat completion. Measured 2026-10-29 on TASK-33620.10's branch (real `ConsoleAgentBridge`, gateway, `chat_api_call` and adapters against a loopback server that answers every request with a whole JSON completion): through Custom (`chat_with_custom_openai`) and vLLM the run completed with the raw JSON text (`{"id": "chatcmpl-1", "object": "chat.completion", ...`) as the assistant reply, after one request. A named custom endpoint (`custom-ep:<slug>`, run by the hosted engine) does not show the JSON: the engine cannot read the answer as a stream and the reply fails as "unexpected provider error"; it is not retried (TASK-37421 leaves that key out).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A streamed request to Custom or vLLM answered with a whole JSON chat completion shows the completion's content as the reply, not the JSON text
- [ ] #2 A test drives such a server through the real adapters for Custom and vLLM and asserts the reply text; it fails before the change
<!-- AC:END -->
