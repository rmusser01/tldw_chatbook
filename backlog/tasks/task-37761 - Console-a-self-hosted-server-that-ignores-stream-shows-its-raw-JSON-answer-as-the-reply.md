---
id: TASK-37761
title: 'Console: a self-hosted server that ignores stream shows its raw JSON answer
  as the reply'
status: Done
assignee: []
created_date: 2026-10-29 17:10
labels:
- console
- streaming
- local-models
- p2
dependencies: []
priority: medium
updated_date: 2026-11-01 08:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since TASK-33620.10 every Console provider streams by default. Some OpenAI-compatible servers do not support streaming and simply ignore `"stream": true`, answering with one whole `application/json` chat completion. Measured 2026-10-29 on TASK-33620.10's branch (real `ConsoleAgentBridge`, gateway, `chat_api_call` and adapters against a loopback server that answers every request with a whole JSON completion): through Custom (`chat_with_custom_openai`) and vLLM the run completed with the raw JSON text (`{"id": "chatcmpl-1", "object": "chat.completion", ...`) as the assistant reply, after one request. A named custom endpoint (`custom-ep:<slug>`, run by the hosted engine) does not show the JSON: the engine cannot read the answer as a stream and the reply fails as "unexpected provider error"; it is not retried (TASK-37421 leaves that key out).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A streamed request to Custom or vLLM answered with a whole JSON chat completion shows the completion's content as the reply, not the JSON text
- [x] #2 A test drives such a server through the real adapters for Custom and vLLM and asserts the reply text; it fails before the change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
PR3304 integration: ADR required: no new ADR; existing shared adapter and ADR097/134/156 contracts apply. Follow Docs/superpowers/plans/2026-11-01-pr3304-review-repair.md. Reproduce whole-JSON loss/regeneration with real Custom/vLLM adapters; normalize application/json preserving content/tools/usage in one request; verify targeted Stop/usage contracts and obtain independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented shared application/json normalization for Custom/vLLM streamed requests, preserving content, native tools, usage and response-close ownership without a second HTTP generation. The rebased causal RED returned no usable reply and regenerated it; this supersedes the older raw-JSON observation for the integrated tree. Real adapters and negative malformed-JSON/close controls pass in r-66c563d6 (200 tests). Existing ADR097/134/156; no new ADR. Independent review approves. See Docs/superpowers/qa/2026-11-01-pr3304-backlog-integration.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
