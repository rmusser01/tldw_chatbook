---
id: TASK-32369
title: anthropic and cohere handlers emit real HTTP 400 errors for local validation failures
status: To Do
assignee: []
created_date: '2026-09-11 01:55'
labels:
  - chat
  - providers
dependencies: []
priority: low
---

## Description

While fixing the misattributed "provider returned HTTP 400" on the custom-openai path (task-32342), lane A found that the anthropic and cohere handlers in `LLM_Calls/LLM_API_Calls.py` (around L1569 and L2604) raise real 400-shaped errors for failures that never left the client (local validation), so Console again blames the provider.

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.

## Acceptance Criteria

- [ ] Local validation failures in the anthropic and cohere handlers raise `ChatConfigurationError` (status_code None) or an equivalent client-side error, not an HTTP 400
- [ ] Console's provider-failure copy for those cases says the request was rejected locally, naming the field
- [ ] One test per handler pins the local-failure path
