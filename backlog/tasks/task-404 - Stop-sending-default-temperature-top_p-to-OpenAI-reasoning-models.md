---
id: TASK-404
title: Stop sending default temperature/top_p to OpenAI reasoning models
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 07:12'
updated_date: '2026-07-21 14:13'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
chat_with_openai force-includes default temperature and top_p in the request; OpenAI reasoning models (o-series, gpt-5 family) reject these with HTTP 400 'Unsupported parameter', so ANY call routed to them through this handler fails — including the Responses-API branch. Found during task-403 verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reasoning-model requests omit unsupported sampling parameters,Non-reasoning models keep today's parameter behavior,Regression test covers both shapes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
chat_with_openai no longer injects config-backed temperature/top_p for reasoning-family models: new _is_openai_reasoning_model predicate (o1/o3/o4/gpt-5 families, boundary-safe — o365-copilot and olmo never match) gates the sampling block, together with the Responses-API branch; explicit caller values on that path are dropped with a warning (the API rejects any value). Non-reasoning models keep byte-identical behavior. 17 new payload-seam tests (fake requests.Session capturing json) covering both branches + boundary matrix; live proof: the previously-400ing gpt-5-mini reasoning_effort call now succeeds end-to-end through chat_api_call against the real API.
<!-- SECTION:NOTES:END -->
