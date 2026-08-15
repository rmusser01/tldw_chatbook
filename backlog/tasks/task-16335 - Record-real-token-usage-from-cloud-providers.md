---
id: TASK-16335
title: Record real token usage from cloud providers
status: To Do
assignee:
  - '@robert'
created_date: '2026-08-15 22:39'
labels:
  - research
  - budget
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-16331 records exact token counts for local OpenAI-compatible providers (their raw dict responses carry usage), but cloud paths still fall back to estimates: chat_with_openai and chat_with_anthropic receive usage in their API responses and discard it while returning content strings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A provider-side usage reporting seam lets handlers publish exact prompt and completion token counts for the current call,chat_api_call consumes published usage into the active recorder instead of estimates and clears it per call,OpenAI and Anthropic non-streaming paths publish usage when their responses carry it,Paths without usage keep the estimate fallback unchanged,Tests with mocked handlers or HTTP cover publication, dispatcher consumption, and the fallback
<!-- AC:END -->
