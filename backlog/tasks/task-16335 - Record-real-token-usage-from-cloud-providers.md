---
id: TASK-16335
title: Record real token usage from cloud providers
status: Done
assignee:
  - '@robert'
created_date: '2026-08-15 22:39'
updated_date: '2026-08-15 22:45'
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
- [x] #1 A provider-side usage reporting seam lets handlers publish exact prompt and completion token counts for the current call,chat_api_call consumes published usage into the active recorder instead of estimates and clears it per call,OpenAI and Anthropic non-streaming paths publish usage when their responses carry it,Paths without usage keep the estimate fallback unchanged,Tests with mocked handlers or HTTP cover publication, dispatcher consumption, and the fallback
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Survey chat_with_openai and chat_with_anthropic non-streaming response paths for where usage is available
2. TDD a provider-side usage publication seam in usage_recorder (contextvar hand-off consumed by chat_api_call per call) plus dispatcher consumption replacing estimates
3. Wire OpenAI and Anthropic non-streaming paths to publish usage when their responses carry it
4. Tests with mocked handlers plus lint plus task close
ADR required: no - completes the task-16331 recorder seam for cloud providers
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Survey finding that shrank the task: the cloud handlers ALREADY publish usage — `chat_with_openai` (classic and Responses-API, via `_normalize_openai_responses_payload`) and `chat_with_anthropic` (via its OpenAI-shape normalization) all return OpenAI-shaped dicts carrying `usage`, and the task-16331 dict normalization in `chat_api_call` already extracts content and records exact counts. The handler-side seam the AC anticipated turned out to be the return-dict itself; no handler changes were needed.
- The one real gap: Anthropic preserves its own usage field names (`input_tokens`/`output_tokens`) inside the normalized dict, which the recorder's gate (OpenAI names only) did not recognize — those calls silently fell back to estimates. `chat_api_call`'s recording block now maps both naming schemes (OpenAI names win when both are present, which also pins the mixed-key case).
- Verified TDD: 3 new tests written first and watched failing (Anthropic-style keys recorded exactly; OpenAI-style still exact; mixed keys prefer OpenAI names); `Tests/Chat/ + Tests/Research/` = 116 passed; ruff clean. Estimate fallback for usage-less paths is pinned by the pre-existing tests.
<!-- SECTION:NOTES:END -->
