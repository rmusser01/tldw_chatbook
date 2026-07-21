---
id: TASK-403
title: Verify response prefill against Gemini and OpenAI Responses-API branch
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-21 03:48'
updated_date: '2026-07-21 07:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec section 5/8 verification items left open (no key/config available at implementation time): Gemini role-remap with trailing model turn, and the OpenAI use_responses_api input-shape branch. If either rejects trailing-assistant payloads, build the skip+warn incompatible-provider guard (payload AND display seed dropped, inline warning) — note the guard mechanism was superseded for llama.cpp and does not exist yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Gemini prefilled send verified accepted or guarded
- [x] #2 Responses-API branch verified accepted or guarded
- [x] #3 Spec section 8 updated with outcomes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Responses-API leg VERIFIED ACCEPTED 2026-07-21: raw /v1/responses call with the handler's exact trailing-assistant input shape (gpt-5-mini, reasoning.effort=low) succeeds; influence-not-literal continuation like Chat Completions; no guard needed. Through-the-handler check blocked by pre-existing unrelated bug (handler force-sends temperature/top_p which reasoning models 400) — filed as task-404. Gemini leg remains BLOCKED: no Google API key in this environment; AC #1 left open, task stays In Progress until a key is available.
<!-- SECTION:NOTES:END -->
