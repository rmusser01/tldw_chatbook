---
id: TASK-403
title: Verify response prefill against Gemini and OpenAI Responses-API branch
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 03:48'
updated_date: '2026-07-21 14:03'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec section 5/8 verification items left open (no key/config available at implementation time): Gemini role-remap with trailing model turn, and the OpenAI use_responses_api input-shape branch. If either rejects trailing-assistant payloads, build the skip+warn incompatible-provider guard (payload AND display seed dropped, inline warning) — note the guard mechanism was superseded for llama.cpp and does not exist yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Gemini prefilled send verified accepted or guarded
- [x] #2 Responses-API branch verified accepted or guarded
- [x] #3 Spec section 8 updated with outcomes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
COMPLETE 2026-07-21. Gemini leg: ACCEPTED with literal continuation, verified both raw (generateContent, trailing model turn) and through chat_with_google's remap/alternation path on gemini-flash-lite-latest (free-tier 429s are quota noise; gemini-2.5-flash-lite 404s as retired-for-new-users — use -latest aliases). Responses-API leg: verified accepted earlier (see notes below). No provider requires the skip+warn guard; it remains unbuilt by design. Spec §8 records all outcomes.
<!-- SECTION:NOTES:END -->
