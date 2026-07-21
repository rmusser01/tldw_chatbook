---
id: TASK-403
title: Verify response prefill against Gemini and OpenAI Responses-API branch
status: To Do
assignee: []
created_date: '2026-07-21 03:48'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec section 5/8 verification items left open (no key/config available at implementation time): Gemini role-remap with trailing model turn, and the OpenAI use_responses_api input-shape branch. If either rejects trailing-assistant payloads, build the skip+warn incompatible-provider guard (payload AND display seed dropped, inline warning) — note the guard mechanism was superseded for llama.cpp and does not exist yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Gemini prefilled send verified accepted or guarded,Responses-API branch verified accepted or guarded,Spec section 8 updated with outcomes
<!-- AC:END -->
