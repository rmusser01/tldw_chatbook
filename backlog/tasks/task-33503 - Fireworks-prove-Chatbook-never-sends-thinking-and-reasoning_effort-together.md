---
id: TASK-33503
title: 'Fireworks: prove Chatbook never sends thinking and reasoning_effort together'
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-29 19:28'
updated_date: '2026-09-29 20:02'
labels:
  - providers
  - engine
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fireworks' reasoning guide says a request carrying both `thinking` and `reasoning_effort` fails validation. Chatbook's Fireworks preset sends neither today, but nothing pins that, and the registry comment describing Fireworks reasoning is inaccurate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test fails if any engine preset's request can carry both `thinking` and `reasoning_effort`
- [x] #2 The Fireworks registry comment describes its reasoning contract accurately, with sources
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Registry-wide test: no engine preset can build a request carrying both thinking and reasoning_effort (extra_body_fields vs reasoning_effort flag, and a built payload).
2. Correct the Fireworks registry comment with the documented contract and sources.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The conflict cannot occur today: FIREWORKS has reasoning_effort=False and no extra body, and the engine never invents a thinking field. Pinned registry-wide: for every engine-driven record, extra_body_fields never holds thinking alongside reasoning_effort=True, and a built payload never carries both (test_no_engine_preset_can_send_thinking_with_reasoning_effort). The registry comment (and the Settings/Console guides and a test docstring) wrongly said Fireworks "hides reasoning behind its own API surface"; corrected per docs.fireworks.ai/guides/reasoning: reasoning_content, required back on interleaved tool turns (proprietary disposition). Not done: enabling Fireworks reasoning_effort (would need a per-record value allowlist; its guide and API reference disagree on the thinking/effort conflict).
<!-- SECTION:NOTES:END -->
