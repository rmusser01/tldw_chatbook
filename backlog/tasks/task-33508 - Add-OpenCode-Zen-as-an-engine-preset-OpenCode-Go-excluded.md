---
id: TASK-33508
title: Add OpenCode Zen as an engine preset (OpenCode Go excluded)
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-29 19:26'
updated_date: '2026-09-29 20:03'
labels:
  - providers
  - engine
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
OpenCode Zen is a pay-per-request model gateway whose Chat Completions models (DeepSeek, GLM, Kimi, MiniMax, Qwen and free models) any client can call with a Zen key. OpenCode Go is a subscription meant for coding agents and requires per-conversation session headers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user with an OpenCode Zen key can chat with seeded Chat Completions models through the engine
- [x] #2 Zen's documented response extras parse, and unknown extras still fail closed
- [x] #3 OpenCode Go's exclusion and its reason are recorded
- [x] #4 The Settings guide documents setup and that Responses- or Messages-only models are not offered
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. OPENCODE_ZEN record: opencode.ai/zen/v1, Bearer, seeded Chat Completions models only, top-level cost allowance, proprietary reasoning.
2. Record OpenCode Go exclusion (per-conversation session header, coding-agent-only terms).
3. Hand lists, config, tests, Settings guide.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
OPENCODE_ZEN record (key opencode_zen): opencode.ai/zen/v1, Bearer OPENCODE_API_KEY (the name Zen's docs use), seeded with the 15 Zen models documented AND listed on the chat/completions protocol (Zen does not translate protocols; /models has no protocol field, so discovery is off). Top-level cost allowed; the streamed cost frame after [DONE] is never read; reasoning_content private and replayed on tool turns. Upstream passthrough means other extras stay strict until a live capture. OpenCode Go excluded and recorded (registry + Settings guide): coding-agent subscription requiring a per-conversation x-opencode-session header.
<!-- SECTION:NOTES:END -->
