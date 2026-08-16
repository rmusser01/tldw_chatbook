---
id: TASK-16319
title: Console thinking levels and budget for local providers
status: In Progress
assignee:
  - '@Robert'
created_date: '2026-08-15 18:02'
updated_date: '2026-08-16 03:14'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qwen3.8-27B exposes adjustable thinking (reasoning_effort low/medium/xhigh) and separate thinking-token caps. The Console already has Reasoning/Budget fields but drops them for local providers: the direct llama.cpp path forwards sampling params only, and PROVIDER_PARAM_MAP has no entries for local keys. Wire the existing fields through to llama.cpp (chat_template_kwargs + per-request reasoning_budget), vLLM/Custom OpenAI (top-level reasoning_effort), and MLX-LM (pending live verification), keep thinking output out of the visible reply on the llama.cpp direct path, and warn (not block) on values the model family does not consume.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console Reasoning/Budget fields reach local provider requests with the per-provider wire format from ADR-066
- [ ] #2 Reasoning effort none sends enable_thinking false and prefill still forces enable_thinking false
- [ ] #3 Qwen3.8-27B verified live against llama-server --jinja: low/medium/xhigh change thinking depth and reasoning_budget truncates thinking (checked with and without --reasoning-format)
- [ ] #4 Visible Console reply never contains think tags or reasoning content on the llama.cpp direct path regardless of the server's reasoning-format flag
- [ ] #5 Values outside the model-family hint set warn without blocking and placeholders list consumed values
- [ ] #6 Request preview shows the composed chat_template_kwargs and reasoning_budget fields
- [ ] #7 Unit and integration tests and lint pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Wire-format table (ADR-066)\n2. Direct llama.cpp payload wiring\n3. Start-anchored think filter\n4. Wire filter into direct path\n5. Adapter-path param maps + shared builder + handlers\n6. Hints, warnings, placeholder, preview\n7. Full verification and wrap-up
<!-- SECTION:PLAN:END -->
