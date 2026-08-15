---
id: TASK-16319
title: Console thinking levels and budget for local providers
status: To Do
assignee: []
created_date: '2026-08-15 18:02'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qwen3.8-27B exposes adjustable thinking (reasoning_effort low/medium/xhigh) and separate thinking-token caps. The Console already has Reasoning/Budget fields but drops them for local providers: the direct llama.cpp path forwards sampling params only, and PROVIDER_PARAM_MAP has no entries for local keys. Wire the existing fields through to llama.cpp (chat_template_kwargs + per-request reasoning_budget), vLLM/Custom OpenAI (top-level reasoning_effort), and MLX-LM (pending live verification), keep thinking output out of the visible reply on the llama.cpp direct path, and warn (not block) on values the model family does not consume.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Console Reasoning/Budget fields reach local provider requests with the per-provider wire format from ADR-066
- [ ] Reasoning effort none sends enable_thinking false and prefill still forces enable_thinking false
- [ ] Qwen3.8-27B verified live against llama-server --jinja: low/medium/xhigh change thinking depth and reasoning_budget truncates thinking (checked with and without --reasoning-format)
- [ ] Visible Console reply never contains think tags or reasoning content on the llama.cpp direct path regardless of the server's reasoning-format flag
- [ ] Values outside the model-family hint set warn without blocking and placeholders list consumed values
- [ ] Request preview shows the composed chat_template_kwargs and reasoning_budget fields
- [ ] Unit and integration tests and lint pass
<!-- AC:END -->
