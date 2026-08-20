---
id: TASK-19051
title: >-
  Reproduce unconditional-sampling-param behavior for the five unprobeable
  summarize_with_ providers (openrouter, groq, huggingface, deepseek, mistral)
status: To Do
assignee: []
created_date: '2026-08-20 16:07'
labels:
  - llm
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-19021 probed what it could: cohere (working key) accepts its unconditional temperature on the current flagship command-a-plus-05-2026, the function default command-a-03-2025, and command-a-reasoning-08-2025 -- all HTTP 200 with the exact summarize_with_cohere payload shape -- so that function was left unchanged. The remaining five sites could not be probed for lack of working credentials: the repo-root openrouter key returns 401 'User not found' on both /api/v1/key and chat completions (dead key, matching the prior session's caveat), and there are no groq/huggingface/deepseek/mistral keys in the repo root or environment. Suspect sites (line numbers on origin/dev 25500ad87): summarize_with_groq :1570 temperature; summarize_with_openrouter :1794/:1883 temperature (routed vendor-prefixed ids may inherit the upstream OpenAI/Anthropic 400s TASK-18802/19020 fixed -- probe a routed openai/gpt-5 and anthropic/claude-sonnet-5 first; if the rejection is just the upstream rule, prefer stripping the vendor prefix and consulting the EXISTING openai/anthropic predicates in model_capabilities.py over minting an openrouter-specific table); summarize_with_huggingface :1980-1985 max_tokens+temperature; summarize_with_deepseek :2174 temperature; summarize_with_mistral :2357-2359 temperature AND top_p together plus max_tokens (the both-at-once shape that 400s on Anthropic Claude 4.x per TASK-19020). Additionally, credential-free probing established that summarize_with_huggingface's endpoint host api-inference.huggingface.co no longer exists in DNS (NXDOMAIN while router.huggingface.co and huggingface.co resolve), so that function fails at connect for every caller regardless of params -- its real fix is an endpoint migration, not only a param gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each provider's modern-model behavior toward its function's exact unconditional-param payload shape is probed and recorded before any code change, once a working credential exists
- [ ] #2 Any reproduced rejection consults a model_capabilities predicate per the TASK-18414/18802 design; for openrouter, prefix-stripping onto the existing openai/anthropic predicates is preferred over a new table if the rejection is the upstream rule
- [ ] #3 summarize_with_huggingface's dead api-inference.huggingface.co endpoint is migrated to a served host or its failure mode made honest
- [ ] #4 Providers whose models accept the params are left unchanged
<!-- AC:END -->
