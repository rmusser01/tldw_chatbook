---
id: TASK-19021
title: >-
  Remaining summarize_with_* providers send unconditional sampling params with
  no capability gate
status: To Do
assignee: []
created_date: '2026-08-20 15:16'
labels:
  - llm
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-18802 fixed the Anthropic and OpenAI halves of the summarization path and swept the rest of Summarization_General_Lib.py for the same unconditional-params shape. CODE-READ, not probe-verified -- each needs a reproduction before being fixed. Sites (line numbers as of the 18802 branch): summarize_with_cohere :1307 temperature; summarize_with_groq :1555 temperature; summarize_with_openrouter :1779 and :1868 temperature (openrouter proxies OpenAI/Anthropic models, so a routed gpt-5 or claude-sonnet-5 may inherit exactly the 400s 18802 fixed -- worth probing first); summarize_with_huggingface :1967-1969 max_tokens + temperature; summarize_with_deepseek :2159 temperature; summarize_with_mistral :2342-2344 temperature AND top_p together plus max_tokens (the both-at-once shape that 400s on Anthropic Claude 4.x per TASK-19020 -- check whether Mistral restricts the combination). summarize_with_google sends none (commented out) and is unaffected. None of these providers' modern-model behavior has been probed; if a provider rejects nothing, record that and close. Follow the model_capabilities predicate design from TASK-18414/18802 for any that reproduce.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each provider's modern-model behavior toward its unconditional params is probed and recorded before any code change
- [ ] #2 Any provider that reproduces a rejection consults a model_capabilities predicate instead of sending the param unconditionally
- [ ] #3 Providers whose models accept the params are left unchanged
<!-- AC:END -->
