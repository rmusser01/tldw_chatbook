---
id: TASK-41522
title: "First-party cloud adapters send to an unchecked base-URL override in their own config table when the caller pins nothing"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- security
- providers
priority: high
references:
- tldw_chatbook/LLM_Calls/LLM_API_Calls.py
- tldw_chatbook/Provider_Contract
dependencies:
- TASK-41131
- TASK-41132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the endpoint review (round 15) and repeated in endpoint fix r1: the first-party cloud adapters (OpenAI, Anthropic, Cohere, Google, HF, Groq, OpenRouter, DeepSeek, Mistral, Moonshot, Z.ai, QwenCloud) send to a base-URL override in their own config table without going through the provider endpoint contract when the caller pins nothing. This was out of scope for the endpoint round. Related: TASK-41131 and TASK-41132.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed adapter validates a configured base-URL override through the endpoint contract before any request leaves
- [ ] #2 A refused override surfaces a named refusal, not a silent fallback
- [ ] #3 A test per adapter family is red before the fix
<!-- AC:END -->
