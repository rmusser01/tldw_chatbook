---
id: TASK-41523
title: "Custom OpenAI API 2 reports ready without a key, but its handler requires one"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- providers
- settings
priority: medium
references:
- tldw_chatbook/LLM_Calls/LLM_API_Calls.py
- tldw_chatbook/Provider_Contract/provider_setup_facts.py
dependencies:
- TASK-41143
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in endpoint fix r1 (round 15). Readiness says Custom OpenAI API 2 is ready without a key, but `chat_with_custom_openai_2` requires one unless the caller resolved it, so keyless Dreams and ingest sends fail with "API Key required" (pre-existing as "api_ip required").
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Readiness and the handler agree on whether a key is needed for a configured endpoint
- [ ] #2 A keyless endpoint that does not need a key sends from Dreams and ingest, or readiness says Needs key
- [ ] #3 A test covers both
<!-- AC:END -->
