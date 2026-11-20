---
id: TASK-42019
title: "A Cohere base saved with /v1 posts to /v1/v2/chat"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:54
labels:
- follow-up
- providers
priority: low
references:
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/LLM_Calls/LLM_API_Calls.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review r17cu-rev1 (round 17). Typing `https://api.cohere.com/v1` still saves `/v1` (the rule for a path whose `v1` is last), and the Cohere adapter appends `/v2/chat`, so it posts `/v1/v2/chat`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Cohere base ending in `/v1` on Cohere's own host saves as the bare host, the same rewrite TASK-42002 applies to values already saved
- [ ] #2 A test covers it, red before the fix
<!-- AC:END -->
