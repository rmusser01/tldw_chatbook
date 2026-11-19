---
id: TASK-42021
title: "QwenCloud's contract accepts a /models base its adapter refuses"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- providers
priority: low
references:
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/LLM_Calls
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41722 (round 17). The endpoint contract accepts a QwenCloud `…/v1/models` endpoint, but the adapter's `normalize_qwencloud_base_url` refuses a base ending in `/models`, so Console reads Ready and the send fails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The contract and the adapter agree on a QwenCloud `/models` endpoint, with a test for each side
<!-- AC:END -->
