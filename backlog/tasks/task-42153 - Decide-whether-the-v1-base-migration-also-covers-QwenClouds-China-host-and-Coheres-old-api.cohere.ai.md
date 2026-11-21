---
id: TASK-42153
title: "Decide whether the /v1 base migration also covers QwenCloud's China host and Cohere's old api.cohere.ai"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
labels:
- follow-up
- providers
- settings
- decision
priority: low
references:
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- tldw_chatbook/config.py
- tldw_chatbook/provider_registry.py
- Docs/User_Guide/settings.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review of the providers stream's TASK-42002 (round 18; owner note). TASK-42002 rewrites bases an earlier Settings saved with `/v1` appended, and its AC#1 limits the rewrite to the provider's own host: on the round-18 providers branch, `_documented_own_path` in `Provider_Contract/provider_endpoint_contract.py` returns nothing unless the saved host equals the host of the registry's `default_base_url`. Two saved shapes are therefore left as they are:
- QwenCloud's China host, `https://dashscope.aliyuncs.com/v1`. The registry default is `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` (`tldw_chatbook/provider_registry.py:1649`), and the China host's compatible base also ends in `/compatible-mode/v1`.
- Cohere's old host, `https://api.cohere.ai/v1`. The default is `https://api.cohere.com` (`provider_registry.py:1582`); with `/v1` kept, the adapter posts to `/v1/v2/chat`, the shape TASK-42019 covers for `api.cohere.com`.

Whether to extend the migration to these hosts is an owner decision.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision is recorded: whether the migration also rewrites these two shapes
- [ ] #2 If extended: each shape is rewritten to its documented base when config loads and saved once, any other host is still left alone, and tests for both are red before the change
- [ ] #3 `Docs/User_Guide/settings.md` lists the shapes the migration covers
<!-- AC:END -->
