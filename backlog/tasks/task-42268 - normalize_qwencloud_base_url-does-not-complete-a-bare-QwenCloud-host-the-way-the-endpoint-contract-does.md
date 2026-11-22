---
id: TASK-42268
title: "normalize_qwencloud_base_url does not complete a bare QwenCloud host the way the endpoint contract does"
status: To Do
assignee: []
created_date: 2026-11-22 00:17
updated_date: 2026-11-22 00:17
labels:
- follow-up
- providers
- qwencloud
- bug
priority: medium
references:
- tldw_chatbook/LLM_Calls/qwencloud.py
- tldw_chatbook/LLM_Calls/qwencloud_url.py
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
- backlog/tasks/task-42153 - Decide-whether-the-v1-base-migration-also-covers-QwenClouds-China-host-and-Coheres-old-api.cohere.ai.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review of the providers2 stream (round 19). Run at e5626ea043 through the scratch wrapper: the endpoint contract's `resolve_provider_endpoint("qwencloud", "https://dashscope-intl.aliyuncs.com")` completes the bare intl host to `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`, but `normalize_qwencloud_base_url` returns the bare host, and `https://dashscope.aliyuncs.com/compatible-mode` unchanged. The adapter builds its URL as `f"{final_base}/responses"` (`tldw_chatbook/LLM_Calls/qwencloud.py:1191`), so a stored bare host makes the adapter post `<host>/responses`, while the contract (and Console's shipped default, `console_provider_endpoints.py:70`) names `/compatible-mode/v1`. At e5626ea043 the contract reads the China host `dashscope.aliyuncs.com` as `/v1`; TASK-42153 (round 19, same batch) makes it complete the bare China host and `…/compatible-mode` to `/compatible-mode/v1` as well, so the adapter has the same gap on both hosts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A stored bare `https://dashscope-intl.aliyuncs.com` or `https://dashscope.aliyuncs.com`, or either host with `/compatible-mode`, makes the adapter send to the same URL the contract resolves, with a test per shape that is red before the fix
- [ ] #2 The contract and the adapter resolve the same stored value to the same URL
- [ ] #3 A base on another host is left as entered
<!-- AC:END -->
