---
id: TASK-36516
title: Settings saves koboldcpp's suggested root endpoint as an OpenAI chat route that the native koboldcpp send cannot use
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- settings
- providers
- bug
priority: medium
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/Chat/provider_endpoint_contract.py
- tldw_chatbook/Chat/provider_setup_persistence.py
- tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py
dependencies:
- TASK-36380
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36380's review round 2 (PR #3224, 2026-10-23). That PR fixed only the Discover models hint. The bug is older than the PR.

A user who types the endpoint Settings suggests for koboldcpp gets a provider that cannot send:
- **The placeholder is a bare root.** `PROVIDER_ENDPOINT_PLACEHOLDERS["koboldcpp"]` is `http://127.0.0.1:5001` (settings_screen.py:1710).
- **A bare root is saved as the OpenAI chat route.** The endpoint contract reads a root as form `origin`, with `chat_url = <root>/v1/chat/completions` and `persisted_endpoint = chat_url` (provider_endpoint_contract.py:290-292). The Settings save writes that value to `[api_settings.koboldcpp] api_url` (provider_setup_persistence.py:889-897 and :1363-1373). A probe on dev 5b5920b8e2: `resolve_provider_endpoint("koboldcpp", "http://127.0.0.1:5001")` gives `persisted_endpoint == "http://127.0.0.1:5001/v1/chat/completions"`.
- **The koboldcpp send is KoboldAI-native.** `chat_with_kobold` (LLM_API_Calls_Local.py:944) reads that `api_url` (:978). It posts the native generate payload (`prompt`, `max_length`, …; :1071) to the URL unchanged (:1127), then requires a `results` list (:1157-1164) and otherwise raises ChatProviderError "Unexpected response structure". An OpenAI chat completions route answers with `choices`.

Only the native route `/api/v1/generate`, the template default (config.py:5262), works end to end. Not checked against a running koboldcpp server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every koboldcpp endpoint Settings suggests or saves leads to a Console send that posts a payload the saved route accepts; otherwise Settings refuses the endpoint and names the native /api/v1/generate route
- [ ] #2 A koboldcpp endpoint already on the native route is saved unchanged, and a send to it still posts KoboldAI's native payload
- [ ] #3 A test pins the URL and payload shape a koboldcpp send uses after Settings saves a bare root
- [ ] #4 Other providers' saved endpoints are unchanged
<!-- AC:END -->
