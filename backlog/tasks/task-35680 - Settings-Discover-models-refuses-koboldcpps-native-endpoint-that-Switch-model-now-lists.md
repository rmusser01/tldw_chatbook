---
id: TASK-35680
title: Settings Discover models refuses koboldcpp's native endpoint that Switch model now lists
status: To Do
created_date: 2026-10-20 03:58
dependencies:
- TASK-35010
labels:
- follow-up
- settings
- providers
- model-discovery
priority: low
references:
- tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py
- tldw_chatbook/Chat/provider_endpoint_contract.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-35010 (2026-10-20). Under the owner's ruling "All 5, endpoint too", koboldcpp's template endpoint `/api/v1/generate` now resolves through the endpoint contract: its models URL is the root's `/v1/models`, and Switch model lists a running koboldcpp from it. Settings ▸ Discover models still refuses the same URL as a native endpoint (`_NATIVE_ENDPOINT_PATHS_BY_PROVIDER` in openai_compatible_model_discovery.py). So on a profile left at the template, Switch model can list koboldcpp's models while Discover models in Settings reports that the endpoint is unsupported, and the user cannot save those models from Settings without changing the URL.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Discover models for koboldcpp at its template endpoint lists the models the server serves at the root's /v1/models, the same listing Switch model shows
- [ ] #2 Save selected then saves them to koboldcpp's list, and koboldcpp's send URL stays the native route
- [ ] #3 An endpoint with an API suffix before the native route is still refused as ambiguous
<!-- AC:END -->
