---
id: TASK-36380
title: Settings Discover models refuses koboldcpp root and prefix-root endpoints that Switch model probes
status: To Do
created_date: 2026-10-22 23:58
labels:
- follow-up
- settings
- providers
- model-discovery
priority: low
references:
- tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py
- tldw_chatbook/Chat/provider_endpoint_contract.py
dependencies:
- TASK-35832
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the reviews of TASK-35832 (PR #3213, 2026-10-22). Once TASK-35832 lands, Discover models and Switch model agree on koboldcpp's native route `/api/v1/generate`. They still disagree on two other endpoint shapes the endpoint contract accepts:
- **Bare root.** For `http://localhost:5001`, Switch model probes the root's `/v1/models`. Discover refuses it, because koboldcpp is not in `_BASE_URL_INFERABLE_PROVIDER_KEYS`.
- **Prefix roots.** For `/kobold` or `/API/V1/GENERATE`, the contract reads the path as a prefix root and Switch model probes `<root>/v1/models`. Discover's generic path rules refuse it.

The refusal also shows Settings' generic copy ("This endpoint is not a valid OpenAI-compatible models endpoint…"), even where the contract has a specific reason, such as "Endpoint path has ambiguous API suffixes." So a user can see koboldcpp's models in Switch model but can't save them from Settings, and the message doesn't say why.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Discover models for koboldcpp lists from the same models URL Switch model probes, for a bare root and for a prefix root, or the task records why a shape stays refused
- [ ] #2 Where Discover refuses an endpoint the contract refuses with a reason, the Settings message names that reason
- [ ] #3 Other providers' discovery eligibility and models URLs are unchanged
<!-- AC:END -->
