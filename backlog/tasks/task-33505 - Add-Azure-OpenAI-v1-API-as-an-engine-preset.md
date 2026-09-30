---
id: TASK-33505
title: Add Azure OpenAI (v1 API) as an engine preset
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-29 19:28'
updated_date: '2026-09-29 20:03'
labels:
  - providers
  - engine
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Azure OpenAI is the common enterprise route to OpenAI models. The Hermes/oh-my-pi comparison deferred it because its responses carry content-filter annotations the strict engine cannot parse yet, its URL is per resource, and newer deployments require `max_completion_tokens`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user who sets their Azure resource URL and key can chat with a deployment through the engine, streaming and non-streaming, on the documented v1 API with the api-key header
- [x] #2 Azure's documented content-filter annotations parse; for every preset, unknown fields and non-empty unexpected arrays still fail closed (an allowlisted field may also be an empty list, dropped like null)
- [x] #3 Deployments that require max_completion_tokens can be used
- [x] #4 A content-filter stop is reported as a provider error
- [x] #5 Readiness asks for the resource URL, and the Settings guide documents setup and the unsupported asynchronous-filter mode
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Engine: optional max_tokens_key (max_completion_tokens); a stream annotation key whose choice-less, usage-less frames are validated and dropped; an allowlisted field may be an empty list.
2. AZURE record: user-supplied resource host + /openai/v1 suffix, api-key header, deployment names as models (seeded-only), documented content-filter allowances, content_filter finish = provider error.
3. Readiness: base URL required with an Azure example; modal collects it.
4. Hand lists, config tables, tests (fixtures from captured Azure shapes), Settings guide incl. unsupported asynchronous filter.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AZURE record (key azure): v1 API, api-key header (auth_scheme api_key_header), default_base_url None + base_url_suffix /openai/v1 (the user sets the resource host, Databricks pattern), seeded-only (models are deployment names; the models route lists base models), content_filter finish = provider error. Engine additions, each opt-in or data-neutral: max_tokens_key (Azure sends max_completion_tokens, which v1/o-series/gpt-5 require); stream_annotation_key (HostedChatStream accepts and drops a choice-less, usage-less frame carrying the key -- Azure's leading prompt_filter_results frame, which otherwise fails as misplaced usage); and the level value rule now accepts an empty list (no data, dropped like null) so annotations: [] parses -- unknown keys and non-empty arrays still fail closed for every preset (AC reworded to state this exactly). Allowances come from Azure docs plus two public captures (see the registry comment). Readiness: azure joins PROVIDERS_REQUIRING_BASE_URL_KEYS with its own reason/example ("Missing resource URL", mapped to endpoint_missing); Databricks copy byte-identical. Not supported (documented): Asynchronous Filter mode, Entra ID tokens. Tests: Tests/LLM_Calls/test_followup_presets.py (captured response + stream, negative controls); mutations of each engine change turn tests red. No live call (no key).
<!-- SECTION:NOTES:END -->
