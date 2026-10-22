---
id: TASK-35832
title: Settings Discover models refuses koboldcpp's native endpoint that Switch model now lists
status: Done
assignee:
- '@claude'
created_date: 2026-10-20 03:58
updated_date: 2026-10-22 20:25
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
- [x] #1 Discover models for koboldcpp at its template endpoint lists the models the server serves at the root's /v1/models, the same listing Switch model shows
- [x] #2 Add to model list then saves them to koboldcpp's list, and koboldcpp's send URL stays the native route
- [x] #3 An endpoint with an API suffix before the native route is still refused as ambiguous
<!-- AC:END -->

## Implementation Plan

1. In openai_compatible_model_discovery.py, stop refusing koboldcpp's native `/api/v1/generate` outright: for a koboldcpp path ending in that route, eligibility and the models URL come from `resolve_provider_endpoint` (the contract Switch model probes), so a route the contract refuses (an API suffix before it) stays unsupported. Ollama's `/api/tags` refusal is unchanged.
2. Flip the unit tests that pinned the refusal (`test_native_kobold_generate_is_not_eligible`, the koboldcpp half of `test_malformed_endpoint_reports_distinct_error_kind`) and add the ambiguous-suffix refusal and the request URL.
3. Extend the census-listed koboldcpp journey in Tests/UI/test_settings_saved_lists_follow_file.py: on a private profile left at the template, Settings Discover models lists what the stub serves at `/v1/models`, Add to model list writes them to koboldcpp's list, and `api_settings.koboldcpp.api_url` stays the native route.
4. One sentence in Docs/User_Guide/settings.md; run the discovery, contract and Settings discovery tests with -n 4, preflight.

## Implementation Notes

Discovery now asks the endpoint contract about koboldcpp's native route instead of refusing it, and the contract alone decides which endpoints that route covers.
- The contract gained `is_native_generate_route(resolution)`. It uses the same `_is_kobold_native_generate` check the resolver uses on canonical, percent-decoded segments.
- In `openai_compatible_model_discovery.py`, koboldcpp left `_NATIVE_ENDPOINT_PATHS_BY_PROVIDER` (Ollama's `/api/tags` stays). Discovery keeps no copy of the route.
- `supports_openai_compatible_model_discovery` and `build_models_url` resolve the endpoint first. When the contract reads it as the native route, the endpoint is eligible and lists at the contract's `models_url`, the URL Switch model probes. So `http://localhost:5001/api/v1/generate` and `.../gen%65rate` list from `http://localhost:5001/v1/models`, and `/proxy/api/v1/generate` lists from `/proxy/v1/models`.
- A form the contract refuses stays `unsupported_endpoint` under the generic path rules, which list no path ending in the route. Examples: `/v1/api/v1/generate` ("Endpoint path has ambiguous API suffixes."), `/v1/models/api/v1/generate`, or one with a query.
- `/API/V1/GENERATE` is not the route to the contract, which reads it as a plain prefix root. Switch model would probe `/API/V1/GENERATE/v1/models` for it. Discover refuses it under the generic rules, as it refuses any unknown prefix root such as `/kobold`. That gap between the two surfaces predates this task and is not specific to the native route.
- The `supports_...` docstring names the exception.
- The discovery cache fingerprint stays the native URL. Add to model list writes only `[providers]`, so `api_settings.koboldcpp.api_url` is untouched.

Tests, all rewritten on purpose:
- `test_native_kobold_generate_is_not_eligible` became `test_native_kobold_generate_lists_where_the_endpoint_contract_says`, with four endpoints including `%65`.
- `test_kobold_generate_the_endpoint_contract_refuses_is_not_eligible` (AC#3) asserts that the contract refuses each case and that discovery does too.
- `test_kobold_uppercase_generate_is_a_prefix_root_that_generic_rules_refuse` covers the uppercase case.
- `test_kobold_native_endpoint_discovers_from_its_roots_v1_models` uses a MockTransport.
- The koboldcpp half of `test_malformed_endpoint_reports_distinct_error_kind` uses the ambiguous URL.
- The census-listed `Tests/UI/test_settings_saved_lists_follow_file.py::test_koboldcpps_native_endpoint_lists_at_v1_models_and_sends_unchanged`:
  - asserts AC#3 (`/v1/api/v1/generate` is refused by the contract and by discovery);
  - runs the Settings journey on a private profile at the template: Discover models through the real service and client with a MockTransport, then Add to model list;
  - asserts the single request is the contract's `models_url`, the receipt, `[providers].koboldcpp == ["None", "kobold-served-a", "kobold-served-b"]` in the file, and `api_url` still the native route.
  - The census needs no edit.

Red/green evidence:
- At dev 28beeb9ed5, the journey failed at `_press_discover` with "This endpoint is not a valid OpenAI-compatible models endpoint. ...", and the unit test failed with `assert 'unsupported' == 'success'`.
- At d38aac57c7, which still had the raw-path pre-filter, the `%65` case failed with `assert False is True`.
- Negative controls, all reverted:
  - accepting any path ending in the route turned the journey red on the new AC#3 assertion, plus 3 cases of the refusal test;
  - trusting any error-free contract resolution turned the uppercase test red.
- Runs after the review fixes (`-n 4`):
  - `Tests/LLM_Provider_Catalog`, the contract and local-server-discovery tests, `test_settings_saved_lists_follow_file.py` and `test_settings_pick_joins_saved_list.py`: 710 passed, 1 failed. The failure is `test_probe_handles_recursive_json_as_bounded_failure` ("network policy"), which fails the same way at base.
  - The picker, listing-dropped, key-check and switcher-probe files: 92 passed.
- In the first round's wider run, 68 failures (`RecoveryRequired: raw_source_selection_changed` on plain Tests/UI nodes, and "network policy" on the endpoint probes) were identical in names and reasons with the source change reverted.
- `preflight.sh` passes.

Files: `tldw_chatbook/Chat/provider_endpoint_contract.py`, `tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py`, `Tests/LLM_Provider_Catalog/test_openai_compatible_model_discovery.py`, `Tests/UI/test_settings_saved_lists_follow_file.py`, `Docs/User_Guide/settings.md`.
