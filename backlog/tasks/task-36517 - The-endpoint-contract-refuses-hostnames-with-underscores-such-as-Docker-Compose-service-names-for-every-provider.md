---
id: TASK-36517
title: The endpoint contract refuses hostnames with underscores, such as Docker Compose service names, for every provider
status: To Do
created_date: 2026-10-24 00:26
labels:
- follow-up
- settings
- providers
priority: low
references:
- tldw_chatbook/Chat/provider_endpoint_contract.py
- tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py
dependencies:
- TASK-36380
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36380's review (PR #3224, 2026-10-23), and checked on dev 5b5920b8e2. A local server reached by a container name that contains an underscore, such as `http://kobold_cpp:5001`, cannot be configured. Docker Compose service names may contain underscores, and httpx accepts the URL (`httpx.URL("http://kobold_cpp:5001/v1").host == "kobold_cpp"`).

The endpoint contract is the cause. `_valid_dns_label` (provider_endpoint_contract.py:548-554) allows only letters, digits and hyphens, so `_canonical_host` returns None and the contract answers "Endpoint URL must include a valid host and port." (:239-243). The message is wrong: the URL has both.

The refusal is not limited to koboldcpp. A probe through the scratch wrapper on dev resolved `http://kobold_cpp:5001/v1` for every key in `URL_BASED_PROVIDER_KEYS` (14 keys) and for openai, deepinfra and anthropic, 17 providers in all. Every one refused it, and every one accepted the same URL spelled `kobold-cpp`.

Two readers of the same endpoint disagree. Discovery's own URL parse accepts underscores (`supports_openai_compatible_model_discovery("vllm", "http://vllm_server:8000/v1")` and the same for `custom` return True). So Discover models can list from an endpoint that the provider save then refuses with "Endpoint is invalid." On PR #3224's branch, koboldcpp's discovery follows the contract and refuses it too.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An http or https endpoint whose host contains an underscore resolves for every provider the contract serves, or is refused with a message that names the underscore
- [ ] #2 Discover models and the provider save make the same decision on such an endpoint
- [ ] #3 Hosts the contract refuses today for any other reason are still refused, and a test pins both directions
<!-- AC:END -->
