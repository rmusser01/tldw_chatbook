---
id: TASK-36517
title: The endpoint contract refuses hostnames with underscores, such as Docker Compose service names, for every provider
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:26
updated_date: '2026-11-05 21:17'
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
- [x] #1 An http or https endpoint whose host contains an underscore resolves for every provider the contract serves, or is refused with a message that names the underscore
- [x] #2 Discover models and the provider save make the same decision on such an endpoint
- [x] #3 Hosts the contract refuses today for any other reason are still refused, and a test pins both directions
<!-- AC:END -->

## Implementation Plan

1. Resolve, not refuse: `_valid_dns_label` in `Provider_Contract/provider_endpoint_contract.py` accepts `_` anywhere in a label, like a letter. Discovery's own URL parse (`validate_url`) already accepts such hosts, so once the contract does, the save and Discover models agree, including koboldcpp, whose Discover follows the contract (AC#1, AC#2).
2. Keep every other host rule as it is: ASCII only, no `%`, no empty or `..` labels, a hyphen never first or last, labels of 1-63 characters, a host of at most 253, not all digits and dots (AC#3).
3. Tests in `Tests/Chat/test_local_server_discovery.py`, which a PR lane runs whole (dreams-boundary). For every provider the contract serves (`URL_BASED_PROVIDER_KEYS` plus openai, deepinfra, anthropic), the save builder and Discover's eligibility both accept `kobold_cpp` exactly as they accept `kobold-cpp`; hosts refused today, with and without an underscore, are still refused with the same message. Red before the fix.

## Implementation Notes

The endpoint contract accepts an underscore in a host, so a Docker Compose name such as `http://kobold_cpp:5001/v1` resolves and saves for every provider it serves, as Discover models already listed from it.
- `_valid_dns_label` (`Provider_Contract/provider_endpoint_contract.py`) allows `_` anywhere in a label. Every other host rule is unchanged: ASCII only, no `%`, no empty label, a hyphen never first or last, 1-63 characters per label, at most 253 per host, and not all digits and dots. Discovery needed no change. Its own parse (`validate_url`) already accepted such hosts, and koboldcpp's Discover follows the contract (TASK-36380), so it now accepts the native route on such a host too.
- How the criteria were read: AC#1 and AC#3 overlap on a host such as `-kobold_cpp`, which has an underscore and is also refused for another reason (here a leading hyphen). AC#3 decides it: such a host stays refused with the message it gets today, so AC#1 and AC#2 cover hosts whose only departure from the old rule is the underscore. Discovery's generic rules still accept some hosts the contract refuses for those other reasons, such as `-kobold`, `kob*old` or `1.2.3`, with or without an underscore. That disagreement is older than this task and is listed as a proposed follow-up, not changed here.
- Tests, in `Tests/Chat/test_local_server_discovery.py` (run whole by the dreams-boundary lane):
  - `test_an_underscore_host_saves_and_discovers_like_its_hyphen_spelling`: 19 providers, the 16 `URL_BASED_PROVIDER_KEYS` plus openai, deepinfra and anthropic; koboldcpp on `/api/v1/generate`. The contract resolves `kobold_cpp`, `build_provider_setup_mutation` saves exactly what it saves for `kobold-cpp` with the host respelled, and `supports_openai_compatible_model_discovery` accepts both spellings.
  - `test_hosts_refused_for_another_reason_are_still_refused`: 14 hosts, with and without `_` (leading or trailing hyphen, `..`, `%`, `*`, all digits and dots, a 64-character label, a 304-character host), for all 19 providers. Each still gets "Endpoint URL must include a valid host and port." and the save refuses it.
  - `Tests/Chat/test_provider_endpoint_contract.py` pinned `http://bad_host/v1` as rejected. It is rewritten on purpose to `http://-bad_host/v1`, which is still rejected.
- Red before the fix: with the base contract, all 19 underscore cases failed with `assert ('Endpoint UR...t and port.',) == ()`. The refusal test pins rules dev already keeps. Under a temporary edit that dropped the hyphen-edge rule and allowed `*`, it failed for 6 hosts (`-kobold`, `kobold-`, `-kobold_cpp`, `kobold_cpp-`, `kob*old`, `kob_*old`). Green after the fix.
- Covering run (`-n 4`, run.sh, load average 43-54): the contract, discovery, console endpoints, setup persistence, setup facts, switcher probe targets, both local-server discovery files, evidence hardening, custom endpoint registry and engine swap, Kimi/Z.ai, QwenCloud and Strata provider files, and two wizard endpoint files. Result: 1129 passed, 1 failed. The failure is `test_qwencloud_provider_contract.py::test_chat_api_call_forwards_qwencloud_mode_base_and_tools` (`assert 'api_mode' not in {...}`), and it fails the same way with the base contract module.

Files: `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py`, `Tests/Chat/test_local_server_discovery.py`, `Tests/Chat/test_provider_endpoint_contract.py`.

