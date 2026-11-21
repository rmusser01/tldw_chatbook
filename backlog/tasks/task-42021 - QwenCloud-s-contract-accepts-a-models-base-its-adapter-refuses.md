---
id: TASK-42021
title: "QwenCloud's contract accepts a /models base its adapter refuses"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 16:11
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
- [x] #1 The contract and the adapter agree on a QwenCloud `/models` endpoint, with a test for each side
<!-- AC:END -->

## Implementation Plan

1. The adapter (`normalize_qwencloud_base_url`) refuses a base ending in `/models` on purpose (pinned in `Tests/LLM_Calls/test_qwencloud.py`), so the contract refuses a QwenCloud path ending in `/models` with `ENDPOINT_ROUTE_REFUSAL`, as it reads `/responses` the adapter's way (TASK-41722).
2. Tests: the contract refuses it and the adapter refuses the same values; red before on the contract side.

## Implementation Notes

- The adapter's refusal of a `/models` base is deliberate (pinned by `Tests/LLM_Calls/test_qwencloud.py::test_base_url_rejects_unsafe_or_malformed_values`), so the contract moved to it: `_resolve_provider_endpoint` refuses a `qwencloud` path whose last segment is `models` with `ENDPOINT_ROUTE_REFUSAL`, before the TASK-41722 `/responses` strip (the adapter's order). Before, the cloud reading dropped it as a pasted `/models` and Console read Ready.
- Test: `Tests/Chat/test_local_server_discovery.py::test_console_and_the_qwencloud_adapter_agree_on_a_models_endpoint` (8 nodes) asserts both sides per value: `endpoint_refusal`, the Settings save (`persisted_endpoint_for_key`) and `normalize_qwencloud_base_url` refuse the same five `/models` values (own host, trailing slash, `/responses/models`, bare host, a gateway) and accept the base, a pasted `/responses` and a pasted `/chat/completions` alike.
- Pin rewritten on purpose: `Tests/Chat/test_provider_setup_persistence.py::test_persisted_endpoint_shape_matches_provider_and_owned_key`'s QwenCloud row saved `.../proxy/v1/models` as `.../proxy/v1`; a save now refuses it, so the row pastes `/chat/completions` instead (same saved base).
- Red/green: the 5 refused nodes failed against the previous contract (`assert '' == 'Endpoint path names a request route...'`; the 3 accepted ones are invariants), 8 passed after.
- Docs: `Docs/Architecture/llm-providers.md` and `Docs/User_Guide/settings.md` (which also gains TASK-42020's preset/Z.ai route refusal).
- Files: `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py`, `Tests/Chat/test_local_server_discovery.py`, `Tests/Chat/test_provider_setup_persistence.py`, the two docs.
