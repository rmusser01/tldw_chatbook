---
id: TASK-41146
title: "Named custom endpoints' sampling params table has no User Guide home, though the not-loaded message sends users to config.toml for it"
status: Done
assignee: ['@claude']
updated_date: 2026-11-15 18:20
created_date: 2026-11-15 06:30
labels:
- follow-up
- docs
- user-guide
- custom-endpoints
priority: low
references:
- Docs/User_Guide/settings.md
- tldw_chatbook/Chat/custom_endpoint_registry.py
- tldw_chatbook/Provider_Contract/sampling_params.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top d4e8d51ca8 (endpoint stream b3c75c8234, on dev ecad404bd4); line numbers are that tree's.

**Why.** When a named endpoint's sampling params are invalid, Chatbook says "Provider blocked: this custom endpoint is not loaded: … Fix it in config.toml, or delete the entry in Settings ▸ Custom endpoints." The user then has no page that says which keys `[custom_endpoints.<slug>.params]` takes or what ranges `validate_sampling_params` accepts. Found by review r14ep-rev4 (declined there, #9).

**Evidence.** `custom_endpoint_registry.py:325-337` chooses the config.toml wording when the refusal reason comes from `validate_sampling_params(params)` (`Provider_Contract/sampling_params.py:90`). `Docs/User_Guide/settings.md`'s Custom endpoints section (from `:1061`) documents the `[api_settings.custom]` sampler keys but not the named entries' `params` table; only `console/agent-runs-and-tools.md:2123` names it, in passing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Custom endpoints section of the Settings guide documents [custom_endpoints.<slug>.params]: the keys it reads, their valid ranges, and that an invalid value keeps the entry from loading
- [x] #2 The not-loaded message's config.toml advice points at a page that covers it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read custom_endpoint_registry.py (how [custom_endpoints.<slug>.params] is loaded and which refusal produces the config.toml wording) and Provider_Contract/sampling_params.py (validate_sampling_params: accepted keys, types, ranges).
2. Add a subsection to the Settings guide's Custom endpoints section quoting the real keys and ranges, and stating that an invalid value keeps the entry from loading, with the message the user sees.
3. Check the not-loaded message's config.toml advice leads to that page (message wording or a link from where the message is documented); add a docs pin only if a cheap one fits an existing lane.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Documented the named entries' `params` table where the not-loaded message now sends users, and made the message name that page.

- **Docs** (`Docs/User_Guide/settings.md`, Custom endpoints): a **Sampling params** paragraph, a TOML example and a key table quoting `KNOWN_SAMPLING_PARAM_KEYS` and the ranges/enums of `validate_sampling_params` (`Provider_Contract/sampling_params.py`): float keys 0.0-2.0 / 0.0-1.0 / -2.0-2.0, int minima 0/1/0/1024, the four string enumerations. It says what the code does, not more: the values apply only when an agent run routes work to the entry (`Agents/agent_routing.resolve_child_params` and `console_chat_controller._resolve_new_chat_routing` / `execute_agent_chat_create` pass them through the ADR-344 `extra_sources` seam; an ordinary Console session never does), ranking below `[console.provider_defaults]` and above `[chat_defaults]`; the New endpoint form's **Params** box writes the table and refuses bad values before Create; **Edit** and **Rename** keep it unchanged (`build_entry_edit_mutation`, `build_entry_mutation(replace(...))`); an invalid value keeps the entry from loading, with the Settings row and chat copy quoted; a non-table `params` drops the whole entry (pydantic `_params_reject_non_tables` -> "malformed entry", so it is not listed and a chat reads "no longer configured"). One sentence after the family table gives the `family` config values (`ENDPOINT_FAMILIES`), since an unknown family gets the same config.toml advice.
- **Copy** (`Chat/custom_endpoint_registry.missing_custom_endpoint_copy`): "Fix it in config.toml" now reads "Fix it in config.toml (see 'Custom endpoints' in the User Guide's Settings page), or delete the entry in Settings ▸ Custom endpoints.", following the `NO_SERVER_HELP` precedent ("see 'Local servers' in the User Guide's First-Run Setup page"). The URL-refusal branch ("Fix it in Settings ▸ Custom endpoints.") is unchanged.
- **Evidence.** Red: the two config.toml cases of `Tests/Chat/test_local_server_discovery.py::test_a_custom_endpoint_not_loaded_says_where_its_reason_can_be_fixed` failed on the new expected copy before the source change (2 failed, 1 passed); green after; the whole file 884 passed (`-n 4`, load 42). A wrapper probe over real TOML confirmed every documented claim: `temperature = 1` and `presence_penalty = -2` load; `max_tokens = 4096.0`, `seed = nan`, `temperature = inf`, `stop`, `streaming = true`, `top_k = true`, `verbosity = "LOW"`, `thinking_budget_tokens = 1023` are refused with the quoted reasons on both the Settings row and the chat copy; `params = "x"` drops the entry; `family = "vllm"` reads "not loaded: Unknown endpoint family: vllm."; the Edit mutation keeps `params`. Ruff gate OK.
- Files: `tldw_chatbook/Chat/custom_endpoint_registry.py`, `Tests/Chat/test_local_server_discovery.py`, `Docs/User_Guide/settings.md`.
<!-- SECTION:NOTES:END -->
