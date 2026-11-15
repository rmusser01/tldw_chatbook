---
id: TASK-41146
title: "Named custom endpoints' sampling params table has no User Guide home, though the not-loaded message sends users to config.toml for it"
status: To Do
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
- [ ] #1 The Custom endpoints section of the Settings guide documents [custom_endpoints.<slug>.params]: the keys it reads, their valid ranges, and that an invalid value keeps the entry from loading
- [ ] #2 The not-loaded message's config.toml advice points at a page that covers it
<!-- AC:END -->
