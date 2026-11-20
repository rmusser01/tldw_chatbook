---
id: TASK-42020
title: "Engine presets and Z.ai accept a request route as their base"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 16:09
labels:
- follow-up
- providers
priority: low
references:
- tldw_chatbook/Provider_Contract/provider_endpoint_contract.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by review r17cu-rev1 (round 17). TASK-41727 refuses a route-ending path for first-party cloud keys, but the preset reading (TASK-34100.6) still accepts one, for example `https://api.z.ai/api/paas/v4/responses`, which is then sent with the adapter's own route appended.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A preset or Z.ai base ending in a request route is refused with `ENDPOINT_ROUTE_REFUSAL`
- [x] #2 Tests cover the engine presets and Z.ai, red before the fix
<!-- AC:END -->

## Implementation Plan

1. In `_resolve_provider_endpoint`, refuse a route-ending path (`_names_a_route`, TASK-41727's rule) with `ENDPOINT_ROUTE_REFUSAL` before the engine-preset / route-owning-preset reading.
2. Check no preset's shipped default or suffix trips it.
3. Tests in `Tests/Chat/test_local_server_discovery.py`: every engine preset and Z.ai refuse a base ending in a route, red before.

## Implementation Notes

- `_resolve_provider_endpoint` now refuses a route-ending path (`_names_a_route`, TASK-41727's rule) with `ENDPOINT_ROUTE_REFUSAL` before the engine-preset / route-owning-preset reading, so every engine preset and Z.ai refuse e.g. `https://api.z.ai/api/paas/v4/responses`. The documented base, with or without a pasted `/chat/completions` or `/models`, is still accepted; no shipped default trips it (the test checks each record's default).
- At base the adapters (Z.ai's resolver, the hosted engine) already refused such a base at send; the contract was what accepted it, so Console read Ready and a save wrote it.
- Test: `Tests/Chat/test_local_server_discovery.py::test_a_preset_or_zai_base_ending_in_a_route_is_refused` (33 nodes: every engine preset plus Z.ai; Together and Z.ai also dispatch and assert nothing is sent).
- Red/green: 33 failed (`AssertionError: responses`) against the previous contract, 33 passed after.
- Files: `tldw_chatbook/Provider_Contract/provider_endpoint_contract.py`, `Tests/Chat/test_local_server_discovery.py`.
