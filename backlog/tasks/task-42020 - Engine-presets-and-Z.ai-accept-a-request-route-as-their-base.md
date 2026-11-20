---
id: TASK-42020
title: "Engine presets and Z.ai accept a request route as their base"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:54
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
- [ ] #1 A preset or Z.ai base ending in a request route is refused with `ENDPOINT_ROUTE_REFUSAL`
- [ ] #2 Tests cover the engine presets and Z.ai, red before the fix
<!-- AC:END -->
