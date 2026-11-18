---
id: TASK-41727
title: "A cloud provider base whose request route has no v1 before it is accepted as an origin"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
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
Found by TASK-41700's review (round 16), predates it. The generic rule reads a cloud path with no `v1` segment as an origin, so `https://api.openai.com/responses` is accepted and the adapter appends its own route after `/responses`. Close to TASK-41704's scope (the generic rule's `/v1` handling for native cloud bases).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A cloud base ending in a request route with no `v1` before it is refused with the route refusal
- [ ] #2 A test pins one such base per adapter family, red before the fix
<!-- AC:END -->
