---
id: TASK-32474
title: Console custom endpoint registry (named endpoints from templates)
status: To Do
assignee: []
created_date: '2026-09-11 03:43'
labels: []
dependencies: []
---

## Renumbering provenance

Renumbered from TASK-32474 on 2026-09-11: the id collided with a task that
arrived on dev while this branch was in review (owner rule TASK-19601 — the
older arrival keeps the id). No dependencies referenced the old id.


## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the named custom-endpoint registry per Docs/superpowers/specs/2026-09-10-console-custom-endpoint-registry-design.md: config-owned [custom_endpoints.<slug>] entries mapped onto existing execution families, creation-from-template flow in Conversation Settings, F9 Settings management, optional one-way convert for the custom/custom_2 slots. Integrates with the TASK-30012 connection-first modal recomposition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Registry entries persist in [custom_endpoints.<slug>] and survive restart,custom-ep:<slug> ids resolve through the family execution path with pinned endpoint,Endpoint created from any provider template inside the modal without config editing,Selecting a registry entry never hits the unsaved-endpoint block,F9 Settings rename/edit/delete with reference guard + detach,custom and custom_2 keep working; optional convert action,Unit + Pilot tests per spec testing section,ADR authored and linked before implementation
<!-- AC:END -->

## Implementation Plan

ADR: backlog/decisions/146-console-custom-endpoint-registry.md


## Implementation Notes

**Persistent-diagnostic pin (CI, Derived Artifacts guard).** The registry's
load boundary adds three WARNING diagnostics in
`tldw_chatbook/Chat/custom_endpoint_registry.py` (malformed section / malformed
entry / validation reasons, each naming only the slug). Reviewed per the
guard's procedure: the malformed-entry line deliberately logs only the
Pydantic field/type taxonomy -- never the exception object, whose messages
embed rejected input values (a malformed credential field would otherwise
write the secret into the persistent log). No user content, secrets, paths,
or URLs reach a persistent sink. Inventory pin updated via
`check_persistent_diagnostic_inventory.py --write` and committed with this
note.

