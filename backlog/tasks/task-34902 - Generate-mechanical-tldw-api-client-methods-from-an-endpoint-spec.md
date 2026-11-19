---
id: TASK-34902
title: Generate mechanical tldw_api client methods from an endpoint spec
status: To Do
created_date: 2026-10-12 17:53
labels:
- simplification
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_api/client.py is 16.8k LOC with 1203 methods of which 809 are pure _request plus model_validate boilerplate. Combined with the Interop layer every endpoint is written four times by hand; a declarative endpoint table (verb path template request and response schema) makes the mechanical 809 generated and drift-diffable.

Evidence (verified on dev df4985e): Measured on dev df4985e: 809 of 1203 async defs are exactly response = await self._request(...) plus Schema.model_validate(response) (example block client.py:3233-3331). Schemas already live in separate *_schemas.py files. This is a client for the external tldw server not an in-app router. Coordinate with the interop-core task: one endpoint map should eventually feed both ends.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Declarative endpoint spec table added (method verb path template request and response schema),Mechanical client methods generated from the table,The 394 long-tail methods with real logic stay hand-written,Public import surface (tldw_api __init__ re-exports) unchanged,Generated output remains diffable against external server API drift
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
