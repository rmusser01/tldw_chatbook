---
id: TASK-31586
title: Repair MCP documentation inventory contract
status: Done
assignee: []
created_date: '2026-09-05 05:22'
updated_date: '2026-09-05 05:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile the MCP documentation inventory contract with the current private Library tool surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP inventory documentation and contract agree exactly
- [x] #2 MCP documentation contract module passes in full
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compare the runtime private Library tool inventory, documentation inventories, and test contract. 2. Correct the stale source of truth with the smallest consistent change. 3. Run the full MCP documentation contract module and lint/diff checks. ADR required: no. ADR path: N/A. Reason: this synchronizes documentation and tests with an existing tool surface.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the three standalone inventory documents and the local Library tool guide to
the current 21-tool descriptor surface after retirement of the generic Collections
list/get/search trio. The full MCP documentation contract passes with 75 tests. ADR
required: no; this documents an already-decided and implemented surface retirement.
<!-- SECTION:NOTES:END -->
