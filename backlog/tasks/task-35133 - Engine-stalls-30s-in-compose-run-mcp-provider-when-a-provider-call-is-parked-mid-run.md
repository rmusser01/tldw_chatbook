---
id: TASK-35133
title: Engine stalls 30s+ in compose_run_mcp_provider when a provider call is parked
  mid-run
status: To Do
created_date: 2026-10-17 01:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified against dev base (b912d334 stash): parking a provider call mid-run stalls the engine's second agent step inside compose_run_mcp_provider -> UnifiedMCPControlPlaneService.local_external_catalog, 30s+ under the stall watchdog, status STREAMING. Surfaced during the 2026-10-16 burndown T2 scoping test; NOT a regression of that branch (stash-verified). Needs a dedicated investigation task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
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
