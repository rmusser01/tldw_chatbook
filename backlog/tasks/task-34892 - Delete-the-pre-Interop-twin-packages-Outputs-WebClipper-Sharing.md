---
id: TASK-34892
title: Delete the pre-Interop twin packages (Outputs WebClipper Sharing)
status: Done
created_date: 2026-10-12 17:52
labels:
- simplification
- dead-code
priority: high
assignee:
- '@zcode'
updated_date: 2026-10-13 02:04
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Interop migration superseded the Outputs WebClipper and Sharing packages but left the originals behind with zero production importers kept green by their own tests (~1.5k prod plus ~1.2k test LOC).

Evidence (verified on dev df4985e): Verified: production imports only the *_Interop twins (app_service_wiring.py); exact-name greps find no importers of the core packages. Smoking gun: Outputs_Panel.py:75 reads server_outputs_scope_service which no code ever sets (wiring sets outputs_scope_service) so the seam has never been exercised. ADR-187 names the Sharing_Interop seam. Tests/Outputs Tests/WebClipper Tests/Sharing/test_server_sharing_service.py pin the twins only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Outputs WebClipper and Sharing packages and their test directories removed,App wiring continues to use the Interop twins unchanged,The never-set getattr in UI/Outputs_Panel.py:75 fixed or removed,Import graph grep-clean for the removed package names
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify exact-name importers of the three packages at HEAD 2. Delete Outputs WebClipper Sharing packages and their pinning tests 3. Fix the Outputs_Panel never-set getattr to the real Interop attribute after checking interface compatibility 4. Targeted test run diffed vs baseline 5. Notes + AC tick
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Deleted the pre-Interop originals Outputs/ WebClipper/ Sharing/ (~1.45k LOC). app_service_wiring constructs same-named classes but imports them from the Interop packages - only the class NAMES collide; the core packages had zero importers outside their own tests. The three twin-parity test files were trimmed to Interop-only. Outputs_Panel now reads outputs_scope_service (the attribute wiring sets) with the legacy name as fallback. CORRECTION (deletion-wave review, 2026-10-12): the first pass only fixed the attribute NAME - the panel still called the deleted twin's method names, so every server-mode operation would have raised AttributeError; the review round repointed all 12 operation names to the Interop API (list_templates create_template get/update/delete/preview_template list/list_deleted/get/create/update/delete_artifact) and added Tests/Outputs/test_outputs_panel_seam.py pinning each name against the real OutputsScopeService. Note: OutputsPanel itself has zero production importers (follow-up task filed).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
