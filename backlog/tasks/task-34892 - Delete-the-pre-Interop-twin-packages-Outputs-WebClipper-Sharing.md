---
id: TASK-34892
title: Delete the pre-Interop twin packages (Outputs WebClipper Sharing)
status: To Do
created_date: 2026-10-12 17:52
labels:
- simplification
- dead-code
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Interop migration superseded the Outputs WebClipper and Sharing packages but left the originals behind with zero production importers kept green by their own tests (~1.5k prod plus ~1.2k test LOC).

Evidence (verified on dev df4985e): Verified: production imports only the *_Interop twins (app_service_wiring.py); exact-name greps find no importers of the core packages. Smoking gun: Outputs_Panel.py:75 reads server_outputs_scope_service which no code ever sets (wiring sets outputs_scope_service) so the seam has never been exercised. ADR-187 names the Sharing_Interop seam. Tests/Outputs Tests/WebClipper Tests/Sharing/test_server_sharing_service.py pin the twins only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Outputs WebClipper and Sharing packages and their test directories removed,App wiring continues to use the Interop twins unchanged,The never-set getattr in UI/Outputs_Panel.py:75 fixed or removed,Import graph grep-clean for the removed package names
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
