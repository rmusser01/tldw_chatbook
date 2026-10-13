---
id: TASK-34896
title: Fix the Sharing_Panel server-service seam (six legacy operation names)
status: To Do
created_date: 2026-10-13 02:03
labels:
- simplification
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing (surfaced by the deletion-wave review): Sharing_Panel duck-calls six operations by the deleted twin's names (create_share_token list_share_tokens revoke_share_token preview_public_share verify_public_share_password import_public_share) while Sharing_Interop's SharingScopeService exposes *_link / *_public_link names. Same class of bug as the Outputs_Panel mismatch fixed in task-34892 - in server mode every panel operation would raise AttributeError. Review the Interop API and repoint or alias.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Panel operation names match the SharingScopeService API or the service carries documented aliases,Seam test pins every panel operation name against the real Interop scope service (mirror Tests/Outputs/test_outputs_panel_seam.py),Sharing tests pass
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
