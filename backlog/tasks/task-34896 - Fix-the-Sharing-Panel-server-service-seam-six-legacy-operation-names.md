---
id: TASK-34896
title: Fix the Sharing_Panel server-service seam (six legacy operation names)
status: Done
created_date: 2026-10-13 02:03
labels:
- simplification
assignee:
- '@zcode'
updated_date: 2026-10-14 18:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-existing (surfaced by the deletion-wave review): Sharing_Panel duck-calls six operations by the deleted twin's names (create_share_token list_share_tokens revoke_share_token preview_public_share verify_public_share_password import_public_share) while Sharing_Interop's SharingScopeService exposes *_link / *_public_link names. Same class of bug as the Outputs_Panel mismatch fixed in task-34892 - in server mode every panel operation would raise AttributeError. Review the Interop API and repoint or alias.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Panel operation names match the SharingScopeService API or the service carries documented aliases,Seam test pins every panel operation name against the real Interop scope service (mirror Tests/Outputs/test_outputs_panel_seam.py),Sharing tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Map Sharing_Panel's six operation names vs SharingScopeService API 2. Repoint or alias 3. Seam test mirroring test_outputs_panel_seam 4. Tests green + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Repointed the six link/token operations to the SharingScopeService API (create_share_token->create_link list_share_tokens->list_links revoke_share_token->revoke_link preview_public_share->inspect_public_link verify_public_share_password->verify_public_link_password import_public_share->import_public_link); kwargs were already 1:1 with the Interop signatures (verified against server_sharing_service.py:96-150). The ten workspace-share operations were already Interop-named and correct. New Tests/Sharing/test_sharing_panel_seam.py (16 params, mirroring the Outputs seam test) pins EVERY panel operation name against the real SharingScopeService so name drift cannot silently regress the seam again. Verification: 16/16 seam tests + the Sharing directory green.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
