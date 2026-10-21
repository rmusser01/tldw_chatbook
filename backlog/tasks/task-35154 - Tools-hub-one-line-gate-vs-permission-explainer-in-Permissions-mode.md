---
id: TASK-35154
title: 'Tools hub: one-line gate-vs-permission explainer in Permissions mode'
status: Done
created_date: 2026-10-16 20:15
updated_date: 2026-10-21 05:57
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec amendment A7 item 2: gate off means the tool does not exist for agents; Allow/Ask/Off governs only tools that pass the gate. Users flip a gate, see no change in the matrix, and conclude it is broken. Add one sentence at the top of Permissions mode in the Tools & MCP hub.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Permissions mode shows the one-line explainer,Copy passes the TASK-35018/35019 plain-language rules,Test asserts the line renders
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate Permissions mode header rendering in mcp_permissions_mode.py (compose order) 2. TDD: add test asserting the explainer sentence renders verbatim 3. Implement one plain-language sentence reusing existing hint classes/id 4. Carryover A: 35153 helper absent from base (verified by grep) - leave local_tools label as-is and note it 5. Carryover B: migrate two servers-mode tests to the saved-dict loader seam 6. Run permissions-mode, servers-mode, settings-tools-hub, token-governance suites 7. Commit
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented: one-line gate-vs-permission explainer pinned in mcp_permissions_mode.py (first child, hint-tier styling, zero new tokens), test asserts exact copy; carryover A left as-is (35153 not in base); carryover B migrated two tests to the loader seam with RED dead-patch proof. Suites green or stash-verified baseline-identical.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Permissions mode opens with the one-line gate-vs-permission explainer (spec amendment A7 item 2), pinned verbatim and test-asserted; both review-carryover seam items from 35153 dispositioned (A left as-is because 35153 is not in this base; B migrated to the live loader seam with a RED dead-patch proof). All named suites green or baseline-identical.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
