---
id: TASK-63
title: Implement workspace domain foundation
status: Done
labels:
- workspaces
- foundation
- pr-a
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR A from the workspace operating context plan: local workspace models, eligibility rules, registry persistence, and app-owned service wiring needed before Console workspace UI work can begin.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace domain models validate required identity, authority, membership, runtime binding, and eligibility fields.
- [x] #2 Eligibility rules preserve global browse/search visibility while blocking active Console context operations for cross-workspace items with recovery copy.
- [x] #3 Local workspace registry persists workspace records, active workspace, memberships, and runtime bindings without storing secrets.
- [x] #4 The app exposes an app-owned workspace_registry_service without changing existing pending Console or Notes workspace context behavior.
- [x] #5 Focused workspace foundation tests and relevant screen navigation smoke tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Review existing DB/service/app setup seams for workspace-friendly integration points.
2. Write failing tests for workspace models and eligibility rules, then implement the smallest domain package to pass.
3. Write failing tests for local workspace registry persistence, memberships, runtime binding redaction, and app-owned service wiring.
4. Implement `WorkspaceDB` and `LocalWorkspaceRegistryService` using existing `BaseDB` and app service patterns.
5. Run focused tests, update task acceptance criteria/notes, and commit PR A changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added the `tldw_chatbook.Workspaces` foundation package with validated workspace records, memberships, runtime bindings, secret metadata scrubbing, and pure active-context eligibility rules.
- Added `WorkspaceDB` plus `LocalWorkspaceRegistryService` for local workspace records, active workspace persistence, item memberships, and runtime binding persistence.
- Wired `workspace_registry_service` into `TldwCli` through a new configurable `workspaces_db_path` while preserving existing pending Console and Notes workspace state.
- Verification: `python -m pytest -q Tests/Workspaces/test_workspace_models.py Tests/Workspaces/test_workspace_eligibility.py Tests/Workspaces/test_workspace_registry_service.py Tests/UI/test_screen_navigation.py --tb=short`
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR A foundation is implemented: workspace domain models, active-context eligibility, local registry persistence, and app-owned service wiring are in place for later Console/Library UI slices.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
