---
id: TASK-73.5
title: Add Settings server sync workspace and handoff defaults
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-31 22:41
labels:
- settings
- server
- sync
- workspaces
- handoff
dependencies:
- TASK-73.1
documentation:
- Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
parent_task_id: TASK-73
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the source-of-truth contracts Settings may read for server, sync, workspace, and handoff status/defaults, then expose only sourced status/default rows while keeping execution, conflict recovery, workspace switching, and ACP task/run ownership in their dedicated surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The task identifies concrete source services/files for server profile, sync safety, workspace context, handoff policy, and ACP handoff readiness before UI status rows are added.
- [x] #2 Settings only renders status/defaults from identified source contracts, or explicit WIP/read-only owner copy when a source contract is missing.
- [x] #3 Settings shows active server profile/defaults and local/server authority without rewriting runtime ownership.
- [x] #4 Settings shows sync safety, dry-run/blocked state, and recovery copy consistent with Home, Console, and Library.
- [x] #5 Workspace defaults and handoff policy are visible without hiding cross-workspace Library visibility rules.
- [x] #6 ACP task/run handoff defaults are represented as defaults/status only; ACP remains the runtime/session owner.
- [x] #7 Cross-screen tests verify Settings, Home, Console, and Library status language agrees where source contracts exist.
- [x] #8 Actual CDP/Textual-web screenshot QA verifies the server/sync/workspace/handoff Settings surface and is approved before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory current source-of-truth contracts for server profile, sync safety, workspace context, handoff policy, and ACP readiness.
2. Add failing focused Settings/cross-surface tests for sourced or explicit WIP/read-only status rows.
3. Implement the smallest Settings adapter/UI changes to expose status/default rows without moving runtime ownership into Settings.
4. Verify owner-boundary copy against Home, Console, Library, and ACP expectations.
5. Run focused pytest suites, diff hygiene, and capture Textual-web/CDP QA evidence for approval.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Settings Overview server/sync/workspace/handoff status block from explicit source contracts. The UI reads runtime policy state, sync promotion states, workspace registry/context, Library visibility copy, and ACP runtime session readiness while keeping runtime control, sync execution, workspace switching, and ACP session management in their owning destinations. Added focused Settings and cross-surface regressions, updated the Settings plan with the source-contract matrix, and recorded approved textual-web/CDP screenshot evidence.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Settings now exposes server profile/authority, sync safety/recovery, workspace default/Library visibility, handoff policy, and ACP handoff readiness as read-only status/default rows. Verification: `python -m pytest -q Tests/UI/test_settings_configuration_hub.py Tests/UI/test_home_screen.py Tests/UI/test_console_persistent_rails.py Tests/UI/test_destination_shells.py Tests/UI/test_destination_visual_parity_correction.py::test_workflows_screen_matches_approved_procedure_columns --tb=short` reported 290 passed with one existing dependency warning; `git diff --check` passed; approved screenshot is `Docs/superpowers/qa/product-maturity/screen-qa/settings/settings-server-sync-defaults-2026-05-31-large.png`.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Unit, mounted, and integration tests relevant to changed behavior pass
- [x] #3 Static analysis and diff hygiene checks pass
- [x] #4 Actual app QA walkthrough completed with screenshots
- [x] #5 User approval recorded for visible Settings changes
- [x] #6 Documentation and task notes updated
- [x] #7 Task status moved to Done after implementation notes are added
<!-- DOD:END -->
