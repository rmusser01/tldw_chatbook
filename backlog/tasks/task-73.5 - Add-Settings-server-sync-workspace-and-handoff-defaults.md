---
id: TASK-73.5
title: Define Settings server sync workspace and handoff source contracts
status: To Do
labels:
- settings
- server
- sync
- workspaces
- handoff
dependencies:
- TASK-73.1
priority: high
parent_task_id: TASK-73
documentation:
- Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the source-of-truth contracts Settings may read for server, sync, workspace, and handoff status/defaults, then expose only sourced status/default rows while keeping execution, conflict recovery, workspace switching, and ACP task/run ownership in their dedicated surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The task identifies concrete source services/files for server profile, sync safety, workspace context, handoff policy, and ACP handoff readiness before UI status rows are added.
- [ ] #2 Settings only renders status/defaults from identified source contracts, or explicit WIP/read-only owner copy when a source contract is missing.
- [ ] #3 Settings shows active server profile/defaults and local/server authority without rewriting runtime ownership.
- [ ] #4 Settings shows sync safety, dry-run/blocked state, and recovery copy consistent with Home, Console, and Library.
- [ ] #5 Workspace defaults and handoff policy are visible without hiding cross-workspace Library visibility rules.
- [ ] #6 ACP task/run handoff defaults are represented as defaults/status only; ACP remains the runtime/session owner.
- [ ] #7 Cross-screen tests verify Settings, Home, Console, and Library status language agrees where source contracts exist.
- [ ] #8 Actual CDP/Textual-web screenshot QA verifies the server/sync/workspace/handoff Settings surface and is approved before PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Unit, mounted, and integration tests relevant to changed behavior pass
- [ ] #3 Static analysis and diff hygiene checks pass
- [ ] #4 Actual app QA walkthrough completed with screenshots
- [ ] #5 User approval recorded for visible Settings changes
- [ ] #6 Documentation and task notes updated
- [ ] #7 Task status moved to Done after implementation notes are added
<!-- DOD:END -->
