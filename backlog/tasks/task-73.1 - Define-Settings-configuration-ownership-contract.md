---
id: TASK-73.1
title: Define Settings configuration ownership contract
status: Done
labels:
- settings
- configuration
- planning
- ux
dependencies: []
priority: high
parent_task_id: TASK-73
documentation:
- Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define and expose the Settings ownership contract so users and implementers can tell what Settings owns, what it only observes, what is read-only or WIP, and which destination owns runtime execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings documents global defaults, persisted config, runtime status, and runtime-owner boundaries for each category.
- [x] #2 Overview and inspector copy make MCP, ACP, Console, sync, and workspace ownership explicit without duplicating long prose.
- [x] #3 Read-only and WIP categories have visible owner/recovery copy and do not look fully actionable.
- [x] #4 Mounted regressions cover ownership copy, category presence, and MCP/ACP boundary language.
- [x] #5 Actual CDP/Textual-web screenshot QA verifies the ownership/overview surface and is approved before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing mounted/model regressions for the Settings ownership contract, including category presence and MCP/ACP/runtime boundary language.
2. Add a focused SettingsOwnershipRecord model and ownership matrix helpers without changing runtime ownership.
3. Render ownership summaries in Overview/detail/inspector so read-only and WIP categories expose owner/recovery copy.
4. Update the Settings configuration hub design spec with the ownership matrix.
5. Run focused Settings tests and diff hygiene.
6. Capture actual Textual-web/CDP screenshot evidence and wait for user approval before PR closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a `SettingsOwnershipRecord` contract and per-category ownership matrix for persisted config sections, observed runtime state, runtime owners, read-only/WIP state, and recovery copy.
- Updated Settings overview/detail/inspector rendering so ownership boundaries are visible without claiming Console, MCP, ACP, sync, or workspace runtime ownership.
- Updated the configuration hub design spec with the ownership matrix and added mounted regressions for category coverage, boundary copy, and provider inspector scope.
- Captured approved Textual-web/CDP QA evidence at `Docs/superpowers/qa/settings-configuration-hub/stage-1-ownership-overview-cdp-1200x1240.png`.
- Addressed PR review follow-up by aligning ownership metadata with actual provider and Console save paths, centralizing Overview ownership rows, caching the ownership matrix, and adding a safe missing-record fallback.
- Verification: `python -m pytest -q Tests/UI/test_settings_configuration_hub.py --tb=short` passed with 86 tests; Backlog ID harness passed; `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Settings now has an explicit ownership contract that distinguishes persisted defaults from runtime execution surfaces. The visible overview and inspector copy make Console, MCP, ACP, sync, and workspace boundaries clear, and the spec/task evidence records the approved CDP screenshot.
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
