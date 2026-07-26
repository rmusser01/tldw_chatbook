---
id: TASK-720
title: Storage shows workspaces DB template path and resolved path as conflicting truths
status: Done
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - settings
  - storage
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Storage category's Workspaces input displays the default template path while the caption below shows the actually-resolved per-profile path - two different paths for a restart-gated, data-loss-adjacent setting (cap-27). Users cannot tell which one is live. Finding m4.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The input and its caption cannot be read as two different current locations
- [x] #2 The actually-active DB path is unambiguous in the Storage category
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Label the two layers explicitly and explain the difference inline; lock with a hub test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The inputs edit config.toml values while the caption list shows runtime-resolved files; with a [general].users_name profile these legitimately differ (per-profile directory), which read as two conflicting current locations. Storage now labels the sections "Database paths (configured)" and "Active files (resolved this session)" and adds a note (#settings-storage-configured-note) explaining the per-profile resolution. Assertions added to test_storage_category ordering test; 16 storage-filtered hub tests green.
<!-- SECTION:NOTES:END -->
