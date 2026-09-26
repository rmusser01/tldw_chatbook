---
id: TASK-32722
title: Add tuning-only llama.cpp launch profiles
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 16:11'
updated_date: '2026-09-17 16:57'
labels:
  - llamacpp
  - lab
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users save and reuse common llama.cpp tuning without copying raw command lines or persisting local source paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Strict versioned named profiles persist tuning only and reject stale or corrupt writes.
- [x] #2 Structured controls cover context, GPU layers, threads, parallel slots, flash attention, KV cache types and batch sizes with runtime-default reset.
- [x] #3 Managed flags conflict with expert flags explicitly; profiles do not change Console sampling or saved provider endpoints.
- [x] #4 Mounted keyboard flows and targeted persistence and command tests cover saved profiles and malformed input.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/165-llamacpp-tuning-profiles-and-startup-diagnostics.md
Reason: implements ADR-114 with explicit tuning persistence and bounded diagnostics under ADR-165.

Follow the corresponding slice of Docs/superpowers/plans/2026-09-17-llamacpp-management-milestone.md and its linked design. Write failing targeted behavioral tests, implement the minimal owned interfaces, integrate the mounted flow, and verify the affected lifecycle, source, snapshot and settings boundaries. Record review and qualification limits before closeout.

## Renumbering provenance

The Backlog CLI assigned 32713 below the live all-ref/all-worktree maximum 32720. Renumbered immediately before implementation to 32722 to avoid existing branch reservations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented strict immutable tuning values and bounded version-1 JSON profiles with private atomic CAS writes and interprocess locking. Mounted profile Save/Delete/Reload/default-reset and common tuning fields preserve session drafts and reject expert ownership conflicts. No source path, endpoint, raw arguments or sampling fields are persisted in profiles. Fixed short-alias admission and initial hydration/selection regressions after independent review. ADR: backlog/decisions/165-llamacpp-tuning-profiles-and-startup-diagnostics.md. Files: llamacpp_profiles.py, llamacpp_setup_view.py, launcher integration and focused tests. Evidence: Docs/superpowers/reviews/2026-09-17-llamacpp-milestone-verification.md. User guide: Docs/User_Guide/lab.md; Textual incident recorded in backlog/docs/lessons-textual.md.
<!-- SECTION:NOTES:END -->
