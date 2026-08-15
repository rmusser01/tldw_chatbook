---
id: TASK-16320
title: 'Trajectory import: open shared traces read-only'
status: To Do
assignee: []
created_date: '2026-08-15 13:53'
labels:
  - trajectory
  - import
dependencies:
  - TASK-16319
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Import a trajectory export file (task-16319 format) and render it in the existing TrajectoryScreen as a detached, read-only view. Imported traces must never write into the local conversations/messages/sidecar tables -- they exist only as an ephemeral snapshot for viewing, preserving sync integrity. Malformed or wrong-version files are rejected with actionable errors. ADR: extend the task-16319 export-format ADR rather than creating a new one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Imported file renders in the TrajectoryScreen (ledger, inspector, timeline) without any DB writes,Malformed files fail with actionable error messages,Version mismatches are detected and reported,Import action accessible from the Console trajectory surface,Tests cover happy path, malformed input, and version mismatch
<!-- AC:END -->
