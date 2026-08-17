---
id: TASK-16320
title: 'Trajectory import: open shared traces read-only'
status: Done
assignee: []
created_date: '2026-08-15 13:53'
updated_date: '2026-08-15 17:48'
labels:
  - trajectory
  - import
dependencies:
  - TASK-16813
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Import a trajectory export file (task-16813 format) and render it in the existing TrajectoryScreen as a detached, read-only view. Imported traces must never write into the local conversations/messages/sidecar tables -- they exist only as an ephemeral snapshot for viewing, preserving sync integrity. Malformed or wrong-version files are rejected with actionable errors. ADR: extend the task-16813 export-format ADR rather than creating a new one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Imported file renders in the TrajectoryScreen (ledger, inspector, timeline) without any DB writes,Malformed files fail with actionable error messages,Version mismatches are detected and reported,Import action accessible from the Console trajectory surface,Tests cover happy path, malformed input, and version mismatch
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Chat/trajectory_import.py: load file -> validate_trajectory_export (ADR-067 seam) -> map export sections to derive_trajectory inputs -> TrajectorySnapshot; no DB writes anywhere. 2. UI: import action on the trajectory surface (single-letter ADR-031 binding + file open), renders read-only snapshot; errors surfaced as notifications. 3. Tests: mapping unit tests, malformed/version rejection, no-write assertion (DB row counts unchanged), UI pilot test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Import seam: Chat/trajectory_import.py (pure, no DB imports) reuses validate_trajectory_export, maps sections to derive_trajectory inputs. UI: 'o' binding on TrajectoryScreen opens file -> pushes read-only TrajectoryScreen (no conversation_id/revision providers -> no polling). Inspector shows redaction marker for redacted payloads. Tests: Tests/Chat/test_trajectory_import.py (mapping incl compaction/variants/usage/redaction, malformed+version rejection, structural no-DB proof) and Tests/UI/test_trajectory_import_ui.py (pilot 'o' import, no-live-polling, error notifications, behavioral no-write via unchanged DB row counts).
<!-- SECTION:NOTES:END -->
