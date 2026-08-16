---
id: TASK-16813
title: 'Trajectory export: versioned JSON trace format and writer'
status: Done
assignee: []
created_date: '2026-08-15 13:53'
updated_date: '2026-08-15 17:35'
labels:
  - trajectory
  - export
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users export a conversation's trajectory (trace) to a shareable, self-contained file, building on the Console trajectory view (task-16311..16315, ADR-066). Purpose: sharing/debugging agent runs outside the local DB. Export folds the same inputs as the projection (messages incl. usage_json, sidecar rows, variant sets where available, compaction records) into one versioned JSON document. Privacy: tool payloads may contain file contents -- export defaults to redacted payload previews, full payloads only behind an explicit opt-in flag. ADR required: yes (export format is a data contract) -- create before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Export produces a schema-versioned single-file JSON trace of one conversation,Tool payloads redacted by default; full payloads only with explicit opt-in flag,Import validator round-trips the exported file (task-2 seam),Export of a conversation lacking sidecar rows still succeeds (legacy fallback),Unit tests cover format, redaction, and edge cases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. ADR-067 export format contract (versioned JSON, redaction default). 2. Chat/trajectory_export.py: build/write/validate; validator is the import seam. 3. Round-trip tests (export->validate->derive_trajectory renders), redaction, legacy no-sidecar, malformed/version rejection, atomic write. 4. Task notes + Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Export module shipped: build/write/validate per ADR-067; review Approved (minors: redacted-flag cross-check deferred to task-16320 import validation)
<!-- SECTION:NOTES:END -->
