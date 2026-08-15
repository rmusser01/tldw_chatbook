---
id: TASK-16311
title: Schema v38 trajectory metadata sidecar
status: To Do
assignee: []
created_date: '2026-08-15 00:10'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Local-only message_trajectory_metadata table (turn_id, seq, timing, tool payload_json) with migration, accessors, and tests. Per ADR-066 and Docs/superpowers/specs/2026-08-14-console-trajectory-view-design.md; plan task 1 in Docs/superpowers/plans/2026-08-14-console-trajectory-view.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Migration v37->v38 runs idempotently,Accessors roundtrip rows,Per-migration tests pass
<!-- AC:END -->
