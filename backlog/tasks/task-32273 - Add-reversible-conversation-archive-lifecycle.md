---
id: TASK-32273
title: Add reversible conversation archive lifecycle
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 15:33'
updated_date: '2026-09-10 16:36'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Saved conversations need reversible archive state without losing history or identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Individual and bulk archive preserve conversation identity and messages with reversible version-checked changes.
- [x] #2 Active Archived and All queries are correctly filtered before pagination and counts.
- [x] #3 Targeted persistence and migration tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md
Reason: durable lifecycle and query contract.

1. Read spec and existing migration/service patterns.
2. Write failing real SQLite lifecycle/pagination/migration tests.
3. Implement archive persistence, version-checked bulk mutation, archive scope and metadata projection.
4. Run targeted tests and lint; document evidence.

Plan: Docs/superpowers/plans/2026-09-10-console-archive-recovery.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added indexed local archive state through schema migration 70→71, optimistic version-checked batch archive/restore, active/archived/all query filtering before counts and pagination, and bounded archive-state reads. Exact reads and exports retain archived history. Shared async mutations keep reservations through SQLite settlement and reject conflicting sends, drafts and live work.

ADR: backlog/decisions/147-conversation-archive-and-exact-resume.md. User guide: Docs/User_Guide/console/sessions-tabs-workspaces.md. Targeted verification and limitations: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md.

Integrated onto an isolated branch from current dev; original checkout changes are excluded. Task and ADR IDs were reassigned to avoid published collisions.
<!-- SECTION:NOTES:END -->
