---
id: TASK-30019
title: Decide legacy Collections migration or retirement
status: To Do
assignee: []
created_date: '2026-09-03 03:04'
updated_date: '2026-09-03 03:05'
labels:
  - library
  - collections
  - legacy
  - adr
  - decision
dependencies:
  - TASK-18919
references:
  - TASK-18919
  - backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md
  - Docs/superpowers/specs/2026-09-01-collections-followup-backlog-design.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Inventory the v1 schema, code reachability, compatibility promises, and downgrade paths using only repository evidence and synthetic fixtures. No real-user database, telemetry, row count, path, identifier, name, item text, URL, or membership is inspected or collected. Compare at least: retained export-only recovery, explicit user-approved migration into captures, and retirement after a defined release/notice window. The task produces a new accepted ADR defining authority mapping, canonical-URL collisions, membership handling, consent, backup/export requirements, rollback, retention, privacy boundaries, and removal gates. No legacy data is mutated or deleted in this decision task. Atomic implementation tasks are created only after the ADR selects an outcome.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Repository code/schema/recovery reachability, compatibility promises, and downgrade behavior are inventoried with synthetic fixtures only; no real-user data or telemetry is inspected.
- [ ] #2 Retained export-only recovery, explicit user-approved capture migration, and time-bounded retirement are compared with evidence and stated trade-offs.
- [ ] #3 A new accepted ADR selects the lifecycle and defines authority mapping, canonical collisions, memberships, consent, backup/export, rollback, retention, privacy, and removal gates.
- [ ] #4 The decision task mutates or deletes no legacy data and changes no production lifecycle.
- [ ] #5 Any approved implementation is decomposed into atomic Backlog tasks only after the ADR selects an outcome; those tasks do not reference uncreated future IDs.
- [ ] #6 Decision evidence, ADR linkage, documentation checks, and task closeout satisfy repository DoD.
<!-- AC:END -->
