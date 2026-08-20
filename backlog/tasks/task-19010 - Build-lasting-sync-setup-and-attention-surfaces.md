---
id: TASK-19010
title: Build lasting sync setup and attention surfaces
status: To Do
assignee: []
created_date: '2026-08-20 07:51'
labels:
  - notes
  - sync
  - ux
  - accessibility
dependencies:
  - TASK-19003
  - TASK-19004
  - TASK-19005
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add feature-gated Textual flows for relationship choice, root setup, dry-run review, activation receipts, root management, manual reconciliation, and attention resolution against typed services.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `Add from files…` first distinguishes `Import once` from `Keep a folder synced` before any picker or scan.
- [ ] #2 Lasting-root setup captures display name, folder, local destination, direction, and capability state; server-backed setup remains visibly disabled pending its external ADR and versioned capability.
- [ ] #3 Check and manual Sync now are mutation-free reviewed flows with safe actions, attention, skips, managed placements, stale-review detection, progress, and durable receipts.
- [ ] #4 Root management exposes status, pause/resume, review attention, retarget, disconnect, and contextual actions without a global conflict winner or auto-sync toggle.
- [ ] #5 Conflict, deletion, move, partial, offline, passive, failed, and recovery states use explicit effects and bounded next actions; disconnect never deletes either authority.
- [ ] #6 The UI talks only to typed runtime projections/messages, stays feature-gated and inert, preserves legacy Sync, and is keyboard/readability tested at 60x20.
<!-- AC:END -->

## Decision Record Check

ADR required: no new ADR
ADR paths: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`, `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`, `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`
Reason: this task renders the accepted local runtime contract against typed fakes and does not activate a new owner or define the external server contract.
