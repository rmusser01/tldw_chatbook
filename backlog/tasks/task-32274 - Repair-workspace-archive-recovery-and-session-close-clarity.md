---
id: TASK-32274
title: Repair workspace archive recovery and session close clarity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 15:35'
updated_date: '2026-09-10 16:36'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Workspace archive is discoverable and reversible in Console and Settings, including name collisions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Archived rows visibly identify their state and can be restored from Console.
- [x] #2 Archive provides Undo and View archived; restoration provides explicit feedback.
- [x] #3 Name conflicts can be resolved while restoring and close copy distinguishes saved history from actual losses.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md
Reason: implement shared lifecycle and recovery design.

Inspect lifecycle and close flows; pin failing collision/status/feedback tests; implement reversible workspace controls and accurate close impact; run targeted mounted checks.

Plan: Docs/superpowers/plans/2026-09-10-console-archive-recovery.md
Spec: Docs/superpowers/specs/2026-09-10-console-archive-recovery-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added atomic workspace Restore as, visible archived rows, Show archived, and Undo/View archived receipts in Console and Settings. Restore does not activate the workspace. Close copy distinguishes retained saved history from unsaved messages, drafts, attachments and live work; current dev tree-wide loss accounting is preserved.

ADR: backlog/decisions/147-conversation-archive-and-exact-resume.md. User guide: Docs/User_Guide/console/sessions-tabs-workspaces.md. Targeted verification and limitations: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md.

Integrated onto an isolated branch from current dev; original checkout changes are excluded. Task and ADR IDs were reassigned to avoid published collisions.
<!-- SECTION:NOTES:END -->
