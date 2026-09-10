---
id: TASK-32275
title: Build searchable conversation archive review and bulk recovery
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
Saved chats can be searched, reviewed, archived and restored individually or in batches from Library.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Active Archived and All scopes provide exact paged results and preserve query through review.
- [x] #2 A bounded read-only transcript with match navigation supports review before resume.
- [x] #3 Single and bulk archive restore and undo act on captured identities and expose partial failures.
- [x] #4 Resume and Use as source are distinct actions with responsive keyboard-operable layout.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md
Reason: implement shared lifecycle and recovery design.

Inspect Library paging and reader seams; add failing lifecycle/preview tests; implement scoped search, bounded reader and bulk receipts; verify keyboard/compact/wide behavior.

Plan: Docs/superpowers/plans/2026-09-10-console-archive-recovery.md
Spec: Docs/superpowers/specs/2026-09-10-console-archive-recovery-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Active/Archived/All search scopes, workspace/archive labels, captured-ID single and bulk archive/restore, and version-checked Undo receipts. The current dev Library Reader remains the canonical read-only transcript/Find surface, with distinct Resume, Restore only and Use as source actions. Existing adaptive panes and single-page pager suppression are preserved.

ADR: backlog/decisions/147-conversation-archive-and-exact-resume.md. User guide: Docs/User_Guide/console/sessions-tabs-workspaces.md. Targeted verification and limitations: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md.

Integrated onto an isolated branch from current dev; original checkout changes are excluded. Task and ADR IDs were reassigned to avoid published collisions.
<!-- SECTION:NOTES:END -->
