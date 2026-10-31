---
id: TASK-32240
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
ADR path: backlog/decisions/144-conversation-archive-and-exact-resume.md
Reason: implement shared lifecycle and recovery design.

Inspect Library paging and reader seams; add failing lifecycle/preview tests; implement scoped search, bounded reader and bulk receipts; verify keyboard/compact/wide behavior.

Plan: Docs/superpowers/plans/2026-09-10-console-archive-recovery.md
Spec: Docs/superpowers/specs/2026-09-10-console-archive-recovery-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented scoped Active/Archived/All search, workspace labels, a bounded read-only transcript with long-message paging and cross-window Find, captured-ID bulk archive/restore and versioned partial-success Undo. Resume retains original identity; Use as source is a separate action. Wide list/preview columns and compact scroll/focus support replace the distant preview and suppress a single-page pager. Final integrated evidence: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md. User guide updated at Docs/User_Guide/console/sessions-tabs-workspaces.md. ADR-144 linked. Compact/wide mounted flows and final cancellation-safe lifecycle run pass (29 checks); targeted FTS/recovery run passes (75); six close tests pass. Independent final review reports no remaining blockers. No full suite, live provider calls, commit or unrelated checkout cleanup.
<!-- SECTION:NOTES:END -->
