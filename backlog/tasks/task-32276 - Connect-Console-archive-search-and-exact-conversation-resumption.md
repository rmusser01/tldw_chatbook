---
id: TASK-32276
title: Connect Console archive search and exact conversation resumption
status: Done
assignee:
  - '@codex'
created_date: '2026-09-10 15:35'
updated_date: '2026-09-10 16:44'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console and Library complete the same original-conversation recovery journey.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library Resume reopens the original conversation and active branch, preserving unrelated drafts and reusing open sessions.
- [x] #2 Console exposes archive and full archived conversation search and filters archived chats from ordinary history.
- [x] #3 Active or queued work is guarded and archive never silently cancels it.
- [x] #4 Documentation and targeted integrated lifecycle tests cover the complete workflow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md
Reason: implement shared lifecycle and recovery design.

Inspect pending handoff and Console activation; pin exact resume/draft/busy guard tests; add typed resume and archive/search entry points; run integrated verification and update guide.

Plan: Docs/superpowers/plans/2026-09-10-console-archive-recovery.md
Spec: Docs/superpowers/specs/2026-09-10-console-archive-recovery-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Connected Console archive/search controls to Library and added a typed original-ID resume handoff. Recovery reuses open sessions or hydrates the persisted conversation and branch, retains workspace/global ownership and unrelated drafts, and guards archive/send admission. Current dev asynchronous hydration and switcher modes remain intact.

ADR: backlog/decisions/147-conversation-archive-and-exact-resume.md. User guide: Docs/User_Guide/console/sessions-tabs-workspaces.md. Targeted verification and limitations: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md.

Integrated onto an isolated branch from current dev; original checkout changes are excluded. Task and ADR IDs were reassigned to avoid published collisions.
<!-- SECTION:NOTES:END -->
