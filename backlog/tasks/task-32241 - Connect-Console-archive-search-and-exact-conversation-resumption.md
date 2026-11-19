---
id: TASK-32241
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
ADR path: backlog/decisions/144-conversation-archive-and-exact-resume.md
Reason: implement shared lifecycle and recovery design.

Inspect pending handoff and Console activation; pin exact resume/draft/busy guard tests; add typed resume and archive/search entry points; run integrated verification and update guide.

Plan: Docs/superpowers/plans/2026-09-10-console-archive-recovery.md
Spec: Docs/superpowers/specs/2026-09-10-console-archive-recovery-design.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented typed original-conversation resume and Console archive/full-search routes. Resume preserves durable global/workspace ownership and active branch, reuses open tabs, drains superseding requests, and keeps drafts. Shared archive guards refuse live/queued/unsaved work; send/archive reservations survive canceled UI workers until actual DB settlement. Protected receipts offer Undo/View archived. Final integrated evidence: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md. User guide updated at Docs/User_Guide/console/sessions-tabs-workspaces.md. ADR-144 linked. Compact/wide mounted flows and final cancellation-safe lifecycle run pass (29 checks); targeted FTS/recovery run passes (75); six close tests pass. Independent final review reports no remaining blockers. No full suite, live provider calls, commit or unrelated checkout cleanup.

Follow-up verification requested: exercise the actual Restore and resume confirmation, workspace name collision, and a deterministic provider turn persisted into the original conversation; keep all fixtures isolated.

Follow-up complete: four mounted Console archive/recovery tests pass, including two new actual Restore-and-resume confirmation → next-send scenarios (standalone chat and archived workspace name collision). Deterministic provider output persists to the original conversation with original history included and unrelated draft intact. Ruff and format checks pass. No further application-code changes needed; QA report updated.
<!-- SECTION:NOTES:END -->
