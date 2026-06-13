---
id: TASK-10.4
title: 'Product Maturity Phase 3.4: Source-Selected Study Generation'
status: Done
assignee: []
created_date: '2026-05-07 00:00'
updated_date: '2026-05-07 00:00'
labels:
  - product-maturity
  - phase-3-knowledge-study
  - study-generation
dependencies: []
parent_task_id: TASK-10
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow a user who enters Study from visible Library sources to launch a server-backed study-pack generation job from those concrete source items, while keeping local-mode recovery honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library-originated Study context carries concrete note and media source items for generation while preserving visible source titles.
- [x] #2 Study Dashboard in server mode can queue a study-pack generation job from selected Library source items.
- [x] #3 Local mode explains the server-mode requirement and does not dispatch generation.
- [x] #4 Repo-tracked QA evidence, roadmap, and parent task notes record the Phase 3.4 verification and residual risk.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing Phase 3.4 product-maturity regression for Library source item handoff, server generation dispatch, local-mode recovery, and tracking evidence.
2. Extend Study scope models so concrete source items can survive Library to Study navigation and saved state.
3. Extract eligible note and media source items from the Library snapshot without treating conversation records as message IDs.
4. Add a Study Dashboard generation action that queues server study-pack generation from the selected source items and reports unavailable local/runtime states.
5. Record Phase 3.4 QA evidence and update the tracker, README, and parent task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Phase 3.4 source-selected Study generation gate. Library now carries concrete note and media source items into Study along with the existing visible material titles. Study Dashboard exposes a source study-pack generation action that queues `create_study_pack_job` in server mode using those source items, reports queued job status, and keeps local mode disabled with a clear server requirement. Conversation records remain visible in the material context but are not sent as study-pack source items until message-level selection exists.
<!-- SECTION:NOTES:END -->
