---
id: TASK-22031
title: Share Library adaptive reader shell and migrate Conversations
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-24 23:24'
updated_date: '2026-08-24 23:57'
labels:
  - library
  - ui
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-24-library-destinations-adaptive-reader-design.md
  - backlog/decisions/086-library-adaptive-reader-shell.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extract the shipped Media reader structure into the Library-local adaptive shell and migrate Conversations as its first additional consumer. Preserve Media domain behavior while adding the approved list comfort expansion and complete read-only conversation work pane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Media uses the shared Library-local shell with its existing modes actions selection loading recovery and preference compatibility preserved
- [ ] #2 Library and Conversations list are independently collapsible while the conversation work pane remains mounted
- [ ] #3 Collapsing Library expands the destination list toward its comfort cap without changing saved widths
- [ ] #4 Conversations exposes the complete saved transcript with Read and Info modes Find and Open in Console
- [ ] #5 Selected and loaded conversation identity stay truthful under rapid traversal stale workers deletion and retry
- [ ] #6 Shared Library preferences and Media legacy fallback follow ADR-086 without responsive preference writes
- [ ] #7 Automated geometry race capability and Media regression tests pass with a live TUI walkthrough at representative terminal sizes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory Media and Conversations capabilities
2. Extract and prove the shared shell
3. Migrate shared preferences and adaptive geometry
4. Add the fenced Conversations reader
5. Run automated and live verification

ADR required: yes
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md
Reason: implements the accepted Library structural boundary.
<!-- SECTION:PLAN:END -->
