---
id: TASK-16311
title: Standardize Library pager display and harden Conversation paging
status: In Progress
assignee: []
created_date: '2026-08-15 02:44'
updated_date: '2026-08-15 02:59'
labels:
  - library
  - pagination
  - conversations
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md
  - backlog/decisions/067-library-top-level-pagination-contracts.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every top-level Conversation reachable through a consistent 20-item Library pager while establishing the small pure display convention reused by later source tasks. Preserve full-source search, deterministic deep links, selection safety, and truthful recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Conversation pages contain at most 20 records and expose exact range, total, Previous, Next, loading, disabled-reason, and retry presentation.
- [ ] #2 Full-source Conversation search runs before paging, and deterministic stable ordering makes every matching record reachable.
- [ ] #3 Off-page Conversation navigation opens the target's coherent rank-derived owning page without injecting an extra page-1 row.
- [ ] #4 Conversation count and page rows come from one coherent read transaction and malformed page or locator envelopes fail closed inside the canvas.
- [ ] #5 Conversation selection clears with visible notice on page or scope change, while focus and detail/back behavior follow the approved design.
- [ ] #6 The shared code is limited to one pure pager-display calculation; Conversation retains request, state, worker, widget, and event ownership.
- [ ] #7 Automated state, service, mounted Textual, geometry, race, privacy, mutation, and isolated live verification required by the approved design pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/067-library-top-level-pagination-contracts.md
Reason: TASK-16311 changes the Conversation page/locator service contract and establishes the pure shared display contract.

Detailed plan: Docs/superpowers/plans/2026-08-14-task-16311-library-conversation-pagination.md

1. Add the pure immutable pager-display calculation with exhaustive state tests.
2. Make Conversation count/rows coherent and add a bounded stable-ID owning-page locator.
3. Validate Conversation summaries and integrate the pure display without a generic controller/widget.
4. Harden requested/applied scope, retry, focus, selection, restore, races, unmount, clamping, and deep-link lifecycle.
5. Render and geometry-test the Conversation-specific pager.
6. Run inverse mutations, owner/full gates, isolated live verification, docs, reviews, and task closeout.
<!-- SECTION:PLAN:END -->
