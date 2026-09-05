---
id: TASK-31710
title: Wait for explicit Console journey phases before asserting transitions
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 18:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep atomic settings-transfer and prompt identity-drift journeys deterministic when lifecycle callbacks or mode mounting finish at different speeds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Atomic transfer failures retain one source owner until the test explicitly allows retry
- [ ] #2 Recipe identity drift checks start after the current modal result control is attached
- [ ] #3 Full native Console flow tests and static checks pass without production timing changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only phase synchronization, preserving current lifecycle and ownership contracts.
1. Inspect full-file failures and reproduce focused variants.
2. Keep injected settlement failure armed until explicit retry and await current prompt controls instead of fixed pauses.
3. Run focused races, full native flow file, and static checks; retain all ownership and mutation assertions.
<!-- SECTION:PLAN:END -->
