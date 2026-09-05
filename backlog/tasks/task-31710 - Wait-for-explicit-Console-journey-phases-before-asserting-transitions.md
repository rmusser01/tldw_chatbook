---
id: TASK-31710
title: Wait for explicit Console journey phases before asserting transitions
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:51'
updated_date: '2026-09-05 19:35'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep atomic settings-transfer and prompt identity-drift journeys deterministic when lifecycle callbacks or mode mounting finish at different speeds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Atomic transfer failures retain one source owner until the test explicitly allows retry
- [x] #2 Recipe identity drift checks start after the current modal result control is attached
- [x] #3 Full native Console flow tests and static checks pass without production timing changes
- [x] #4 Settings-return journeys retain same-instance reuse and independently verify fresh reconstruction after explicit unmount, including exact claim release and worker cancellation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-application-session-state-ownership.md
Reason: Test-only synchronization and existing handoff ownership; upstream TASK31520 already changes Console navigation to reuse. ADR033 fresh-screen wording predates that upstream change and is not rewritten here.
1. Inspect full-file failures and reproduce focused variants.
2. Keep injected settlement failure armed until explicit retry and await current prompt controls instead of fixed pauses.
3. Cover the credential return through both ordinary same-instance navigation and explicit installed-screen disposal; use real removal for unmount/cancellation subjects and release held test gates in finally.
4. Run focused races, the full native flow file, and static checks; retain all ownership and mutation assertions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Kept atomic settlement faults armed through explicit retry and waited for freshly queried, attached prompt controls. Adapted the credential journey to verify both same-instance reuse and fresh reconstruction after explicit cached-screen uninstall/remove, preserving store identity and handoff/status assertions. The cancellation fixtures now distinguish suspension from actual removal: suspension releases the exact claim without cancelling the held worker; real removal cancels it and cannot resurrect a superseded or settled handoff. All held test gates release in finally to prevent teardown hangs. ADR033 governs handoffs; its fresh-screen wording predates upstream TASK31520 reuse and was not rewritten in this test-only task. Original post-rebase selection: 32 passed/3 stale lifecycle failures. Focused lifecycle variants: 4 passed. Final complete native-flow file: 349 passed in 552.43s. Ruff lint and changed-region formatting passed; root reviewed the scoped diff and self-review is complete. No production changes or new ADR.
<!-- SECTION:NOTES:END -->
