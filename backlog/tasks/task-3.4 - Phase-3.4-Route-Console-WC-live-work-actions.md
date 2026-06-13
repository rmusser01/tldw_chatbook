---
id: TASK-3.4
title: 'Phase 3.4: Route Console W+C live-work actions'
status: Done
assignee: []
created_date: '2026-05-03 22:23'
updated_date: '2026-05-03 22:27'
labels:
  - unified-shell
  - phase-3
  - console
  - watchlists
dependencies: []
parent_task_id: TASK-3
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Console live-work status card action for W+C watchlist run launches route to the existing W+C run detail surface, so the Console card supports recovery follow-through instead of only displaying static action copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console live-work status-card state exposes an actionable primary button only when the launch payload has a supported route context.
- [x] #2 Clicking the W+C Console action stages the selected watchlist run context and navigates to the W+C runs surface.
- [x] #3 Unsupported launch payloads remain non-actionable status cards with recovery copy.
- [x] #4 Focused automated tests and Phase 3 QA evidence verify the `console-live-work-primary-action` route wiring and roadmap/task updates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for actionable Console live-work card state, W+C button click navigation, unsupported payload fallback, and Phase 3.4 evidence tracking.
2. Extend the Console live-work card contract with optional primary action metadata derived only from supported payload contexts.
3. Render a Console action button when action metadata exists and route W+C watchlist run launches through the existing subscription watchlist run context.
4. Add Phase 3.4 QA evidence plus roadmap and task updates.
5. Run focused Console, navigation, W+C detail, and diff hygiene verification before commit/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a typed Console live-work primary action contract that only resolves W+C watchlist run payloads with a supported `target_id`. The Console status card now renders `console-live-work-primary-action` for those launches, and the app routes the action through the existing subscription watchlist-run context before navigating to W+C. Unsupported payloads remain non-actionable with recovery copy. Added focused regression tests plus Phase 3.4 roadmap, evidence, and Backlog tracking.
<!-- SECTION:NOTES:END -->
