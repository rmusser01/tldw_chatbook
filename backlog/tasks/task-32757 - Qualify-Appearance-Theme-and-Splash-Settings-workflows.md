---
id: TASK-32757
title: Qualify Appearance Theme and Splash Settings workflows
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-17 21:19'
updated_date: '2026-09-17 22:01'
labels:
  - ui
  - settings
  - design-system
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the design-system feature review for Interface settings so users can edit appearance defaults, manage themes, and preview splash cards with reliable keyboard access and visible controls at wide and compact terminal sizes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Appearance draft validation, preview, Save, Revert and navigation preserve their existing persistence boundaries.
- [ ] #2 Theme browse, edit, preview, Apply, Save and launch-default workflows keep editor state, runtime theme and persisted preferences distinct.
- [ ] #3 Splash controls and previews remain usable by keyboard and accurately reflect successful or failed instant preference writes.
- [ ] #4 Production CSS keeps reviewed controls and actions visible in dark and light themes at wide and compact sizes; confirmed defects are fixed with token-backed styles.
- [ ] #5 Targeted tests and native private-profile journeys record persistence, rendering and lifecycle evidence; the completion ledger and PR are updated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; existing ADR-150 and ADR-161
Reason: Qualifies and repairs existing Interface settings within their established commit models and token/component system; no new ownership, persistence or application boundary.
1. Read prior Appearance/Theme tasks, inspect all three Interface categories, and reproduce baseline failures.
2. Move affected reviewed tests to pre-import private profiles, preserving their behavioral assertions.
3. Exercise real Settings keyboard, persistence, preview and compact geometry journeys; add regressions for confirmed defects and apply minimal fixes.
4. Run targeted regressions, token/build checks, scoped static checks and native dark/light wide/compact journeys with private-profile lifecycle receipts.
5. Review the diff, update user guidance and the completion ledger, then commit and push to PR #2704.
6. Reconcile four incoming dev commits through c97a64eba5 before the final push: preserve both ingestion guide additions, review automatic code merges, and run focused ingestion/clipboard regressions and artifact guards.
<!-- SECTION:PLAN:END -->
