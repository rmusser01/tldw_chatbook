---
id: TASK-32757
title: Qualify Appearance Theme and Splash Settings workflows
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-17 21:19'
updated_date: '2026-09-17 22:16'
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
- [x] #1 Appearance draft validation, preview, Save, Revert and navigation preserve their existing persistence boundaries.
- [x] #2 Theme browse, edit, preview, Apply, Save and launch-default workflows keep editor state, runtime theme and persisted preferences distinct.
- [x] #3 Splash controls and previews remain usable by keyboard and accurately reflect successful or failed instant preference writes.
- [x] #4 Production CSS keeps reviewed controls and actions visible in dark and light themes at wide and compact sizes; confirmed defects are fixed with token-backed styles.
- [ ] #5 Targeted tests and native private-profile journeys record persistence, rendering and lifecycle evidence; the completion ledger and PR are updated.
- [x] #6 Integration with current dev preserves ingestion option validation and editor identity, including clearing a repaired provider warning.
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
7. Four incoming structural tests reproduce the documented raw_source_selection_changed fixture failure. Use the established pre-import private-profile helper for those cases, retain their assertions, and rerun the affected integration selection.
8. Resolve the reviewed integration interaction between incoming STT Select validation and existing in-place option synchronization; verify invalid-to-Auto recovery clears its warning while retaining other editor identities.
9. The expanded option identity/event-order selection reproduced the same pre-import profile failure in 20 parameter cases (four functions). Apply only the existing isolation wrapper, preserve all original assertions, and rerun those cases.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Appearance, Theme and Splash keep controls and complete values visible at wide/compact sizes through token-backed styles. Splash saves use the correct effects section, distinguish file/cache outcomes, and retain focus/newer input. Theme launch-default writes rebase Appearance without discarding explicit drafts. Updated guidance and review ledger; 98 affected Settings cases, 32 governance cases and four final native journeys with 28 inspected captures pass. Native exit/isolation and 11 private database integrity checks pass.
Reconciled four dev commits through c97a64eba5, preserving both Import guide additions. Fixed the discovered Select-warning interaction with in-place option updates; 102 distinct integration tests pass after repairing confirmed pre-import profile fixtures. Original assertions retained. All seven preflight checks pass; no new lint/format debt, scoped fatal checks and follow-up CSS/inventory checks pass. Independent review accepted the fixes. QA and limits: Docs/superpowers/qa/2026-09-17-settings-interface/README.md.
ADR required: no; follows existing ADR-033 (Settings commit models), ADR-150 and ADR-161. Integration repairs preserve existing boundaries. A semantic auto-merge testing lesson is recorded in backlog/docs/lessons-testing-evidence.md. Full-suite, external-provider and full animation qualification are not claimed.
<!-- SECTION:NOTES:END -->
