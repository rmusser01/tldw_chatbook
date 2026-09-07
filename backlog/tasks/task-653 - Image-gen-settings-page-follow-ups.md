---
id: TASK-653
title: >-
  Image Gen settings page follow-ups
status: To Do
assignee: []
created_date: '2026-07-25 21:30'
updated_date: '2026-07-25 21:30'
labels:
  - image-generation
  - settings
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-blocking follow-ups from the Settings ▸ Image Gen page reviews (spec `Docs/superpowers/specs/2026-07-25-image-gen-settings-page-design.md`; final whole-branch review + live smoke, 2026-07-25). None are shipped defects — Critical/Important findings were fixed pre-PR. Also tracks the deferred v2 scope (style-template management UI).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Probing with an empty/unset base_url reports a precise badge (e.g. "No base URL configured") instead of the technically-true-but-confusing "Unreachable: blocked by egress policy" (currently unreachable from the UI because form values fall back to effective placeholders, so low priority).
- [ ] The probe stale-badge race (category switched away mid-probe; session-counter guard drops the stale callback) gets a regression test — the logic is trace-verified sound but untested.
- [ ] The default-backend Select disables (or visually marks) non-enabled backends instead of relying solely on the blocked-save message.
- [ ] A secret field emptied while dirty signals its pending deletion (placeholder/source-line update) instead of still reading "Local config key saved" until Save.
- [ ] Other Settings categories are audited for the DestinationHarness-has-no-CSS_PATH blind spot that hid this page's zero-width controls (a real-app-CSS width regression test now exists for Image Gen — extend the pattern or spot-check siblings).
- [ ] Terminals narrower than 120 cols: the backend row (worst case: the ★ Default row) either fits, wraps, or scrolls with an affordance — currently verified only at ≥120.
- [ ] v2: style-template management UI (create/edit/delete user templates) per the spec's deferred scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
