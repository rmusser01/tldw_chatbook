---
id: TASK-37270
title: Gate real project-chat journeys with an executable UAT pilot
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-28 23:58'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent project-chat feature failures from passing component-only checks by exercising the real application journey and retaining actionable evidence. Publish a wider workflow catalog and an initial deterministic project-chat pilot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The reviewed catalog defines user actions, independent success criteria, and evidence for core journeys while clearly labeling unimplemented coverage.
- [ ] #2 An isolated real-app pilot creates a workspace, binds a fixture folder, starts a scoped chat through the UI, and reads actual project evidence through the production provider and tool path.
- [ ] #3 Fenced, native, compact schema recovery, and wrong-family argument recovery variants validate the final assistant answer and durable run; negative controls reject false success.
- [ ] #4 Only the owned loopback provider can be contacted; private profile isolation, bounded waits, and failure/cleanup artifacts are verified.
- [ ] #5 The exact pilot variants are enrolled in an existing required CI lane with complete no-skip result checking and retained diagnostic artifacts after targeted qualification.
- [ ] #6 Documentation records exact commands and measured platform coverage, distinguishing scripted Windows/Linux runs from unperformed live Fedora qualification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-069/174/241/253 apply. Review and tighten the design, then write the implementation plan. Build the HTTP script and independent oracle tests first, exercise real startup and UI creation/send in the pilot, qualify negative controls, and enroll exact cases with retained artifacts in the existing CI gate. No full-suite run or live model is part of this first pilot.
<!-- SECTION:PLAN:END -->
