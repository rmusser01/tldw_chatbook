---
id: TASK-14
title: 12-screen actual screenshot QA campaign
status: Done
assignee: []
created_date: '2026-05-09 03:44'
updated_date: '2026-05-09 03:45'
labels:
  - ux
  - screen-qa
  - product-maturity
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-05-08-12-screen-screenshot-qa-campaign.md
  - Docs/superpowers/specs/2026-05-08-12-screen-screenshot-qa-campaign-design.md
  - >-
    Docs/superpowers/specs/2026-05-08-destination-visual-parity-correction-design.md
  - Docs/superpowers/trackers/product-maturity-roadmap.md
  - Docs/superpowers/qa/product-maturity/screen-qa/README.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Coordinate the actual rendered screenshot QA campaign for all 12 top-level destination screens. This parent tracks the campaign-level evidence and merge sequence; each child screen task owns one focused PR and requires user approval from an actual running-app screenshot before PR creation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Baseline screenshot evidence exists for each of 12 screens
- [x] #2 Final screenshot evidence exists for each of 12 screens
- [x] #3 User approval is recorded for each screen
- [x] #4 One PR is opened and merged per screen
- [x] #5 Campaign README and per-screen notes are updated
<!-- AC:END -->

## Implementation Plan

1. Run the 12-screen actual screenshot campaign one top-level destination at a time.
2. Capture baseline and final actual rendered screenshots for each screen.
3. Require explicit user approval of the actual final screenshot before opening each screen PR.
4. Keep per-screen notes, Backlog child tasks, and the campaign README synchronized.
5. Merge each focused screen PR before advancing unless explicitly overridden.

## Implementation Notes

Completed the 12-screen actual screenshot QA campaign across Console, Home, Library, Artifacts, Personas, Watchlists, Schedules, Workflows, MCP, ACP, Skills, and Settings. Each screen has baseline and final actual screenshot evidence, explicit user approval, focused verification, and a merged PR. Campaign tracking was reconciled after Settings PR #306 merged so the README and child Backlog tasks reflect the final merged state.
