---
id: TASK-16483
title: Console selection feedback actions phase 3
status: Done
assignee: []
created_date: '2026-08-16 05:27'
updated_date: '2026-08-17 03:29'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review feedback actions (Request changes / LGTM / Comment) on selections in agent output, routed as the next user message via the prompt queue
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Diff rows support line-granularity selection (unified-diff projection, snap to whole diff lines, reverse-video strip)
- [x] #2 Menu offers Request changes / LGTM / Comment only when the selection sits in agent output (widened per product decision 2026-08-16: assistant prose, tool, and diff rows; USER rows excluded)
- [x] #3 Without an active run, Request changes + LGTM render disabled with a visible hint; Comment stays enabled
- [x] #4 Comment modal collects an optional comment (empty submit sends without a comment; cancel/escape abandons)
- [x] #5 Feedback composes header + quoted selection + optional comment and routes via the prompt queue as the next user message (queues behind an active run; composer draft untouched)
- [x] #6 All selection/feedback/dismissal/transcript suites green; no new failures vs pre-existing baselines
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Per Docs/superpowers/plans/2026-08-15-console-selection-feedback-phase3.md (tasks 1-6: diff-row selection protocol → menu feedback entries + run gating → transcript wiring → comment modal → screen handler + prompt-queue dispatch → wrap-up)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
LIVE SPIKE ADDENDUM (2026-08-16, user-verified in a real terminal): four live rounds on feat/console-selection-feedback hardened the geometry and surfaced one pre-existing phase-1 layout bug, all fixed with RED->GREEN regression tests (233 selection/feedback tests green at tip 8ff227c15):

- 01fe56883: menu clamps within the OWNING TRANSCRIPT's box, never the bare screen (a bottom release painted it over the composer).
- e2dc272e4: compact single-row buttons + shrink guard (feedback menu 24 -> 9 rows; base 11 -> 5; fits boxes down to 6 rows by dropping border+hint).
- 2567e9a1e: ANSI-color mode keeps disabled feedback buttons borderless with labels (per-ID specificity over textual's ansi disabled rule).
- 395fd6882 + ee5a5dd9c: on bottom overflow the menu hops entirely above the selected row (then touching) so the reverse-video highlight stays visible; stale selection_top clamped to the owner box.
- d2b4d2630 (the 'black bar under the composer'): the screen-mounted menu consumed its own height from the screen's 1fr budget (textual 8.2.8 feeds position:absolute children's heights into the fr denominator) -> overlay:screen added; ADR-068 Amendment 3 records the rule (screen-mounted overlays need position:absolute AND overlay:screen). Root-caused via a temporary F12 whole-screen-children layout dump, removed in 8ff227c15. Lessons recorded in backlog/docs/lessons-live-verification.md.
<!-- SECTION:NOTES:END -->
