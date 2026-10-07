---
id: TASK-33625
title: 'Console unification: width-aware priority model so safety controls never clip'
status: To Do
assignee: []
created_date: '2026-09-30 03:04'
labels:
  - console
  - ux-review-2026-09-29
  - unification
dependencies: []
references:
  - qa/console-ux-review-2026-09-29/report.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: the Console has six horizontal regions, and each one degrades differently when width runs short. In every one of them the most important item goes first. The status strip scrolls in a fixed order, cuts values mid-token and can show a wrong value, so Approvals, cost and sources disappear below 160 columns. The footer keeps a fixed prefix and strips labels. The composer row's fixed budget clips Stop out of view for every run (P0), and there is no key, palette or slash alternative. The two rails use separate budgets that are never combined, have no hysteresis and leave a dead zone around 118-128 columns. The tab strip hides the active tab. The approval card clips Deny at 80x24. The focused Terminal cuts its first column and shrinks to a 44x6 viewport.

What unifying means here: one width-aware priority model shared by every horizontal region (status strip, footer, composer row, tab strip, rails, approval card, Terminal). Each item declares a priority, a compact form and a visibility condition. The lowest priority is dropped first. Nothing is clipped mid-token, and safety controls (Stop, Deny, Approvals, run state) are never clipped. A focusable '+N' overflow always appears when something is hidden. The rails share one combined budget that keeps a transcript floor and uses enter and exit margins (hysteresis). Geometry tests at 80, 100, 120, 160 and 235 columns assert on rendered regions, not on constants. This umbrella is the parent of the theme's P0/P1 fixes (G1-01, GAP4-07, GAP4-20).

Evidence: qa/console-ux-review-2026-09-29/report.md (theme 'Width is not budgeted by priority') and qa/console-ux-review-2026-09-29/findings.md / findings.json.

Related P2/P3 in the ledger: G1-05, G1-22, G1-21, G1-18, G1-34, G4-15, G4-35, G4-37, G4-34, GAP2-13, G4-64, G1-23, G4-36, GAP1-10, G1-45, G4-38, G1-42, G2-31.

Source: Console UX review 2026-09-29 — qa/console-ux-review-2026-09-29/report.md (themes) and findings.md (ledger).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 80, 100, 120, 160 and 235 columns, Stop (during a run), Deny (on an approval card), the Approvals indicator and the run state are fully visible and operable, verified by geometry tests on rendered regions
- [ ] #2 No Console horizontal region cuts a label or value mid-token at any tested width; when items do not fit, the lowest-priority items are dropped first and a focusable overflow control reveals them
- [ ] #3 The active tab is always visible in the tab strip at every tested width and tab count
- [ ] #4 The two rails never together reduce the transcript below its minimum width, and resizing back and forth across a threshold by one column does not toggle a rail
- [ ] #5 The focused Terminal shows its full first column at every tested width
- [ ] #6 All P0/P1 child tasks under this umbrella are Done
<!-- AC:END -->
