---
id: TASK-382
title: Fix the tab-bar tooltip painting borderless text over the Console header
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With the pointer resting over the '1 Home' tab, its tooltip ('Open dashboard, notifications, and active work.') renders as bare text mixed into the Console header area, producing interleaved fragments like 'Open dashboard, notifications, and / active work. / control actions.' on screen.

**Repro:** Hover the mouse over the '1 Home' tab in the top bar and look at the Console header block beneath it.

**Verifier note:** Confirmed in j3-79: the '1 Home' tab tooltip ('Open dashboard, notifications, and active work.') renders as bare borderless text over the Console header, leaving the interleaved fragment 'control actions.' — the Tooltip surface lacks an opaque bordered style in the app theme. Not a harness artifact (it is the app's own tooltip rendering) and not covered by any ledger item or task. P3 correct.

**Source:** Console UX expert review 2026-07-20 (finding j3-tab-tooltip-garbles-header; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J3 attachments journey. Evidence: `j3-79-view-modal.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tooltips need an opaque bordered surface that fully covers (or avoids) underlying text
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: Textual's default `Tooltip` is borderless and fills with `$panel`,
so over the Console header the text bled into the widget underneath as
interleaved fragments. Added a global `Tooltip` rule to utilities/_overrides.tcss
giving it a `round $primary` border + explicit opaque `$panel` fill + `$text`
color, so the tooltip is a distinct surface that fully COVERS what it overlaps.
Verified: computed border_top=('round', cyan), background alpha=1.0. Fixed
alongside task-386 (same surface). CSS-source contract test + regenerated bundle.
<!-- SECTION:NOTES:END -->
