---
id: TASK-386
title: Prevent hover tooltips occluding the content they describe on setup surfaces
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Test Provider tooltip ('Run a local readiness check...') painted over the provider-test result line while it was being read, garbling both (j1-15). The Save-selected tooltip ('Append selected discovered model IDs to the local provider list.') covered the discovered-model checkbox row itself — while the cursor rests on the very button that acts on that row (j1-25). A tab tooltip similarly covered the left half of the tab bar. Tooltips are also the only carrier of key guidance like 'URL-based local providers also get a short live endpoint probe', invisible to keyboard users.

**Repro:** Hover 'Test Provider' after clicking it -> tooltip overlaps result text; hover 'Save selected' after discovery -> tooltip covers the model row; park mouse on the Home tab -> tooltip covers tab labels.

**Verifier note:** No prior art in the ledger or backlog (searched 'tooltip' across tasks; the phase plans specify tooltip CONTENT, never placement). The observations are credible from the captures (Test Provider tooltip over the result line j1-15, Save-selected tooltip over the model row j1-25) and follow from Textual's default cursor-adjacent tooltip placement — the app does not offset tooltips away from result/target content. The strongest sub-point is hover-only guidance (e.g. the live-probe explanation exists only as tooltip text), invisible to keyboard users. Downgraded P2→P3: transient, mouse-only, self-dismissing on move; a placement/static-copy polish item rather than a flow blocker.

**Source:** Console UX expert review 2026-07-20 (finding j1-tooltip-occludes-content; P3, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J1 new-user cold start journey. Evidence: `j1-15-test-bad-url-result.png`, `j1-25-save-selected.png`, `j1-18-discover-bad-url-result.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tooltips should position away from the control's related content (and never over the data a click just produced)
- [ ] #2 Essential explanations should exist as static text, not hover-only
<!-- AC:END -->
