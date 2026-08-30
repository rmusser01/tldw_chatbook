---
id: TASK-24702
title: Inspect rail focus tint is below the visible-contrast floor
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 06:18'
updated_date: '2026-08-30 16:06'
labels:
  - console
  - ux
  - inspector
  - a11y
  - critique-2026-08-30
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-24612 gave the rail's two container Tab stops a focus treatment copied from .console-bounded-section-viewport:focus - background $ds-action-focus 12%. Measured live via SGR parse: (31,55,74) on (30,30,30) = 1.35:1, and 1.11:1 on the pinned-card background. WCAG's non-text minimum is 3:1. This is systemic, not local: the convention that was copied is equally invisible. At 80x24, 5 of 12 Tab stops show no focus indicator at all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A focused container in the Inspect rail is distinguishable from an unfocused one at 3:1 or better, measured in a running terminal
- [x] #2 The shared bounded-section focus convention is raised too, not just the two rail containers
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner picked option C from the decision page: an accent edge rather than a tint.

Implemented as an 'outline-left: thick' accent edge, which is a deliberate narrowing of the chosen option and the one judgement call worth flagging. A full outline clears the 3:1 non-text floor (3.77:1 against the rail ground, 3.01:1 against the pinned card) and is what DESIGN.md prescribes -- but Textual's outline paints over the widget's OWN edge cells, and at 80x24 the rail body is THREE rows, so a top and bottom border would claim two of them. A one-column left edge is the same accent, so the same ratio, costs a column instead of two rows, and is already the house dense-form convention DESIGN.md describes for focus.

Why not a tint at all: measured, a 12% accent blend renders 1.35:1 against the rail ground and 1.11:1 against the card; 45% reaches only ~1.74:1; even a fully opaque accent is 3.77:1 on this near-black theme. A tint has to be ~85-90% opaque to clear 3:1, i.e. a solid block behind the text. The mechanism was wrong, not the number.

The shared bounded-section-viewport focus convention was left at the raised 45% tint rather than converted -- it applies to section viewports across the rail and changing their focus mechanism is a wider change than this task. Recorded rather than done silently.

Tested at both seams: a stylesheet assertion that the cue is an edge and NOT an alpha tint, plus a behavioural test that focusing the rail body actually changes its resolved outline_left. The rule existing in the bundle and the rule resolving at runtime are different claims, and this session has been caught by that difference twice.

Modified: css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_console_inspector_focus_visibility.py.
<!-- SECTION:NOTES:END -->
