---
id: TASK-31981
title: 'Library: a blocked action explains itself only on mouse hover, not inline'
status: Done
assignee: []
created_date: '2026-09-07 22:48'
updated_date: '2026-09-08 01:52'
labels:
  - library
  - media
  - ux
  - accessibility
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #6 P1, both assessors. Clicking a disabled `○ Generate` in the reader's Analysis pane produces a byte-identical screen; the reason (`No analysis provider is configured.`) appears only on mouse hover, unreachable by a keyboard-first user. The same silence covers `○ Analyze` in select mode. The Export gate already does this correctly two clicks away: `No destination chosen` is printed directly beneath the blocked `○ Export bundle (.zip)`. The product contract says explain why unavailable actions are unavailable, and the design doc says use recovery callouts instead of silent disabled controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A blocked Generate/Analyze action shows its reason as an always-visible line adjacent to the control, not only in a hover tooltip
- [x] #2 The reason names a next step where one exists (e.g. Set a provider in Settings)
- [x] #3 The pattern matches the existing Export gate's inline-reason treatment
- [x] #4 A painted pin asserts the reason text is present on the screen with no hover
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both blocked actions already resolved their reason through one seam
(`_library_media_analysis_provider_reason` → `analysis_unavailable_reason`)
and surfaced it only as a tooltip. Fix: render it inline too.

- **Viewer** (`Widgets/Library/library_media_viewer.py`): after the Generate
  toolbar, yield a `Static(reason, id="library-media-analysis-generate-reason",
  classes="library-media-action-reason")` when a reason is set.
- **Select-mode Analyze** (`Widgets/Library/library_media_canvas.py`): after
  the analyze row, yield the same-class `Static`
  (`id="library-media-analyze-selected-reason"`).
- **Next step (AC#2)** (`Library/ingest_analysis.py`): the no-provider case of
  `analysis_unavailable_reason` now returns "No analysis provider is configured
  · Set one in Settings ▸ Providers & Models." (established app grammar; the
  Settings label verified by grep, not invented). The provider-not-ready branch
  stays bare — its fix depends on the specific gap. Persisted job `short_reason`
  and the ingest `hint` are untouched.
- **CSS**: `.library-media-action-reason` added to
  `css/components/_agentic_terminal.tcss` (mirrors `.library-export-quiet-line`),
  bundle rebuilt.
- **Precedent matched**: the Export gate's `Static` "No destination chosen"
  line under `○ Export bundle (.zip)` (`library_export_canvas.py:174`).
- **TDD**: painted-text pins at 235x52 and 100x30 in
  `Tests/UI/test_library_media_render_fixes.py` — red (NoMatches on the reason
  id, tooltip-only) → green. Pre-existing string assertions updated to the
  enriched copy. 7 unrelated render_fixes reds are a baseline `█` scrollbar
  artifact (confirmed identical on baseline).
- **Docs**: `Docs/User_Guide/library/media-and-conversations.md` updated + stamp.

## Renumbering provenance

Filed as TASK-31977 during critique #6's fix wave; renumbered to TASK-31981 because a concurrent session landed its own TASK-31977 on dev first (2026-08-21 owner rule, TASK-19601: older arrival keeps the id). No other task references this one.
<!-- SECTION:NOTES:END -->
