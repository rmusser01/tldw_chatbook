---
id: TASK-21201
title: Move Console speech toggles into the status header
status: Done
assignee:
  - '@codex'
created_date: '2026-08-23 21:21'
updated_date: '2026-08-23 22:52'
labels:
  - console
  - ui
  - speech
dependencies:
  - TASK-3070.10
references:
  - >-
    Docs/superpowers/specs/2026-08-23-console-auto-speak-ownership-and-header-controls-design.md
parent_task_id: TASK-3070
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Speak replies and Hands-free controls visible beside the Console status while reclaiming the former control-bar row and preserving narrow-terminal usability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Speak replies and Hands-free remain on the same row immediately left of the Console status at every supported width
- [x] #2 The Console subtitle truncates before the title, speech controls, or status, and Ready remains right-aligned
- [x] #3 Compact-height mode keeps the speech controls reachable without reducing the normal transcript/composer vertical budget
- [x] #4 Retry and Resume speech recovery remain reachable without crowding the header
- [x] #5 Tab and F6 navigation include the relocated controls
- [x] #6 Focused geometry, interaction, and speech-control tests pass at 60-column and representative wide layouts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm no open PR overlaps the Console header/control-bar files and inspect current production composition, CSS, and navigation contracts.
2. ADR required: no; ADR path: N/A; reason: routine Console UI layout polish preserving existing state/event/application boundaries.
3. Add production-CSS geometry, recovery-height, navigation, and interaction tests first and record the expected red failures.
4. Add the focused ConsoleSpeechControls widget, DestinationHeader before_status seam, and late-bound wiring while preserving message IDs and silent programmatic sync.
5. Make ConsoleControlBar the sole dynamic-height owner, preserve focus mode, keep the header in compact mode, and regenerate CSS.
6. Run only focused tests, Ruff/format, CSS parity, diagnostic inventory, diff checks, and live rendered-frame verification at 60x18, 80x24, and 140x42.
7. Self-review, complete backlog notes/criteria, then commit, push, open a separate PR, address Qodo, rebase on latest dev, wait for required checks, and merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved the existing Speak replies and Hands-free event contracts into a focused one-row header widget and added an optional pre-status slot to the shared destination header. The subtitle now owns all width compression, the speech controls keep intrinsic width with a two-cell status gap, and compact-height mode retains the header while the lower action bar uses one row normally and a second row only for retry/resume recovery. Existing state owners, event handlers, focus mode, and speech recovery behavior remain unchanged; Tab and F6 now include the relocated switches. Added production-bundle geometry coverage from 60 columns through wide layouts, silent-sync and single-event interaction tests, recovery-height and navigation coverage, and isolated the Hands-free UI test from native audio/model initialization. ADR required: no; ADR path: N/A; this is layout polish within existing application and event boundaries.
<!-- SECTION:NOTES:END -->
