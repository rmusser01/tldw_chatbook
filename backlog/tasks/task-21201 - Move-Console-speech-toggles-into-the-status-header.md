---
id: TASK-21201
title: Move Console speech toggles into the status header
status: To Do
assignee:
  - '@codex'
created_date: '2026-08-23 21:21'
labels:
  - console
  - ui
  - speech
dependencies:
  - TASK-3070.10
references:
  - Docs/superpowers/specs/2026-08-23-console-auto-speak-ownership-and-header-controls-design.md
parent_task_id: TASK-3070
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Speak replies and Hands-free controls visible beside the Console status while reclaiming the former control-bar row and preserving narrow-terminal usability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Speak replies and Hands-free remain on the same row immediately left of the Console status at every supported width
- [ ] #2 The Console subtitle truncates before the title, speech controls, or status, and Ready remains right-aligned
- [ ] #3 Compact-height mode keeps the speech controls reachable without reducing the normal transcript/composer vertical budget
- [ ] #4 Retry and Resume speech recovery remain reachable without crowding the header
- [ ] #5 Tab and F6 navigation include the relocated controls
- [ ] #6 Focused geometry, interaction, and speech-control tests pass at 60-column and representative wide layouts
<!-- AC:END -->
