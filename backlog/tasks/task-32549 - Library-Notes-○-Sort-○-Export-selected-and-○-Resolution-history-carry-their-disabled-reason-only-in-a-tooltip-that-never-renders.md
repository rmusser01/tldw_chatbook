---
id: TASK-32549
title: >-
  Library Notes: "○ Sort", "○ Export selected" and "○ Resolution history" carry
  their disabled reason only in a tooltip that never renders
status: In Progress
assignee: []
created_date: '2026-09-13 06:47'
updated_date: '2026-09-14 16:00'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, persona Sam. D13. Task-32257 fixed Import selected items / Check selection through `_disabled_action_label`; task-32362 fixed Export bundle / Find; task-32261 AC#3 keeps the ○ glyph by design (Library-wide marker, task-32235). These three controls remain reason-less.

**What happened.** "○ Sort: Newest" while a filter shows (B 36, 37), "○ Export selected" with 0 selected (B 43), "○ Resolution history" (B 39) — glyph only; the reason lives in a tooltip that does not render in the TUI. Only "○ Server notes" carries its reason on screen. Captures: B 36, 37, 39, 43.

**Cause.** PROVEN pattern (reason on tooltip only), sites INFERRED.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the three controls states its disabled reason as text at the control through the shared _disabled_action_label seam
- [ ] #2 A test pins the three labels in their disabled states
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce '○ Sort: Newest' while filtered and '○ Export selected' with none selected at 235x52.
2. Measure whether each reason fits in its control's label at every pane width the control is reachable at.
3. State each reason on screen through ONE shared seam.
4. Pin the three disabled states.
<!-- SECTION:PLAN:END -->
