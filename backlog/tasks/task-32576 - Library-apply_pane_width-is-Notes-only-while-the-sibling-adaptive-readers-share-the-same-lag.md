---
id: TASK-32576
title: >-
  Library: apply_pane_width is Notes-only while the sibling adaptive readers
  share the same lag
status: To Do
assignee: []
created_date: '2026-09-14 22:45'
labels:
  - library
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recorded by wave-4 group 8 (task-32544 / task-32557). A pane that narrows used to keep the toolbar shape the previous, wider size chose, so shrinking the terminal left labels painted against the grip ('Add from' at 60 columns). apply_pane_width fixes that for Library Notes by pushing the contract width into the canvas on a shrink. The sibling adaptive readers — Media, Conversations, Prompts, Skills — resolve their layouts the same way and were not swept, so the same lag is presumably live on each of them and nobody has looked. Also unswept: toolbar_action_rows covers only the FOLDER actions; the browse and transfer rows are still measured for their current labels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each sibling adaptive reader is checked for the same shrink lag, at 235 to 60 columns, and the finding recorded per reader
- [ ] #2 Readers that have the lag get the same treatment, or a recorded reason why they do not need it
- [ ] #3 The browse and transfer toolbar rows are covered by toolbar_action_rows, or the gap is recorded with what it costs
- [ ] #4 Shrink-only remains deliberate and documented (growth records the width without re-shaping), and the round-trip pin still passes
<!-- AC:END -->
