---
id: TASK-32621
title: >-
  Library Notes: Folder files asserts Saved over no file selected and silently
  drops non-note entries from its tree
status: To Do
assignee: []
created_date: '2026-09-15 06:43'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A section 11 and assessor B section 6f, persona Riley, Obsidian workflow.

What happened. With no file selected, the Folder-files right pane shows the chip 'Saved' and a fully drawn, apparently editable body box beside the words 'No file selected' (A cap 29) -- a save state asserted over nothing, on the one surface whose entire promise is that it edits the real file. And the tree silently omits notes.csv, meta.yaml, a Canvas folder and an attachments folder, with no line explaining what is shown (A cap 29; B confirms the same omissions at cap 35 and reads them as correct behaviour -- which is the point: the behaviour is right and unexplained).

Related asymmetry worth stating in the same place: lasting sync silently ignores the same .csv and .yaml sources that Import once at least reports under Failed and Skipped, and the sync review's '0 need attention' hides it (A section 11).

Cause PROVEN by capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With no file selected the right pane shows no save state and no editable body
- [ ] #2 The Folder-files tree states what it lists, so a missing file is explained rather than silently absent
- [ ] #3 The two vault-reading paths agree on whether an unreadable source is reported or ignored, or each says which it does
<!-- AC:END -->
