---
id: TASK-32614
title: >-
  Library Notes: Folder files wastes most of its width at 235x52 and wraps a
  path mid-token at 100x30
status: To Do
assignee: []
created_date: '2026-09-15 06:41'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D7, persona Alex, Obsidian workflow.

What happened. At 235x52 (B cap 35) the file-tree pane is about 55 columns wide and rows 6 through 26 are entirely blank -- the Search box sits at row 27 and the tree begins at row 30, pinned to the bottom. Twenty of the pane's rows are dead space while the tree is cut off at ten visible nodes. At 100x30 (B cap 47) the tree pane still claims about 58 of 100 columns for three rows of content while the detail pane is squeezed to about 40 columns and wraps a file path mid-token across three lines.

The 60x24 layout collapses the tree to a vertical rail and gives the detail pane the full width with a back cue (B cap 48) -- correct responsive behaviour -- which is what makes the two failures hard to excuse: the mechanism exists and is simply not engaged at these widths.

Cause INFERRED (CSS not traced). Not re-observed by A, who did not exercise 100x30. Wave 4's Folder-files work (task-32552, PR #2682) covered keys, Git truth and hidden folders, not geometry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 235x52 the Folder-files tree pane fills the height it claims, or claims less
- [ ] #2 At 100x30 the two panes divide the width so that a file path is not wrapped mid-token
- [ ] #3 The breakpoint behaviour that works at 60x24 is applied at 100x30 as well
- [ ] #4 Layout assertions cover all three widths for this pane
<!-- AC:END -->
