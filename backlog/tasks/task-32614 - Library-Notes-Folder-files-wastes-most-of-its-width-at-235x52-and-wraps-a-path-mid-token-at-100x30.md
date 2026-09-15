---
id: TASK-32614
title: >-
  Library Notes: Folder files wastes most of its width at 235x52 and wraps a
  path mid-token at 100x30
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-15 06:41'
updated_date: '2026-09-15 18:27'
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
- [x] #1 At 235x52 the Folder-files tree pane fills the height it claims, or claims less
- [x] #2 At 100x30 the two panes divide the width so that a file path is not wrapped mid-token
- [x] #3 The breakpoint behaviour that works at 60x24 is applied at 100x30 as well
- [x] #4 Layout assertions cover all three widths for this pane
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the Folder-files pane at 235x52, 100x30 and 60x24 through the production Library harness -- regions, not guesses.
2. Find what owns the blank rows and give it the height it actually costs.
3. Decide the 100x30 pane split from the resolver's own rule rather than a new breakpoint.
4. Stop both identity lines folding a path mid-token.
5. Pin all three widths, and see every pin fail first.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three findings, one cause each, all measured through the production Library harness (`_production_workspace_context`) rather than eyeballed.

**The blank rows are a default.** `#file-notes-tree-header` is a `Horizontal`, and Textual's `Horizontal` defaults to `height: 1fr` -- so the navigator's title row and the tree split the pane's spare rows between them. Measured before the fix: 19 rows at 235x52, 9 at 100x30, 6 at 60x24, all but one blank. That is the capture's 'rows 6 through 26 are entirely blank, the Search box sits at row 27 and the tree begins at row 30' to the row (search row measured at y=27, tree at y=30). One CSS rule (`height: auto; min-height: 1`) gives those rows to the tree: 20 -> 38 visible tree rows at 235x52, 10 -> 18 at 100x30.

**The 100x30 split was an outlier profile, not a missing breakpoint.** `LIBRARY_FILE_NOTES_READER_PROFILE` carried `work_min_width=30` -- the lowest floor of any Library destination (every other is 44-48) -- which is what let the resolver divide 100 columns 56/34. Dropping the override for the shared default (44) makes it 46/44. Checked across widths with the pure resolver before changing anything: 235x52 and 60x24 are byte-identical either way; the only other change is that 80-89 columns now drops the list pane, which is what Notes, Prompts, Skills and Collections already do at that width. AC#3 is answered by applying that floor -- the same rule that collapses 60x24 -- rather than by making 100x30 narrow: the resolver, not a second breakpoint, is where this destination disagreed with its peers.

**Neither identity line survived a real path.** `#file-notes-breadcrumb` and `#file-notes-exact-path` both had `height: auto` with default wrapping, and a path is one long token, so Textual folded them wherever the column ran out: measured 6 rows for the absolute path at 100x30 broken at '...m80n2j152t'/'9gw3w8qwk...', and 4 rows for the breadcrumb broken at '2026'/'-09-14'. Both are now one row, nowrap, and fitted by `_fit_path_surfaces` through the existing `elide_path_middle` (basename kept -- a tail cut spends the row on '/private/var/folders/...' and hides the name).

Two things that cost a round and are worth knowing:
* the breadcrumb's width is NOT the work pane's -- it shares its row with the Edit/Manage chips (31 of 49 cells at 60x24) until `-stack-editor-actions` gives it the row to itself (50). A value fitted to the wrong one of those loses the basename;
* that width therefore MOVES during settling, so the fit memoises the widths it used and re-fits once after a refresh when they change, stopping as soon as two consecutive fits agree. A one-shot 'fit when unmeasured' guard was not enough and was seen failing.

**Modified:** `Widgets/Library/library_file_notes_workspace.py` (tree-header + path-line CSS, `_breadcrumb_copy`/`_exact_path_copy`/`_fit_path_surfaces`/`_refit_path_surfaces`, five raw writers routed through it, standalone-harness profile), `UI/Library_Modules/screen_constants.py`, `Tests/UI/test_library_notes_w5_folder_files_layout.py` (new).

**Red first:** 'the title row took 19 rows at (235, 52)'; '#file-notes-exact-path took 2 rows at (235, 52)'; '#file-notes-breadcrumb took 4 rows at (100, 30)'; `work_min_width == 44` assert failed at 30.
<!-- SECTION:NOTES:END -->
