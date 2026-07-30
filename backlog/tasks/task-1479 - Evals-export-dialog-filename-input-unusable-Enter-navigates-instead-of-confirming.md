---
id: TASK-1479
title: >-
  Evals export dialog: filename input unusable, Enter navigates instead of confirming
status: Done
assignee: []
created_date: '2026-07-30 10:00'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by live UAT (2026-07-30) at 235x52. Export (`e`) opens `Third_Party.textual_fspicker.FileSave`, where three failures compounded: the filename input rendered ~5 cells wide and empty (the pre-filled default `<bench>.json` was invisible), Enter with list focus navigated directories (activated `..`) instead of confirming, and text typed after clicking the input never appeared. A keyboard-driven export could not be completed at all — despite `e` being a keyboard affordance advertised in the footer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] The filename input is wide enough to show a typical default filename, and the pre-filled default is visible on open
- [x] For a save dialog, initial focus lands in the filename input and Enter there confirms the save
- [x] The keyboard-only path `e` then Enter exports with the default name into the dialog's directory
- [x] Directory navigation in the file list still works (Enter on a directory still navigates)
- [x] A test or scripted verification covers the keyboard-only export path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the three dialog failures with painted-geometry tests under the real CSS bundle
2. Fix width at bundle tier, FileSave mount focus, and the ambiguous Input query
3. Document deviations in the vendored package's ENHANCEMENTS.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Commit 393fb7dab. Three defects, not two: (1) the filename input collapsed to ~5 cells because features/_conversations.tcss ships a bare `Select { width: 100%; }` that outranks widget DEFAULT_CSS through the bundle — fixed with a scoped `FileSave/FileOpen InputBar` rule in components/_dialogs.tcss (plus a defensive DEFAULT_CSS change for non-bundle consumers); (2) FileSave now focuses the filename input on mount (FileOpen keeps list focus); (3) `self.query_one(Input)` in _select_file/_confirm_file silently matched the hidden #path-input — every filename read is now InputBar-scoped, which alone had made Enter-to-confirm impossible. Five new tests assert painted geometry (region.width) under the real bundle and the keyboard-only e-then-Enter path. Verified live: full-width input with visible default, Enter exported a 20.6KB JSON with 4 cells. Latent sibling risk filed as task-1514 (EnhancedFileDialog unpinned).
<!-- SECTION:NOTES:END -->
